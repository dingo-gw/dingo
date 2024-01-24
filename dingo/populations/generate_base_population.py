import argparse
import copy
import hashlib
import textwrap
import time

import pandas as pd
import torch
import yaml
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

from dingo.core.models import PosteriorModel
from dingo.core.utils import fix_random_seeds
from dingo.gw.dataset import WaveformDataset
from dingo.gw.domains import build_domain
from dingo.gw.prior import build_prior_with_defaults, autocomplete_full_prior_dict
from dingo.gw.training import set_train_transforms
from dingo.gw.transforms import AddWhiteNoiseComplex, UnpackDict
from dingo.gw.waveform_generator.waveform_generator import build_waveform_generator
from dingo.populations.population_dataset import BasePopulationDataset


def generate_base_population(
    event_model_path,
    asd_dataset_path,
    size,
    batch_size=None,
    device="cuda",
    num_workers=0,
):
    """
    Generate a "base" population, meaning a population drawn from a Dingo model prior:
        (1) Sample event parameters from the prior.
        (2) For each set of event parameters, simulate detector data.
        (3) For each set of detector data, generate a Dingo embedding.
    These tasks are carried out based on a Dingo event model and its data generative
    process, all of which is contained within the model file and its metadata.

    Later, we will sample subpopulations based on a population likelihood. These
    subpopulations will contain weighted events.

    Parameters
    ----------
    event_model_path : str
        Dingo single-event model. This determines the population prior,
        event generation, and embedding for each event.
    asd_dataset_path : str
        Path to an ASD dataset. This is needed for simulating noise for events.
    size : int
        Desired population size
    batch_size : int
    device : str
        "cuda" or "cpu"
    num_workers : int

    Returns
    -------
    BasePopulationDataset
    """
    event_model = PosteriorModel(
        model_filename=event_model_path, device=device, load_training_info=False
    )
    # Make sure this is not a GNPE network.
    event_model.set_embedding_only()

    # (1) Build a waveform dataset / dataloader. This should generate samples in real
    # time during population generation.

    waveform_dataset_dict = {
        "settings": copy.deepcopy(event_model.metadata["dataset_settings"])
    }
    waveform_dataset_dict["settings"]["num_samples"] = size
    del waveform_dataset_dict["settings"]["compression"]

    prior = build_prior_with_defaults(
        waveform_dataset_dict["settings"]["intrinsic_prior"]
    )
    waveform_dataset_dict["parameters"] = pd.DataFrame(prior.sample(size))

    domain_update = event_model.metadata["train_settings"]["data"].get("domain_update")

    # This WaveformDataset contains parameters but no waveforms.
    waveform_dataset = WaveformDataset(
        dictionary=waveform_dataset_dict,
        precision="single",
        domain_update=domain_update,
    )

    # Build the WaveformGenerator and save as an attribute of the WaveformDataset. This
    # will auto-generate waveforms when called via __getitem__(). Maybe move this to
    # the WaveformDataset code, if it does not lead to a circular import. Or put it in
    # the waveform_generator module.
    wfg_domain = build_domain(waveform_dataset.settings["domain"])
    waveform_dataset.waveform_generator = build_waveform_generator(
        waveform_dataset.settings["waveform_generator"], wfg_domain
    )

    # (2) Build the transforms to produce detector waveforms. We use the same
    # transforms that are used to train the Dingo model, with minor changes as below.
    set_train_transforms(
        waveform_dataset,
        event_model.metadata["train_settings"]["data"],
        asd_dataset_path,
    )
    for t in waveform_dataset.transform.transforms:
        if isinstance(t, AddWhiteNoiseComplex):
            # We need the S/N ratio to apply selection effects.
            t.store_snr = True
        if isinstance(t, UnpackDict):
            # Save all the parameters.
            t.selected_keys[0] = "parameters"

    # (3) Data generation loop, similar to the training loop for a single-event
    # network, but with no backward pass. For each event, save the parameters,
    # embedding, and any auxiliary information (e.g., S/N).
    data_loader = DataLoader(
        waveform_dataset,
        batch_size=batch_size,
        shuffle=False,  # Critical to keep the ordering.
        pin_memory=True,
        num_workers=num_workers,  # Update this after debugging
        worker_init_fn=fix_random_seeds,
    )
    parameters = []
    embeddings = []
    print(f"Generating {size} embeddings.")
    time_start = time.time()
    network_time = 0.0
    with threadpool_limits(limits=1, user_api="blas"):
        with torch.no_grad():
            event_model.model.eval()
            for _, data in enumerate(data_loader):
                parameters.append(data[0])
                waveform = data[1].to(event_model.device, non_blocking=True)
                network_start = time.time()
                embeddings.append(event_model.model(waveform).to("cpu"))
                network_time += time.time() - network_start
    total_time = time.time() - time_start
    print(
        f"Done. This took {total_time} seconds, of which {network_time} s doing forward "
        f"passes."
    )
    parameters = pd.DataFrame(
        {k: torch.cat([p[k] for p in parameters]) for k in parameters[0].keys()}
    )
    embeddings = torch.cat(embeddings).numpy()

    # (4) Return PopulationDataset, containing this data, as well as relevant metadata
    # (e.g., prior, event generation process, pointer to Dingo model including hash).

    prior_dict = autocomplete_full_prior_dict(
        {
            **waveform_dataset_dict["settings"]["intrinsic_prior"],
            **event_model.metadata["train_settings"]["data"]["extrinsic_prior"],
        }
    )

    settings = {
        "event_model_path": event_model_path,
        "md5sum": hashlib.md5(open(event_model_path, "rb").read()).hexdigest(),
        "asd_dataset_path": asd_dataset_path,
        "size": size,
        "prior": prior_dict,
    }

    # Append settings, calculate log_probs under the full prior, also S/N
    population_dataset_dict = {
        "parameters": parameters,
        "embeddings": embeddings,
        "settings": settings,
    }

    return BasePopulationDataset(dictionary=population_dataset_dict)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """Generate a population model based on a settings file."""
        ),
    )
    parser.add_argument(
        "--settings_file",
        type=str,
        required=True,
        help="YAML file containing population configuration.",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="population_dataset.hdf5",
        help="Name of file for storing dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load settings
    with open(args.settings_file, "r") as f:
        settings = yaml.safe_load(f)

    dataset = generate_base_population(**settings)
    dataset.to_file(args.out_file)


if __name__ == "__main__":
    main()
