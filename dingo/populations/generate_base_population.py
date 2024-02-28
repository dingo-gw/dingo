import argparse
import copy
import hashlib
import textwrap
import time

import pandas as pd
import torch
import yaml
from tqdm import tqdm
from bilby.gw.prior import BBHPriorDict
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

from dingo.core.models import PosteriorModel
from dingo.core.utils import fix_random_seeds
from dingo.gw.dataset import WaveformDataset
from dingo.gw.domains import build_domain
from dingo.gw.prior import build_prior_with_defaults, autocomplete_full_prior_dict
from dingo.gw.training import set_train_transforms
from dingo.gw.transforms import (
    AddWhiteNoiseComplex,
    UnpackDict,
    SampleExtrinsicParameters,
)
from dingo.gw.waveform_generator.waveform_generator import build_waveform_generator
from dingo.populations.population_models import build_population_model
from dingo.populations.training.population_dataset import EventEmbeddingsDataset


def generate_base_population(
    event_model_path,
    asd_dataset_path,
    size,
    batch_size=None,
    device="cuda",
    num_workers=0,
    population_model=None,
    population_prior=None,
):
    """
    Generate a "base" population, meaning a population drawn from the prior:
        (1) Sample event parameters from the population prior and likelihood,
        or optionally the event model prior.
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
    population_model : str
        Type of population model. The population model is used to generate parameters
        for the embeddings, ensuring good coverage for later training. (Optional)
    population_prior : dict
        Prior for the population model. (Optional)

    Returns
    -------
    EventEmbeddingsDataset
    """
    event_model = PosteriorModel(
        model_filename=event_model_path, device=device, load_training_info=False
    )
    event_model_metadata = copy.deepcopy(event_model.metadata)
    # TODO: Make sure this is not a GNPE network.
    event_model.set_embedding_only()

    # (1) Build a waveform dataset / dataloader. This should generate samples in real
    # time during population generation.

    waveform_dataset_dict = {
        "settings": copy.deepcopy(event_model.metadata["dataset_settings"])
    }
    waveform_dataset_dict["settings"]["num_samples"] = size
    del waveform_dataset_dict["settings"]["compression"]

    intrinsic_prior = build_prior_with_defaults(
        waveform_dataset_dict["settings"]["intrinsic_prior"]
    )
    full_prior_dict = autocomplete_full_prior_dict(
        {
            **waveform_dataset_dict["settings"]["intrinsic_prior"],
            **event_model.metadata["train_settings"]["data"]["extrinsic_prior"],
        }
    )

    if population_model is not None:
        full_prior = BBHPriorDict(copy.deepcopy(full_prior_dict))
        population_model = build_population_model(
            population_model=population_model,
            population_prior=population_prior,
            event_model_prior=full_prior,
        )
        parameters_intrinsic = []
        for _ in tqdm(range(size // batch_size)):
            # We batch the sampling of parameters because preparing the cosmologies can
            # be slow. For convenience, we generate populations of size batch_size.
            # TODO: Deal with any remainder in the batching.
            # TODO: Parallelize.
            hyperparameters = population_model.prior.sample()
            event_generation_func = population_model.get_event_generator(
                hyperparameters
            )
            for _ in range(batch_size):
                # Use the event model intrinsic prior, but updated with the population
                # event parameters.
                p = intrinsic_prior.sample()
                p.update(event_generation_func())
                parameters_intrinsic.append(p)
        waveform_dataset_dict["parameters"] = pd.DataFrame(parameters_intrinsic)
        # TODO: Ensure that the samples lie within the model prior, e.g., for masses.

    else:
        waveform_dataset_dict["parameters"] = pd.DataFrame(intrinsic_prior.sample(size))

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
        if isinstance(t, SampleExtrinsicParameters) and population_model is not None:
            for p in population_model.event_parameters:
                if p in t.extrinsic_prior_dict:
                    t.extrinsic_prior_dict.pop(p)
                    t.prior.pop(p)

    # (3) Data generation loop, similar to the training loop for a single-event
    # network, but with no backward pass. For each event, save the parameters,
    # embedding, and any auxiliary information (e.g., S/N).
    data_loader = DataLoader(
        waveform_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=fix_random_seeds,
    )
    parameters = []
    embeddings = []
    print(f"Generating {size} embeddings.")
    time_start = time.time()
    with threadpool_limits(limits=1, user_api="blas"):
        with torch.no_grad():
            event_model.model.eval()
            for i, data in enumerate(data_loader):
                parameters.append(copy.deepcopy(data[0]))
                waveform = data[1].to(event_model.device, non_blocking=True)
                network_start = time.time()
                embeddings.append(event_model.model(waveform).detach().to("cpu"))
                waveform_time = network_start - time_start
                network_time = time.time() - network_start
                time_start = time.time()
                print(
                    f"Batch {i} [{batch_size*i}/{size}]  Waveform generation time: "
                    f"{waveform_time:.5f} s   Network time: {network_time:.5f} s"
                )
    parameters = pd.DataFrame(
        {k: torch.cat([p[k] for p in parameters]) for k in parameters[0].keys()}
    )
    embeddings = torch.cat(embeddings).numpy()

    # (4) Return PopulationDataset, containing this data, as well as relevant metadata
    # (e.g., prior, event generation process, pointer to Dingo model including hash).

    settings = {
        "event_model_path": event_model_path,
        "md5sum": hashlib.md5(open(event_model_path, "rb").read()).hexdigest(),
        "asd_dataset_path": asd_dataset_path,
        "size": size,
        "prior": full_prior_dict,
        "population_model": population_model.model_type,
        "population_prior": population_prior,
        "full_event_model_metadata": event_model_metadata,
    }

    # Append settings, calculate log_probs under the full prior, also S/N
    population_dataset_dict = {
        "parameters": parameters,
        "embeddings": embeddings,
        "settings": settings,
    }

    return EventEmbeddingsDataset(dictionary=population_dataset_dict)


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
