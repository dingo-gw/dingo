import numpy as np
from gwpy.timeseries import TimeSeries
import argparse
import os
from os.path import dirname, join
import yaml
from chainconsumer import ChainConsumer
import scipy

from dingo.core.models import PosteriorModel
from dingo.core.dataset import DingoDataset
from dingo.core.utils import load_data_from_file
from dingo.gw.inference.data_download import download_raw_data
from dingo.gw.gwutils import get_window
from dingo.gw.domains import FrequencyDomain
from dingo.gw.inference.utils import *


def load_raw_data(time_event, settings, event_dataset=None):
    """
    Load raw event data.

    * If event_dataset is provided and event data is saved in it, load and return the data
    * Else, event data is downloaded. If event_dataset is provided, the event data is
      additionally saved to the file.

    Parameters
    ----------
    time_event: float
        gps time of the events
    settings: dict
        dict with the settings
    event_dataset: str
        name of the event dataset file

    Returns
    -------

    """
    event = str(time_event)

    # first try to load the event data from the saved dataset
    if event_dataset is not None:
        data = load_data_from_file(event_dataset, event, settings=settings)
        if data is not None:
            print(f"Data for event at {event} found in {event_dataset}.")
            return data

    # if this did not work, download the data
    print(f"Downloading data for event at {event}.")
    data = download_raw_data(time_event, **settings)

    # optionally save this data to event_dataset
    if event_dataset is not None:
        dataset = DingoDataset(
            dictionary={event: data, "settings": settings}, data_keys=[event]
        )
        print(f"Saving data for event at {event} to {event_dataset}.")
        dataset.to_file(event_dataset, mode="a")

    return data


def parse_settings_for_raw_data(model_metadata, time_psd, time_buffer):
    domain_type = model_metadata["dataset_settings"]["domain"]["type"]

    if domain_type == "FrequencyDomain":
        data_settings = model_metadata["train_settings"]["data"]
        settings = {
            "window": data_settings["window"],
            "detectors": data_settings["detectors"],
            "time_segment": data_settings["window"]["T"],
            "time_psd": time_psd,
            "time_buffer": time_buffer,
            "f_s": data_settings["window"]["f_s"],
        }
    else:
        raise NotImplementedError(f"Unknown domain type {domain_type}")

    return settings


def data_to_domain(raw_data, settings_raw_data, domain, **kwargs):
    """

    Parameters
    ----------
    raw_data
    settings_raw_data
    model_metadata

    Returns
    -------
    data: dict
        dict with domain_data

    """

    if type(domain) == FrequencyDomain:
        window = get_window(kwargs["window"])
        data = {"waveform": {}, "asds": {}}
        # convert event strains to frequency domain
        for det, event_strain in raw_data["strain"].items():
            event_strain = TimeSeries(event_strain, dt=1 / settings_raw_data["f_s"])
            event_strain = event_strain.to_pycbc()
            event_strain = (event_strain * window).to_frequencyseries()
            event_strain = event_strain.cyclic_time_shift(
                settings_raw_data["time_buffer"]
            )
            event_strain = domain.update_data(np.array(event_strain))
            data["waveform"][det] = event_strain

        # convert psds to asds
        for det, psd in raw_data["psd"].items():
            asd = psd ** 0.5
            asd = domain.update_data(asd, low_value=1.0)
            data["asds"][det] = asd

        return data

    else:
        raise NotImplementedError(f"Unknown domain type {type(domain)}")


def sample_posterior_of_event(
    time_event,
    model,
    model_init=None,
    event_dataset=None,
    event_dataset_init=None,
    time_buffer=2.0,
    time_psd=1024,
    device="cpu",
    num_samples=50_000,
    samples_init=None,
    num_gnpe_iterations=30,
):
    # get init_samples if requested (typically for gnpe)
    if model_init is not None:
        # use event_dataset if no separate one is provided for initialization;
        # in that case, the data settings of model and model_init need to be compatible
        if event_dataset_init is None:
            event_dataset_init = event_dataset
        samples_init = sample_posterior_of_event(
            time_event,
            model_init,
            event_dataset=event_dataset_init,
            time_buffer=time_buffer,
            time_psd=time_psd,
            device=device,
            num_samples=num_samples,
        )

    # load model
    if not type(model) == PosteriorModel:
        model = PosteriorModel(model, device=device, load_training_info=False)
    # currently gnpe only implemented for time shifts
    gnpe = "gnpe_time_shifts" in model.metadata["train_settings"]["data"]

    # step 1: download raw event data
    settings_raw_data = parse_settings_for_raw_data(
        model.metadata, time_psd, time_buffer
    )
    raw_data = load_raw_data(
        time_event, settings=settings_raw_data, event_dataset=event_dataset
    )

    # step 2: prepare the data for the network domain
    domain_data = data_to_domain(
        raw_data,
        settings_raw_data,
        model.build_domain(),
        window=model.metadata["train_settings"]["data"]["window"],
    )

    if not gnpe:
        assert samples_init is None, "samples_init can only be used for gnpe."
        samples = sample_with_npe(domain_data, model, num_samples)

    else:
        samples = sample_with_gnpe(
            domain_data,
            model,
            samples_init,
            num_gnpe_iterations=num_gnpe_iterations,
        )

    # TODO: apply post correction of sky position here

    return samples


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer the posterior of a GW event using a trained dingo model.",
    )
    parser.add_argument(
        "--out_directory",
        type=str,
        default=None,
        help="Directory for output of analysis (samples, plots).",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained dingo model.",
    )
    parser.add_argument(
        "--model_init",
        type=str,
        default=None,
        help="Path to trained dingo model for initialization of gnpe. Only required if "
        "gnpe is used.",
    )
    parser.add_argument(
        "--gps_time_event",
        type=float,
        required=True,
        nargs="+",
        help="List of GPS times of the events to be analyzed. Used to download the GW "
        "event data, or search for it in the dataset file.",
    )
    parser.add_argument(
        "--event_dataset",
        type=str,
        default=None,
        help="Path to dataset for GW event data. If set, GW event data is cached in "
        "this dataset, such that it only needs to be downloaded once.",
    )
    parser.add_argument(
        "--event_dataset_init",
        type=str,
        default=None,
        help="Path to dataset for GW event data, for gnpe initialization model "
        "model_init. If set, GW event data is cached in this dataset, such that it "
        "only needs to be downloaded once.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50_000,
        help="Number of posterior samples per event.",
    )
    parser.add_argument(
        "--num_gnpe_iterations",
        type=int,
        default=30,
        help="Number of gnpe iterations.",
    )
    parser.add_argument(
        "--reference_file",
        type=str,
        default=None,
        help="File with path to reference samples (e.g., bilby, LALInference). If set, "
        "the corresponding samples are loaded and compared in a cornerplot.",
    )
    # settings for raw GW data
    parser.add_argument(
        "--time_psd",
        type=float,
        default=1024,
        help="Time in seconds used for PSD estimation.",
    )
    parser.add_argument(
        "--time_buffer",
        type=float,
        default=2,
        help="Buffer time in seconds. The analyzed strain segment extends up to "
        "gps_time_event + time_buffer.",
    )

    args = parser.parse_args()

    return args


def analyze_event():
    args = parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = PosteriorModel(args.model, device=device, load_training_info=False)
    epoch = model.epoch
    wf_model = model.metadata["dataset_settings"]["waveform_generator"]["approximant"]

    # create analysis directory
    if args.out_directory is None:
        args.out_directory = join(dirname(args.model), f"analysis_epoch-{epoch}")
    os.makedirs(args.out_directory, exist_ok=True)

    #
    if args.reference_file is not None:
        with open(args.reference_file, "r") as f:
            ref = yaml.safe_load(f)[wf_model]
    else:
        ref = None

    # sample posterior for events
    for time_event in args.gps_time_event:

        print(f"Analyzing event at gps time {time_event}.")
        samples = sample_posterior_of_event(
            time_event=time_event,
            model=model,
            model_init=args.model_init,
            time_psd=args.time_psd,
            event_dataset=args.event_dataset,
            event_dataset_init=args.event_dataset_init,
            num_samples=args.num_samples,
            num_gnpe_iterations=args.num_gnpe_iterations,
        )

        # if no reference samples are available, simply save the dingo samples
        if ref is None or time_event not in ref:
            pd.DataFrame(samples).to_pickle(
                join(args.out_directory, f"dingo_samples_gps-{time_event}.pkl")
            )

        # if reference samples are available, save dingo samples and additionally
        # compare to the reference method in a corner plot
        else:
            name_event = ref[time_event]["event_name"]
            ref_samples_file = ref[time_event]["reference_samples"]["file"]
            ref_method = ref[time_event]["reference_samples"]["method"]

            pd.DataFrame(samples).to_pickle(
                join(args.out_directory, f"dingo_samples_{name_event}.pkl")
            )

            ref_samples = load_ref_samples(ref_samples_file)

            generate_cornerplot(
                {"name": ref_method, "samples": ref_samples, "color": "blue"},
                {"name": "dingo", "samples": pd.DataFrame(samples), "color": "orange"},
                filename=join(args.out_directory, f"cornerplot_{name_event}.pdf"),
            )


def generate_cornerplot(*sample_sets, filename=None):
    parameters = [
        p
        for p in sample_sets[0]["samples"].keys()
        if p in set.intersection(*tuple(set(s["samples"].keys()) for s in sample_sets))
    ]
    N = len(sample_sets)

    c = ChainConsumer()
    for s in sample_sets:
        c.add_chain(s["samples"][parameters], color=s["color"], name=s["name"])
    c.configure(
        linestyles=["-"] * N,
        linewidths=[1.5] * N,
        sigmas=[np.sqrt(2) * scipy.special.erfinv(x) for x in [0.5, 0.9]],
        shade=[False] + [True] * (N - 1),
        shade_alpha=0.3,
        bar_shade=False,
        label_font_size=10,
        tick_font_size=10,
        usetex=False,
        legend_kwargs={"fontsize": 30},
        kde=False,
    )
    c.plotter.plot(filename=filename)


def load_ref_samples(ref_samples_file):
    # Todo: this function should be made more flexible for other formats and parameters
    from bilby.gw.conversion import component_masses_to_chirp_mass

    columns = [
        "mass_1",
        "mass_2",
        "phase",
        "geocent_time",
        "luminosity_distance",
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "phi_jl",
        "theta_jn",
        "psi",
        "ra",
        "dec",
    ]
    samples = np.load(ref_samples_file, allow_pickle=True)["samples"]
    samples = pd.DataFrame(data=samples, columns=columns)
    # add chirp mass and mass ratio.
    Mc = component_masses_to_chirp_mass(samples["mass_1"], samples["mass_2"])
    q = samples["mass_2"] / samples["mass_1"]
    samples.insert(loc=0, column="chirp_mass", value=Mc)
    samples.insert(loc=0, column="mass_ratio", value=q)
    samples.drop(columns="geocent_time", inplace=True)
    return samples


if __name__ == "__main__":

    analyze_event()

    if False:
        from os.path import join
        import pandas as pd
        from chainconsumer import ChainConsumer

        # --model /Users/maxdax/Documents/Projects/GW-Inference/dingo/dingo-devel/tutorials/02_gwpe/train_dir_max/cluster_models/model_Pv2.py
        # --model_init /Users/maxdax/Documents/Projects/GW-Inference/dingo/dingo-devel/tutorials/02_gwpe/train_dir_max/cluster_models/model_Pv2_init.py
        # --time_event 1126259462.4
        # --event_dataset /Users/maxdax/Documents/Projects/GW-Inference/dingo/dingo-devel/tutorials/02_gwpe/datasets/strain_data/events_dataset.hdf5
        # --reference_file /Users/maxdax/Documents/Projects/GW-Inference/dingo/dingo-devel/tutorials/02_gwpe/train_dir_max/cluster_models/reference_file.yaml

        model = "XPHM"
        event = "GW150914"
        time_event = 1126259462.4
        dir = (
            "/Users/maxdax/Documents/Projects/GW-Inference/dingo/dingo-devel/tutorials/"
            "02_gwpe/train_dir_max/cluster_models/"
        )
        model_name_init = join(dir, f"model_{model}_init.pt")
        model_name = join(dir, f"model_{model}.pt")

        events_file = (
            "/Users/maxdax/Documents/Projects/GW-Inference/dingo/dingo-devel/"
            "tutorials/02_gwpe/datasets/strain_data/events_dataset.hdf5"
        )

        samples = sample_posterior_of_event(
            time_event=time_event,
            model=model_name,
            model_init=model_name_init,
            time_psd=1024,
            event_dataset=events_file,
            num_samples=1_000,
        )

    # pd.DataFrame(samples).to_pickle(join(dir, f"dingo_{model}.pkl"))
    #
    # c = ChainConsumer()
    # c.add_chain(pd.DataFrame(samples))
    # c.configure(usetex=False)
    # fig = c.plotter.plot(filename=join(dir, f"{model}.pdf"))
