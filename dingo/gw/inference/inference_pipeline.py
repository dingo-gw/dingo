import numpy as np
from gwpy.timeseries import TimeSeries

from dingo.core.models import PosteriorModel
from dingo.core.dataset import DingoDataset
from dingo.core.utils import load_data_from_file
from dingo.gw.inference.data_download import download_raw_data
from dingo.gw.gwutils import get_window


def load_raw_data(time_event, settings, file_name=None):
    """
    Load raw event data.

    * If file_name is provided and event data is saved in it, load and return the data
    * Else, event data is downloaded. If file_name is provided, the event data is
      additionally saved to the file.

    Parameters
    ----------
    time_event: float
        gps time of the events
    settings: dict
        dict with the settings
    file_name: str
        name of the event dataset file

    Returns
    -------

    """
    event = str(time_event)

    # first try to load the event data from the saved dataset
    if file_name is not None:
        data = load_data_from_file(file_name, event, settings=settings)
        if data is not None:
            print(f"Data for event at {event} found in {file_name}.")
            return data

    # if this did not work, download the data
    print(f"Downloading data for event at {event}.")
    data = download_raw_data(time_event, **settings)

    # optionally save this data to file_name
    if file_name is not None:
        dataset = DingoDataset(
            dictionary={event: data, "settings": settings}, data_keys=[event]
        )
        print(f"Saving data for event at {event} to {file_name}.")
        dataset.to_file(file_name, mode="a")

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


def data_to_domain(raw_data, settings_raw_data, model_metadata):
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
    domain_type = model_metadata["dataset_settings"]["domain"]["type"]

    if domain_type == "FrequencyDomain":
        window = get_window(model_metadata["train_settings"]["data"]["window"])
        data = {"waveform": {}, "asds": {}}
        # convert event strains to frequency domain
        for det, event_strain in raw_data["strain"].items():
            event_strain = TimeSeries(event_strain, dt=1 / settings_raw_data["f_s"])
            event_strain = event_strain.to_pycbc()
            event_strain = (event_strain * window).to_frequencyseries()
            event_strain = event_strain.cyclic_time_shift(
                settings_raw_data["time_buffer"]
            )
            data["waveform"][det] = np.array(event_strain)

        # convert psds to asds
        for det, psd in raw_data["psd"].items():
            data["asds"][det] = psd ** 0.5

        return data

    else:
        raise NotImplementedError(f"Unknown domain type {domain_type}")


def inference(
    time_event,
    model,
    model_pose=None,
    file_name=None,
    time_buffer=2.0,
    time_psd=1024,
    device="cpu",
):
    # load model
    model = PosteriorModel(model, device=device)

    # step 1: download raw event data
    settings_raw_data = parse_settings_for_raw_data(
        model.metadata, time_psd, time_buffer
    )
    raw_data = load_raw_data(
        time_event, settings=settings_raw_data, file_name=file_name
    )

    # step 2: prepare the data for the network domain
    domain_data = data_to_domain(raw_data, settings_raw_data, model.metadata)

    import h5py
    ref_strains = h5py.File(
        "/Users/maxdax/Documents/Projects/GW-Inference/dingo/dingo-devel/tutorials"
        "/02_gwpe/datasets/strain_data/old_data/strain_FD_whitened.hdf5", "r"
    )

    det = "H1"
    a = ref_strains[det][:]
    b = domain_data["waveform"][det] / domain_data["asds"][det]
    b = b[:8193]
    assert np.all(np.abs(((a/b)[160:]).imag) < 1e-8)
    assert np.all(np.abs(((a/b)[160:]).real - 1) < 1e-8)
    print("done")
    # step 3: prepare the data for neural network
    # this includes whitening, repackaging and


if __name__ == "__main__":
    event = "GW150914"
    time_event = 1126259462.4
    model_name = (
        "/Users/maxdax/Documents/Projects/GW-Inference/dingo/dingo-devel"
        "/tutorials/02_gwpe/train_dir_max/model_latest.pt"
    )
    events_file = (
        "/Users/maxdax/Documents/Projects/GW-Inference/dingo/dingo-devel/"
        "tutorials/02_gwpe/datasets/strain_data/events_dataset.hdf5"
    )
    inference(
        time_event=time_event, model=model_name, time_psd=1024, file_name=events_file
    )
