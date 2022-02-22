import numpy as np
import torch
from torchvision.transforms import Compose
from gwpy.timeseries import TimeSeries
from bilby.gw.detector.networks import InterferometerList

import matplotlib

matplotlib.use("module://backend_interagg")
import matplotlib.pyplot as plt

from dingo.core.models import PosteriorModel
from dingo.core.dataset import DingoDataset
from dingo.core.utils import load_data_from_file
from dingo.gw.inference.data_download import download_raw_data
from dingo.gw.gwutils import get_window, get_window_factor
from dingo.gw.domains import build_domain, FrequencyDomain
from dingo.gw.transforms import (
    WhitenAndScaleStrain,
    RepackageStrainsAndASDS,
    SelectStandardizeRepackageParameters,
    GNPEShiftDetectorTimes,
    TimeShiftStrain,
    UnpackDict,
    PostCorrectGeocentTime,
    GetDetectorTimes,
    CopyToExtrinsicParameters,
)


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


def inference(
    time_event,
    model,
    file_name=None,
    time_buffer=2.0,
    time_psd=1024,
    device="cpu",
    num_samples=10_000,
    init_context_parameters=None,
    num_gnpe_iterations=30,
):
    # load model
    model = PosteriorModel(model, device=device)
    domain = build_domain(model.metadata["dataset_settings"]["domain"])
    data_settings = model.metadata["train_settings"]["data"]
    domain.update(data_settings["domain_update"])
    domain.window_factor = get_window_factor(data_settings["window"])
    detectors = data_settings["detectors"]
    # standardization = {
    #     k: {kk: np.float32(vv) for kk, vv in v.items()}
    #     for k, v in data_settings["standardization"].items()
    # }
    # correct loaded model
    # data_settings["inference_parameters"] = \
    #     data_settings["selected_parameters"]

    inference_parameters = data_settings["inference_parameters"]

    # check if is_gnpe
    if "gnpe_time_shifts" in data_settings:
        gnpe = True
        try:
            del data_settings["gnpe_time_shifts"]["kernel_kwargs"]
        except:
            pass
        # model.metadata["train_settings"]["data"]["gnpe_time_shifts"]["kernel"] = \
        #     "bilby.core.prior.Uniform(minimum=-0.001, maximum=0.001)"
    else:
        gnpe = False

    # step 1: download raw event data
    settings_raw_data = parse_settings_for_raw_data(
        model.metadata, time_psd, time_buffer
    )
    raw_data = load_raw_data(
        time_event, settings=settings_raw_data, file_name=file_name
    )

    # step 2: prepare the data for the network domain
    domain_data = data_to_domain(
        raw_data,
        settings_raw_data,
        domain,
        window=model.metadata["train_settings"]["data"]["window"],
    )

    # if False:
    #     import h5py
    #     ref_strains = h5py.File(
    #         "/Users/maxdax/Documents/Projects/GW-Inference/dingo/dingo-devel/tutorials"
    #         "/02_gwpe/datasets/strain_data/old_data/strain_FD_whitened.hdf5",
    #         "r",
    #     )
    #     det = "H1"
    #     a = ref_strains[det][:]
    #     b = domain_data["waveform"][det] / domain_data["asds"][det]
    #     a = domain.update_data(a)
    #     a, b = a[domain.min_idx :], b[domain.min_idx :]
    #     assert np.all(np.abs((a / b).imag) < 1e-8)
    #     assert np.all(np.abs((a / b).real - 1) < 1e-8)

    # step 3: prepare the data for neural network
    # this includes whitening, repackaging and
    transforms = Compose(
        [
            WhitenAndScaleStrain(domain.noise_std),
            RepackageStrainsAndASDS(detectors, first_index=domain.min_idx),
            # UnpackDict(selected_keys=["waveform"]),
        ]
    )
    nn_data = transforms(domain_data)

    x_ = torch.from_numpy(nn_data["waveform"])
    model.model.eval()

    post_transform = SelectStandardizeRepackageParameters(
        {"inference_parameters": inference_parameters},
        data_settings["standardization"],
        inverse=True,
        as_type="dict",
    )

    if not gnpe:
        # print(torch.std(x[..., :2, :]))
        # print(torch.mean(x[..., :2, :]))

        # add batch dimension
        x = x_[None, :]
        y = model.model.sample(x, num_samples=num_samples).detach()
        samples = post_transform({"parameters": y})

    else:
        """
        a) get proxies
        b) shift by time
        c) preprocessing exact equiv
        d) get samples
        e) post processing tc
        f) get H1_time, L1_time
        """
        ifo_list = InterferometerList(detectors)
        gnpe_settings = model.metadata["train_settings"]["data"]["gnpe_time_shifts"]

        pre_transforms = []

        pre_transforms.append(
            GNPEShiftDetectorTimes(
                ifo_list,
                gnpe_settings["kernel"],
                gnpe_settings["exact_equiv"],
                inference=True,
            )
        )
        pre_transforms.append(TimeShiftStrain(ifo_list, domain))
        pre_transforms.append(
            SelectStandardizeRepackageParameters(
                {"context_parameters": data_settings["context_parameters"]},
                data_settings["standardization"],
            )
        )
        pre_transforms.append(
            UnpackDict(
                selected_keys=["extrinsic_parameters", "waveform", "context_parameters"]
            )
        )
        pre_transforms = Compose(pre_transforms)

        num_samples = len(list(init_context_parameters.values())[0])

        x_ = x_.expand(num_samples, *x_.shape)

        #################
        # Loop
        #################
        extrinsic_parameters = init_context_parameters
        extrinsic_parameters_ = init_context_parameters
        for i in range(num_gnpe_iterations):

            sample = {
                "parameters": {},
                "extrinsic_parameters": {**extrinsic_parameters, "geocent_time": 0},
                "waveform": x_,
            }
            extrinsic_parameters, *x = pre_transforms(sample)

            y = model.model.sample(*x, num_samples=1).detach()

            post_transforms = Compose(
                [
                    SelectStandardizeRepackageParameters(
                        {"inference_parameters": inference_parameters},
                        data_settings["standardization"],
                        inverse=True,
                        as_type="dict",
                    ),
                    PostCorrectGeocentTime(),
                    CopyToExtrinsicParameters("ra", "dec", "geocent_time"),
                    GetDetectorTimes(ifo_list, data_settings["ref_time"]),
                ]
            )

            samples = post_transforms(
                {"parameters": y, "extrinsic_parameters": extrinsic_parameters}
            )
            extrinsic_parameters = {
                k: samples["extrinsic_parameters"][k]
                for k in init_context_parameters.keys()
            }

            delta = np.array(
                samples["extrinsic_parameters"]["H1_time"]
                - extrinsic_parameters_["H1_time"]
            )
            extrinsic_parameters_ = extrinsic_parameters
            print(f"99th percentile: {np.percentile(np.abs(delta), 99)*1000:.2f} ms")

    return samples["parameters"]


if __name__ == "__main__":
    from os.path import join
    import pandas as pd
    from chainconsumer import ChainConsumer

    model = "Pv2"
    model = "XPHM"
    event = "GW150914"
    time_event = 1126259462.4
    # time_event = 1126259462.4
    dir = "/Users/maxdax/Documents/Projects/GW-Inference/dingo/dingo-devel/tutorials/" \
          "02_gwpe/train_dir_max/cluster_models/"
    # model_name = (
    #     "/Users/maxdax/Documents/Projects/GW-Inference/dingo/dingo-devel"
    #     "/tutorials/02_gwpe/train_dir_max/model_latest.pt"
    # )
    model_name_init = join(dir, f"model_{model}_init.pt")
    model_name = join(dir, f"model_{model}.pt")

    events_file = (
        "/Users/maxdax/Documents/Projects/GW-Inference/dingo/dingo-devel/"
        "tutorials/02_gwpe/datasets/strain_data/events_dataset.hdf5"
    )
    init_samples = inference(
        time_event=time_event,
        time_psd=1024,
        file_name=events_file,
        model=model_name_init,
        num_samples=10_000,
    )

    samples = inference(
        time_event=time_event,
        time_psd=1024,
        file_name=events_file,
        model=model_name,
        init_context_parameters=init_samples,
    )

    pd.DataFrame(samples).to_pickle(join(dir, f"dingo_{model}.pkl"))

    c = ChainConsumer()
    c.add_chain(pd.DataFrame(samples))
    c.configure(usetex=False)
    fig = c.plotter.plot(filename=join(dir, f"{model}.pdf"))



    print("done")
