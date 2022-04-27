import numpy as np
from torchvision.transforms import Compose
from bilby.gw.detector.networks import InterferometerList
import time
from astropy.time import Time

from dingo.gw.transforms import (
    WhitenAndScaleStrain,
    RepackageStrainsAndASDS,
    SelectStandardizeRepackageParameters,
    GNPECoalescenceTimes,
    TimeShiftStrain,
    PostCorrectGeocentTime,
    GetDetectorTimes,
    CopyToExtrinsicParameters,
    ExpandStrain,
    ToTorch,
    ResetSample,
    GNPEChirp,
)
from dingo.core.models import PosteriorModel
from dingo.gw.inference.data_preparation import get_event_data_and_domain
from dingo.gw.domains import build_domain_from_model_metadata


def get_transforms_for_npe(model, num_samples, as_type="dict"):
    domain = build_domain_from_model_metadata(model.metadata)

    # preprocessing transforms:
    #   * whiten and scale strain (since the inference network expects standardized data)
    #   * repackage strains and asds from dicts to an array
    #   * convert array to torch tensor on the correct device
    #   * expand strain num_sample times
    #   * extract only strain/waveform from the sample
    transforms_pre = Compose(
        [
            WhitenAndScaleStrain(domain.noise_std),
            RepackageStrainsAndASDS(
                model.metadata["train_settings"]["data"]["detectors"],
                first_index=domain.min_idx,
            ),
            ToTorch(device=model.device),
            ExpandStrain(num_samples),
        ]
    )

    # postprocessing transforms:
    #   * de-standardize data and extract inference parameters
    inference_params = model.metadata["train_settings"]["data"]["inference_parameters"]
    transforms_post = SelectStandardizeRepackageParameters(
        {"inference_parameters": inference_params},
        model.metadata["train_settings"]["data"]["standardization"],
        inverse=True,
        as_type=as_type,
    )

    return transforms_pre, transforms_post


def sample_with_npe(
    event_data,
    model,
    num_samples,
    as_type="dict",
    batch_size=None,
    get_log_prob=False,
):
    # get transformations for preprocessing
    transforms_pre, transforms_post = get_transforms_for_npe(
        model, num_samples, as_type
    )

    # prepare data for inference network
    x = transforms_pre(event_data)["waveform"]

    # sample from inference network
    if not get_log_prob:
        y = model.sample(x, batch_size=batch_size)
        # post process samples
        samples = transforms_post({"parameters": y})["parameters"]
    else:
        y, log_prob = model.sample(x, batch_size=batch_size, get_log_prob=True)
        # post process samples
        samples = transforms_post({"parameters": y})["parameters"]
        if as_type == "dict":
            samples["log_prob"] = log_prob
        else:
            raise NotImplementedError()

    return samples


def get_transforms_for_gnpe(model, init_parameters, as_type="dict"):
    # get model settings
    data_settings = model.metadata["train_settings"]["data"]
    ifo_list = InterferometerList(data_settings["detectors"])
    domain = build_domain_from_model_metadata(model.metadata)

    gnpe_time_settings = model.metadata["train_settings"]["data"].get(
        "gnpe_time_shifts"
    )
    gnpe_chirp_settings = model.metadata["train_settings"]["data"].get("gnpe_chirp")
    if not gnpe_time_settings and not gnpe_chirp_settings:
        raise KeyError(
            "GNPE inference requires network trained for either chirp mass "
            "or coalescence time GNPE."
        )

    # transforms for gnpe loop, to be applied prior to sampling step:
    #   * reset the sample (e.g., clone non-gnpe transformed waveform)
    #   * blurring detector times to obtain gnpe proxies
    #   * shifting the strain by - gnpe proxies
    #   * repackaging & standardizing proxies to sample['context_parameters']
    #     for conditioning of the inference network
    gnpe_transforms_pre = [ResetSample(extrinsic_parameters_keys=init_parameters)]
    if gnpe_time_settings:
        gnpe_transforms_pre.append(
            GNPECoalescenceTimes(
                ifo_list,
                gnpe_time_settings["kernel"],
                gnpe_time_settings["exact_equiv"],
                inference=True,
            )
        )
        gnpe_transforms_pre.append(TimeShiftStrain(ifo_list, domain))
    if gnpe_chirp_settings:
        gnpe_transforms_pre.append(
            GNPEChirp(
                gnpe_chirp_settings["kernel"],
                domain,
                gnpe_chirp_settings.get("order", 0),
            )
        )
    gnpe_transforms_pre.append(
        SelectStandardizeRepackageParameters(
            {"context_parameters": data_settings["context_parameters"]},
            data_settings["standardization"],
            device=model.device,
        )
    )

    gnpe_transforms_pre = Compose(gnpe_transforms_pre)

    # transforms for gnpe loop, to be applied after sampling step:
    #   * de-standardization of parameters
    #   * post correction for geocent time (required for gnpe with exact equivariance)
    #   * computation of detectortimes from parameters (required for next gnpe iteration)
    gnpe_transforms_post = Compose(
        [
            SelectStandardizeRepackageParameters(
                {"inference_parameters": data_settings["inference_parameters"]},
                data_settings["standardization"],
                inverse=True,
                as_type=as_type,
            ),
            PostCorrectGeocentTime(),
            CopyToExtrinsicParameters(
                "ra", "dec", "geocent_time", "chirp_mass", "mass_ratio"
            ),
            GetDetectorTimes(ifo_list, data_settings["ref_time"]),
        ]
    )

    return gnpe_transforms_pre, gnpe_transforms_post


def sample_with_gnpe(
    event_data,
    model,
    samples_init,
    num_gnpe_iterations=None,
    batch_size=None,
):
    # prepare data for inference network, and add initial samples as extrinsic parameters
    transforms_pre, _ = get_transforms_for_npe(
        model, num_samples=len(list(samples_init.values())[0])
    )
    data = {
        "waveform_": transforms_pre(event_data)["waveform"],
        "extrinsic_parameters": samples_init,
        "parameters": {},
    }

    # get transformations for gnpe loop
    gnpe_transforms_pre, gnpe_transforms_post = get_transforms_for_gnpe(
        model,
        init_parameters=samples_init.keys(),
    )

    model.model.eval()

    print("iteration / network time / processing time")
    for idx in range(num_gnpe_iterations):
        time_start = time.time()

        data = gnpe_transforms_pre(data)
        x = [data["waveform"], data["context_parameters"]]

        time_network_start = time.time()
        data["parameters"] = model.sample(*x, batch_size=batch_size)
        time_network = time.time() - time_network_start

        data = gnpe_transforms_post(data)

        time_processing = time.time() - time_start - time_network
        print(f"{idx:03d}  /  {time_network:.2f} s  /  {time_processing:.2f} s")

    samples = data["parameters"]
    return samples


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
    batch_size=None,
    get_log_prob=False,
    post_correct_ra=True,
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
            batch_size=batch_size,
            # correct ra only in last step, until then use ref_time of model
            post_correct_ra=False,
        )

    # load model
    if not type(model) == PosteriorModel:
        model = PosteriorModel(model, device=device, load_training_info=False)

    gnpe = (
        "gnpe_time_shifts" in model.metadata["train_settings"]["data"]
        or "gnpe_chirp_mass" in model.metadata["train_settings"]["data"]
    )

    # get raw event data, and prepare it for the network domain
    event_data, _ = get_event_data_and_domain(
        model.metadata, time_event, time_psd, time_buffer, event_dataset
    )

    if not gnpe:
        if samples_init is not None:
            raise ValueError("samples_init can only be used for gnpe.")
        samples = sample_with_npe(
            event_data,
            model,
            num_samples,
            batch_size=batch_size,
            get_log_prob=get_log_prob,
        )

    else:
        if get_log_prob:
            raise ValueError("GNPE does not provide access to log_prob.")
        samples = sample_with_gnpe(
            event_data,
            model,
            samples_init,
            num_gnpe_iterations=num_gnpe_iterations,
            batch_size=batch_size,
        )

    # post correction of sky position
    if post_correct_ra:
        samples["ra"] = get_corrected_sky_position(
            samples["ra"],
            time_event,
            model.metadata["train_settings"]["data"]["ref_time"],
        )

    return samples


def get_corrected_sky_position(ra, t_event, t_ref):
    """
    Calculate the corrected sky position of an event. This is necessary, since the
    model was trained with waveform projections assuming a particular reference time
    t_ref. The corrected sky position takes into account the time difference between
    the event and t_ref.

    Parameters
    ----------
    ra:
        right ascension parameter of the event
    t_event:
        gps time of the event
    t_ref: float
        gps time, used as reference time for the model

    Returns
    -------
    ra_corr: float
        corrected right ascension parameter of the event

    """
    time_reference = Time(t_ref, format="gps", scale="utc")
    time_event = Time(t_event, format="gps", scale="utc")
    longitude_event = time_event.sidereal_time("apparent", "greenwich")
    longitude_reference = time_reference.sidereal_time("apparent", "greenwich")
    delta_longitude = longitude_event - longitude_reference
    ra_correction = delta_longitude.rad
    ra_corr = (ra + ra_correction) % (2 * np.pi)
    return ra_corr
