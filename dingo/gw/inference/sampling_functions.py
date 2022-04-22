import torch
from torchvision.transforms import Compose
from bilby.gw.detector.networks import InterferometerList
import time

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
    GNPEChirpMass,
)
from dingo.core.models import PosteriorModel
from dingo.gw.inference.data_preparation import (
    parse_settings_for_raw_data,
    load_raw_data,
    data_to_domain,
)
from dingo.gw.domains import build_domain_for_model


def get_transforms_for_npe(model, num_samples, as_type="dict"):
    domain = build_domain_for_model(model)

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


def sample_with_npe(domain_data, model, num_samples, as_type="dict", batch_size=None):
    # get transformations for preprocessing
    transforms_pre, transforms_post = get_transforms_for_npe(
        model, num_samples, as_type
    )

    # prepare data for inference network
    x = transforms_pre(domain_data)["waveform"]

    # sample from inference network
    model.model.eval()
    y = model.sample(x, batch_size=batch_size)

    # post process samples
    samples = transforms_post({"parameters": y})

    return samples


def get_transforms_for_gnpe(model, init_parameters, as_type="dict"):
    # get model settings
    data_settings = model.metadata["train_settings"]["data"]
    ifo_list = InterferometerList(data_settings["detectors"])
    domain = build_domain_for_model(model)

    gnpe_time_settings = model.metadata["train_settings"]["data"].get(
        "gnpe_time_shifts"
    )
    gnpe_chirp_settings = model.metadata["train_settings"]["data"].get(
        "gnpe_chirp_mass"
    )
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
        gnpe_transforms_pre.append(GNPEChirpMass(gnpe_chirp_settings["kernel"], domain))
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
            CopyToExtrinsicParameters("ra", "dec", "geocent_time", "chirp_mass"),
            GetDetectorTimes(ifo_list, data_settings["ref_time"]),
        ]
    )

    return gnpe_transforms_pre, gnpe_transforms_post

def sample_with_gnpe(
    domain_data,
    model,
    samples_init,
    num_gnpe_iterations=None,
    batch_size=None,
):
    # prepare data for inference network, and add initial samples as extrinsic parameters
    transforms_pre, _ = get_transforms_for_npe(
        model, num_samples=len(list(samples_init["parameters"].values())[0])
    )
    data = {
        "waveform_": transforms_pre(domain_data)["waveform"],
        "extrinsic_parameters": samples_init["parameters"],
        "parameters": {},
    }

    # get transformations for gnpe loop
    gnpe_transforms_pre, gnpe_transforms_post = get_transforms_for_gnpe(
        model,
        init_parameters=samples_init["parameters"].keys(),
    )

    model.model.eval()

    
    sampler = list(torch.utils.data.BatchSampler(range(data["waveform_"].shape[0]), batch_size=batch_size, drop_last=False))
    print("iteration / network time / processing time")
    time_network = 0
    for idx in range(num_gnpe_iterations):
        torch.cuda.empty_cache()
        time_start = time.time()
        time_network = 0

        batch_list = []
        for i in range(len(sampler)):
            # Taking a batch of the full data dict according to the sampler
            batch_data =  {
                    "waveform_": data["waveform_"][sampler[i], ...],
                    "extrinsic_parameters": {k:v[sampler[i], ...] for k,v in data["extrinsic_parameters"].items()},
                    "parameters": {}
                }
            batch_data = gnpe_transforms_pre(batch_data)
            x = [batch_data["waveform"], batch_data["context_parameters"]]

            time_network_start = time.time()
            batch_data["parameters"] = model.sample(*x)
            time_network += time.time() - time_network_start

            batch_data = gnpe_transforms_post(batch_data)
            del batch_data["waveform_"]
            del batch_data["waveform"]
            batch_list.append(batch_data)

        # Combining Batches into a full data dict
        # Copying the old waveform
        data = {
            "waveform_": data["waveform_"]
        }
        data.update({
            key:(torch.cat([batch[key] for batch in batch_list]) if isinstance(val, torch.Tensor)
            else {
                k:torch.cat(
                    [
                    batch[key][k] for batch in batch_list
                    ]) for k in batch_list[0][key].keys()
                })
            for key, val in batch_list[0].items()
        })

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
        )

    # load model
    if not type(model) == PosteriorModel:
        model = PosteriorModel(model, device=device, load_training_info=False)

    gnpe = ("gnpe_time_shifts" in model.metadata["train_settings"]["data"] or
            'gnpe_chirp_mass' in model.metadata["train_settings"]["data"])

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
        build_domain_for_model(model),
        window=model.metadata["train_settings"]["data"]["window"],
    )

    if not gnpe:
        if samples_init is not None:
            raise ValueError("samples_init can only be used for gnpe.")
        samples = sample_with_npe(
            domain_data, model, num_samples, batch_size=batch_size
        )

    else:
        samples = sample_with_gnpe(
            domain_data,
            model,
            samples_init,
            num_gnpe_iterations=num_gnpe_iterations,
            batch_size=batch_size,
        )

    # TODO: apply post correction of sky position here

    return samples

def sample_posterier_of_injection(
    domain_data,
    model,
    model_init=None,
    device="cpu",
    num_samples=50_000,
    samples_init=None,
    num_gnpe_iterations=30,
    batch_size=None,
):
    # get init_samples if requested (typically for gnpe)
    if model_init is not None:
        samples_init = sample_posterier_of_injection(
            domain_data,
            model_init,
            device=device,
            num_samples=num_samples,
            batch_size=batch_size,
        )

    # load model
    if not type(model) == PosteriorModel:
        model = PosteriorModel(model, device=device, load_training_info=False)
    # currently gnpe only implemented for time shifts
    gnpe = "gnpe_time_shifts" in model.metadata["train_settings"]["data"]

    if not gnpe:
        if samples_init is not None:
            raise ValueError("samples_init can only be used for gnpe.")
        samples = sample_with_npe(
            domain_data, model, num_samples, batch_size=batch_size
        )

    else:
        samples = sample_with_gnpe(
            domain_data,
            model,
            samples_init,
            num_gnpe_iterations=num_gnpe_iterations,
            batch_size=batch_size,
        )

    return samples
