import pandas as pd
import torch
from torchvision.transforms import Compose
from bilby.gw.detector.networks import InterferometerList

from dingo.gw.transforms import (
    WhitenAndScaleStrain,
    RepackageStrainsAndASDS,
    SelectStandardizeRepackageParameters,
    GNPEShiftDetectorTimes,
    TimeShiftStrain,
    PostCorrectGeocentTime,
    GetDetectorTimes,
    CopyToExtrinsicParameters,
    ExpandStrain,
    ToTorch,
    ResetSample,
)


def get_transforms_for_npe(model, num_samples, as_type="dict"):
    domain = model.build_domain()

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


def sample_with_npe(domain_data, model, num_samples, as_type="dict"):
    # get transformations for preprocessing
    transforms_pre, transforms_post = get_transforms_for_npe(
        model, num_samples, as_type
    )

    # prepare data for inference network
    x = transforms_pre(domain_data)["waveform"]

    # sample from inference network
    model.model.eval()
    y = model.model.sample(x).detach()

    # post process samples
    samples = transforms_post({"parameters": y})

    return samples


def get_transforms_for_gnpe_time(model, init_parameters, as_type="dict"):
    # get model settings
    data_settings = model.metadata["train_settings"]["data"]
    ifo_list = InterferometerList(data_settings["detectors"])
    gnpe_settings = model.metadata["train_settings"]["data"]["gnpe_time_shifts"]

    # transforms for gnpe loop, to be applied prior to sampling step:
    #   * reset the sample (e.g., clone non-gnpe transformed waveform)
    #   * blurring detector times to obtain gnpe proxies
    #   * shifting the strain by - gnpe proxies
    #   * repackaging & standardizing proxies to sample['context_parameters']
    #     for conditioning of the inference network
    gnpe_transforms_pre = Compose(
        [
            ResetSample(extrinsic_parameters_keys=init_parameters),
            GNPEShiftDetectorTimes(
                ifo_list,
                gnpe_settings["kernel"],
                gnpe_settings["exact_equiv"],
                inference=True,
            ),
            TimeShiftStrain(ifo_list, model.build_domain()),
            SelectStandardizeRepackageParameters(
                {"context_parameters": data_settings["context_parameters"]},
                data_settings["standardization"],
            ),
        ]
    )

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
            CopyToExtrinsicParameters("ra", "dec", "geocent_time"),
            GetDetectorTimes(ifo_list, data_settings["ref_time"]),
        ]
    )

    return gnpe_transforms_pre, gnpe_transforms_post


def sample_with_gnpe(
    domain_data,
    model,
    samples_init,
    num_gnpe_iterations=None,
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
    gnpe_transforms_pre, gnpe_transforms_post = get_transforms_for_gnpe_time(
        model, init_parameters=samples_init["parameters"].keys(),
    )

    model.model.eval()

    for idx in range(num_gnpe_iterations):
        data = gnpe_transforms_pre(data)
        x = [data["waveform"], data["context_parameters"]]
        data["parameters"] = model.model.sample(*x, num_samples=1).detach()
        data = gnpe_transforms_post(data)

        Mc = data["parameters"]["chirp_mass"]
        print(torch.mean(Mc), torch.std(Mc))

    samples = data["parameters"]
    return samples
