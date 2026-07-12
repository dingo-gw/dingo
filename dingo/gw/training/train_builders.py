from typing import List, Optional
import copy

import torchvision
from bilby.gw.detector import InterferometerList

from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.domains import build_domain
from dingo.gw.transforms import (
    ProjectOntoDetectors,
    SampleNoiseASD,
    WhitenAndScaleStrain,
    AddWhiteNoiseComplex,
    SelectStandardizeRepackageParameters,
    RepackageStrainsAndASDS,
    SelectKeys,
    GNPECoalescenceTimes,
    SampleExtrinsicParameters,
    GetDetectorTimes,
    CropMaskStrainRandom,
    StrainTokenization,
)
from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.prior import default_inference_parameters
from dingo.gw.gwutils import *
from dingo.core.utils import *


TIME_ALIGNMENT_REQUIRED_CONDITIONING = ("ra", "dec", "geocent_time")


def validate_time_alignment_settings(data_settings: dict) -> None:
    """
    Validate that the data_settings dict is consistent with
    ``data.time_alignment=True``.

    The aligned model targets the factorisation
        q(theta | d) = q(theta_hat | d_aligned, ra, dec, geocent_time)
                       * q(ra, dec, geocent_time | d).
    The transform pipeline this function gates must:
      * have {ra, dec, geocent_time} as conditioning parameters (so the network
        receives them as inputs), and
      * NOT have them as inference targets (the aligned model does not predict
        sky/time -- the sky-position model does).
    It must also not coexist with ``gnpe_time_shifts``, which manipulates the
    same per-detector arrival times stochastically.
    """
    required = set(TIME_ALIGNMENT_REQUIRED_CONDITIONING)
    conditioning_set = set(data_settings.get("conditioning_parameters", []))
    missing = required - conditioning_set
    if missing:
        raise ValueError(
            f"data.time_alignment=True requires {sorted(required)} to be in "
            f"data.conditioning_parameters; missing: {sorted(missing)}."
        )
    overlap = required & set(data_settings["inference_parameters"])
    if overlap:
        raise ValueError(
            f"data.time_alignment=True is incompatible with having "
            f"{sorted(overlap)} in data.inference_parameters; these are "
            f"conditioning quantities, not inference targets."
        )
    if "gnpe_time_shifts" in data_settings:
        raise ValueError(
            "data.time_alignment=True is incompatible with data.gnpe_time_shifts; "
            "both manipulate per-detector arrival times."
        )


def build_dataset(
    data_settings: dict,
    leave_waveforms_on_disk: Optional[bool] = False,
) -> WaveformDataset:
    """Build a dataset based on a settings dictionary. This should contain the path of
    a saved waveform dataset.

    This function also truncates the dataset as necessary.

    Parameters
    ----------
    data_settings : dict
    leave_waveforms_on_disk: bool
        If provided, the values associated with the waveforms will not be loaded into memory during initialization.
        Instead, they will be loaded from disk when the dataset is accessed. This is useful for reducing the memory
        load of large datasets, but can slow down data preprocessing.

    Returns
    -------
    WaveformDataset
    """

    # Build and truncate datasets
    domain_update = data_settings.get("domain_update", None)
    wfd = WaveformDataset(
        file_name=data_settings["waveform_dataset_path"],
        precision="single",
        domain_update=domain_update,
        svd_size_update=data_settings.get("svd_size_update"),
        leave_waveforms_on_disk=leave_waveforms_on_disk,
    )
    return wfd


def set_train_transforms(wfd, data_settings, asd_dataset_path, omit_transforms=None):
    """
    Set the transform attribute of a waveform dataset based on a settings dictionary.
    The transform takes waveform polarizations, samples random extrinsic parameters,
    projects to detectors, adds noise, and formats the data for input to the neural
    network. It also implements optional GNPE transformations.

    Note that the WaveformDataset is modified in-place, so this function returns nothing.

    Parameters
    ----------
    wfd : WaveformDataset
    data_settings : dict
    asd_dataset_path : str
        Path corresponding to the ASD dataset used to generate noise.
    omit_transforms :
        List of sub-transforms to omit from the full composition.
    """

    print(f"Setting train transforms.")
    if omit_transforms is not None:
        print("Omitting \n\t" + "\n\t".join([t.__name__ for t in omit_transforms]))

    # By passing the wfd domain when instantiating the noise dataset, this ensures the
    # domains will match. In particular, it truncates the ASD dataset beyond the new
    # f_max, and sets it to 1 below f_min.
    asd_dataset = ASDDataset(
        asd_dataset_path,
        ifos=data_settings["detectors"],
        precision="single",
        domain_update=wfd.domain.domain_dict,
    )
    assert wfd.domain == asd_dataset.domain
    domain = wfd.domain

    extrinsic_prior_dict = get_extrinsic_prior_dict(data_settings["extrinsic_prior"])
    if data_settings["inference_parameters"] == "default":
        data_settings["inference_parameters"] = default_inference_parameters

    ref_time = data_settings["ref_time"]
    # Build detector objects
    ifo_list = InterferometerList(data_settings["detectors"])

    # Build transforms.
    transforms = [
        SampleExtrinsicParameters(extrinsic_prior_dict),
        GetDetectorTimes(ifo_list, ref_time),
    ]

    extra_context_parameters = []
    if "gnpe_time_shifts" in data_settings:
        d = data_settings["gnpe_time_shifts"]
        transforms.append(
            GNPECoalescenceTimes(
                ifo_list,
                d["kernel"],
                d["exact_equiv"],
                inference=False,
            )
        )
        extra_context_parameters += transforms[-1].context_parameters

    # User-declared conditioning parameters for conditional NPE. These are added
    # to the context_parameters tensor alongside any GNPE proxies and share the
    # same standardization / repackaging path. They are assumed to be parameters
    # already present in either the waveform dataset (intrinsic) or the
    # extrinsic prior, so no additional sampling transform is required.
    extra_context_parameters += data_settings.get("conditioning_parameters", [])

    # Chained-NPE time alignment: the network sees data with no detector-frame
    # time-of-arrival info, and conditions on (ra, dec, geocent_time) to recover
    # antenna-pattern dependence. Implemented by skipping the per-detector time
    # shift in ProjectOntoDetectors.
    time_alignment = data_settings.get("time_alignment", False)
    if time_alignment:
        validate_time_alignment_settings(data_settings)

    # Add the auto-derived and user-declared context parameters to
    # context_parameters the first time the transforms are constructed. We do not
    # want to overwrite the ordering of the parameters in subsequent runs.
    if "context_parameters" not in data_settings:
        data_settings["context_parameters"] = []
    for p in extra_context_parameters:
        if p not in data_settings["context_parameters"]:
            data_settings["context_parameters"].append(p)

    # If the standardization factors have already been set, use those. Otherwise,
    # calculate them, and save them within the data settings.
    #
    # Standardizations are calculated at this point because the present set of
    # transforms is sufficient for generating samples of all regression and context
    # parameters.
    try:
        standardization_dict = data_settings["standardization"]
        print("Using previously-calculated parameter standardizations.")
    except KeyError:
        print("Calculating new parameter standardizations.")
        standardization_dict = get_standardization_dict(
            extrinsic_prior_dict,
            wfd,
            data_settings["inference_parameters"] + data_settings["context_parameters"],
            torchvision.transforms.Compose(transforms),
        )
        data_settings["standardization"] = standardization_dict

    transforms.append(
        ProjectOntoDetectors(
            ifo_list, domain, ref_time, apply_time_shift=not time_alignment
        )
    )
    transforms.append(SampleNoiseASD(asd_dataset))
    transforms.append(WhitenAndScaleStrain(domain.noise_std))
    # We typically add white detector noise. For debugging purposes, this can be turned
    # off with zero_noise option in data_settings.
    if not data_settings.get("zero_noise", False):
        transforms.append(AddWhiteNoiseComplex())
    transforms.append(
        SelectStandardizeRepackageParameters(
            {
                k: data_settings[k]
                for k in ["inference_parameters", "context_parameters"]
            },
            standardization_dict,
        )
    )
    transforms.append(
        RepackageStrainsAndASDS(data_settings["detectors"], first_index=domain.min_idx)
    )
    if "random_strain_cropping" in data_settings:
        transforms.append(
            CropMaskStrainRandom(domain, **data_settings["random_strain_cropping"])
        )
    if "tokenization" in data_settings:
        tokenization = data_settings["tokenization"]
        transforms.append(
            StrainTokenization(
                domain=domain,
                token_size=tokenization.get("token_size"),
                num_tokens_per_block=tokenization.get("num_tokens_per_block"),
                drop_last_token=tokenization.get("drop_last_token", False),
            )
        )

    selected_keys = ["inference_parameters", "waveform"]
    if "tokenization" in data_settings:
        selected_keys += ["position", "drop_token_mask"]
    if data_settings["context_parameters"]:
        selected_keys += ["context_parameters"]

    transforms.append(SelectKeys(selected_keys=selected_keys))

    # Drop transforms that are not desired. This is useful for generating, e.g.,
    # noise-free data, or for producing data not formatted for input to the network.
    if omit_transforms is not None:
        transforms = [t for t in transforms if type(t) not in omit_transforms]

    wfd.transform = torchvision.transforms.Compose(transforms)


def initialization_dataloader(
    wfd: WaveformDataset,
    data_settings: dict,
    asd_dataset_path: str,
    spec: dict,
    batch_size: int = 1000,
):
    """
    Build a dataloader answering an embedding network's init_data_spec (see the
    contract in dingo.core.nn.enets): a transform-stack variation of the training
    data, used for data-driven weight initialization (e.g. seeding the SVD
    projection layer from clean waveforms).

    This replaces the waveform dataset's transforms; call set_train_transforms
    again afterwards to restore the training configuration.

    Parameters
    ----------
    wfd : WaveformDataset
    data_settings : dict
        The train data settings; not modified (the spec is applied to a copy).
    asd_dataset_path : str
        Waveforms are whitened with respect to these ASDs.
    spec : dict
        The data variation requested by the network:
        * "noise": bool -- if False, no noise is added to the waveforms.
        * "network_format": bool -- if False, samples are not repackaged /
          standardized for network input; they remain dicts with per-detector
          complex strains under "waveform".
        * "fix_parameters": dict -- prior parameters pinned to a fixed value,
          e.g. {"luminosity_distance": 100.0}.
        * "num_samples": int -- number of samples the initialization consumes.
    batch_size : int

    Returns
    -------
    torch.utils.data.DataLoader
    """
    data_settings = copy.deepcopy(data_settings)
    for name, value in spec.get("fix_parameters", {}).items():
        data_settings["extrinsic_prior"][name] = str(value)

    omit_transforms = []
    if not spec.get("noise", True):
        omit_transforms.append(AddWhiteNoiseComplex)
    if not spec.get("network_format", True):
        omit_transforms += [
            RepackageStrainsAndASDS,
            SelectStandardizeRepackageParameters,
            SelectKeys,
            CropMaskStrainRandom,
            StrainTokenization,
        ]
    set_train_transforms(
        wfd, data_settings, asd_dataset_path, omit_transforms=omit_transforms or None
    )

    num_samples = spec["num_samples"]
    if num_samples > len(wfd):
        raise IndexError(
            f"Network initialization requests {num_samples} samples, but the "
            f"waveform dataset only contains {len(wfd)}."
        )
    return DataLoader(
        wfd,
        batch_size=batch_size,
        num_workers=0,
        worker_init_fn=fix_random_seeds,
    )
