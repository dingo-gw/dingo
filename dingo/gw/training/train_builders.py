from typing import List, Optional
import copy

import torch.multiprocessing
import torchvision
from threadpoolctl import threadpool_limits
from bilby.gw.detector import InterferometerList

from dingo.core.utils import *
from dingo.gw.SVD import SVDBasis
from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.domains import build_domain
from dingo.gw.gwutils import *
from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.prior import default_inference_parameters
from dingo.gw.transforms import (
    ProjectOntoDetectors,
    SampleNoiseASD,
    WhitenAndScaleStrain,
    AddWhiteNoiseComplex,
    SelectStandardizeRepackageParameters,
    RepackageStrainsAndASDS,
    UnpackDict,
    GNPECoalescenceTimes,
    SampleExtrinsicParameters,
    GetDetectorTimes,
    StrainTokenization,
    DropFrequencyValues,
    DropDetectors,
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


def set_train_transforms(
    wfd: WaveformDataset,
    data_settings: dict,
    asd_dataset_path: str,
    omit_transforms=None,
    print_output: bool = True,
):
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
    print_output : bool
        Whether to write print statements to the console.
    """
    if print_output:
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
        print_output=print_output,
    )
    assert wfd.domain == asd_dataset.domain

    # Add window factor to domain, so that we can compute the noise variance.
    # TODO: we want to set `domain = wfd.domain`. This does not work at the moment,
    #  because this requires updating the window factor of the wfd.domain (instead of
    #  just the local domain object). This causes trouble if the
    #  set_train_transforms function is called multiple times, since the second time
    #  the domain_update = wfd.domain.domain_dict contains a window factor, which will
    #  cause an error in domain_update.
    domain = build_domain(wfd.domain.domain_dict)
    domain.window_factor = get_window_factor(data_settings["window"])

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

    # Add the GNPE context to context_parameters the first time the transforms are
    # constructed. We do not want to overwrite the ordering of the parameters in
    # subsequent runs.
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
        if print_output:
            print("Using previously-calculated parameter standardizations.")
    except KeyError:
        if print_output:
            print("Calculating new parameter standardizations.")
        standardization_dict = get_standardization_dict(
            extrinsic_prior_dict,
            wfd,
            data_settings["inference_parameters"] + data_settings["context_parameters"],
            torchvision.transforms.Compose(transforms),
        )
        data_settings["standardization"] = standardization_dict

    transforms.append(ProjectOntoDetectors(ifo_list, domain, ref_time))
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
        RepackageStrainsAndASDS(
            ifos=data_settings["detectors"],
            first_index=domain.min_idx,
            drop_asd_channel=data_settings.get("drop_asd_channel", False),
        )
    )

    if data_settings["context_parameters"]:
        selected_keys = ["inference_parameters", "waveform", "context_parameters"]
    else:
        selected_keys = ["inference_parameters", "waveform"]

    try:
        norm_freq = data_settings["tokenization"].get(
            "normalize_frequency_for_positional_encoding", False
        )
        num_tokens = data_settings["tokenization"].get("num_tokens", None)
        token_size = data_settings["tokenization"].get("token_size", None)
        single_tokenizer = data_settings["tokenization"].get("single_tokenizer", False)
        transforms.append(
            StrainTokenization(
                domain,
                num_tokens_per_block=num_tokens,
                token_size=token_size,
                normalize_frequency=norm_freq,
                single_tokenizer=single_tokenizer,
                print_output=print_output,
            )
        )
        selected_keys.append("position")
        selected_keys.append("drop_token_mask")

        # Randomly drop frequency ranges or detectors during training
        if "drop_frequency_range" in data_settings["tokenization"]:
            transforms.append(
                DropFrequencyValues(
                    domain,
                    drop_f_settings=data_settings["tokenization"][
                        "drop_frequency_range"
                    ],
                    print_output=print_output,
                )
            )
        if "drop_detectors" in data_settings["tokenization"]:
            transforms.append(
                DropDetectors(
                    num_blocks=len(data_settings["detectors"]),
                    p_drop_012_detectors=data_settings["tokenization"][
                        "drop_detectors"
                    ].get("p_drop_012_detectors", None),
                    p_drop_hlv=data_settings["tokenization"]["drop_detectors"].get(
                        "p_drop_hlv", None
                    ),
                    print_output=print_output,
                )
            )

    except KeyError:
        print(
            "No tokenization information found, omitting StrainTokenization transform."
        )

    transforms.append(UnpackDict(selected_keys=selected_keys))

    # Drop transforms that are not desired. This is useful for generating, e.g.,
    # noise-free data, or for producing data not formatted for input to the network.
    if omit_transforms is not None:
        transforms = [t for t in transforms if type(t) not in omit_transforms]

    wfd.transform = torchvision.transforms.Compose(transforms)


def build_svd_for_embedding_network(
    wfd: WaveformDataset,
    data_settings: dict,
    asd_dataset_path: str,
    size: int,
    num_training_samples: int,
    num_validation_samples: int,
    num_workers: int = 0,
    batch_size: int = 1000,
    out_dir: Optional[str] = None,
) -> List:
    """
    Construct SVD matrices V based on clean waveforms in each interferometer. These
    will be used to seed the weights of the initial projection part of the embedding
    network.

    It first generates a number of training waveforms, and then produces the SVD.

    Parameters
    ----------
    wfd : WaveformDataset
    data_settings : dict
    asd_dataset_path : str
        Training waveforms will be whitened with respect to these ASDs.
    size : int
        Number of basis elements to include in the SVD projection.
    num_training_samples : int
    num_validation_samples : int
    num_workers : int
    batch_size : int
    out_dir : str
        SVD performance diagnostics are saved here.

    Returns
    -------
    list of numpy arrays
        The V matrices for each interferometer. They are ordered as in data_settings[
        'detectors'].
    """
    # Building the transforms can alter the data_settings dictionary. We do not want
    # the construction of the SVD to impact this, so begin with a fresh copy of this
    # dictionary.
    data_settings = copy.deepcopy(data_settings)

    # This is needed to prevent an occasional error when loading a large dataset into
    # memory using a dataloader. This removes a limitation on the number of "open files".
    old_sharing_strategy = torch.multiprocessing.get_sharing_strategy()
    torch.multiprocessing.set_sharing_strategy("file_system")

    # Fix the luminosity distance to a standard value, just in order to generate the SVD.
    data_settings["extrinsic_prior"]["luminosity_distance"] = "100.0"

    # Build the dataset, but with certain transforms omitted. In particular, we want to
    # build the SVD based on zero-noise waveforms. They should still be whitened though.
    set_train_transforms(
        wfd,
        data_settings,
        asd_dataset_path,
        omit_transforms=[
            AddWhiteNoiseComplex,
            RepackageStrainsAndASDS,
            SelectStandardizeRepackageParameters,
            UnpackDict,
        ],
    )

    print("Generating waveforms for embedding network SVD initialization.")
    time_start = time.time()
    ifos = list(wfd[0]["waveform"].keys())
    waveform_len = len(wfd[0]["waveform"][ifos[0]])
    num_waveforms = num_training_samples + num_validation_samples
    if num_waveforms > len(wfd):
        raise IndexError(
            f"Requested {num_waveforms} samples for generating SVD for embedding "
            f"network, but waveform dataset only contains {len(wfd)} samples."
        )
    waveforms = {
        ifo: np.empty((num_waveforms, waveform_len), dtype=np.complex128)
        for ifo in ifos
    }
    parameters = pd.DataFrame()

    loader = DataLoader(
        wfd,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=fix_random_seeds,
    )
    with threadpool_limits(limits=1, user_api="blas"):
        for idx, data in enumerate(loader):
            # This is for handling the last batch, which may otherwise push the total
            # number of samples above the number requested.
            lower = idx * batch_size
            n = min(batch_size, num_waveforms - lower)

            parameters = pd.concat(
                [parameters, pd.DataFrame(data["parameters"]).iloc[:n]],
                ignore_index=True,
            )
            strain_data = data["waveform"]
            for ifo, strains in strain_data.items():
                waveforms[ifo][lower : lower + n] = strains[:n]
            if lower + n == num_waveforms:
                break
    print(f"...done. This took {time.time() - time_start:.0f} s.")

    # Reset the standard sharing strategy.
    torch.multiprocessing.set_sharing_strategy(old_sharing_strategy)

    print("Generating SVD basis for ifo:")
    time_start = time.time()
    basis_dict = {}
    for ifo in ifos:
        basis = SVDBasis()
        basis.generate_basis(waveforms[ifo][:num_training_samples], size)
        basis_dict[ifo] = basis
        print(f"...{ifo} done.")
    print(f"...this took {time.time() - time_start:.0f} s.")

    if out_dir is not None:
        print(f"Testing SVD basis matrices.")
        for ifo, basis in basis_dict.items():
            print(f"...{ifo}:")
            basis.compute_test_mismatches(
                waveforms[ifo][num_training_samples:],
                parameters=parameters.iloc[num_training_samples:].reset_index(
                    drop=True
                ),
                verbose=True,
            )
            basis.to_file(os.path.join(out_dir, f"svd_{ifo}.hdf5"))
    print("Done")

    # Return V matrices in standard order. Drop the elements below domain.min_idx,
    # since the neural network expects data truncated below these. The dropped elements
    # should be 0.
    print(f"Truncating SVD matrices below index {wfd.domain.min_idx}.")
    print("...V matrix shapes:")
    V_rb_list = []
    for ifo in data_settings["detectors"]:
        V = basis_dict[ifo].V
        assert np.allclose(V[: wfd.domain.min_idx], 0)
        V = V[wfd.domain.min_idx :]
        print("      " + str(V.shape))
        V_rb_list.append(V)
    print("\n")
    return V_rb_list
