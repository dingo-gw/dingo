import copy
import torchvision
from torch.utils.data import DataLoader
from bilby.gw.detector import InterferometerList

from dingo.gw.SVD import SVDBasis
from dingo.gw.dataset import WaveformDataset

from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.domains import build_domain
from dingo.gw.transforms import (
    SampleExtrinsicParameters,
    GetDetectorTimes,
    ProjectOntoDetectors,
    SampleNoiseASD,
    WhitenAndScaleStrain,
    AddWhiteNoiseComplex,
    SelectStandardizeRepackageParameters,
    RepackageStrainsAndASDS,
    UnpackDict,
    GNPEDetectorTimes,
    GNPEChirpMass,
)
from dingo.gw.ASD_dataset.noise_dataset import ASDDataset
from dingo.gw.prior import default_params
from dingo.gw.gwutils import *
from dingo.core.utils import *


def build_dataset(data_settings):
    """Build a dataset based on a settings dictionary. This should contain the path of
    a saved waveform dataset.

    This function also truncates the dataset as necessary.

    Parameters
    ----------
    data_settings : dict

    Returns
    -------
    WaveformDataset
    """
    # Build and truncate datasets
    domain_update = data_settings.get('domain_update', None)
    wfd = WaveformDataset(
        file_name=data_settings["waveform_dataset_path"],
        precision="single",
        domain_update=domain_update,
    )
    # wfd.truncate_dataset_domain(data_settings["conditioning"]["frequency_range"])
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

    print(f"Setting train transforms. Omitting {omit_transforms}.")

    asd_dataset = ASDDataset(asd_dataset_path, ifos=data_settings["detectors"])
    asd_dataset.truncate_dataset_domain(
        data_settings["conditioning"]["frequency_range"]
    )
    # check compatibility of datasets
    if wfd.domain.domain_dict != asd_dataset.domain.domain_dict:
        raise ValueError(f'wfd.domain: {wfd.domain.domain_dict} \n!= '
                         f'asd_dataset.domain: {asd_dataset.domain.domain_dict}')

    # Add window factor to domain. Can this just be added directly rather than
    # using a second domain instance?
    domain = build_domain(wfd.domain.domain_dict)
    domain.window_factor = get_window_factor(
        data_settings["conditioning"]["window_kwargs"]
    )

    extrinsic_prior_dict = get_extrinsic_prior_dict(data_settings["extrinsic_prior"])
    if data_settings["selected_parameters"] == "default":
        data_settings["selected_parameters"] = default_params

    # If the standardization factors have already been set, use those. Otherwise,
    # calculate them, and save them within the data settings. Note that the order that
    # parameters appear in standardization_dict is the same as the order in the neural
    # network.
    try:
        standardization_dict = data_settings["standardization"]
        print("Using previously-calculated parameter standardizations.")
    except KeyError:
        print("Calculating new parameter standardizations.")
        standardization_dict = get_standardization_dict(
            extrinsic_prior_dict, wfd, data_settings["selected_parameters"]
        )
        data_settings["standardization"] = standardization_dict

    ref_time = data_settings["ref_time"]
    # Build detector objects
    ifo_list = InterferometerList(data_settings["detectors"])

    # Build transforms.
    gnpe_proxy_dim = 0
    transforms = []
    transforms.append(SampleExtrinsicParameters(extrinsic_prior_dict))
    transforms.append(GetDetectorTimes(ifo_list, ref_time))
    # gnpe time shifts
    if "gnpe_time_shifts" in data_settings:
        d = data_settings["gnpe_time_shifts"]
        transforms.append(
            GNPEDetectorTimes(
                ifo_list,
                d["kernel_kwargs"],
                d["exact_equiv"],
                std=standardization_dict["std"]["geocent_time"],
            )
        )
        gnpe_proxy_dim += transforms[-1].gnpe_proxy_dim
    # gnpe chirp mass
    if "gnpe_chirp_mass" in data_settings:
        d = data_settings["gnpe_chirp_mass"]
        transforms.append(
            GNPEChirpMass(
                domain.sample_frequencies_truncated,
                d["kernel_kwargs"],
                mean=standardization_dict["std"]["chirp_mass"],
                std=standardization_dict["std"]["chirp_mass"],
            )
        )
        gnpe_proxy_dim += transforms[-1].gnpe_proxy_dim
    transforms.append(ProjectOntoDetectors(ifo_list, domain, ref_time))
    transforms.append(SampleNoiseASD(asd_dataset))
    transforms.append(WhitenAndScaleStrain(domain.noise_std))
    transforms.append(AddWhiteNoiseComplex())
    transforms.append(SelectStandardizeRepackageParameters(standardization_dict))
    transforms.append(RepackageStrainsAndASDS(data_settings["detectors"]))
    if gnpe_proxy_dim == 0:
        selected_keys = ["parameters", "waveform"]
    else:
        selected_keys = ["parameters", "waveform", "gnpe_proxies"]
    transforms.append(UnpackDict(selected_keys=selected_keys))

    # Drop transforms that are not desired. This is useful for generating, e.g.,
    # noise-free data, or for producing data not formatted for input to the network.
    if omit_transforms is not None:
        transforms = [t for t in transforms if type(t) not in omit_transforms]

    wfd.transform = torchvision.transforms.Compose(transforms)


def build_train_and_test_loaders(
    wfd: WaveformDataset, train_fraction: float, batch_size: int, num_workers: int
):
    """
    Split the dataset into train and test sets, and build corresponding DataLoaders.
    The random split uses a fixed seed for reproducibility.

    Parameters
    ----------
    wfd : WaveformDataset
    train_fraction : float
        Fraction of dataset to use for training. The remainder is used for testing.
        Should lie between 0 and 1.
    batch_size : int
    num_workers : int

    Returns
    -------
    (train_loader, test_loader)
    """

    # Split the dataset. This function uses a fixed seed for reproducibility.
    train_dataset, test_dataset = split_dataset_into_train_and_test(wfd, train_fraction)

    # Build DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=lambda _: np.random.seed(
            int(torch.initial_seed()) % (2 ** 32 - 1)
        ),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=lambda _: np.random.seed(
            int(torch.initial_seed()) % (2 ** 32 - 1)
        ),
    )

    return train_loader, test_loader


def build_svd_for_embedding_network(
    wfd: WaveformDataset,
    data_settings: dict,
    asd_dataset_path: str,
    size: int,
    num_training_samples: int,
    num_validation_samples: int,
    num_workers: int = 0,
    batch_size: int = 1000,
    out_dir=None,
):
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
    waveforms = {
        ifo: np.empty((num_waveforms, waveform_len), dtype=np.complex128)
        for ifo in ifos
    }
    loader = DataLoader(
        wfd,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=lambda _: np.random.seed(
            int(torch.initial_seed()) % (2 ** 32 - 1)
        ),
    )
    for idx, data in enumerate(loader):
        strain_data = data["waveform"]
        lower = idx * batch_size
        n = min(batch_size, num_waveforms - lower)
        for ifo, strains in strain_data.items():
            waveforms[ifo][lower : lower + n] = strains[:n]
        if lower + n == num_waveforms:
            break
    print(f"...done. This took {time.time() - time_start:.0f} s.")

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
        print(f"Testing SVD basis matrices, saving stats to {out_dir}")
        for ifo, basis in basis_dict.items():
            basis.test_basis(
                waveforms[ifo][num_training_samples:],
                outfile=os.path.join(out_dir, f"SVD_{ifo}_stats.npy"),
            )
    print("Done")

    # Return V matrices in standard order.
    return [basis_dict[ifo].V for ifo in data_settings["detectors"]]