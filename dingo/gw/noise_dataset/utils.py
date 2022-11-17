from os.path import join
from typing import List

import numpy as np

from dingo.gw.noise_dataset.ASD_dataset import ASDDataset


def get_path_raw_data(data_dir, run, detector):
    """
    Return the directory where the PSD data is to be stored
    Parameters
    ----------
    data_dir : str
        Path to the directory where the PSD dataset will be stored
    run : str
        Observing run that is used for the PSD dataset generation
    detector : str
        Detector that is used for the PSD dataset generation

    Returns
    -------
    the path where the data is stored
    """
    return join(data_dir, "tmp", run, detector)


def get_psds_from_params_dict(params, frequencies, scale_factor, smoothen=False):

    xs = params["x_positions"]
    ys = params["y_values"]
    spectral_features = params["spectral_features"]
    variance = params["variance"]

    num_psds = ys.shape[0]
    num_spectral_segments = params["spectral_features"].shape[1]
    frequency_segments = np.array_split(
        np.arange(frequencies.shape[0]), num_spectral_segments
    )

    psds = np.zeros((num_psds, len(frequencies)))
    for i in range(num_psds):
        spline = scipy.interpolate.CubicSpline(xs, ys[i, :])
        base_noise = spline(frequencies)

        lorentzians = np.array([])
        for j, seg in enumerate(frequency_segments):
            f0, A, Q = spectral_features[i, j, :]
            lorentzian = lorentzian_eval(frequencies[seg], f0, A, Q)
            # small amplitudes are not modeled to maintain smoothness
            if np.max(lorentzian) <= 3 * variance:
                lorentzian = np.zeros_like(frequencies[seg])
            lorentzians = np.concatenate((lorentzians, lorentzian), axis=0)
        assert lorentzians.shape == frequencies.shape

        if smoothen:
            psds[i, :] = np.exp(base_noise + lorentzians) / scale_factor
        else:
            psds[i, :] = (
                np.exp(np.random.normal(base_noise + lorentzians, variance))
                / scale_factor
            )
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(16,9))
        # plt.xlim(20, 2000)
        # plt.ylim(1.e-47, 1.e-42)
        # # plt.loglog(frequencies, np.exp(np.random.normal(base_noise + lorentzians, variance)) / scale_factor)
        # plt.loglog(frequencies, psds[i, :])
        # plt.savefig("/work/jwildberger/dingo-dev/review/ASD_datasets/debug.pdf")
        # exit(1)
    return psds


# TODO: make this more generic to load arbitrary numpy files as dict
def load_params_from_files(psd_paths):
    times = np.zeros(len(psd_paths))
    # initialize parameter dict
    psd = np.load(join(psd_paths[0]), allow_pickle=True).item()

    parameters = {
        "x_positions": psd["parameters"]["x_positions"],
        "y_values": np.zeros(
            (len(psd_paths), psd["parameters"]["x_positions"].shape[0])
        ),
        "spectral_features": np.zeros(
            (
                len(psd_paths),
                psd["parameters"]["spectral_features"].shape[0],
                psd["parameters"]["spectral_features"].shape[1],
            )
        ),
        "variance": psd["parameters"]["variance"],
    }

    for ind, filename in enumerate(psd_paths):
        psd = np.load(filename, allow_pickle=True).item()
        parameters["spectral_features"][ind, :, :] = psd["parameters"][
            "spectral_features"
        ]
        parameters["y_values"][ind, :] = psd["parameters"]["y_values"]
        times[ind] = psd["time"][0]

    return parameters, times


def merge_datasets(dataset_list: List[ASDDataset]) -> ASDDataset:
    """
    Merge a collection of datasets into one.

    Parameters
    ----------
    dataset_list : list[ASDDataset]
        A list of ASDDatasets. Each item should be a dictionary containing
        parameters and polarizations.

    Returns
    -------
    ASDDataset containing the merged data.
    """

    print(f"Merging {len(dataset_list)} datasets into one.")

    # This ensures that all of the keys are copied into the new dataset. The
    # "extensive" parts of the dataset (parameters, waveforms) will be overwritten by
    # the combined datasets, whereas the "intensive" parts (e.g., SVD basis, settings)
    # will take the values in the *first* dataset in the list.
    merged_dict = copy.deepcopy(dataset_list[0].to_dictionary())

    merged_dict["parameters"] = pd.concat([d.parameters for d in dataset_list])
    merged_dict["polarizations"] = {}
    polarizations = list(dataset_list[0].polarizations.keys())
    for pol in polarizations:
        # We pop the data array off of each of the polarizations dicts to save memory.
        # Otherwise this operation doubles the total amount of memory used. This is
        # destructive to the original datasets.
        merged_dict["polarizations"][pol] = np.vstack(
            [d.polarizations.pop(pol) for d in dataset_list]
        )

    # Update the settings based on the total number of samples.
    merged_dict["settings"]["num_samples"] = len(merged_dict["parameters"])

    merged = ASDDataset(dictionary=merged_dict)

    return merged


def merge_datasets_cli():
    """
    Command-line function to combine a collection of datasets into one. Used for
    parallelized waveform generation.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
        Combine a collection of datasets into one.

        Datasets must be in sequentially-labeled HDF5 files with a fixed prefix. 
        The settings for the new dataset will be based on those of the first file. 
        Optionally, replace the settings with those specified in a YAML file.
        """
        ),
    )
    parser.add_argument(
        "--prefix", type=str, required=True, help="Prefix of sequential files names."
    )
    parser.add_argument(
        "--num_parts",
        type=int,
        required=True,
        help="Total number of datasets to merge.",
    )
    parser.add_argument(
        "--out_file", type=str, required=True, help="Name of file for new dataset."
    )
    parser.add_argument(
        "--settings_file", type=str, help="YAML file containing new dataset settings."
    )
    args = parser.parse_args()

    dataset_list = []
    for i in range(args.num_parts):
        file_name = args.prefix + str(i) + ".hdf5"
        dataset_list.append(ASDDataset(file_name=file_name))
    merged_dataset = merge_datasets(dataset_list)

    # Optionally, update the settings file based on that provided at command line.
    if args.settings_file is not None:
        with open(args.settings_file, "r") as f:
            settings = yaml.safe_load(f)
        # Make sure num_samples is correct
        settings["num_samples"] = len(merged_dataset.parameters)
        merged_dataset.settings = settings

    merged_dataset.to_file(args.out_file)
    print(
        f"Complete. New dataset consists of {merged_dataset.settings['num_samples']} "
        f"samples."
    )


def build_svd_cli():
    """
    Command-line function to build an SVD based on an uncompressed dataset file.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
        Build an SVD basis based on a set of waveforms.
        """
        ),
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="HDF5 file containing training waveforms.",
    )
    parser.add_argument(
        "--size", type=int, required=True, help="Number of basis elements to keep."
    )
    parser.add_argument(
        "--out_file", type=str, required=True, help="Name of file for saving SVD."
    )
    parser.add_argument(
        "--num_train",
        type=int,
        help="Number of waveforms to use for training SVD. "
             "Remainder are used for validation.",
    )
    args = parser.parse_args()

    dataset = ASDDataset(file_name=args.dataset_file)
    if args.num_train is None:
        n_train = len(ASDDataset)
    else:
        n_train = args.num_train

    basis, n_train, n_test = train_svd_basis(dataset, args.size, n_train)
    # FIXME: This is not an ideal treatment. We should update the waveform generation
    #  to always provide the requested number of waveforms.
    print(
        f"SVD basis trained based on {n_train} waveforms and validated on {n_test} "
        f"waveforms. Note that if this differs from number requested, it will not be "
        f"reflected in the settings file. This is likely due to EOB failure to "
        f"generate certain waveforms."
    )
    basis.to_file(args.out_file)
