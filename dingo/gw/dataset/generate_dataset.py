import copy
import textwrap
import yaml
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from threadpoolctl import threadpool_limits
from torchvision.transforms import Compose
from bilby.gw.prior import BBHPriorDict

from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.prior import build_prior_with_defaults, default_intrinsic_dict
from dingo.gw.domains import build_domain, build_domain_from_wfd_settings
from dingo.gw.transforms import WhitenFixedASD
from dingo.gw.waveform_generator import (
    WaveformGenerator,
    NewInterfaceWaveformGenerator,
    generate_waveforms_parallel,
)
from dingo.gw.SVD import SVDBasis, ApplySVD


def generate_parameters_and_polarizations(
    waveform_generator: WaveformGenerator,
    prior: BBHPriorDict,
    num_samples: int,
    num_processes: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Generate a dataset of waveforms based on parameters drawn from the prior.

    Parameters
    ----------
    waveform_generator : WaveformGenerator
    prior : Prior
    num_samples : int
    num_processes : int

    Returns
    -------
    pandas DataFrame of parameters
    dictionary of numpy arrays corresponding to waveform polarizations
    """
    print("Generating dataset of size " + str(num_samples))
    parameters = pd.DataFrame(prior.sample(num_samples))

    if num_processes > 1:
        with threadpool_limits(limits=1, user_api="blas"):
            with Pool(processes=num_processes) as pool:
                polarizations = generate_waveforms_parallel(
                    waveform_generator, parameters, pool
                )
    else:
        polarizations = generate_waveforms_parallel(waveform_generator, parameters)

    # Find cases where waveform generation failed and only return data for successful ones
    wf_failed = np.any(np.isnan(polarizations["h_plus"]), axis=1)
    if wf_failed.any():
        idx_failed = np.where(wf_failed)[0]
        idx_ok = np.where(~wf_failed)[0]
        polarizations_ok = {k: v[idx_ok] for k, v in polarizations.items()}
        parameters_ok = parameters.iloc[idx_ok]
        failed_percent = 100 * len(idx_failed) / len(parameters)
        print(
            f"{len(idx_failed)} out of {len(parameters)} configuration ({failed_percent:.1f}%) failed to generate."
        )
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(parameters.iloc[idx_failed])
        print(
            f"Only returning the {len(idx_ok)} successfully generated configurations."
        )
        return parameters_ok, polarizations_ok

    return parameters, polarizations


def train_svd_basis(dataset: WaveformDataset, size: int, n_train: int):
    """
    Train (and optionally validate) an SVD basis.

    Parameters
    ----------
    dataset : WaveformDataset
        Contains waveforms to be used for building SVD.
    size : int
        Number of elements to keep for the SVD basis.
    n_train : int
        Number of training waveforms to use. Remaining are used for validation. Note
        that the actual number of training waveforms is n_train * len(polarizations),
        since there is one waveform used for each polarization.

    Returns
    -------
    SVDBasis, n_train, n_test
        Since EOB waveforms can fail to generate, provide also the number used in
        training and validation.
    """
    # Prepare data for training and validation.
    train_data = np.vstack([val[:n_train] for val in dataset.polarizations.values()])
    test_data = np.vstack([val[n_train:] for val in dataset.polarizations.values()])
    test_parameters = pd.concat(
        [
            # I would like to save the polarization, but saving the dataframe with
            # string columns causes problems. Fix this later.
            # dataset.parameters.iloc[n_train:].assign(polarization=pol)
            dataset.parameters.iloc[n_train:]
            for pol in dataset.polarizations
        ]
    )
    test_parameters.reset_index(drop=True, inplace=True)

    print("Building SVD basis.")
    basis = SVDBasis()
    basis.generate_basis(train_data, size)

    assert np.allclose(basis.V[: dataset.domain.min_idx], 0)

    # Since there is a possibility that the size of the dataset returned by
    # generate_parameters_and_polarizations is smaller than requested, we don't assume
    # that there are n_test samples. Instead we just look at the size of the test
    # dataset.
    if test_data.size != 0:
        basis.compute_test_mismatches(
            test_data, parameters=test_parameters, verbose=True
        )

    # Return also the true number of samples. Some EOB waveforms may have failed to
    # generate, so this could be smaller than the number requested.
    n_ifos = len(dataset.polarizations)
    n_train = len(train_data) // n_ifos
    n_test = len(test_data) // n_ifos

    return basis, n_train, n_test


def generate_dataset(settings: Dict, num_processes: int) -> WaveformDataset:
    """
    Generate a waveform dataset.

    Parameters
    ----------
    settings : dict
        Dictionary of settings to configure the dataset
    num_processes : int

    Returns
    -------
    A WaveformDataset based on the settings.
    """

    prior = build_prior_with_defaults(settings["intrinsic_prior"])
    domain = build_domain(settings["domain"])

    new_interface_flag = settings["waveform_generator"].get("new_interface", False)
    if new_interface_flag:
        waveform_generator = NewInterfaceWaveformGenerator(
            domain=domain,
            **settings["waveform_generator"],
        )
    else:
        waveform_generator = WaveformGenerator(
            domain=domain,
            **settings["waveform_generator"],
        )

    dataset_dict = {"settings": settings}

    if "compression" in settings:
        compression_transforms = []

        if "whitening" in settings["compression"]:
            compression_transforms.append(
                WhitenFixedASD(
                    domain,
                    asd_file=settings["compression"]["whitening"],
                    inverse=False,
                )
            )

        if "svd" in settings["compression"]:
            svd_settings = settings["compression"]["svd"]

            # Load an SVD basis from file, if specified.
            if "file" in svd_settings:
                basis = SVDBasis(file_name=svd_settings["file"])

            # Otherwise, generate the basis based on simulated waveforms.
            else:
                # If using whitened waveforms, then the SVD should be based on these.
                waveform_generator.transform = Compose(compression_transforms)

                n_train = svd_settings["num_training_samples"]
                n_test = svd_settings.get("num_validation_samples", 0)
                parameters, polarizations = generate_parameters_and_polarizations(
                    waveform_generator,
                    prior,
                    n_train + n_test,
                    num_processes,
                )
                svd_dataset_settings = copy.deepcopy(settings)
                svd_dataset_settings["num_samples"] = len(parameters)
                del svd_dataset_settings["compression"]["svd"]

                # We build a WaveformDataset containing the SVD-training waveforms
                # because when constructed, it will automatically zero the waveforms
                # below f_min. This is useful for EOB waveforms, which are Fourier
                # transformed from time domain, and hence are nonzero below f_min. The
                # waveforms need to be zeroed below f_min because this corresponds to
                # setting the lower bound of the likelihood integral.

                svd_dataset = WaveformDataset(
                    dictionary={
                        "parameters": parameters,
                        "polarizations": polarizations,
                        "settings": svd_dataset_settings,
                    }
                )
                basis, n_train, n_test = train_svd_basis(
                    svd_dataset, svd_settings["size"], n_train
                )
                # Reset the true number of samples, in case this has changed due to
                # failure to generate some EOB waveforms.
                svd_settings["num_training_samples"] = n_train
                svd_settings["num_validation_samples"] = n_test

            compression_transforms.append(ApplySVD(basis))
            dataset_dict["svd"] = basis.to_dictionary()

        waveform_generator.transform = Compose(compression_transforms)

    # Generate main dataset
    parameters, polarizations = generate_parameters_and_polarizations(
        waveform_generator, prior, settings["num_samples"], num_processes
    )
    dataset_dict["parameters"] = parameters
    dataset_dict["polarizations"] = polarizations
    # Update to take into account potentially failed configurations
    dataset_dict[settings["num_samples"]] = len(parameters)

    dataset = WaveformDataset(dictionary=dataset_dict)
    return dataset


def autocomplete_wfd_settings(wfd_settings: dict, num_processes: int = 1):
    """
    Autocomplete dict with settings for waveform dataset generation. This includes
    the prior settings ('default' priors are explicitly set) and the domain
    (e.g., band generation for MultibandedFrequencyDomain).
    """

    wfd_settings = copy.deepcopy(wfd_settings)

    # complete default values for prior
    for k, v in wfd_settings["intrinsic_prior"].items():
        if v == "default":
            wfd_settings["intrinsic_prior"][k] = default_intrinsic_dict[k]

    # build domain
    domain = build_domain_from_wfd_settings(wfd_settings, num_processes)
    wfd_settings["domain"] = domain.domain_dict

    return wfd_settings


def autocomplete_wfd_settings_cli():
    """
    Command line script for autocompletion of wfd settings.
    """
    parser = argparse.ArgumentParser(description="Autocomplete wfd settings.")
    parser.add_argument("--settings_file", type=str, required=True)
    parser.add_argument("--num_processes", type=int, default=1)
    args = parser.parse_args()
    with open(args.settings_file, "r") as f:
        wfd_settings = yaml.safe_load(f)
    wfd_settings_autocompleted = autocomplete_wfd_settings(
        wfd_settings, args.num_processes
    )
    out_file = (
        ".".join(args.settings_file.split(".")[:-1])
        + "_autocompleted."
        + args.settings_file.split(".")[-1]
    )
    with open(out_file, "w") as f:
        yaml.dump(wfd_settings_autocompleted, f, default_flow_style=False)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
        Generate a waveform dataset based on a settings file.
        """
        ),
    )
    parser.add_argument(
        "--settings_file",
        type=str,
        required=True,
        help="YAML file containing database settings",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes to use in pool for parallel waveform generation",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="waveform_dataset.hdf5",
        help="Name of file for storing dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load settings
    with open(args.settings_file, "r") as f:
        settings = yaml.safe_load(f)

    dataset = generate_dataset(settings, args.num_processes)
    dataset.to_file(args.out_file)


if __name__ == "__main__":
    main()
