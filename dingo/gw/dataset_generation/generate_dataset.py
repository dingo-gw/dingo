import yaml
import argparse
from multiprocessing import Pool
import pandas as pd
import numpy as np
import h5py

from ..prior import build_prior_with_defaults
from ..domains import build_domain
from ..waveform_generator import WaveformGenerator, generate_waveforms_parallel
from torchvision.transforms import Compose
from ..SVD import SVDBasis, ApplySVD


def generate_parameters_and_polarizations(
    waveform_generator, prior, num_samples, num_processes
):
    parameters = pd.DataFrame(prior.sample(num_samples))
    with Pool(processes=num_processes) as pool:
        polarizations = generate_waveforms_parallel(
            waveform_generator, parameters, pool
        )
    return parameters, polarizations


def generate_dataset(settings, num_processes):

    prior = build_prior_with_defaults(settings["intrinsic_prior"])
    domain = build_domain(settings["domain"])
    waveform_generator = WaveformGenerator(
        settings["waveform_generator"]["approximant"],
        domain,
        settings["waveform_generator"]["f_ref"],
    )

    dataset = {}

    if "compression" in settings:
        compression_transforms = []

        if "svd" in settings["compression"]:
            svd_settings = settings["compression"]["svd"]

            # Load an SVD basis from file, if specified.
            if 'file' in svd_settings:
                basis = SVDBasis()
                basis.from_file(svd_settings['file'])

            # Otherwise, generate the basis based on simulated waveforms.
            else:
                parameters, polarizations = generate_parameters_and_polarizations(
                    waveform_generator,
                    prior,
                    svd_settings["num_training_samples"],
                    num_processes,
                )
                train_data = np.vstack(list(polarizations.values()))
                basis = SVDBasis()
                basis.generate_basis(train_data, svd_settings["size"])

            compression_transforms.append(ApplySVD(basis))
            dataset["svd_V"] = basis.V

        waveform_generator.transform = Compose(compression_transforms)

    # Generate main dataset
    parameters, polarizations = generate_parameters_and_polarizations(
        waveform_generator, prior, settings["num_samples"], num_processes
    )
    dataset["parameters"] = parameters
    dataset["polarizations"] = polarizations

    return dataset


def recursive_hdf5_save(group, d):
    for k, v in d.items():
        if isinstance(v, dict):
            next_group = group.create_group(k)
            recursive_hdf5_save(next_group, v)
        else:
            group.create_dataset(k, data=v)


def save_dataset(dataset, settings, file_name):
    f = h5py.File(file_name, "w")
    recursive_hdf5_save(f, dataset)
    f.attrs["settings"] = str(settings)
    f.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--settings_file",
        type=str,
        required=True,
        help="Directory containing waveform data, basis, and parameter file.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes to use in pool for parallel waveform generation",
    )
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--logdir", type=str, default="log")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load settings
    with open(args.settings_file, "r") as f:
        settings = yaml.safe_load(f)

    dataset = generate_dataset(settings, args.num_processes)

    save_dataset(dataset, settings, args.out_file)


if __name__ == "__main__":
    main()

# File from Nihar
#
# cwd = "/home/n/Documents/Research/dingo/dingo-devel/tutorials/03_aligned_spin"
# settings_file = f"{cwd}/datasets/waveforms/settings.yaml"
#
# with open(settings_file, "r") as fp:
#     settings = yaml.safe_load(fp)
#
#
# wf_dict = {
#     "waveforms_directory": f"{cwd}/datasets/waveforms",
#     "env_path": "/home/n/Documents/Research/dingo/dingo-devel/venv",
#     "num_threads": 1,
#     "logdir": f"{cwd}/../../logs",
#     "settings_file": settings_file,
#     "parameters_basis": f"{cwd}/datasets/waveforms/parameters_basis.npy",
#     "parameters_dataset": f"{cwd}/datasets/waveforms/parameters_dataset.npy",
#     "process_id": 0,
#     "num_wf_per_process": 1000,
#     "polarization_basis_fname": "polarization_basis.npy",
#     "dataset_fname": "waveform_dataset.hdf5",
# }
# # Adding the waveform_dataset_generation_settings to wf_dict for use in step 3
# wf_dict = dict(wf_dict, **settings["waveform_dataset_generation_settings"])
#
#
# # Step (1): Generate parameter files
# generate_parameters_(
#     waveforms_directory=wf_dict["waveforms_directory"],
#     settings_file=wf_dict["settings_file"],
#     parameters_file=wf_dict["parameters_basis"],
#     n_samples=wf_dict["num_wfs_basis"],
# )
#
# # Parameter files for dataset
# generate_parameters_(
#     waveforms_directory=wf_dict["waveforms_directory"],
#     settings_file=wf_dict["settings_file"],
#     parameters_file=wf_dict["parameters_dataset"],
#     n_samples=wf_dict["num_wfs_dataset"],
# )
#
# # Step (2): Generate waveforms for SVD basis
# generate_waveforms_(
#     waveforms_directory=wf_dict["waveforms_directory"],
#     settings_file=wf_dict["settings_file"],
#     basis_file=None,
#     num_threads=wf_dict["num_threads"],
#     num_wf_per_process=wf_dict["num_wf_per_process"],
#     parameters_file=wf_dict["parameters_basis"],
#     process_id=wf_dict["process_id"],
#     use_compression=False,
# )
#
# # Step (3): Build SVD basis from polarizations"
# build_svd_basis_(
#     waveforms_directory=wf_dict["waveforms_directory"],
#     basis_file=wf_dict["polarization_basis_fname"],
#     parameters_file=wf_dict["parameters_basis"],
#     rb_max=wf_dict["rb_max"],
#     rb_train_fraction=wf_dict["rb_train_fraction"],
# )
#
# # Step (4): Generate production waveforms and project onto SVD basis
# generate_waveforms_(
#     waveforms_directory=wf_dict["waveforms_directory"],
#     settings_file=wf_dict["settings_file"],
#     basis_file=wf_dict["polarization_basis_fname"],
#     num_threads=wf_dict["num_threads"],
#     num_wf_per_process=wf_dict["num_wf_per_process"],
#     parameters_file=wf_dict["parameters_dataset"],
#     process_id=wf_dict["process_id"],
#     use_compression=True,
# )
#
# # Step (5): Consolidate waveform dataset
# consolidate_dataset_(
#     waveforms_directory=wf_dict["waveforms_directory"],
#     parameters_file=wf_dict["parameters_dataset"],
#     basis_file=wf_dict["polarization_basis_fname"],
#     dataset_file=wf_dict["dataset_fname"],
# )
