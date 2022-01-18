# File from Nihar

import yaml
import argparse
from torchvision.transforms import Compose
from multiprocessing import Pool
import pandas as pd
import h5py

from .dataset_utils import structured_array_from_dict_of_arrays
from ..prior import build_prior_with_defaults
from ..domains import build_domain
from ..waveform_generator import WaveformGenerator
from .generate_waveforms import generate_dataset_old


def generate_dataset(settings, num_processes):

    prior = build_prior_with_defaults(settings['intrinsic_prior'])
    domain = build_domain(settings['domain'])
    waveform_generator = WaveformGenerator(settings['waveform_generator']['approximant'],
                                           domain,
                                           settings['waveform_generator']['f_ref'])

    # if 'compression' in settings:
    #     compression_transforms = []
    #     if 'SVD' in settings['compression']:
    #         compression_transforms.append(init_SVD_compression(prior,
    #                                                            waveform_generator,
    #                                                            settings['compression']['SVD']))
    #     waveform_generator.transform = Compose(compression_transforms)

    parameter_samples = pd.DataFrame(prior.sample(settings['num_samples']))
    with Pool(processes=num_processes) as pool:
        polarizations = generate_dataset_old(waveform_generator, parameter_samples, pool)

    return parameter_samples, polarizations


def save_dataset(parameters, polarizations, settings, file_name):

    f = h5py.File(file_name, 'w')

    f.create_dataset('parameters', data=parameters)

    grp = f.create_group('polarizations')
    for pol, waveforms in polarizations.items():
        grp.create_dataset(pol, data=waveforms)

    f.attrs['settings'] = str(settings)

    f.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings_file', type=str, required=True,
                        help='Directory containing waveform data, basis, and parameter file.')
    parser.add_argument('--num_processes', type=int, default=1,
                        help='Number of threads to use in pool for parallel waveform generation')
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--logdir', type=str, default='log')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load settings
    with open(args.settings_file, 'r') as f:
        settings = yaml.safe_load(f)

    parameters, polarizations = generate_dataset(settings, args.num_processes)

    save_dataset(parameters, polarizations, settings, args.out_file)


if __name__ == "__main__":
    main()

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
