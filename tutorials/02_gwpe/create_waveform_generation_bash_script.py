"""
Generate a shell script for the 5 steps in waveform dataset generation.

This is to be run on a single machine and is an alternative to the DAG
generation script which does not require condor.


Workflow:
  1. (a) Parameter file for waveforms used to build the SVD basis:
        python3 ./generate_parameters.py --waveforms_directory ./datasets/waveforms/ --parameters_file parameters.npy --n_samples 2000
     (b) Parameter file for the production waveforms:
        python3 ./generate_parameters.py --waveforms_directory ./datasets/waveforms/ --parameters_file parameters_prod.npy --n_samples 10000

  2. Generate waveforms for SVD basis:
        python3 ./generate_waveforms.py --waveforms_directory ./datasets/waveforms/ --parameters_file parameters.npy --num_wf_per_process 200 --process_id 0
        ... repeat for other chunks

  3. Build SVD basis from polarizations
        python3 ./build_SVD_basis.py --waveforms_directory ./datasets/waveforms/ --parameters_file parameters.npy --basis_file polarization_basis.npy --rb_max 50

  4. Generate production waveforms and project onto SVD basis
        python3 ./generate_waveforms.py --waveforms_directory ./datasets/waveforms/ --parameters_file parameters_prod.npy  --use_compression --basis_file polarization_basis.npy --num_wf_per_process 200 --process_id 0
        ... repeat for other chunks

  5. Consolidate waveform dataset
        python3 ./collect_waveform_dataset.py --waveforms_directory ./datasets/waveforms/ --parameters_file parameters_prod.npy --settings_file settings.yaml --dataset_file waveform_dataset.hdf5

Example command line:
    python3 ./create_waveform_generation_bash_script.py --waveforms_directory ./datasets/waveforms/ --parameters_file_basis parameters_basis.npy --parameters_file_dataset parameters_dataset.npy --basis_file polarization_basis.npy --settings_file settings.yaml --dataset_file waveform_dataset.hdf5 --num_wfs_basis 200 --num_wfs_dataset 500 --rb_max 50 --env_path /home/mpuer/projects/dingo-devel/dingo-devenv
"""


import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="""
        Collect compressed waveform polarizations and parameters.
        Save consolidated waveform dataset in HDF5 format.
    """)
    # dingo script arguments
    parser.add_argument('--waveforms_directory', type=str, required=True,
                        help='Directory containing waveform data, basis, and parameter file.')
    parser.add_argument('--parameters_file_basis', type=str, required=True,
                        help='Parameter file for basis waveforms.')
    parser.add_argument('--parameters_file_dataset', type=str, required=True,
                        help='Parameter file for compressed production waveforms.')
    parser.add_argument('--basis_file', type=str, default='polarization_basis.npy')
    parser.add_argument('--settings_file', type=str, default='settings.yaml')
    parser.add_argument('--dataset_file', type=str, default='waveform_dataset.hdf5')
    parser.add_argument('--num_wfs_basis', type=int, default=1000,
                        help='Number of waveforms to generate for building the SVD basis')
    parser.add_argument('--num_wfs_dataset', type=int, default=10000,
                        help='Number of waveforms to generate for the waveform dataset.')
    parser.add_argument('--num_wf_per_process', type=int, default=1,
                        help='Number of waveforms to generate per process.')
    parser.add_argument('--rb_max', type=int, default=0,
                        help='Truncate the SVD basis at this size. No truncation if zero.')
    parser.add_argument('--num_threads', type=int, default=1,
                        help='Number of threads to use in pool for parallel waveform generation')
    parser.add_argument('--env_path', type=str, help='Absolute path to the dingo Python environment. '
                                                     'We will execute "env_path/bin/activate".')
    parser.add_argument('--script_name', type=str, default='waveform_generation_script.sh')
    parser.add_argument('--logdir', type=str, default='log')

    return parser.parse_args()


def generate_parameter_command(n_samples: int, parameters_file: str, args: argparse.Namespace):
    script = 'generate_parameters.py'
    id_str = script.split('.')[0] + '_' + os.path.basename(parameters_file).split('.')[0]
    out_file = os.path.join(args.logdir, id_str+'.log')

    return f'''$SCRIPT_DIR/{script} \\
    --waveforms_directory {args.waveforms_directory} \\
    --settings_file {args.settings_file} \\
    --parameters_file {parameters_file} \\
    --n_samples {n_samples} > {out_file} 2>&1\n'''

def generate_waveforms_command(parameters_file: str, args: argparse.Namespace,
                               use_compression=False, basis_file=None):
    script = 'generate_waveforms.py'
    id_str = script.split('.')[0] + '_' + os.path.basename(parameters_file).split('.')[0]
    out_file = os.path.join(args.logdir, id_str+'.log')

    cmd = f'''$SCRIPT_DIR/{script} \\
    --waveforms_directory {args.waveforms_directory} \\
    --settings_file {args.settings_file} \\
    --parameters_file {parameters_file} \\
    --num_threads {args.num_threads}'''
    if use_compression and basis_file is not None:
        cmd += f''' \\
    --use_compression \\
    --basis_file {args.basis_file}'''
    cmd += f' > {out_file} 2>&1\n'
    return cmd

def generate_basis_command(parameters_file: str, args: argparse.Namespace):
    script = 'build_SVD_basis.py'
    out_file = os.path.join(args.logdir, script.split('.')[0]+'.log')

    return f'''$SCRIPT_DIR/{script} \\
    --waveforms_directory {args.waveforms_directory} \\
    --parameters_file {parameters_file} \\
    --basis_file {args.basis_file} \\
    --rb_max {args.rb_max} > {out_file} 2>&1\n'''

def collect_waveform_dataset(args: argparse.Namespace):
    script = 'collect_waveform_dataset.py'
    out_file = os.path.join(args.logdir, script.split('.')[0]+'.log')

    return f'''$SCRIPT_DIR/{script} \\
    --waveforms_directory {args.waveforms_directory} \\
    --parameters_file {args.parameters_file_dataset} \\
    --basis_file {args.basis_file} \\
    --settings_file {args.settings_file} \\
    --dataset_file {args.dataset_file} > {out_file} 2>&1\n'''


if __name__ == "__main__":
    args = parse_args()
    script_dir = f'{args.env_path}/bin'

    with open(args.script_name, 'w') as fp:
        doc = f'#!/bin/bash\n\nsource {args.env_path}/bin/activate\n'
        doc += f'\nmkdir -p {args.logdir}\n'
        doc += f'SCRIPT_DIR={script_dir}\n'

        doc += '\necho "Step (1): Generate parameter files"\n'
        doc += generate_parameter_command(args.num_wfs_basis,
                                          args.parameters_file_basis, args)
        doc += '\n'
        doc += generate_parameter_command(args.num_wfs_dataset,
                                          args.parameters_file_dataset, args)

        doc += '\necho "Step (2): Generate waveforms for SVD basis"\n'
        doc += generate_waveforms_command(args.parameters_file_basis, args)

        doc += '\necho "Step (3): Build SVD basis from polarizations"\n'
        doc += generate_basis_command(args.parameters_file_basis, args)

        doc += '\necho "Step (4): Generate production waveforms and project onto SVD basis"\n'
        doc += generate_waveforms_command(args.parameters_file_dataset, args,
                                          use_compression=True, basis_file=args.basis_file)

        doc += '\necho "Step (5): Consolidate waveform dataset"\n'
        doc += collect_waveform_dataset(args)
        doc += '\n'
        fp.writelines(doc)

