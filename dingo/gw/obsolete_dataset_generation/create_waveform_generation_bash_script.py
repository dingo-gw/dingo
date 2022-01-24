"""
Generate a shell script for the 5 steps in waveform dataset generation.
"""


import argparse
import os
import textwrap
import yaml

from .generate_parameters import PARAMETERS_FILE_BASIS, PARAMETERS_FILE_DATASET,\
    BASIS_FILE, SETTINGS_FILE, DATASET_FILE


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
        Build a shell script for waveform dataset generation.
        
        About the generated shell script:
        
          * It samples parameters from the intrinsic prior, generates
            waveform polarizations, projects them onto an SVD basis,
            and saves the consolidated waveform dataset in HDF5 format.
        
          * It runs on a single machine / node and is an alternative to 
            the DAG generation script which requires condor. It can
            use parallelization via a thread pool.
        
        
        Workflow:
        
          1. (a) Parameter file for waveforms used to build the SVD basis:
                generate_parameters.py
             (b) Parameter file for the production waveforms:
                generate_parameters.py
        
          2. Generate waveforms for SVD basis:
                generate_waveforms.py
                There is only a single data chunk since we are running on a single node.
                I.e. we have a single data file for each polarization.
        
          3. Build SVD basis from polarizations:
                build_SVD_basis.py
        
          4. Generate production waveforms and project onto SVD basis
                generate_waveforms.py
                Again, there is only a single data chunk and a single data file for each polarization.
        
          5. Consolidate waveform dataset
                collect_waveform_dataset.py
        
        
        Example invocation:
        
            ./env/bin/create_waveform_generation_bash_script
                --waveforms_directory ./datasets/waveforms/
                --env_path ./env
                --num_threads 4
                
            In addition to command line arguments this script requires a yaml
            settings_file to be present in the waveforms_directory which
            specifies physical parameters.
        
            After executing this script run the generated bash script which 
            carries out the five steps of the workflow described above.
    """)
    )

    # dingo script arguments
    parser.add_argument('--waveforms_directory', type=str, required=True,
                        help='Directory containing waveform data, basis, and parameter file.')
    parser.add_argument('--num_threads', type=int, default=1,
                        help='Number of threads to use in pool for parallel waveform generation')
    parser.add_argument('--env_path', type=str, help='Absolute path to the dingo Python environment. '
                                                     'We will execute scripts in  "env_path/bin".')
    parser.add_argument('--script_name', type=str, default='waveform_generation_script.sh')
    parser.add_argument('--logdir', type=str, default='log')

    return parser.parse_args(args=args)


def generate_parameter_command(n_samples: int, parameters_file: str,
                               args: argparse.Namespace, log_file):
    """
    Generate command string for 'generate_parameters' task.
    """
    script = 'generate_parameters'
    id_str = script + '_' + os.path.basename(parameters_file).split('.')[0]

    return f'''python3 $SCRIPT_DIR/{script} \\
    --waveforms_directory {args.waveforms_directory} \\
    --settings_file {SETTINGS_FILE} \\
    --parameters_file {parameters_file} \\
    --n_samples {n_samples} > {log_file} 2>&1\n'''


def generate_waveforms_command(parameters_file: str, num_wfs: int,
                               args: argparse.Namespace, log_file,
                               use_compression=False, basis_file=None):
    """
    Generate command string for 'generate_waveforms' task.
    """
    script = 'generate_waveforms'
    id_str = script + '_' + os.path.basename(parameters_file).split('.')[0]

    cmd = f'''python3 $SCRIPT_DIR/{script} \\
    --waveforms_directory {args.waveforms_directory} \\
    --settings_file {SETTINGS_FILE} \\
    --parameters_file {parameters_file} \\
    --process_id 0 \\
    --num_wf_per_process {num_wfs} \\
    --num_threads {args.num_threads}'''
    if use_compression and basis_file is not None:
        cmd += f''' \\
    --use_compression \\
    --basis_file {basis_file}'''
    cmd += f' > {log_file} 2>&1\n'
    return cmd


def generate_basis_command(parameters_file: str,
                           args: argparse.Namespace, log_file):
    """
    Generate command string for 'build_SVD_basis' task.
    """
    script = 'build_SVD_basis'

    return f'''python3 $SCRIPT_DIR/{script} \\
    --waveforms_directory {args.waveforms_directory} \\
    --parameters_file {parameters_file} \\
    --basis_file {BASIS_FILE} \\
    --rb_max {args.rb_max} \\
    --rb_train_fraction {args.rb_train_fraction} > {log_file} 2>&1\n'''


def collect_waveform_dataset(args: argparse.Namespace, log_file):
    """
    Generate command string for 'collect_waveform_dataset' task.
    """

    script = 'collect_waveform_dataset'

    return f'''python3 $SCRIPT_DIR/{script} \\
    --waveforms_directory {args.waveforms_directory} \\
    --parameters_file {PARAMETERS_FILE_DATASET} \\
    --basis_file {BASIS_FILE} \\
    --settings_file {SETTINGS_FILE} \\
    --dataset_file {DATASET_FILE} > {log_file} 2>&1\n'''


def generate_workflow(args):
    script_dir = f'{args.env_path}/bin'
    log_file = os.path.join(args.logdir, 'create_waveform_generation_bash_script.log')

    settings_path = os.path.join(args.waveforms_directory, SETTINGS_FILE)
    with open(settings_path, 'r') as fp:
        settings = yaml.safe_load(fp)

    # Add the parameters from the settings file to args
    args.__dict__.update(settings['waveform_dataset_generation_settings'])

    with open(args.script_name, 'w') as fp:
        doc = f'#!/bin/bash\n\nsource {args.env_path}/bin/activate\n'
        doc += f'\nmkdir -p {args.logdir}\n'
        doc += f'SCRIPT_DIR={script_dir}\n'

        doc += '\necho "Step (1): Generate parameter files"\n'
        doc += generate_parameter_command(args.num_wfs_basis,
                                          PARAMETERS_FILE_BASIS, args, log_file)
        doc += '\n'
        doc += generate_parameter_command(args.num_wfs_dataset,
                                          PARAMETERS_FILE_DATASET, args, log_file)

        doc += '\necho "Step (2): Generate waveforms for SVD basis"\n'
        doc += generate_waveforms_command(PARAMETERS_FILE_BASIS,
                                          args.num_wfs_basis, args, log_file)

        doc += '\necho "Step (3): Build SVD basis from polarizations"\n'
        doc += generate_basis_command(PARAMETERS_FILE_BASIS, args, log_file)

        doc += '\necho "Step (4): Generate production waveforms and project onto SVD basis"\n'
        doc += generate_waveforms_command(PARAMETERS_FILE_DATASET, args.num_wfs_dataset,
                                          args, log_file, use_compression=True,
                                          basis_file=BASIS_FILE)

        doc += '\necho "Step (5): Consolidate waveform dataset"\n'
        doc += collect_waveform_dataset(args, log_file)
        doc += '\n'
        fp.writelines(doc)

    print(f'Workflow written to {args.script_name}.')


def main():
    args = parse_args()
    generate_workflow(args)


if __name__ == "__main__":
    main()

