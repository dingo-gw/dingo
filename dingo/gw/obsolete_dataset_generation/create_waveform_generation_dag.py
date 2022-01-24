"""
Setup a condor DAG for waveform dataset generation using pycondor.
"""


import argparse
import os
import textwrap
from typing import Dict
from pycondor import Job, Dagman

os.environ['OMP_NUM_THREADS'] = str(1)
os.environ['MKL_NUM_THREADS'] = str(1)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
        Build a condor DAG for waveform dataset generation.
        
        About the generated directed acyclic graph (DAG):
        
          * It samples parameters from the intrinsic prior, generates
            waveform polarizations, projects them onto an SVD basis,
            and saves the consolidated waveform dataset in HDF5 format.
        
          * The waveform generation tasks are parallelized over the 
            specified number of compute nodes. On each node it can
            use parallelization via a thread pool.
        
        
        Workflow:
        
          1. (a) Parameter file for waveforms used to build the SVD basis:
                generate_parameters.py
             (b) Parameter file for the production waveforms:
                generate_parameters.py
        
          2. Generate waveforms for SVD basis:
                generate_waveforms.py
                The parameter array is divided into chunks so that we
                generate a specified number waveforms per node.
                The polarization data on each node is saved into
                one file for h_plus and h_cross.
        
          3. Build SVD basis from polarizations:
                build_SVD_basis.py
        
          4. Generate production waveforms and project onto SVD basis
                generate_waveforms.py
                As above data is saved as many chunks as there are nodes.
        
          5. Consolidate waveform dataset
                collect_waveform_dataset.py
        
        
        Example invocation:
        
            ./env/bin/create_waveform_generation_dag
              --waveforms_directory ./datasets/waveforms/
              --parameters_file_basis parameters_basis.npy
              --parameters_file_dataset parameters_dataset.npy
              --basis_file polarization_basis.npy
              --settings_file settings.yaml
              --dataset_file waveform_dataset.hdf5
              --num_wfs_basis 5000
              --num_wfs_dataset 10000
              --num_wf_per_process 500
              --rb_max 500
              --env_path ./env
        
            In addition to command line arguments this script requires a yaml
            settings_file to be present in the waveforms_directory which
            specifies physical parameters.
        
            After executing this script submit the generated condor DAG
            in the submit directory with `condor_submit_dag`.
            It will carry out the five steps of the workflow described above. 
    """)
    )

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
    parser.add_argument('--env_path', type=str, required=True,
                        help='Absolute path to the dingo Python environment. '
                             'We will execute scripts in "env_path/bin/".')

    # condor arguments
    parser.add_argument('--request_cpus', type=int, default=None)
    parser.add_argument('--request_memory', type=int, default=None)
    parser.add_argument('--error', type=str, default='condor/error')
    parser.add_argument('--output', type=str, default='condor/output')
    parser.add_argument('--log', type=str, default='condor/log')
    parser.add_argument('--submit', type=str, default='condor/submit')

    return parser.parse_args()


def modulus_check(a: int, b: int,
                  a_label: str, b_label: str):
    """
    Raise error if a % b != 0.
    """
    if a % b != 0:
        raise ValueError(f'Expected {a_label} mod {b_label} to be zero. '
                         f'But got {a} mod {b} = {a % b}.')


def create_args_string(args_dict: Dict):
    """
    Generate argument string from dictionary of
    argument names and arguments.
    """
    return ''.join([f'--{k} {v} ' for k, v in args_dict.items()])


def create_dag(args):
    """
    Create a Condor DAG from command line arguments to
    carry out the five steps in the workflow.
    """
    kwargs = {'request_cpus': args.request_cpus, 'request_memory': args.request_memory,
              'submit': args.submit, 'error': args.error, 'output': args.output,
              'log': args.log, 'getenv': True}

    # Number of chunks the datasets are split into by
    # producing args.num_wf_per_process waveforms per node
    num_chunks_basis = args.num_wfs_basis // args.num_wf_per_process
    num_chunks_dataset = args.num_wfs_dataset // args.num_wf_per_process

    # scripts are installed in the env's bin directory
    path = os.path.join(args.env_path, 'bin')

    # DAG ---------------------------------------------------------------------
    dagman = Dagman(name='example_dagman', submit=args.submit)

    # 1(a) generate_parameters_basis ------------------------------------------
    executable = os.path.join(path, 'generate_parameters')
    args_dict = {'waveforms_directory': args.waveforms_directory,
                 'settings_file': args.settings_file,
                 'parameters_file': args.parameters_file_basis,
                 'n_samples': args.num_wfs_basis
                 }
    args_str = create_args_string(args_dict)
    generate_parameters_basis = Job(name='generate_parameters_basis',
                                    executable=executable, dag=dagman,
                                    arguments=args_str, **kwargs)

    # 1(b) generate_parameters_dataset ----------------------------------------
    executable = os.path.join(path, 'generate_parameters')
    args_dict = {'waveforms_directory': args.waveforms_directory,
                 'settings_file': args.settings_file,
                 'parameters_file': args.parameters_file_dataset,
                 'n_samples': args.num_wfs_dataset
                 }
    args_str = create_args_string(args_dict)
    generate_parameters_dataset = Job(name='generate_parameters_dataset',
                                      executable=executable, dag=dagman,
                                      arguments=args_str, **kwargs)

    # 2. generate_waveforms_basis ---------------------------------------------
    executable = os.path.join(path, 'generate_waveforms')
    args_dict = {'waveforms_directory': args.waveforms_directory,
                 'settings_file': args.settings_file,
                 'parameters_file': args.parameters_file_basis,
                 'num_wf_per_process': args.num_wf_per_process,
                 'num_threads': args.num_threads,
                 'process_id $(Process)': ''
                 }
    args_str = create_args_string(args_dict)
    generate_waveforms_basis = Job(name=f'generate_waveforms_basis',
                                   queue=num_chunks_basis,
                                   executable=executable, dag=dagman,
                                   arguments=args_str, **kwargs)
    generate_waveforms_basis.add_parent(generate_parameters_basis)

    # 3. build_SVD_basis ------------------------------------------------------
    executable = os.path.join(path, 'build_SVD_basis')
    args_dict = {'waveforms_directory': args.waveforms_directory,
                 'parameters_file': args.parameters_file_basis,
                 'basis_file': args.basis_file,
                 'rb_max': args.rb_max,
                 'rb_train_fraction': 1.0
                 }
    args_str = create_args_string(args_dict)
    build_SVD_basis = Job(name='build_SVD_basis',
                          executable=executable,
                          dag=dagman, arguments=args_str, **kwargs)
    build_SVD_basis.add_parent(generate_waveforms_basis)

    # 4. generate_waveforms_dataset -------------------------------------------
    executable = os.path.join(path, 'generate_waveforms')
    args_dict = {'waveforms_directory': args.waveforms_directory,
                 'settings_file': args.settings_file,
                 'parameters_file': args.parameters_file_dataset,
                 'num_wf_per_process': args.num_wf_per_process,
                 'num_threads': args.num_threads,
                 'process_id $(Process)': '',
                 'use_compression': '',
                 'basis_file': args.basis_file
                 }
    args_str = create_args_string(args_dict)
    generate_waveforms_dataset = Job(name=f'generate_waveforms_dataset',
                                     queue=num_chunks_dataset,
                                     executable=executable, dag=dagman,
                                     arguments=args_str, **kwargs)
    generate_waveforms_dataset.add_parent(build_SVD_basis)
    generate_waveforms_dataset.add_parent(generate_parameters_dataset)


    # 5. collect_waveform_dataset ---------------------------------------------
    executable = os.path.join(path, 'collect_waveform_dataset')
    args_dict = {'waveforms_directory': args.waveforms_directory,
                 'parameters_file': args.parameters_file_dataset,
                 'basis_file': args.basis_file,
                 'settings_file': args.settings_file,
                 'dataset_file': args.dataset_file
                 }
    args_str = create_args_string(args_dict)
    collect_waveform_dataset = Job(name='collect_waveform_dataset',
                                   executable=executable, dag=dagman,
                                   arguments=args_str, **kwargs)
    collect_waveform_dataset.add_parent(generate_waveforms_dataset)

    return dagman


def main():
    args = parse_args()

    # Sanity checks
    modulus_check(args.num_wfs_basis, args.num_wf_per_process,
                  'num_wfs_basis', 'num_wf_per_process')
    modulus_check(args.num_wfs_dataset, args.num_wf_per_process,
                  'num_wfs_dataset', 'num_wf_per_process')

    dagman = create_dag(args)

    try:
        dagman.visualize('waveform_dataset_generation_workflow.png')
    except:
        pass

    dagman.build()
    print(f'DAG submission file written to {args.submit}.')


if __name__ == "__main__":
    main()

