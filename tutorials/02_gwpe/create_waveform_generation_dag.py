#!/home/mpuer/projects/dingo-devel/dingo-devenv/bin/python3

"""
Setup DAG and jobs for the 5 steps in waveform dataset generation using pycondor

Write submission files to disk.

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
    python3 ./create_waveform_generation.dag.py --waveforms_directory ./datasets/waveforms/ --parameters_file_basis parameters_basis.npy --parameters_file_dataset parameters_dataset.npy --basis_file polarization_basis.npy --settings_file settings.yaml --dataset_file waveform_dataset.hdf5 --num_wfs_basis 200 --num_wfs_dataset 500 --num_wf_per_process 100 --rb_max 50
"""


import argparse
import os
from pycondor import Job, Dagman


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
    parser.add_argument('--python_binary', type=str, default='python3')

    # condor arguments
    parser.add_argument('--request_cpus', type=int, default=None)
    parser.add_argument('--request_memory', type=int, default=None)
    parser.add_argument('--error', type=str, default='condor/error')
    parser.add_argument('--output', type=str, default='condor/output')
    parser.add_argument('--log', type=str, default='condor/log')
    parser.add_argument('--submit', type=str, default='condor/submit')

    return parser.parse_args()


def create_dag(args):
    kwargs = {'request_cpus': args.request_cpus, 'request_memory': args.request_memory,
              'submit': args.submit, 'error': args.error, 'output': args.output,
              'log': args.log, 'getenv': True}

    chunk_size_basis = args.num_wfs_basis // args.num_wf_per_process
    chunk_size_dataset = args.num_wfs_dataset // args.num_wf_per_process

    executable = args.python_binary

    # DAG ---------------------------------------------------------------------
    dagman = Dagman(name='example_dagman', submit=args.submit)

    # 1(a) generate_parameters_basis ------------------------------------------
    script_path = os.path.abspath('generate_parameters.py')
    generate_parameters_basis_args = f'{script_path}\
    --waveforms_directory {args.waveforms_directory}\
    --settings_file {args.settings_file}\
    --parameters_file {args.parameters_file_basis}\
    --n_samples {chunk_size_basis}'
    generate_parameters_basis = Job(name='generate_parameters_basis',
                                    executable=executable, dag=dagman,
                                    arguments=generate_parameters_basis_args, **kwargs)

    # 1(b) generate_parameters_dataset ----------------------------------------
    script_path = os.path.abspath('generate_parameters.py')
    generate_parameters_dataset_args = f'{script_path}\
    --waveforms_directory {args.waveforms_directory}\
    --settings_file {args.settings_file}\
    --parameters_file {args.parameters_file_dataset}\
    --n_samples {chunk_size_dataset}'
    generate_parameters_dataset = Job(name='generate_parameters_dataset',
                                      executable=executable, dag=dagman,
                                      arguments=generate_parameters_dataset_args, **kwargs)

    # 2. generate_waveforms_basis ---------------------------------------------
    script_path = os.path.abspath('generate_waveforms.py')
    generate_waveforms_basis_args = f'{script_path}\
    --waveforms_directory {args.waveforms_directory}\
    --settings_file {args.settings_file}\
    --parameters_file {args.parameters_file_basis}\
    --num_wf_per_process {args.num_wf_per_process}\
    --num_threads {args.num_threads}\
    --process_id $(Process)'
    generate_waveforms_basis = Job(name=f'generate_waveforms_basis', queue=chunk_size_basis,
                                   executable=executable, dag=dagman,
                                   arguments=generate_waveforms_basis_args, **kwargs)
    generate_waveforms_basis.add_parent(generate_parameters_basis)

    # 3. build_SVD_basis ------------------------------------------------------
    script_path = os.path.abspath('build_SVD_basis.py')
    build_SVD_basis_args = f'{script_path}\
    --waveforms_directory {args.waveforms_directory}\
    --parameters_file {args.parameters_file_basis}\
    --basis_file {args.basis_file}\
    --rb_max {args.rb_max}'
    build_SVD_basis = Job(name='build_SVD_basis',
                          executable=executable,
                          dag=dagman, arguments=build_SVD_basis_args, **kwargs)
    build_SVD_basis.add_parent(generate_waveforms_basis)

    # 4. generate_waveforms_dataset -------------------------------------------
    script_path = os.path.abspath('generate_waveforms.py')
    generate_waveforms_dataset_args = f'{script_path}\
    --waveforms_directory {args.waveforms_directory}\
    --settings_file {args.settings_file}\
    --parameters_file {args.parameters_file_dataset}\
    --num_wf_per_process {args.num_wf_per_process}\
    --num_threads {args.num_threads}\
    --process_id $(Process)\
    --use_compression\
    --basis_file {args.basis_file}'
    generate_waveforms_dataset = Job(name=f'generate_waveforms_dataset', queue=chunk_size_dataset,
                                     executable=executable, dag=dagman,
                                     arguments=generate_waveforms_dataset_args, **kwargs)
    generate_waveforms_dataset.add_parent(build_SVD_basis)
    generate_waveforms_dataset.add_parent(generate_parameters_dataset)


    # 5. collect_waveform_dataset ---------------------------------------------
    script_path = os.path.abspath('collect_waveform_dataset.py')
    collect_waveform_dataset_args = f'{script_path}\
    --waveforms_directory {args.waveforms_directory}\
    --parameters_file {args.parameters_file_dataset}\
    --basis_file {args.basis_file}\
    --settings_file {args.settings_file}\
    --dataset_file {args.dataset_file}'
    collect_waveform_dataset = Job(name='collect_waveform_dataset',
                                   executable=executable, dag=dagman,
                                   arguments=collect_waveform_dataset_args, **kwargs)
    collect_waveform_dataset.add_parent(generate_waveforms_dataset)

    return dagman

if __name__ == "__main__":
    args = parse_args()
    dagman = create_dag(args)

    try:
        dagman.visualize('waveform_dataset_generation_workflow.png')
    except:
        pass

    dagman.build()
