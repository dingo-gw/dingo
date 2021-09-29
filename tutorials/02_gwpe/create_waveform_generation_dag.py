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

# FIXME: use queue argument and pass
# FIXME: setup condor log, stdout, stderr
# FIXME: request_cpus, etc
# FIXME: Use logging instead of print statements as bilby

import argparse
from pycondor import Job, Dagman




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        Collect compressed waveform polarizations and parameters.
        Save consolidated waveform dataset in HDF5 format.
    """)
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
    args = parser.parse_args()


    chunk_size_basis = args.num_wfs_basis // args.num_wf_per_process
    chunk_size_dataset = args.num_wfs_dataset // args.num_wf_per_process

    dagman = Dagman(name='example_dagman')

    generate_parameters_basis = Job(name='generate_parameters_basis',
                                    executable='generate_parameters.py', dag=dagman)
    generate_parameters_basis.add_arg(f'--waveforms_directory {args.waveforms_directory}')
    generate_parameters_basis.add_arg(f'--settings_file {args.settings_file}')
    generate_parameters_basis.add_arg(f'--parameters_file {args.parameters_file_basis}')
    generate_parameters_basis.add_arg(f'--n_samples {chunk_size_basis}')

    generate_parameters_dataset = Job(name='generate_parameters_dataset',
                                    executable='generate_parameters.py', dag=dagman)
    generate_parameters_dataset.add_arg(f'--waveforms_directory {args.waveforms_directory}')
    generate_parameters_dataset.add_arg(f'--settings_file {args.settings_file}')
    generate_parameters_dataset.add_arg(f'--parameters_file {args.parameters_file_dataset}')
    generate_parameters_dataset.add_arg(f'--n_samples {chunk_size_dataset}')


    build_SVD_basis = Job(name='build_SVD_basis', executable='build_SVD_basis.py', dag=dagman)
    build_SVD_basis.add_arg(f'--waveforms_directory {args.waveforms_directory}')
    build_SVD_basis.add_arg(f'--parameters_file {args.parameters_file_basis}')
    build_SVD_basis.add_arg(f'--basis_file {args.basis_file}')
    build_SVD_basis.add_arg(f'--rb_max {args.rb_max}')

    collect_waveform_dataset = Job(name='collect_waveform_dataset',
                                   executable='collect_waveform_dataset.py', dag=dagman)
    collect_waveform_dataset.add_arg(f'--waveforms_directory {args.waveforms_directory}')
    collect_waveform_dataset.add_arg(f'--parameters_file {args.parameters_file_dataset}')
    collect_waveform_dataset.add_arg(f'--basis_file {args.basis_file}')
    collect_waveform_dataset.add_arg(f'--settings_file {args.settings_file}')
    collect_waveform_dataset.add_arg(f'--dataset_file {args.dataset_file}')


    for idx in range(chunk_size_basis):
        generate_waveforms_basis = Job(name=f'generate_waveforms_basis_{idx}',
                                       executable='generate_waveforms.py', dag=dagman)
        generate_waveforms_basis.add_arg(f'--waveforms_directory {args.waveforms_directory}')
        generate_waveforms_basis.add_arg(f'--settings_file {args.settings_file}')
        generate_waveforms_basis.add_arg(f'--parameters_file {args.parameters_file_basis}')
        generate_waveforms_basis.add_arg(f'--num_wf_per_process {args.num_wf_per_process}')
        generate_waveforms_basis.add_arg(f'--process_id {idx}')

        # Connect nodes
        build_SVD_basis.add_parent(generate_waveforms_basis)
        generate_waveforms_basis.add_parent(generate_parameters_basis)


    for idx in range(chunk_size_dataset):
        generate_waveforms_dataset = Job(name=f'generate_waveforms_dataset_{idx}',
                                       executable='generate_waveforms.py', dag=dagman)
        generate_waveforms_dataset.add_arg(f'--waveforms_directory {args.waveforms_directory}')
        generate_waveforms_dataset.add_arg(f'--settings_file {args.settings_file}')
        generate_waveforms_dataset.add_arg(f'--parameters_file {args.parameters_file_dataset}')
        generate_waveforms_dataset.add_arg(f'--num_wf_per_process {args.num_wf_per_process}')
        generate_waveforms_dataset.add_arg(f'--process_id {idx}')
        generate_waveforms_dataset.add_arg(f'--use_compression')
        generate_waveforms_dataset.add_arg(f'--basis_file {args.basis_file}')

        # Connect nodes
        generate_waveforms_dataset.add_parent(build_SVD_basis)
        generate_waveforms_dataset.add_parent(generate_parameters_dataset)
        generate_waveforms_dataset.add_child(collect_waveform_dataset)

    try:
        dagman.visualize('waveform_dataset_generation_workflow.png')
    except:
        pass
    dagman.build()
