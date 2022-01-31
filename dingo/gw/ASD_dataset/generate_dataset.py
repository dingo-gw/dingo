import os
os.environ['OMP_NUM_THREADS'] = str(1)
os.environ['MKL_NUM_THREADS'] = str(1)
import numpy as np
import yaml
from os.path import join
import logging
import argparse
import textwrap
from typing import Dict, List
from functools import partial
from tqdm import tqdm

# from parameterization_utils import get_path_param_data, parameterize_func
from dataset_utils import download_and_estimate_PSDs, create_dataset_from_files


def generate_dataset(data_dir, settings, run: str, ifos: List[str], verbose=False):

    for ifo in ifos:
        print(f'Downloading PSD data for observing run {run} and detector {ifo}')
        download_and_estimate_PSDs(data_dir, run, ifo, settings['dataset_settings'], verbose=verbose)

    create_dataset_from_files(data_dir, run, ifos, settings['dataset_settings'])

def apply_parameterization(data_dir, settings, run, detector, num_processes=1, verbose=True):

    psd_filename = join(data_dir, f'{run}_{detector}_psd.npy')
    metadata_filename = join(data_dir, f'{run}_{detector}_metadata.npy')

    psds = np.load(psd_filename, allow_pickle=True)[:, 160:]
    metadata = np.load(metadata_filename, allow_pickle=True).item()

    T_PSD = settings['dataset_settings']['T_PSD']
    delta_T = settings['dataset_settings']['delta_T']

    param_path = get_path_param_data(data_dir, run, detector, T_PSD, delta_T)
    os.makedirs(param_path, exist_ok=True)

    task_func = partial(parameterize_func, psds=psds, frequencies=metadata['sample_frequencies'],
                        param_path=param_path, settings=settings['parameterization_settings'])

    if num_processes > 1:
        with Pool(processes=num_processes) as pool:
            with tqdm(total=psds.shape[0], disable=not verbose) as pbar:
                for i, _ in enumerate(pool.imap_unordered(task_func, range(psds.shape[0]))):
                    pbar.update()

    else:
        with tqdm(total=psds.shape[0], disable=not verbose) as pbar:
            for i, _ in enumerate(map(task_func, range(psds.shape[1]))):
                pbar.update()


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
        Generate an ASD dataset based on a settings file.
        """
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path where the PSD data is to be stored. Must contain a 'settings.yaml' file.",
    )
    parser.add_argument(
        "--observing_run",
        type=str,
        required=True,
        help="Observing run for which to generate the dataset",
    )
    parser.add_argument(
        "--detectors",
        type=list,
        default=["H1", "L1"],
        help="Detectors for which to generate the dataset",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes to use in pool for parallel parameterisation",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Visualize progress with bars",
    )

    return parser.parse_args()


def main():

    logger = logging.getLogger('pymc3')
    logger.setLevel(logging.ERROR)

    args = parse_args()

    # Load settings
    with open(join(args.data_dir, 'settings.yaml'), "r") as f:
        settings = yaml.safe_load(f)


    generate_dataset(args.data_dir, settings, args.observing_run, args.detectors, verbose=args.verbose)
    # apply_parameterization(args.data_dir, settings, run='O2', detector='H1', num_processes=args.num_processes)


if __name__ == "__main__":
    main()


