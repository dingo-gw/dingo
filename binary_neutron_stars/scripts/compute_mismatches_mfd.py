"""Given dataset settings, compute mismatches between decimated and full waveforms."""
import os
import numpy as np
import yaml
from multiprocessing import Pool
from threadpoolctl import threadpool_limits
import pandas as pd

from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.domains import build_domain
from dingo.gw.waveform_generator import WaveformGenerator
from dingo.gw.dataset.generate_dataset import generate_waveforms_parallel
from dingo.gw.gwutils import get_mismatch
from dingo.gw.transforms.waveform_transforms import factor_fiducial_waveform
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Evaluate multibanded domain mismatches.",
    )
    parser.add_argument(
        "--wfd_directory", type=str, required=True, help="Path to wfd directory."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples for mismatch computation.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=0,
        help="Number of parallel processes for waveform computation.",
    )
    parser.add_argument(
        "--outname", type=str, default=None, help="Name of file to save mismatches."
    )
    args = parser.parse_args()
    return args


def generate_waveforms(waveform_generator, parameters, num_processes=0):
    if num_processes > 1:
        with threadpool_limits(limits=1, user_api="blas"):
            with Pool(processes=num_processes) as pool:
                polarizations = generate_waveforms_parallel(
                    waveform_generator, parameters, pool
                )
    else:
        polarizations = generate_waveforms_parallel(waveform_generator, parameters)
    return polarizations


if __name__ == "__main__":
    args = parse_args()
    wfd_settings = os.path.join(
        args.wfd_directory, "waveform_dataset_settings_autocompleted.yaml"
    )

    with open(wfd_settings, "r") as f:
        wfd_settings = yaml.safe_load(f)

    mfd = build_domain(wfd_settings["domain"])
    ufd = mfd.base_domain
    prior = build_prior_with_defaults(wfd_settings["intrinsic_prior"])
    wfg_ufd = WaveformGenerator(domain=ufd, **wfd_settings["waveform_generator"])
    wfg_mfd = WaveformGenerator(domain=mfd, **wfd_settings["waveform_generator"])

    parameters = pd.DataFrame(prior.sample(args.num_samples))

    polarizations_mfd = generate_waveforms(wfg_mfd, parameters, args.num_processes)
    polarizations_ufd = generate_waveforms(wfg_ufd, parameters, args.num_processes)

    mismatches = []
    for idx in range(args.num_samples):
        for k in polarizations_mfd.keys():
            # extract polarizations at idx
            v1 = polarizations_mfd[k][idx]
            v2 = polarizations_ufd[k][idx]
            # heterodyne
            v1 = factor_fiducial_waveform(v1, mfd, parameters["chirp_mass"][idx])
            v2 = factor_fiducial_waveform(v2, ufd, parameters["chirp_mass"][idx])
            # compute mismatch
            mismatch = get_mismatch(
                v1,
                mfd.decimate(v2),
                domain=mfd,
                asd_file=wfd_settings["compression"]["whitening"],
            )
            mismatches.append(mismatch)

    if args.outname is not None:
        np.save(os.path.join(args.wfd_directory, args.outname), np.array(mismatches))
