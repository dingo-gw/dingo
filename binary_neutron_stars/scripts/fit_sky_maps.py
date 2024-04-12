import numpy as np
from ligo.skymap import kde, io
import argparse

from dingo.gw.result import Result

parser = argparse.ArgumentParser()
parser.add_argument(
    "--samples_filename",
    type=str,
    help="Filename of the samples containing the sky position.",
)
parser.add_argument(
    "--fit_filename",
    type=str,
    help="Filename of the fit for saving.",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=10_000,
    help="Number of samples to use for fit.",
)
parser.add_argument(
    "--num_jobs",
    type=int,
    default=1,
    help="Number of jobs for skymap fit.",
)
parser.add_argument(
    "--num_trials",
    type=int,
    default=5,
    help="Number of trials for skymap fit.",
)
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

# Load result, get unweighted samples with rejection sampling
result = Result(file_name=args.samples_filename)
weights = result.samples["weights"] * result.samples["luminosity_distance"] ** 2
samples = result.samples.sample(args.num_samples, weights=weights, replace=True)
ra_dec_dL = np.array(samples[['ra', 'dec', 'luminosity_distance']])

# Generate skymap fit and save to file
skypost = kde.Clustered2DSkyKDE(ra_dec_dL, trials=args.num_trials, jobs=args.num_jobs)
hpmap = skypost.as_healpix()
io.write_sky_map(args.fit_filename, hpmap, nest=True)