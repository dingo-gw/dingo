"""Apply importance sampling to a dingo result with another result as reference.

This importance- and then rejection-samples a dingo result, based on a 1D marginal of
a dingo reference result. The idea is that the reference result may provide a better
estimate for the 1D marginal, and we can use this to adjust the dingo result before the
likelihood-based importance sampling to provide a better proposal distribution. We here
use this for the geocent time parameter.

Formally, we change the proposal distribution from q(theta|d) to

  q^(theta|d) = q(theta|d) * alpha(theta)
  theta ~ q^(theta|d) <--> theta ~ q(theta|d), w = alpha(theta), rejection sample with w
"""
from dingo.gw.result import Result
import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import splrep, BSpline
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--result_filename",
    type=str,
    help="Filename of the dingo result to be reweighted.",
)
parser.add_argument(
    "--reference_filename",
    type=str,
    help="Filename of the reference result used for reweighting.",
)
parser.add_argument(
    "--parameter_name",
    type=str,
    help="Name of the parameter used for reweighting.",
)
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

# load result and reference
result = Result(file_name=args.result_filename)
result_1D = result.samples[args.parameter_name]
reference = Result(file_name=args.reference_filename)
reference_1D = reference.samples[args.parameter_name]
try:
    reference_weights = reference.samples["weights"]
except KeyError:
    reference_weights = None

# get weights
kde_result = gaussian_kde(result_1D)
kde_reference = gaussian_kde(reference_1D, weights=reference_weights)
x = np.linspace(
    np.min((result_1D, reference_1D)), np.max((result_1D, reference_1D)), 100
)
alpha = lambda theta: kde_reference(theta) / kde_result(theta)
weights = BSpline(*splrep(x, alpha(x)))(result_1D)
weights = weights / np.max(weights)
n_eff = np.sum(weights) ** 2 / np.sum(weights ** 2)
print(f"Effective samples: {n_eff:.0f} ({n_eff / len(weights) * 100:.1f} %)")

# apply rejection sampling
mask = weights >= np.random.uniform(size=len(weights))
n_rej = np.sum(mask)
print(f"Number of rejection samples: {n_rej:.0f} ({n_rej / len(weights) * 100:.1f} %)")
samples_new = result.samples[mask].reset_index(drop=True)
# adjust proposal log prob, was q(theta|d), should now be q(theta|d) * alpha(theta)
samples_new["log_prob"] += np.log(weights)[mask]

# save result
result.samples = samples_new
result.to_file(file_name=args.result_filename[: -len(".hdf5")] + "_reweighted.hdf5")

if args.plot:
    import matplotlib.pyplot as plt

    # plot result
    plt.hist(result_1D, bins=100, density=True)
    plt.plot(x, kde_result(x))
    # plot reference
    plt.hist(reference_1D, bins=100, density=True)
    plt.plot(x, kde_reference(x))
    # plot reweighting
    plt.plot(x, alpha(x))
    plt.show()
