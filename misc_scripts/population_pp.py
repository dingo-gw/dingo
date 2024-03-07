import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm

from dingo.core.models import PosteriorModel
from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.populations.injection import Injection
from dingo.populations.samplers import PopulationSampler


parser = argparse.ArgumentParser(
    description="Make a PP plot for population injections."
)
parser.add_argument(
    "--population-model", type=str, required=True, help="Dingo population model file."
)
parser.add_argument(
    "--event-model",
    type=str,
    required=True,
    help="Dingo event model file. This should contain the Dingo "
    "embedding network that the population model was trained with.",
)
parser.add_argument(
    "--asd-dataset",
    type=str,
    required=True,
    help="ASD dataset containing to be used for generating detector "
    "noise in injections.",
)
parser.add_argument(
    "--num-injections",
    type=int,
    required=True,
    help="Number of injections to perform for the PP plot.",
)
parser.add_argument(
    "--out-file",
    type=str,
    default="pp_plot.pdf",
    help="File name for saving the produced PP plot.",
)
parser.add_argument(
    "--population-size",
    type=int,
    help="Size of populations. Defaults to random population size consistent with model "
    "training.",
)
parser.add_argument("--num_samples", type=int, default=10000)
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

# Load models.
population_model = PosteriorModel(args.population_model, device=args.device)
event_model = PosteriorModel(args.event_model, device=args.device)

# Construct sampler and injection objects.
sampler = PopulationSampler(population_model, event_model)
injection_class = Injection.from_posterior_model_metadata(population_model.metadata)

# Set up detector noise.
ifo_list = [ifo.name for ifo in injection_class.event_injection.ifo_list]
asd_dataset = ASDDataset(args.asd_dataset, ifos=ifo_list)
injection_class.event_injection.asd = asd_dataset

# Generate injections, sample posteriors, and get percentiles.
print("Generating and analyzing injections.")
percentiles = {p: np.empty(args.num_injections) for p in sampler.inference_parameters}
for n in tqdm(range(args.num_injections)):
    injection = injection_class.random_injection(population_size=args.population_size)
    sampler.population = injection
    truths = injection["hyperparameters"]
    samples = sampler.run_sampler(args.num_samples)
    for param, truth in truths.items():
        percentiles[param][n] = stats.percentileofscore(samples[param], truth)

# Produce plot
print("Generating PP plot.")
y = np.linspace(0, 1, args.num_injections + 2)
fig = plt.figure(figsize=(10, 10))
for k, v in percentiles.items():
    p_value = stats.kstest(v / 100, "uniform")[1]
    ordered = np.concatenate(([0.0], np.sort(v / 100), [1.0]))
    plt.step(ordered, y, where="post", label=f"{k} ({p_value:.3g})")
plt.plot(y, y, "k--")
plt.legend()
plt.xlabel(r"$p$")
plt.ylabel(r"$CDF(p)$")
plt.xlim((0, 1))
plt.ylim((0, 1))
size_info = (
    f", population size = {args.population_size}" if args.population_size else ""
)
plt.title(f"{args.num_injections} population injections" + size_info)
ax = fig.gca()
ax.set_aspect("equal", anchor="SW")
plt.savefig(args.out_file)
print(f"Plot saved in {args.out_file}.")
