import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
import io
import warnings
import contextlib
from typing import List, Dict, Any, Tuple, Optional, Callable

from dingo.gw.result import Result
from dingo.core.models import PosteriorModel
from dingo.gw.data.event_dataset import EventDataset
from dingo.gw.inference.gw_samplers import GWSamplerGNPE
from dingo.core.samplers import FixedInitSampler
from dingo.gw.transforms import SelectStandardizeRepackageParameters
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Integration for equation of state likelihood.",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to dingo model."
    )
    parser.add_argument(
        "--event_dataset_path", type=str, required=True, help="Path to even dataset."
    )
    parser.add_argument(
        "--eos_params",
        type=float,
        nargs="+",
        required=True,
        help="Coefficient for EOS polynomical.",
    )
    parser.add_argument(
        "--chirp_mass_grid",
        type=float,
        nargs=3,
        required=True,
        help="Linear grid for chirp mass.",
    )
    parser.add_argument(
        "--mass_ratio_grid",
        type=float,
        nargs=3,
        required=True,
        help="Linear grid for mass_ratio.",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument(
        "--num_samples_is",
        type=int,
        default=None,
        help="Number of IS samples for evidence estimate.",
    )
    parser.add_argument("--outname", type=str, default=None, help="Name for out file.")
    args = parser.parse_args()
    if args.chirp_mass_grid is not None:
        args.chirp_mass_grid[2] = int(args.chirp_mass_grid[2])
    if args.mass_ratio_grid is not None:
        args.mass_ratio_grid[2] = int(args.mass_ratio_grid[2])
    return args


def get_expanded_parameters(
    fixed_parameters: Dict[str, float],
    varied_parameters: Dict[str, List[float]],
    num_repeats_per_variation: int = 1,
) -> Dict[str, torch.Tensor]:
    """Generate a dict of torch tensors for the conditioning parameters.

    Each value of varied_parameters is a list of variations for the respective
    parameters. For each parameter name k in varied parameters, the output contains

    output[k] = [
        varied_parameters[k][0], ..., varied_parameters[k][0],  #  N times
        varied_parameters[k][1], ..., varied_parameters[k][1],  #  N times
        ...
    ]

    with N = num_repeats_per_variation. For each parameter k in fixed_parameters,
    output[k] contains the respective value repeated
    (num_repeats_per_variation * num_variations) times.

    Parameters
    ----------
    fixed_parameters: Dict with fixed conditioning parameters
    varied_parameters: Dict with lists for varied conditioning parameters
    num_repeats_per_variation: number of repeats per element in varied_parameters

    Returns
    -------
    joint_parameters: Dict with torch tensors of expanded, joint parameters
    """
    # number of variations in varied_parameters
    num_variations = len(list(varied_parameters.values())[0])
    assert all([len(v) == num_variations for v in varied_parameters.values()])

    # repeat each varied parameter num_repeats_per_variation times
    joint_parameters = {
        k: torch.tensor(v).repeat_interleave(num_repeats_per_variation)
        for k, v in varied_parameters.items()
    }
    # add fixed conditioning parameters
    ones = torch.ones(num_repeats_per_variation * num_variations)
    for k, v in fixed_parameters.items():
        joint_parameters[k] = v * ones

    return joint_parameters


def prepare_model_input(
    data_settings: Dict[str, Any],
    data: torch.Tensor,
    conditioning_parameters: Dict[str, torch.Tensor],
    device: str,
) -> List[torch.Tensor]:
    """Prepare the input tensors for the dingo model.

    Parameters
    ----------
    data_settings: Model data settings, model.metadata["train_settings"]["data"]
    data: torch tensor with the GW data, without batch dimension
    conditioning_parameters: dict with torch tensor for conditioning parameters
    device: string with torch device

    Returns
    -------
    x: List with input tensors, to be passed to the model.
    """
    context_parameters = SelectStandardizeRepackageParameters(
        {"context_parameters": data_settings["context_parameters"]},
        data_settings["standardization"],
        device=device,
    )({"parameters": conditioning_parameters})["context_parameters"]

    # expand data
    x = data.repeat(len(context_parameters), *[1] * len(data.shape)).to(device)

    return [x, context_parameters]


def convert_to_result(
    sampler: GWSamplerGNPE,
    y: torch.Tensor,
    log_prob: torch.Tensor,
    conditioning_parameters: Dict[str, torch.Tensor],
) -> Result:
    """Parse output of dingo model into Result object."""
    samples = sampler.transform_gnpe_loop_post(
        {"parameters": y, "extrinsic_parameters": {}}
    )["parameters"]
    samples["log_prob"] = log_prob.cpu().numpy()
    for k, v in conditioning_parameters.items():
        samples[k] = v.cpu().numpy()
    samples["chirp_mass"] = samples["chirp_mass_proxy"] + samples["delta_chirp_mass"]
    sampler.samples = pd.DataFrame(samples)
    result = sampler.to_result()

    return result


def compute_is_quantities(
    result: Result, num_samples: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate is quantities (evidence, efficiency) for multiple joint results."""
    log_prob_target = np.array(
        result.samples["log_prior"] + result.samples["log_likelihood"]
    )
    log_prob_proposal = np.array(result.samples["log_prob"])
    log_weights = log_prob_target - log_prob_proposal
    log_weights = np.nan_to_num(log_weights.T.reshape(-1, num_samples), nan=-np.inf)
    weights = np.exp(log_weights - np.max(log_weights, axis=1)[:, None])
    n_eff = np.sum(weights, axis=1) ** 2 / np.sum(weights ** 2, axis=1)
    n_samples = weights.shape[1]
    log_evidence = logsumexp(log_weights, axis=1) - np.log(n_samples)
    log_evidence_std = np.sqrt((n_samples - n_eff) / (n_samples * n_eff))
    eps = n_eff / n_samples

    return log_evidence, log_evidence_std, eps


def compute_log_evidences(
    sampler: GWSamplerGNPE,
    event_dataset: EventDataset,
    data: torch.Tensor,
    fixed_context_parameters: Dict[str, float],
    varied_context_parameters: Dict[str, np.ndarray],
    num_samples: int,
    device: str = "cpu",
    num_processes: int = 0,
):
    """Compute the log evidences for different sets of context parameters."""
    model = sampler.model

    # get conditioning number, expanded num_samples times
    conditioning_parameters = get_expanded_parameters(
        fixed_context_parameters, varied_context_parameters, num_samples
    )

    # prepare the data for the model
    x = prepare_model_input(
        model.metadata["train_settings"]["data"], data, conditioning_parameters, device
    )

    # sample and log prob from model
    model.model.eval()
    with torch.no_grad():
        y, log_prob = model.model.sample_and_log_prob(*x)

    # parse into dingo result object
    result = convert_to_result(sampler, y, log_prob, conditioning_parameters)

    # importance sample
    result.event_metadata = event_dataset.settings
    likelihood_kwargs = dict(
        phase_marginalization_kwargs=dict(approximation_22_mode=True),
        decimate=True,
        phase_heterodyning=True,
    )
    result.importance_sample(num_processes=num_processes, **likelihood_kwargs)

    log_evidence, log_evidence_std, eps = compute_is_quantities(result, num_samples)

    return log_evidence, log_evidence_std, eps


def compute_model_log_prob(
    sampler: GWSamplerGNPE,
    data: torch.Tensor,
    fixed_context_parameters: Dict[str, float],
    evaluation_parameters: Dict[str, np.ndarray],
    batch_size: Optional[int] = None,
    device: str = "cpu",
):
    model = sampler.model

    # standardize parameters
    data_settings = model.metadata["train_settings"]["data"]
    inference_parameters = data_settings["inference_parameters"]
    std_transform = SelectStandardizeRepackageParameters(
        {"inference_parameters": inference_parameters},
        data_settings["standardization"],
        inverse=False,
    )
    y = std_transform(
        {"parameters": {k: torch.tensor(v) for k, v in evaluation_parameters.items()}}
    )["inference_parameters"]
    num_samples = len(y)

    # get conditioning number
    conditioning_parameters = {
        k: v * torch.ones(1, device=device) for k, v in fixed_context_parameters.items()
    }

    # prepare the data for the model
    x = prepare_model_input(
        model.metadata["train_settings"]["data"], data, conditioning_parameters, device
    )

    model.model.eval()
    if batch_size is None:
        batch_size = num_samples
    lower, upper = 0, batch_size
    log_probs = []
    while lower < num_samples:
        upper = min(upper, num_samples)
        with torch.no_grad():
            y_idx = y[lower:upper]
            x_idx = [v.expand(len(y_idx), *v.shape[1:]) for v in x]
            log_probs_idx = model.model.log_prob(y, *x_idx)
            log_probs.append(log_probs_idx)
        lower += batch_size
        upper += batch_size

    return torch.concatenate(log_probs).cpu().numpy()


def build_grid(
    eos: Callable[[float], float], chirp_masses: np.ndarray, mass_ratios: np.ndarray
):
    # chirp_mass, mass_ratio meshgrid
    chirp_masses_m, mass_ratios_m = np.meshgrid(
        chirp_masses, mass_ratios, indexing="ij"
    )
    chirp_mass = chirp_masses_m.ravel()
    mass_ratio = mass_ratios_m.ravel()

    # compute lambdas from eos
    mass_1, mass_2 = chirp_mass_and_mass_ratio_to_component_masses(
        chirp_mass, mass_ratio
    )
    lambda_1 = eos(mass_1)
    lambda_2 = eos(mass_2)

    return {
        "chirp_mass": chirp_mass,
        "mass_ratio": mass_ratio,
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
    }


def batched_execution(fn, batched_kwargs, batch_size: Optional[int] = None):
    if batch_size is None:
        return fn(batched_kwargs)

    N = len(list(batched_kwargs.values())[0])
    output_all = None
    for lower in range(0, N, batch_size):
        upper = min(lower + batch_size, N)
        batched_kwargs_batch = {k: v[lower:upper] for k, v in batched_kwargs.items()}
        output = fn(batched_kwargs_batch)
        if output_all is None:
            output_all = [[v] for v in output]
        else:
            for v_all, v in zip(output_all, output):
                v_all.append(v)

    return [np.concatenate(v) for v in output_all]


def is_marginalized_model(sampler):
    eos_params = {"lambda_2", "lambda_1", "mass_ratio", "delta_chirp_mass"}
    inference_params = set(
        sampler.metadata["train_settings"]["data"]["inference_parameters"]
    )
    return eos_params == inference_params


def compute_eos_integrand_on_grid(
    eos: Callable[[float], float],
    model_path: str,
    event_dataset_path: str,
    fixed_context_parameters: Dict[str, float],
    chirp_masses: np.ndarray,
    mass_ratios: np.ndarray,
    batch_size: Optional[int] = None,
    num_samples_is: Optional[int] = None,
    num_processes: int = 0,
    device: str = "cpu",
):
    # load model and event data, set up sampler and data
    model = PosteriorModel(model_path, device=device)
    event_dataset = EventDataset(event_dataset_path)
    sampler = GWSamplerGNPE(
        model=model,
        init_sampler=FixedInitSampler({}, log_prob=0),
        fixed_context_parameters=fixed_context_parameters,
        num_iterations=1,
    )
    sampler.context = event_dataset.data
    data = sampler.transform_pre(event_dataset.data)

    # build grid
    grid = build_grid(eos, chirp_masses, mass_ratios)
    grid["delta_chirp_mass"] = (
        grid["chirp_mass"] - fixed_context_parameters["chirp_mass_proxy"]
    )

    if is_marginalized_model(sampler):
        # assume marginalized model, so we can still evaluate model log prob on grid
        log_evidence = compute_model_log_prob(
            sampler,
            data,
            fixed_context_parameters,
            grid,
            batch_size=batch_size,
            device=device,
        )
        log_evidence_std, eps = None, None

    else:
        # assume conditional model, so evaluate log_evidences on grid of conditioning
        # parameters
        if batch_size is not None:
            batch_size = int(batch_size / num_samples_is)
        fixed_kwargs = dict(
            sampler=sampler,
            event_dataset=event_dataset,
            data=data,
            fixed_context_parameters=fixed_context_parameters,
            num_samples=num_samples_is,
            device=device,
            num_processes=num_processes,
        )
        fn = lambda x: compute_log_evidences(
            varied_context_parameters=x, **fixed_kwargs
        )
        log_evidence, log_evidence_std, eps = batched_execution(fn, grid, batch_size)

    return log_evidence, log_evidence_std, eps


def polynomial(x, a, *coefficients):
    result = 0
    for p, c in enumerate(coefficients):
        result += c * (x - a) ** p
    return result


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_args()

    fixed_context_parameters = {
        "chirp_mass_proxy": 1.1974457502365112,
        "ra": 3.44616,
        "dec": -0.408084,
    }

    # chirp_masses = np.linspace(1.1972, 1.198, 15)
    # mass_ratios = np.linspace(0.5, 1.0, 10)
    chirp_masses = np.linspace(*args.chirp_mass_grid)
    mass_ratios = np.linspace(*args.mass_ratio_grid)
    log_evidence, log_evidence_std, eps = compute_eos_integrand_on_grid(
        lambda mass: polynomial(mass, *args.eos_params),
        args.model_path,
        args.event_dataset_path,
        fixed_context_parameters,
        chirp_masses,
        mass_ratios,
        num_samples_is=args.num_samples_is,
        batch_size=args.batch_size,
        device=device,
    )

    data = dict(
        log_evidence=log_evidence,
        log_evidence_std=log_evidence_std,
        sample_efficiency=eps,
        chirp_masses=chirp_masses,
        mass_ratios=mass_ratios,
    )
    np.save(args.outname, data)

    import matplotlib.pyplot as plt

    plt.imshow(
        np.exp(log_evidence - np.max(log_evidence)).reshape(
            len(chirp_masses), len(mass_ratios)
        )
    )
    plt.show()
