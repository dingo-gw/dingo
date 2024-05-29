import torch
import numpy as np
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
from bilby.gw.prior import BBHPriorDict
import contextlib
import io
import warnings

from dingo.core.models import PosteriorModel
from dingo.gw.data.event_dataset import EventDataset
from dingo.gw.inference.gw_samplers import GWSamplerGNPE
from dingo.core.samplers import FixedInitSampler


def polynomial(x, a, *coefficients):
    result = 0
    for p, c in enumerate(coefficients):
        result += c * (x - a) ** p
    return result


def run_without_print(func, *args, **kwargs):
    with warnings.catch_warnings():
        with contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore", RuntimeWarning)
            result = func(*args, **kwargs)
    return result


def get_log_evidence(
    chirp_mass,
    mass_ratio,
    lambda_1,
    lambda_2,
    model,
    event_dataset,
    likelihood_kwargs,
    fixed_context_parameters=None,
    num_samples=1_000,
    batch_size=1_000,
    num_processes=10,
):
    # set up conditioning parameters
    if fixed_context_parameters is None:
        fixed_context_parameters = {}
    fixed_context_parameters = fixed_context_parameters.copy()
    fixed_context_parameters["chirp_mass_proxy"] = chirp_mass
    fixed_context_parameters["mass_ratio"] = mass_ratio
    fixed_context_parameters["lambda_1"] = lambda_1
    fixed_context_parameters["lambda_2"] = lambda_2

    # set up sampler with fixed parameters
    sampler = GWSamplerGNPE(
        model=model,
        init_sampler=FixedInitSampler({}, log_prob=0),
        fixed_context_parameters=fixed_context_parameters,
        num_iterations=1,
    )
    sampler.context = event_dataset.data

    # run sampler and generate result object
    sampler.run_sampler(num_samples=num_samples, batch_size=batch_size)
    sampler.samples["chirp_mass"] = sampler.samples["chirp_mass_proxy"]
    result = sampler.to_result()

    # importance sampling
    result.event_metadata = event_dataset.settings
    result.importance_sample(num_processes=num_processes, **likelihood_kwargs)

    return result.log_evidence, result.log_evidence_std, result.effective_sample_size


def get_log_evidences_grid(chirp_masses, mass_ratios, eos, kwargs, prior=None):
    log_evidences = np.zeros((len(chirp_masses), len(mass_ratios)))
    log_evidences_std = np.zeros((len(chirp_masses), len(mass_ratios)))
    neffs = np.zeros((len(chirp_masses), len(mass_ratios)))

    for i, chirp_mass in enumerate(chirp_masses):
        for j, mass_ratio in enumerate(tqdm(mass_ratios)):
            mass_1, mass_2 = chirp_mass_and_mass_ratio_to_component_masses(
                chirp_mass, mass_ratio
            )
            parameters = dict(
                chirp_mass=chirp_mass,
                mass_ratio=mass_ratio,
                lambda_1=eos(mass_1),
                lambda_2=eos(mass_2),
            )
            if prior is None or prior.ln_prob(parameters) > -np.inf:
                log_evidence, log_evidence_std, neff = run_without_print(
                    get_log_evidence, **parameters, **kwargs
                )
            else:
                log_evidence, log_evidence_std, neff = -np.inf, np.nan, 0

            log_evidences[i, j] = log_evidence
            log_evidences_std[i, j] = log_evidence_std
            neffs[i, j] = neff

    return log_evidences, log_evidences_std, neffs


def log_integrate(log_integrand, log_integrand_error, x=None):
    """Compute the integral over log_integrand along the last axis."""
    alpha = np.max(log_integrand, axis=-1)

    # compute integral
    integrand_norm = np.exp(log_integrand - alpha[..., None])
    result_norm = scipy.integrate.simpson(integrand_norm, x=x)

    # compute integral over errors
    integrand_error_norm = log_integrand_error * integrand_norm
    integrand_error_norm[integrand_norm == 0] = 0
    error_norm_sq = scipy.integrate.simpson(integrand_error_norm ** 2, x=x)
    error_norm = np.sqrt(error_norm_sq)

    # compute log_result and log_result_error
    log_result = np.log(result_norm) + alpha
    log_result_error = error_norm / result_norm

    return log_result, log_result_error


def plot_evidences(log_evidences, log_evidences_std, extent=None, alpha=None):
    evidences_norm = np.exp(log_evidences - alpha)
    evidences_norm_std = log_evidences_std * evidences_norm
    evidences_norm_std[np.isnan(evidences_norm_std)] = 0

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(6, 3)

    plot_kwargs = dict(
        vmin=0, vmax=np.max(evidences_norm), extent=extent, aspect="auto"
    )
    ax1.imshow(evidences_norm, **plot_kwargs)
    ax2.imshow(evidences_norm_std, **plot_kwargs)
    ax2.colorbar()
    plt.show()


if __name__ == "__main__":
    model_path = "/Users/maxdax/Documents/Projects/GW-Inference/01_bns/prototyping/equation-of-state/model.pt"
    event_dataset_path = "/Users/maxdax/Documents/Projects/GW-Inference/01_bns/prototyping/equation-of-state/GW170817_event_data.hdf5"
    likelihood_kwargs = dict(
        phase_marginalization_kwargs=dict(approximation_22_mode=True),
        decimate=True,
        phase_heterodyning=True,
    )
    num_samples_is = 100
    num_grid_chirp_mass = 15
    num_grid_mass_ratio = 30

    # load data and model
    event_dataset = EventDataset(file_name=event_dataset_path)
    model = PosteriorModel(
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_filename=model_path,
        load_training_info=False,
    )
    prior = model.metadata["dataset_settings"]["intrinsic_prior"]
    prior = BBHPriorDict(
        {k: v for k, v in prior.items() if "mass" in k or "lambda" in k}
    )

    # setup for integration
    chirp_masses = np.linspace(1.19728, 1.19791, num_grid_chirp_mass)
    mass_ratios = np.linspace(0.05, 1.0, num_grid_mass_ratio)
    # eos_1 = lambda mass: polynomial(mass, 1.4, 500, -4000, 40000)
    eos_1 = lambda mass: polynomial(mass, 1.4, 1000, -6000, 10000)
    eos_2 = lambda mass: polynomial(mass, 1.4, 1000, -1000)

    kwargs = dict(
        model=model,
        event_dataset=event_dataset,
        likelihood_kwargs=likelihood_kwargs,
        fixed_context_parameters=dict(ra=3.44616, dec=-0.408084),
        num_samples=num_samples_is,
        batch_size=1000,
        num_processes=10,
    )

    # compute evidences on grid
    log_evidences_1, log_evidences_std_1, neffs_1 = get_log_evidences_grid(
        chirp_masses, mass_ratios, eos_1, kwargs, prior=prior
    )
    log_evidences_2, log_evidences_std_2, neffs_2 = get_log_evidences_grid(
        chirp_masses, mass_ratios, eos_2, kwargs, prior=prior
    )

    # integrate evidences
    log_res_1, log_error_1 = log_integrate(
        *log_integrate(log_evidences_1, log_evidences_std_1, x=None),
        x=None,
    )
    log_res_2, log_error_2 = log_integrate(
        *log_integrate(log_evidences_2, log_evidences_std_2, x=None),
        x=None,
    )
    print(log_res_1, log_error_1)
    print(log_res_2, log_error_2)
    print(log_res_1 - log_res_2, np.sqrt(log_error_1 ** 2 + log_error_2 ** 2))
    print(
        scipy.special.logsumexp(log_evidences_1)
        - scipy.special.logsumexp(log_evidences_2)
    )

    # plot
    extent = [mass_ratios[0], mass_ratios[-1], chirp_masses[0], chirp_masses[-1]]
    plot_evidences(
        log_evidences_1,
        log_evidences_std_1,
        extent=extent,
        alpha=np.max((log_evidences_1, log_evidences_2)),
    )
    plot_evidences(
        log_evidences_2,
        log_evidences_std_2,
        extent=extent,
        alpha=np.max((log_evidences_1, log_evidences_2)),
    )
    print("a")
