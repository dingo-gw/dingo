from dingo.core.models import PosteriorModel
from dingo.gw.data.event_dataset import EventDataset
from dingo.gw.transforms import (
    HeterodynePhase,
    DecimateWaveformsAndASDS,
    WhitenAndScaleStrain,
    RepackageStrainsAndASDS,
    ApplyFrequencyMasking,
    SelectStandardizeRepackageParameters,
    ToTorch,
)
from dingo.gw.domains import build_domain, build_domain_from_model_metadata
from dingo.gw.likelihood import StationaryGaussianGWLikelihood
from dingo.gw.gwutils import get_extrinsic_prior_dict
from dingo.gw.prior import build_prior_with_defaults

import math
from bilby.gw.prior import PriorDict
import torch
from torchvision.transforms import Compose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Scan for trigger chirp mass.",
    )
    parser.add_argument(
        "--dingo_model", type=str, required=True, help="Path to dingo model."
    )
    parser.add_argument(
        "--event_data", type=str, required=True, help="Path to event data file."
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples per chirp mass."
    )
    parser.add_argument(
        "--f_max", type=float, default=None, help="Upper frequency bound for GW data."
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=0,
        help="Number of parallel processes for likelihood computation.",
    )
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    return args


def get_scan_chirp_masses(model_metadata, overlap_factor=1):
    # extract global prior and chirp mass kernel from model metadata
    prior = PriorDict(model_metadata["dataset_settings"]["intrinsic_prior"])
    prior = prior["chirp_mass"]
    kernel = PriorDict(model_metadata["train_settings"]["data"]["gnpe_chirp"]["kernel"])
    kernel = kernel["chirp_mass"]
    prior_range = prior.maximum - prior.minimum
    kernel_range = kernel.maximum - kernel.minimum
    # uniformly sample in the chirp mass range defined by the prior
    chirp_masses = np.linspace(
        prior.minimum - kernel.minimum,
        prior.maximum - kernel.maximum,
        math.ceil(prior_range / kernel_range * overlap_factor),
    )
    return chirp_masses


def generate_data_for_scan(transform, data, chirp_masses, num_samples=100):
    params_dict = lambda x: dict(parameters=dict(chirp_mass=x, chirp_mass_proxy=x))
    N = len(chirp_masses) * num_samples
    out = transform({**data, **params_dict(chirp_masses[0])})
    waveform = torch.zeros((N, *out["waveform"].shape))
    context_parameters = torch.zeros((N, *out["context_parameters"].shape))
    for idx, chirp_mass in enumerate(chirp_masses):
        out = transform({**data, **params_dict(chirp_mass)})
        lower, upper = idx * num_samples, (idx + 1) * num_samples
        waveform[lower:upper] = out["waveform"]
        context_parameters[lower:upper] = out["context_parameters"]
    return waveform, context_parameters


def get_transforms(model_metadata, f_max=None):
    domain = build_domain_from_model_metadata(model_metadata)
    data_settings = model_metadata["train_settings"]["data"]
    transform_pre = [
        HeterodynePhase(domain=domain.base_domain),
        DecimateWaveformsAndASDS(domain, decimation_mode="whitened"),
        WhitenAndScaleStrain(scale_factor=domain.noise_std),
        RepackageStrainsAndASDS(data_settings["detectors"]),
        ApplyFrequencyMasking(domain=domain, f_max_lower=f_max, deterministic=True),
        SelectStandardizeRepackageParameters(
            {"context_parameters": ["chirp_mass_proxy"]},
            data_settings["standardization"],
        ),
        ToTorch(),
    ]
    transform_pre = Compose(transform_pre)

    transform_post = SelectStandardizeRepackageParameters(
        {"inference_parameters": data_settings["inference_parameters"]},
        data_settings["standardization"],
        inverse=True,
        as_type="dict",
    )

    return transform_pre, transform_post


def build_prior_and_likelihood(model_metadata, event_data, f_max=None):
    # build prior
    intrinsic_prior = model_metadata["dataset_settings"]["intrinsic_prior"]
    extrinsic_prior = get_extrinsic_prior_dict(
        model_metadata["train_settings"]["data"]["extrinsic_prior"]
    )
    prior = build_prior_with_defaults({**intrinsic_prior, **extrinsic_prior})

    # build likelihood
    data_domain = build_domain_from_model_metadata(model_metadata).base_domain
    if f_max is not None:
        data_domain = build_domain(data_domain.domain_dict)
        data_domain.update(dict(f_max=f_max))
        event_data = dict(
            asds={
                k: data_domain.update_data(v, low_value=1.0)
                for k, v in event_data["asds"].items()
            },
            waveform={
                k: data_domain.update_data(v, low_value=0.0)
                for k, v in event_data["waveform"].items()
            },
        )
    likelihood = StationaryGaussianGWLikelihood(
        wfg_kwargs=model_metadata["dataset_settings"]["waveform_generator"],
        wfg_domain=data_domain,
        data_domain=data_domain,
        event_data=event_data,
        t_ref=model_metadata["train_settings"]["data"]["ref_time"],
        phase_marginalization_kwargs=dict(approximation_22_mode=True),
    )

    return prior, likelihood


def main(args):

    # load model and initialize dingo sampler
    model = PosteriorModel(
        device="cpu", model_filename=args.dingo_model, load_training_info=False
    )
    # load data for event
    event_dataset = EventDataset(args.event_data)
    # set data transforms
    transform_pre, transform_post = get_transforms(model.metadata, f_max=args.f_max)
    # get chirp masses for scan
    chirp_masses = get_scan_chirp_masses(model.metadata, overlap_factor=2)
    # generate nn input data for chirp mass scan
    data = generate_data_for_scan(
        transform_pre, event_dataset.data, chirp_masses, num_samples=args.num_samples
    )

    # sample from the model
    t0 = time.time()
    samples = model.sample(*data).detach().cpu().numpy()
    samples = transform_post(dict(parameters=samples))["parameters"]
    delta_chirp_mass = samples.pop("delta_chirp_mass")
    samples["chirp_mass"] = chirp_masses.repeat(args.num_samples) + delta_chirp_mass
    print(f"Wall time for NN sampling: {time.time() - t0:.2f} seconds.")

    # build likelihood
    prior, likelihood = build_prior_and_likelihood(
        model.metadata, event_dataset.data, f_max=args.f_max
    )

    # compute log priors and likelihoods
    theta = pd.DataFrame(samples)
    log_prior = prior.ln_prob(theta, axis=0)
    indices = np.where(log_prior > -np.inf)[0]
    theta = theta.iloc[indices]
    log_prior = log_prior[indices]
    t0 = time.time()
    log_likelihoods = likelihood.log_likelihood_multi(
        theta, num_processes=args.num_processes
    )
    log_probs = log_likelihoods + log_prior
    print(f"Wall time for likelihoods: {time.time() - t0:.2f}")

    # extract chirp mass from max log prob samples
    chirp_mass_trigger = theta["chirp_mass"][np.argmax(log_likelihoods)]
    print(f"Chirp mass trigger: {chirp_mass_trigger:.4f} seconds.")

    # optionally plot
    if args.plot:
        plt.plot(theta["chirp_mass"], np.exp(log_probs - np.max(log_probs)))
        plt.plot(theta["chirp_mass"], np.exp(log_probs - np.max(log_probs)), ".")
        plt.xlim(1.19, 1.23)
        plt.show()

        plt.plot(theta["chirp_mass"], log_likelihoods)
        plt.ylim(np.max(log_likelihoods) - 100, np.max(log_likelihoods) + 10)
        plt.show()

        # assert torch.all(torch.std(data[1].reshape(len(chirp_masses), args.num_samples,
        # *data[1].shape[1:]), dim=1) == 0)
        # plt.plot(
        #     chirp_masses,
        #     np.std(delta_chirp_mass.reshape(len(chirp_masses), args.num_samples),
        #     axis=1),
        # )
        # plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
