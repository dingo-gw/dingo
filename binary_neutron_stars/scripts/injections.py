from dingo.core.models import PosteriorModel
import dingo.gw.injection as injection
from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.inference.gw_samplers import GWSamplerGNPE
from dingo.core.samplers import FixedInitSampler

import yaml
from types import SimpleNamespace
from pp_utils import weighted_percentile_of_score
import numpy as np
import os
import pandas as pd
import torch
from copy import deepcopy
from bilby.core.prior import PriorDict, Uniform, DeltaFunction
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
from ligo.skymap import kde, io
from ligo.skymap.postprocess import crossmatch
from os.path import join
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Evaluate on injections.",
    )
    parser.add_argument(
        "--outdirectory",
        type=str,
        required=True,
        help="Path to outdirectory, containing injection_settings.yaml.",
    )
    parser.add_argument(
        "--process_id", type=int, default=None, help="Index of process for injection."
    )
    args = parser.parse_args()
    with open(join(args.outdirectory, "injection_settings.yaml"), "r") as f:
        args.__dict__.update(yaml.safe_load(f))
    return args


def asd_to_ufd(asd, mfd):
    assert asd.shape == (len(mfd),)
    asd_ufd = np.repeat(asd, mfd._decimation_factors_bands[mfd._band_assignment])
    npad = ((0, 0) * (len(asd.shape) - 1)) + (mfd.base_domain.min_idx, 0)
    asd_ufd = np.pad(asd_ufd, pad_width=npad, mode="constant", constant_values=1)
    assert np.allclose(mfd.decimate(asd_ufd), asd)
    return asd_ufd


def set_chirp_mass(sampler, chirp_mass):
    sampler.fixed_context_parameters = {"chirp_mass_proxy": chirp_mass}
    sampler.transform_pre.transforms[0].fixed_parameters["chirp_mass"] = chirp_mass


def get_skymap_area(
    samples,
    num_samples=None,
    num_trials=1,
    num_jobs=None,
    contour=0.9,
    weights=None,
):
    if weights is None:
        weights = 1
    if num_samples is None:
        num_samples = len(samples)

    # Get unweighted samples with rejection sampling
    # weights *= weights * np.array(samples["luminosity_distance"]) ** 2
    weights *= np.array((0 <= samples["ra"]) * (samples["ra"] <= 2 * np.pi))
    weights *= np.array((-np.pi / 2 <= samples["dec"]) * (samples["dec"] <= np.pi / 2))
    samples = samples.sample(num_samples, weights=weights, replace=True)
    ra_dec_dL = np.array(samples[["ra", "dec", "luminosity_distance"]])

    # Generate skymap fit and save to file
    skypost = kde.Clustered2DSkyKDE(ra_dec_dL, trials=num_trials, jobs=num_jobs)
    hpmap = skypost.as_healpix()
    area = crossmatch(hpmap, contours=[contour])[4][0]
    return area


def insert_component_masses(samples):
    (
        samples["mass_1"],
        samples["mass_2"],
    ) = chirp_mass_and_mass_ratio_to_component_masses(
        samples["chirp_mass"], samples["mass_ratio"]
    )


def update_summary_data(summary_data, args, theta, samples, weights=None, **kwargs):
    data = {}
    insert_component_masses(samples)

    # insert percentiles
    common_parameters = theta.keys() & samples.keys()
    for p in common_parameters:
        key = "percentiles-" + p
        data[key] = weighted_percentile_of_score(
            np.array(samples[p]), theta[p], weights=weights
        )
    # optionally insert sample efficiency
    if weights is not None:
        ESS = np.sum(weights) ** 2 / np.sum(weights ** 2)
        data["sample_efficiencies"] = ESS / len(weights)
    # optionally insert skymap
    if hasattr(args, "skymap"):
        data["skymap_areas"] = get_skymap_area(samples, weights=weights, **args.skymap)

    if weights is not None:
        samples_rej = samples.sample(len(samples), weights=weights, replace=True)
    else:
        samples_rej = samples

    # insert stds
    for p in common_parameters:
        data["std-" + p] = np.std(samples_rej[p])

    # insert fraction of samples above requested thresholds
    if hasattr(args, "fraction_geq"):
        for p, threshold in args.fraction_geq:
            key = f"fraction-geq_{p}_{str(threshold)}"
            data[key] = np.mean(samples_rej[p] >= threshold)

    # insert remaining kwargs
    for k, v in kwargs.items():
        data[k] = v

    if len(summary_data) == 0:
        for k, v in data.items():
            summary_data[k] = [v]
    else:
        for k, v in data.items():
            summary_data[k].append(v)


def get_injection_generator(model_metadata, base_domain_dict, fixed_parameters=None):
    model_metadata = deepcopy(model_metadata)
    # generate injection in base domain
    model_metadata["dataset_settings"]["domain"] = base_domain_dict
    model_metadata["train_settings"]["data"].pop("domain_update")
    injection_generator = injection.Injection.from_posterior_model_metadata(
        model_metadata
    )
    asd_f = model_metadata["train_settings"]["training"]["stage_0"]["asd_dataset_path"]
    asd_dataset = ASDDataset(file_name=asd_f)
    injection_generator.asd = {
        k: asd_to_ufd(v[0], asd_dataset.domain)
        for k, v in asd_dataset.asds.items()
        if k in model_metadata["train_settings"]["data"]["detectors"]
    }
    if fixed_parameters is not None:
        for k, v in fixed_parameters.items():
            injection_generator.prior[k] = DeltaFunction(v)
    return injection_generator


def get_chirp_mass_functions(
    model_metadata=None, injection_generator=None, fixed_chirp_mass=None
):
    if fixed_chirp_mass is not None:
        sample_chirp_mass_proxy = DeltaFunction(fixed_chirp_mass).sample
        get_chirp_mass_prior = lambda chirp_mass_proxy: DeltaFunction(chirp_mass_proxy)
    else:
        # get chirp mass kernel
        chirp_mass_kernel = PriorDict(
            deepcopy(model_metadata["train_settings"]["data"]["gnpe_chirp"]["kernel"])
        )["chirp_mass"]
        if not isinstance(chirp_mass_kernel, Uniform):
            raise NotImplementedError()
        # get chirp mass hyperprior from model metadata
        chirp_mass_hyperprior = PriorDict(
            deepcopy(model_metadata["dataset_settings"]["intrinsic_prior"])
        )["chirp_mass"]
        # exclude chirp mass regions that are forbidden by m1, m2 constraints
        if {"mass_1", "mass_2"}.issubset(injection_generator.prior.keys()):
            m1 = injection_generator.prior["mass_1"]
            m2 = injection_generator.prior["mass_2"]
            mc_max = (m1.maximum * m2.maximum) ** 0.6 / (m1.maximum + m2.maximum) ** 0.2
            mc_min = (m1.minimum * m2.minimum) ** 0.6 / (m1.minimum + m2.minimum) ** 0.2
            chirp_mass_hyperprior.maximum = min(chirp_mass_hyperprior.maximum, mc_max)
            chirp_mass_hyperprior.minimum = max(chirp_mass_hyperprior.minimum, mc_min)
        # decrease chirp mass hyperprior by kernel width to remove boundary effects
        chirp_mass_hyperprior.minimum -= chirp_mass_kernel.minimum
        chirp_mass_hyperprior.maximum -= chirp_mass_kernel.maximum
        # functions for sampling chirp mass proxy and building the corresponding prior
        sample_chirp_mass_proxy = chirp_mass_hyperprior.sample
        get_chirp_mass_prior = lambda chirp_mass_proxy: Uniform(
            minimum=chirp_mass_proxy + chirp_mass_kernel.minimum,
            maximum=chirp_mass_proxy + chirp_mass_kernel.maximum,
        )
    return sample_chirp_mass_proxy, get_chirp_mass_prior


def compute_snr(result):
    """Compute the maximum matched filter signal-to-noise ratio."""
    likelihood = result.likelihood
    theta = result.samples.iloc[np.argmax(result.samples["log_likelihood"])]
    theta = theta.to_dict()

    # recover max-likelihood phase if not already in theta
    if "phase" not in theta:
        pm_flag = likelihood.phase_marginalization
        likelihood.phase_marginalization = False
        phase_grid = np.linspace(0, 2 * np.pi, 100)
        logl = likelihood.log_likelihood_phase_grid(theta, phase_grid)
        theta["phase"] = phase_grid[np.argmax(logl)]
        # logsumexp = lambda a: np.log(
        #     np.mean(np.exp(a - np.max(a)))
        # ) + np.max(a)
        # print(logsumexp(logl_phases) - theta['log_likelihood'])
        # assert np.min(result.likelihood.log_likelihood(theta) - logl) == 0
        likelihood.phase_marginalization = pm_flag

    # compute snr
    snr = likelihood.matched_filter_snr(theta)

    return snr


def aggregate_results(
    outdirectory,
    prefix="summary-dingo_",
    percentiles=(10, 25, 50, 75, 90),
    filename=None,
):
    filenames = [f for f in os.listdir(outdirectory) if f.startswith(prefix)]
    data = pd.concat([pd.read_pickle(join(outdirectory, f)) for f in filenames])
    data["log_bayes_factor"] = data["log_evidence"] - data["log_noise_evidence"]
    keys = data.keys()
    f = sorted(list(set(data["f_max"])))
    data = {f_max: data.iloc[np.where(data["f_max"] == f_max)[0]] for f_max in f}
    summary = {}
    for key in keys:
        summary[key] = {}
        for percentile in percentiles:
            summary[key][percentile] = np.array(
                [np.percentile(d[key], percentile) for d in data.values()]
            )
    summary = {"frequencies": f, "percentiles": summary}
    if filename is not None:
        np.save(filename, summary)
    return summary


def main(args):
    f_max_scan = [None]
    if hasattr(args, "f_max"):
        if isinstance(args.f_max, list):
            f_max_scan = args.f_max
        else:
            f_max_scan = [args.f_max]

    # load model and initialize dingo sampler
    model = PosteriorModel(
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_filename=args.dingo_model,
        load_training_info=False,
    )
    sampler_kwargs = dict(
        model=model,
        init_sampler=FixedInitSampler({}, log_prob=0),
        fixed_context_parameters={"chirp_mass_proxy": np.nan},
        num_iterations=1,
    )
    base_sampler = GWSamplerGNPE(**sampler_kwargs)
    base_domain = base_sampler.domain.base_domain

    # IS kwargs
    likelihood_kwargs = dict(
        phase_marginalization_kwargs=dict(approximation_22_mode=True),
        decimate=True,
        phase_heterodyning=True,
    )
    # synthetic_phase_kwargs = dict(
    #     approximation_22_mode=True, n_grid=1001, num_processes=args.num_processes
    # )
    # likelihood_kwargs_synthetic_phase = {
    #     k: v
    #     for k, v in likelihood_kwargs.items()
    #     if not k.endswith("_marginalization_kwargs")
    # }

    # build injection generator
    injection_generator = get_injection_generator(
        model.metadata, base_domain.domain_dict, getattr(args, "fixed_parameters", None)
    )

    # initialize event metadata
    base_event_metadata = {
        "f_min": injection_generator.data_domain.f_min,
        "f_max": injection_generator.data_domain.f_max,
        **deepcopy(model.metadata["train_settings"]["data"]["window"]),
    }
    base_event_metadata["window_type"] = base_event_metadata.pop("type")

    # get chirp mass hyperprior
    sample_chirp_mass_proxy, get_chirp_mass_prior = get_chirp_mass_functions(
        model.metadata,
        injection_generator,
        getattr(args, "fixed_parameters", {}).get("chirp_mass"),
    )

    summary_dingo = {}
    summary_dingo_is = {}

    for i in range(args.num_injections):
        # sample from hyperprior and set corresponding prior for injection generator
        chirp_mass_proxy = sample_chirp_mass_proxy()
        injection_generator.prior["chirp_mass"] = get_chirp_mass_prior(chirp_mass_proxy)
        # generate an injection
        data = injection_generator.random_injection()
        theta = deepcopy(data["parameters"])
        print(chirp_mass_proxy, theta["chirp_mass"])

        for f_max in f_max_scan:
            print(f"\n\nf_max: {f_max}")
            # set f_max in sampler and event_metadata for importance sampling
            aux = {"f_max": f_max}
            if f_max is not None:
                frequency_update = {"f_max": f_max}
                sampler = GWSamplerGNPE(
                    **sampler_kwargs, frequency_masking=frequency_update
                )
                event_metadata = {**base_event_metadata, **frequency_update}
            else:
                sampler = base_sampler
                event_metadata = base_event_metadata

            sampler.context = data
            set_chirp_mass(sampler, chirp_mass_proxy)

            sampler.run_sampler(
                num_samples=args.num_samples, batch_size=args.batch_size
            )
            result = sampler.to_result()

            update_summary_data(
                summary_dingo, args, theta, result.samples, weights=None, **aux
            )

            if getattr(args, "importance_sampling", False):
                result.event_metadata = event_metadata
                # result.sample_synthetic_phase(
                #     synthetic_phase_kwargs, likelihood_kwargs_synthetic_phase
                # )
                result.importance_sample(
                    num_processes=args.num_processes, **likelihood_kwargs
                )
                print(
                    f"{i:2.0f}: Sample efficiency {result.sample_efficiency * 100:.1f}%"
                )

                aux["snr"] = compute_snr(result)
                aux["log_noise_evidence"] = result.log_noise_evidence
                aux["log_evidence"] = result.log_evidence
                update_summary_data(
                    summary_dingo_is,
                    args,
                    theta,
                    result.samples,
                    weights=np.array(result.samples["weights"]),
                    **aux,
                )

    summary_dingo = pd.DataFrame(summary_dingo)
    summary_dingo_is = pd.DataFrame(summary_dingo_is)
    if args.process_id is not None:
        label = f"_{args.process_id}"
    else:
        label = ""
    summary_dingo.to_pickle(join(args.outdirectory, f"summary-dingo{label}.pd"))
    if len(summary_dingo_is) > 0:
        summary_dingo_is.to_pickle(
            join(args.outdirectory, f"summary-dingo-is{label}.pd")
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
