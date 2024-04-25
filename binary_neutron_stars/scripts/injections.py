from dingo.core.models import PosteriorModel
import dingo.gw.injection as injection
from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.inference.gw_samplers import GWSamplerGNPE
from dingo.core.samplers import FixedInitSampler
from dingo.gw.domains import build_domain
from dingo.gw.data.event_dataset import EventDataset
from dingo.gw.result import Result

import yaml
from types import SimpleNamespace
from pp_utils import weighted_percentile_of_score, make_pp_plot
import numpy as np
import pandas as pd
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
from bilby.core.prior import PriorDict, Uniform, DeltaFunction
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
from ligo.skymap import kde, io
from ligo.skymap.postprocess import crossmatch
from os.path import join
import argparse
import pdb


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
    outdirectory = parser.parse_args().outdirectory
    with open(join(outdirectory, "injection_settings.yaml"), "r") as f:
        args = yaml.safe_load(f)
    args = SimpleNamespace(**args)
    args.outdirectory = outdirectory
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

    samples_rej = samples.sample(len(samples), weights=weights, replace=True)

    # insert chirp mass std
    data["std_chirp_mass"] = np.std(samples_rej["chirp_mass"])

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


def get_chirp_mass_kernel_and_hyperprior(
    model_metadata, injection_generator, fixed_chirp_mass=None
):
    chirp_mass_kernel = PriorDict(
        deepcopy(model_metadata["train_settings"]["data"]["gnpe_chirp"]["kernel"])
    )["chirp_mass"]
    if not isinstance(chirp_mass_kernel, Uniform):
        raise NotImplementedError()
    if fixed_chirp_mass is None:
        chirp_mass_hyperprior = PriorDict(
            deepcopy(model_metadata["dataset_settings"]["intrinsic_prior"])
        )["chirp_mass"]
        if {"mass_1", "mass_2"}.issubset(injection_generator.prior.keys()):
            m1 = injection_generator.prior["mass_1"]
            m2 = injection_generator.prior["mass_2"]
            mc_max = (m1.maximum * m2.maximum) ** 0.6 / (m1.maximum + m2.maximum) ** 0.2
            mc_min = (m1.minimum * m2.minimum) ** 0.6 / (m1.minimum + m2.minimum) ** 0.2
            chirp_mass_hyperprior.maximum = min(chirp_mass_hyperprior.maximum, mc_max)
            chirp_mass_hyperprior.minimum = max(chirp_mass_hyperprior.minimum, mc_min)
        chirp_mass_hyperprior.minimum -= chirp_mass_kernel.minimum
        chirp_mass_hyperprior.maximum -= chirp_mass_kernel.maximum
    else:
        chirp_mass_hyperprior = DeltaFunction(fixed_chirp_mass)
    return chirp_mass_kernel, chirp_mass_hyperprior


def main(args):
    # load model and initialize dingo sampler
    model = PosteriorModel(
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_filename=args.dingo_model,
        load_training_info=False,
    )
    sampler = GWSamplerGNPE(
        model=model,
        init_sampler=FixedInitSampler({}, log_prob=0),
        fixed_context_parameters={"chirp_mass_proxy": np.nan},
        num_iterations=1,
        frequency_masking=getattr(args, "frequency_update", None),
    )

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
        model.metadata,
        sampler.domain.base_domain.domain_dict,
        getattr(args, "fixed_parameters", None),
    )

    # initialize event metadata
    event_metadata = {
        "f_min": injection_generator.data_domain.f_min,
        "f_max": injection_generator.data_domain.f_max,
        **deepcopy(model.metadata["train_settings"]["data"]["window"]),
    }
    event_metadata = {**event_metadata, **getattr(args, "frequency_update", {})}
    event_metadata["window_type"] = event_metadata.pop("type")

    # get chirp mass hyperprior
    chirp_mass_kernel, chirp_mass_hyperprior = get_chirp_mass_kernel_and_hyperprior(
        model.metadata,
        injection_generator,
        getattr(args, "fixed_parameters", {}).get("chirp_mass"),
    )

    summary_dingo = {}
    summary_dingo_is = {}

    for i in range(args.num_injections):
        # sample from hyperprior and set corresponding chirp mass prior for injection generator
        chirp_mass_proxy = chirp_mass_hyperprior.sample()

        if getattr(args, "fixed_parameters", {}).get("chirp_mass") is not None:
            injection_generator.prior["chirp_mass"] = DeltaFunction(
                args.fixed_parameters["chirp_mass"]
            )
        else:
            chirp_mass_prior = Uniform(
                minimum=chirp_mass_proxy + chirp_mass_kernel.minimum,
                maximum=chirp_mass_proxy + chirp_mass_kernel.maximum,
            )
            injection_generator.prior["chirp_mass"] = chirp_mass_prior
            # print(chirp_mass_prior)

        # generate an injection
        data = injection_generator.random_injection()
        theta = deepcopy(data["parameters"])
        print(chirp_mass_proxy, theta["chirp_mass"])

        # sampler.frequency_masking = dict(f_max=40)
        # sampler._initialize_transforms()
        sampler.context = data
        set_chirp_mass(sampler, chirp_mass_proxy)

        sampler.run_sampler(num_samples=args.num_samples, batch_size=args.batch_size)
        result = sampler.to_result()

        update_summary_data(summary_dingo, args, theta, result.samples, weights=None)

        if getattr(args, "importance_sampling", False):
            result.event_metadata = event_metadata
            # result.sample_synthetic_phase(
            #     synthetic_phase_kwargs, likelihood_kwargs_synthetic_phase
            # )
            result.importance_sample(
                num_processes=args.num_processes, **likelihood_kwargs
            )
            print(f"{i:2.0f}: Sample efficiency {result.sample_efficiency * 100:.1f}%")
            update_summary_data(
                summary_dingo_is,
                args,
                theta,
                result.samples,
                weights=np.array(result.samples["weights"]),
                log_evidence=result.log_evidence,
            )

    print(summary_dingo)
    print(summary_dingo_is)

    summary_dingo = pd.DataFrame(summary_dingo)
    summary_dingo_is = pd.DataFrame(summary_dingo_is)
    summary_dingo.to_pickle(join(args.outdirectory, f"summary-dingo_{args.label}.pd"))
    if len(summary_dingo_is) > 0:
        summary_dingo_is.to_pickle(
            join(args.outdirectory, f"summary-dingo-is_{args.label}.pd")
        )

    # if args.plot:
    #     make_pp_plot(
    #         np.stack(list(percentiles.values())).T,
    #         list(percentiles.keys()),
    #         join(args.outdirectory, args.label + "_pp.pdf"),
    #     )
    #     make_pp_plot(
    #         np.stack(list(percentiles_is.values())).T,
    #         list(percentiles_is.keys()),
    #         join(args.outdirectory, args.label + "_pp-is.pdf"),
    #     )


if __name__ == "__main__":
    args = parse_args()
    main(args)
