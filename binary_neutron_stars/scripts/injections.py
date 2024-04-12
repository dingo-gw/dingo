from dingo.core.models import PosteriorModel
import dingo.gw.injection as injection
from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.inference.gw_samplers import GWSamplerGNPE
from dingo.core.samplers import FixedInitSampler
from dingo.gw.domains import build_domain
from dingo.gw.data.event_dataset import EventDataset
from dingo.gw.result import Result

from pp_utils import weighted_percentile_of_score, make_pp_plot
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from os.path import join
import argparse
import pdb


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Evaluate on injections.",
    )
    parser.add_argument(
        "--dingo_model", type=str, required=True,
        help="Path to dingo model."
    )
    parser.add_argument(
        "--outdirectory", type=str, required=True,
        help="Directory where data will be saved."
    )
    parser.add_argument(
        "--label", type=str, required=True,
        help="Label for saving files."
    )
    parser.add_argument(
        "--num_injections", type=int, required=True,
        help="Number of injections to perform."
    )
    parser.add_argument(
        "--num_samples", type=int, default=10_000,
        help="Number of samples per injection."
    )
    parser.add_argument(
        "--batch_size", type=int, default=100_000,
        help="Batch size for sampling."
    )
    parser.add_argument(
        "--num_processes", type=int, default=32,
        help="Number of parallel processes for importance sampling."
    )
    parser.add_argument("--plot", action='store_true')
    args = parser.parse_args()
    return args

def asd_to_ufd(asd, mfd):
    assert asd.shape == (len(mfd), )
    asd_ufd = np.repeat(asd, mfd._decimation_factors_bands[mfd._band_assignment])
    npad = ((0, 0) * (len(asd.shape) - 1)) + (mfd.base_domain.min_idx, 0)
    asd_ufd = np.pad(asd_ufd, pad_width=npad, mode='constant', constant_values=1)
    assert np.allclose(mfd.decimate(asd_ufd), asd)
    return asd_ufd

def set_chirp_mass(sampler, chirp_mass):
    sampler.fixed_context_parameters = {"chirp_mass_proxy": chirp_mass}
    sampler.transform_pre.transforms[0].fixed_parameters['chirp_mass'] = chirp_mass

def adjust_chirp_mass_prior(injection_generator, model_metadata):
    lower = float(model_metadata['train_settings']['data']['gnpe_chirp']['kernel']['chirp_mass'].split("minimum=")[1].split(',')[0])
    upper = float(model_metadata['train_settings']['data']['gnpe_chirp']['kernel']['chirp_mass'].split("maximum=")[1].split(')')[0])
    injection_generator.prior["chirp_mass"].minimum -= lower
    injection_generator.prior["chirp_mass"].maximum -= upper

# model_path = "/fast/groups/dingo/03_binary_neutron_stars/03_models/03_30M_datasets/02_GW190425_lowSpin/02_dataset-g_nn-xl_epochs-400/model_400.pt"
# ed_path = "/fast/groups/dingo/03_binary_neutron_stars/03_models/03_30M_datasets/02_GW190425_lowSpin/02_dataset-g_nn-xl_epochs-400/inference/01_model312_100k/data/GW190425_data0_1240215503-02_generation_event_data.hdf5"
# result_path = "/fast/groups/dingo/03_binary_neutron_stars/03_models/03_30M_datasets/02_GW190425_lowSpin/02_dataset-g_nn-xl_epochs-400/inference/01_model312_100k/result/GW190425_data0_1240215503-02_sampling.hdf5"

def main(args):
    # load model and initialize dingo sampler
    model = PosteriorModel(
        device="cuda",
        model_filename=args.dingo_model, 
        load_training_info=False
    )
    sampler = GWSamplerGNPE(
        model=model, 
        init_sampler=FixedInitSampler({}, log_prob=0),
        fixed_context_parameters={"chirp_mass_proxy": np.nan}, 
        num_iterations=1
    )

    # IS kwargs
    likelihood_kwargs = dict(decimate=True, phase_heterodyning=True)
    synthetic_phase_kwargs = dict(approximation_22_mode=True, n_grid=1001, num_processes=args.num_processes)
    likelihood_kwargs_synthetic_phase = {
        k: v
        for k, v in likelihood_kwargs.items()
        if not k.endswith("_marginalization_kwargs")
    }

    ### build injection generator
    model_metadata = deepcopy(model.metadata)
    # generate injection in base domain
    model_metadata['dataset_settings']['domain'] = sampler.domain.base_domain.domain_dict
    model_metadata['train_settings']['data'].pop('domain_update')
    injection_generator = injection.Injection.from_posterior_model_metadata(model_metadata)
    adjust_chirp_mass_prior(injection_generator, model_metadata)
    asd_fname = model.metadata["train_settings"]["training"]["stage_0"]["asd_dataset_path"]
    asd_dataset = ASDDataset(file_name=asd_fname)
    injection_generator.asd = {k: asd_to_ufd(v[0], asd_dataset.domain) for k,v in asd_dataset.asds.items()}
    event_metadata = {
        'f_min': injection_generator.data_domain.f_min,
        'f_max': injection_generator.data_domain.f_max,
        **model_metadata['train_settings']['data']['window']
    }
    event_metadata['window_type'] = event_metadata.pop('type')

    for i in range(args.num_injections):
        data = injection_generator.random_injection()
        theta = deepcopy(data['parameters'])

        sampler.context = data
        set_chirp_mass(sampler, theta["chirp_mass"])

        sampler.run_sampler(num_samples=args.num_samples, batch_size=args.batch_size)
        result = sampler.to_result()

        if True:
            result.event_metadata = event_metadata
            result.sample_synthetic_phase(synthetic_phase_kwargs, likelihood_kwargs_synthetic_phase)
            result.importance_sample(num_processes=args.num_processes, **likelihood_kwargs)
        else:
            result.samples["weights"] = np.ones(len(result.samples))

        ESS = np.sum(result.samples["weights"]) ** 2 / np.sum(result.samples["weights"] ** 2)
        sample_efficiency = ESS / len(result.samples)
        print(f"\n\n\n{i:2.0f}: Sample efficiency {sample_efficiency * 100:.1f}%\n\n\n")

        # log percentiles and sample efficiency
        if i == 0:
            common_keys = theta.keys() & result.samples.keys()
            percentiles = {k: [] for k in common_keys}
            percentiles_is = {k: [] for k in common_keys}
            sample_efficiencies = []
        for k in common_keys:
            weights = np.array(result.samples["weights"])
            samples = np.array(result.samples[k])
            percentiles[k].append(weighted_percentile_of_score(samples, theta[k]))
            percentiles_is[k].append(weighted_percentile_of_score(samples, theta[k], weights=weights))
        sample_efficiencies.append(sample_efficiency)


    summary_data = {
        "sample_efficiencies": sample_efficiencies,
        **{"dingo:" + k: np.array(v) for k, v in percentiles.items()},
        **{"dingo-is:" + k: np.array(v) for k, v in percentiles_is.items()},
    }
    pd.DataFrame(summary_data).to_pickle(join(args.outdirectory, args.label + ".pd"))

    if args.plot:
        make_pp_plot(np.stack(list(percentiles.values())).T, list(percentiles.keys()), join(args.outdirectory, args.label + "_pp.pdf"))
        make_pp_plot(np.stack(list(percentiles_is.values())).T, list(percentiles.keys()), join(args.outdirectory, args.label + "_pp-is.pdf"))


if __name__ == "__main__":
    args = parse_args()
    main(args)
