import argparse
import os
from os.path import join
import pickle
import wandb
import torch

from dingo.core.posterior_models.build_model import build_model_from_kwargs
from dingo.gw.result import Result
from dingo.gw.inference.gw_samplers import GWSampler
from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw import injection

BATCH_SIZE = 10000


def evaluate_model(
    pm,
    data_dir_base,
    num_samples=10000,
    num_injections=5,
    evaluation_data_path=None,
    phase_marginalization=False,
    use_wandb=False,
):
    logging_dict = {f"inj_{i}": {} for i in range(num_injections)}

    for i in range(num_injections):
        data_dir = join(data_dir_base, f"epoch_{pm.epoch:03}", f"{i:02}")
        os.makedirs(data_dir, exist_ok=True)
        if evaluation_data_path is not None:
            try:
                with open(
                    join(evaluation_data_path, f"{i:02}", "strain_data.pkl"), "rb"
                ) as f:
                    injection_strain = pickle.load(f)
            except FileNotFoundError:
                injection_strain = get_injection(pm, data_dir)
        else:
            injection_strain = get_injection(pm, data_dir)

        run_inference(injection_strain, pm, data_dir, num_samples=num_samples)

        result = run_importance_sampling(
            data_dir, phase_marginalization=phase_marginalization
        )

        if use_wandb:
            is_dir = join(data_dir, "IS")
            wandb.define_metric("epoch")
            wandb.define_metric("*", step_metric="epoch")
            logging_dict[f"inj_{i}"] = {
                "num_samples": len(result.samples),
                "log_evidence": result.log_evidence,
                "effective_samples": result.n_eff,
                "sample_efficiency": 100 * result.sample_efficiency,
                # "cornerplot": wandb.Image(join(is_dir, "corner.png")),
                # "log_probs": wandb.Image(join(is_dir, "log_probs.png")),
                # "weights": wandb.Image(join(is_dir, "weights.png"))
            }
    if use_wandb:
        wandb.log(logging_dict)


def run_inference(strain_data, pm_model, injection_dir, num_samples=1000):
    sampler = GWSampler(model=pm_model)
    sampler.context = strain_data
    sampler.run_sampler(num_samples=num_samples, batch_size=BATCH_SIZE)
    sampler.to_hdf5(outdir=injection_dir)
    return sampler.samples


def run_importance_sampling(injection_dir, phase_marginalization=False):
    samples_path = join(injection_dir, "result.hdf5")
    result = Result(file_name=samples_path)
    is_dir = join(injection_dir, "IS")
    os.makedirs(is_dir, exist_ok=True)

    # We have likelihoods since we didn't use GNPE
    assert "log_prob" in result.samples.columns

    phase_marginalization_kwargs = (
        {"approximation_22_mode": True} if phase_marginalization else None
    )
    result.importance_sample(
        num_processes=30, phase_marginalization_kwargs=phase_marginalization_kwargs
    )
    result.print_summary()
    result.to_file(file_name=join(is_dir, "dingo_samples_weighted.hdf5"))
    try:
        result.plot_corner(filename=join(is_dir, "corner.png"))
        result.plot_log_probs(filename=join(is_dir, "log_probs.png"))
        result.plot_weights(filename=join(is_dir, "weights.png"))
    except:
        pass
    return result


def get_injection(pm, injection_dir):
    asd_path = pm.metadata["train_settings"]["training"]["stage_0"]["asd_dataset_path"]
    asd_dataset = ASDDataset(file_name=asd_path)
    asds = asd_dataset.asds

    os.makedirs(injection_dir, exist_ok=True)

    injection_generator = injection.Injection.from_posterior_model_metadata(pm.metadata)
    detectors = pm.metadata["train_settings"]["data"]["detectors"]
    injection_generator.asd = {det: asds[det][0] for det in detectors}
    strain_data = injection_generator.random_injection()

    with open(join(injection_dir, "strain_data.pkl"), "wb") as f:
        pickle.dump(strain_data, f)
    return strain_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir", required=True, help="Base save directory for the evaluation"
    )
    parser.add_argument("--model_path", required=True, help="Path to posterior model")
    parser.add_argument(
        "--num_samples", type=int, default=10000, help="Path to posterior model"
    )
    parser.add_argument(
        "--num_injections", type=int, default=5, help="Path to posterior model"
    )
    parser.add_argument(
        "--evaluation_data_path",
        default=None,
    )
    parser.add_argument("--phase_marginalization", action="store_true")
    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()

    base_dir = args.save_dir

    pm = build_model_from_kwargs(
        filename=args.model_path, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    evaluate_model(
        pm,
        base_dir,
        num_samples=args.num_samples,
        num_injections=args.num_injections,
        evaluation_data_path=args.evaluation_data_path,
        phase_marginalization=args.phase_marginalization,
        use_wandb=args.wandb,
    )
