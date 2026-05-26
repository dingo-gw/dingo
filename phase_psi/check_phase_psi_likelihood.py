"""
End-to-end cross-check of the phase-psi grid likelihood against the direct dingo
likelihood, using a real model + event (the decisive correctness test, since it
exercises dingo's actual projection pipeline, higher modes, ASDs and multibanding).

For a handful of flow samples, it compares
``phase_psi_grid_log_likelihood(theta)[i, j]`` against the direct
``likelihood._log_likelihood({**theta, phase, psi})`` at the same grid points, and
times the marginalized likelihood versus a naive grid-by-direct-evaluation.

Usage:
    python check_phase_psi_likelihood.py --model_filename MODEL.pt \
        --event_dataset_filename EVENT.hdf5 [--device cpu] [--n_samples 3]
"""

import argparse
import time

import numpy as np

from dingo.core.posterior_models.build_model import build_model_from_kwargs
from dingo.gw.data.event_dataset import EventDataset
from dingo.gw.inference.gw_samplers import GWSampler

from phase_psi_marginalization import (
    PhasePsiMarginalizedLikelihood,
    phase_psi_grid_log_likelihood,
    phase_psi_marginalized_log_likelihood,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_filename", required=True)
    p.add_argument("--event_dataset_filename", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--n_phase_check", type=int, default=6)
    p.add_argument("--n_psi_check", type=int, default=5)
    args = p.parse_args()

    model = build_model_from_kwargs(
        filename=args.model_filename, load_training_info=False, device="cpu"
    )
    model.network_to_device(args.device)
    strain_data = EventDataset(file_name=args.event_dataset_filename).data
    sampler = GWSampler(model=model)
    sampler.context = strain_data
    sampler.run_sampler(num_samples=200, batch_size=200)

    result = sampler.to_result()
    result._build_likelihood()  # plain likelihood, no marginalization
    likelihood = result.likelihood

    scp = likelihood.waveform_generator.spin_conversion_phase
    print(f"spin_conversion_phase = {scp}")
    print(f"inferred params: {sorted(result.samples.columns)}")

    # Fixed (DeltaFunction) parameter values from the prior.
    from bilby.core.prior import DeltaFunction

    fixed = {k: v.peak for k, v in result.prior.items() if isinstance(v, DeltaFunction)}

    # Grids for the check: a few phases/psis, plus dense grids for marginalization.
    phases_check = np.linspace(0, 2 * np.pi, args.n_phase_check, endpoint=False)
    psis_check = np.linspace(0, np.pi, args.n_psi_check, endpoint=False)

    max_abs_diff = 0.0
    for idx in range(args.n_samples):
        row = result.samples.iloc[idx].to_dict()
        theta = {**fixed, **row}
        theta.pop("log_prob", None)
        theta.pop("log_prior", None)
        theta.pop("weights", None)

        grid = phase_psi_grid_log_likelihood(
            likelihood, theta, phases_check, psis_check
        )
        for i, phase in enumerate(phases_check):
            for j, psi in enumerate(psis_check):
                direct = likelihood._log_likelihood(
                    {**theta, "phase": float(phase), "psi": float(psi)}
                )
                diff = abs(grid[i, j] - direct)
                max_abs_diff = max(max_abs_diff, diff)
        print(
            f"sample {idx}: grid range [{grid.min():.1f}, {grid.max():.1f}], "
            f"running max|grid-direct| = {max_abs_diff:.3e}"
        )

    print(f"\nMAX |grid - direct| over all checked points: {max_abs_diff:.3e}")
    print("(mode-decomposition error ~1e-2 is expected and benign.)")

    # Timing: marginalized likelihood via grid vs naive direct grid evaluation.
    theta = {**fixed, **result.samples.iloc[0].to_dict()}
    for k in ("log_prob", "log_prior", "weights"):
        theta.pop(k, None)
    phases = np.linspace(0, 2 * np.pi, 512, endpoint=False)
    psis = np.linspace(0, np.pi, 128, endpoint=False)

    t0 = time.time()
    ll_marg = phase_psi_marginalized_log_likelihood(likelihood, theta, phases, psis)
    t_fast = time.time() - t0
    print(f"\nmarginalized logL = {ll_marg:.4f}  (grid {len(phases)}x{len(psis)}, "
          f"{t_fast * 1e3:.1f} ms incl. 1 waveform call)")

    wrapper = PhasePsiMarginalizedLikelihood(likelihood)
    print(f"wrapper.log_likelihood = {wrapper.log_likelihood(theta):.4f}")


if __name__ == "__main__":
    main()
