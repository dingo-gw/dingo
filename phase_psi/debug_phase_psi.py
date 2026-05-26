"""
Localize the source of the phase-psi grid vs direct-likelihood discrepancy.

Runs three targeted comparisons for a few samples, in one shot:

  (1) PER-MODE psi reconstruction: does cos(2psi)*A_m + sin(2psi)*B_m (from
      signal_m_psi_basis at psi=0, pi/4) equal dingo's own signal_m projected
      directly at psi, mode by mode?  -> isolates signal_m_psi_basis + the antenna
      identity, with NO phase and NO mode-decomposition error.  EXPECT ~1e-13.

  (2) my phase-psi grid (at fixed psi) vs dingo's own phase grid
      (_log_likelihood_phase_grid_mode_decomposed). Both use the per-mode
      (generate_hplus_hcross_m) path, so my method should reduce to dingo's at a
      single psi.  EXPECT ~1e-10.

  (3) dingo's phase grid vs the direct _log_likelihood. This is dingo's *inherent*
      mode-decomposition error (full generate_hplus_hcross path vs the per-mode
      path) and is the same approximation dingo's synthetic phase already uses.

If (1) and (2) are tiny but (3) is large, the discrepancy is dingo's inherent
mode-decomposition error, not a bug in this implementation.

Usage: same args as check_phase_psi_likelihood.py.
"""

import argparse

import numpy as np

from dingo.core.posterior_models.build_model import build_model_from_kwargs
from dingo.gw.data.event_dataset import EventDataset
from dingo.gw.inference.gw_samplers import GWSampler

from phase_psi_marginalization import (
    signal_m_psi_basis,
    phase_psi_grid_log_likelihood,
    PSI_BASIS,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_filename", required=True)
    p.add_argument("--event_dataset_filename", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_samples", type=int, default=3)
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
    result._build_likelihood()
    lk = result.likelihood

    print(f"domain = {type(lk.data_domain).__name__}, "
          f"min_idx = {lk.data_domain.min_idx}, n_freq = {len(lk.data_domain)}")

    from bilby.core.prior import DeltaFunction

    fixed = {k: v.peak for k, v in result.prior.items() if isinstance(v, DeltaFunction)}

    phases = np.linspace(0, 2 * np.pi, 16, endpoint=False)

    for idx in range(args.n_samples):
        row = result.samples.iloc[idx].to_dict()
        theta = {**fixed, **row}
        for k in ("log_prob", "log_prior", "weights"):
            theta.pop(k, None)
        psi_val = float(theta.get("psi", 0.3))

        print(f"\n=== sample {idx} (psi = {psi_val:.4f}) ===")

        # ---- (1) per-mode psi reconstruction vs dingo's direct projection ----
        A, B = signal_m_psi_basis(lk, {**theta, "phase": 0.0}, psi_basis=PSI_BASIS)
        ref = lk.signal_m({**theta, "phase": 0.0, "psi": psi_val})
        c, s = np.cos(2 * psi_val), np.sin(2 * psi_val)
        worst = 0.0
        for m in sorted(A.keys()):
            for ifo in A[m]:
                recon = c * A[m][ifo] + s * B[m][ifo]
                direct = ref[m]["waveform"][ifo]
                denom = np.max(np.abs(direct)) + 1e-30
                worst = max(worst, np.max(np.abs(recon - direct)) / denom)
        print(f"(1) per-mode psi reconstruction max rel err = {worst:.3e}  "
              f"(expect ~1e-13)")

        # ---- (2) my phase-psi grid (fixed psi) vs dingo's phase grid ----
        my_grid = phase_psi_grid_log_likelihood(
            lk, theta, phases, np.array([psi_val])
        )[:, 0]
        dingo_grid = lk._log_likelihood_phase_grid_mode_decomposed(
            {**theta, "psi": psi_val}, phases=phases
        )
        d12 = np.max(np.abs(my_grid - dingo_grid))
        print(f"(2) my grid vs dingo phase grid  max|.| = {d12:.3e}  (expect ~1e-10)")

        # ---- (3) dingo's phase grid vs direct likelihood ----
        direct = np.array(
            [lk._log_likelihood({**theta, "phase": float(ph), "psi": psi_val})
             for ph in phases]
        )
        d3 = np.max(np.abs(dingo_grid - direct))
        d_mine = np.max(np.abs(my_grid - direct))
        print(f"(3) dingo phase grid vs direct   max|.| = {d3:.3e}  "
              f"(dingo's inherent mode-decomp error)")
        print(f"    my grid vs direct            max|.| = {d_mine:.3e}")


if __name__ == "__main__":
    main()
