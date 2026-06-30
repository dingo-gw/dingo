# Strain PPD plotting — design

**Date:** 2026-06-30
**Branch:** `hackathon/ppd-plotting`

## Goal

Given a dingo `Result` HDF5 file, compute and plot the posterior-predictive
distribution (PPD) of the **whitened strain** in both the time domain and the
frequency domain, per detector, and save the figures as PNGs.

Derived from `/work/nihargupte/analysis/dingo/asimov/eccentricity/plotting/
production_plots/strain_posterior.ipynb`, but stripped of the
eccentricity/glitch/prior-hull extras and generalized to a clean, reusable
script.

## Deliverable

Two plotting **methods on the GW `Result` class** (`dingo/gw/result.py`),
mirroring the existing `result.plot_corner` API:

```python
result.plot_ppd_td(filename="ppd_td.png", num_waveforms=1000,
                   num_processes=1, central_time=8.0, zoom=None, ppd=None)
result.plot_ppd_fd(filename="ppd_fd.png", num_waveforms=1000,
                   num_processes=1, ppd=None)
```

They live on the GW `Result` (subclass of `CoreResult`), **not** `core/`,
because they need the GW likelihood, detector projection, and ASD whitening —
`core/` stays domain-agnostic.

Shared expensive computation lives in a private
`_compute_ppd(self, num_waveforms, num_processes, seed)` (used by both methods,
so extraction is justified) returning `{domain, ifos, wf_fd, data_fd}`. Each
public method accepts an optional precomputed `ppd=` dict so a caller can
generate the draws once and render both plots without regenerating waveforms.

Demo run target: GW230709_122727 EAS importance-sampling result
(`prod_o4a/working/S230709bi/Exp19_more_points_1/result/
Exp19_more_points_1_data0_1372940865-2_importance_sampling.hdf5`). A few-line
script loads the `Result` and calls both methods, producing
`GW230709_ppd_td.png` and `GW230709_ppd_fd.png`.

## Pipeline (per detector)

1. `result = Result(file_name=...)`; `result._build_likelihood()`.
   `domain = result.domain.base_domain if hasattr(result.domain, "base_domain")
   else result.domain`.
2. **Sample selection.** Resample `N` parameter rows from `result.samples` with
   replacement, probability ∝ importance `weights`, using a seeded
   `np.random.default_rng`. This is a proper posterior-predictive draw and
   respects the importance weights. (The notebook instead kept the top-90%
   credible set then truncated to `N`; we deliberately diverge.)
3. Generate whitened FD waveforms:
   `apply_func_with_multiprocessing(result.likelihood.signal, samples_df,
   num_processes)`. Each `wf["waveform"][ifo]` is **already whitened**
   (`/asd/noise_std`), the same convention applied to the data, so model and
   data share one y-scale.

## Time domain (`<event>_ppd_td.png`, one row per detector)

- Each draw: phase-shift to place merger at `CENTRAL_TIME`
  (`wf *= exp(2πi·f·CENTRAL_TIME)`), then `one_sided_fd_to_td`, keep real part.
- Bands across draws via `np.nanpercentile`: **median line + shaded 50%
  (25–75) + 90% (5–95)**. (Notebook used a min/max envelope; we deliberately
  diverge to percentile credible bands.)
- Overlay: whitened data trace (`√(4Δf)·context_waveform/asd`, same shift +
  IFFT, light sliding-window average), grey.
- x-axis: time to merger (s), linear, zoomed via `--zoom` (default
  `(-0.7, 0.1)` for GW230709). y-axis: whitened strain (dimensionless, σ units).

## Frequency domain (`<event>_ppd_fd.png`, one row per detector)

- Per draw: `|whitened h(f)|` (magnitude of the complex whitened FD waveform).
- Bands: median + 50%/90% percentiles vs frequency.
- Overlay: `|whitened data|` = `|√(4Δf)·context_waveform/asd|` — the noisy
  ~O(1) floor the signal rises above.
- x-axis: frequency [Hz], **log scale**. y-axis: dimensionless whitened
  amplitude.

## Reused logic

`one_sided_fd_to_td(fd, domain)` is reused verbatim from the notebook (added as
a module-level function in `dingo/gw/result.py`): it builds the full Hermitian
two-sided spectrum (DC zeroed), `np.fft.ifft(...) * sqrt(N)` (normalization tied
to the whitening so noise has unit variance), `dt = 1/(2·f_max)`, output length
`2n-1`. Correct and whitening-consistent.

## Testing / determinism

- Seeded `np.random.default_rng` for resampling → deterministic figures.
- `tests/gw/inference/test_ppd.py`: assert `one_sided_fd_to_td` round-trips —
  a one-sided spectrum with a single populated frequency bin returns a real
  cosine of the expected period (exercises the Hermitian-mirror + normalization
  logic, the one piece of non-trivial new math).

## Boundaries / conventions

- Methods on the GW `Result` in `gw/result.py`; imports `gw/` + `core/`
  (allowed direction). No new classes, no new dependencies.
- Whitening conventions for data (`√(4Δf)/asd`) and model (likelihood's
  `/asd/noise_std`, `noise_std = 1/√(4Δf)`) are identical and must be preserved.
