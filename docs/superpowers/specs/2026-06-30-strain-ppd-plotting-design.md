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
result.plot_ppd_td(filename="ppd_td.png", credible_interval=0.9,
                   num_waveforms=1000, num_processes=1, zoom=None, ppd=None)
result.plot_ppd_fd(filename="ppd_fd.png", credible_interval=0.9,
                   num_waveforms=1000, num_processes=1, ppd=None)
```

They live on the GW `Result` (subclass of `CoreResult`), **not** `core/`,
because they need the GW likelihood, detector projection, and ASD whitening —
`core/` stays domain-agnostic.

Shared expensive computation lives in a private
`_compute_ppd(self, credible_interval, num_waveforms, num_processes, seed)`
(used by both methods, so extraction is justified) returning
`{domain, ifos, wf_fd, data_fd}`. Each public method accepts an optional
precomputed `ppd=` dict so a caller can generate the draws once and render both
plots without regenerating waveforms.

Demo run target: GW230709_122727 EAS importance-sampling result
(`prod_o4a/working/S230709bi/Exp19_more_points_1/result/
Exp19_more_points_1_data0_1372940865-2_importance_sampling.hdf5`). A few-line
script loads the `Result` and calls both methods, producing
`GW230709_ppd_td.png` and `GW230709_ppd_fd.png`.

## Pipeline (per detector)

1. `result = Result(file_name=...)`; `result._build_likelihood()`.
   `domain = result.domain.base_domain if hasattr(result.domain, "base_domain")
   else result.domain`.
2. **Sample selection.** Build the `credible_interval` (default 0.9) credible
   set: normalize the importance/posterior weights, sort descending, and keep
   the highest-weight samples whose cumulative weight ≤ `credible_interval`
   (the X% credible region — same definition as the notebook's
   `select_from_n_percent_interval`). Then draw up to `num_waveforms` rows
   **uniformly at random** from that set (seeded `np.random.default_rng`) so the
   envelope represents the whole interval rather than only the peak (the
   notebook took the top-N highest-weight rows, biasing toward the peak — we
   diverge here).
3. Generate whitened FD waveforms:
   `apply_func_with_multiprocessing(result.likelihood.signal, samples_df,
   num_processes)`. Each `wf["waveform"][ifo]` is **already whitened**
   (`/asd/noise_std`), the same convention applied to the data, so model and
   data share one y-scale.

### Time offset (hidden from the user)

The merger always lands at **t = 0**. Internally each waveform is phase-shifted
by a fixed offset derived from the data segment (`t0 = 1 / domain.delta_f`, the
segment duration `T`, which places the dingo coalescence at the segment edge),
via `wf *= exp(2πi·f·t0)`, then the time axis has `t0` subtracted. `t0` is a
private detail — not a method argument. The implementation verifies the demo
merger sits at 0 and adjusts the offset derivation if not.

## Time domain (`<event>_ppd_td.png`, one row per detector)

- Each draw: phase-shift by `t0`, `one_sided_fd_to_td`, keep real part.
- Band = pointwise **min/max envelope** across the selected draws
  (`np.nanmin` / `np.nanmax`, ignoring NaN waveforms), drawn with
  `fill_between`.
- Overlay: whitened data trace (`√(4Δf)·context_waveform/asd`, same shift +
  IFFT, light sliding-window average), grey.
- x-axis: time to merger (s), linear, zoomed via `zoom=` (default
  `(-0.7, 0.1)` for GW230709). y-axis: whitened strain (dimensionless, σ units).

## Frequency domain (`<event>_ppd_fd.png`, one row per detector)

- Per draw: `|whitened h(f)|` (magnitude of the complex whitened FD waveform).
- Band = min/max envelope of `|whitened h(f)|` across the selected draws.
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
