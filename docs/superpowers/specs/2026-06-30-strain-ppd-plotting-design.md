# Strain PPD plotting — design

**Date:** 2026-06-30
**Branch:** `hackathon/ppd-plotting`

## Goal

Given a dingo `Result` HDF5 file, compute and plot the posterior-predictive
distribution (PPD) of the **whitened strain** in the **time domain**, per
detector, and save the figure as a PNG. (Frequency-domain plotting is out of
scope for this PR.)

Derived from `/work/nihargupte/analysis/dingo/asimov/eccentricity/plotting/
production_plots/strain_posterior.ipynb`, but stripped of the
eccentricity/glitch/prior-hull extras and generalized to a clean, reusable
script.

## Deliverable

Rendering lives in a new GW-specific module **`dingo/gw/utils/plotting.py`**, the
domain-specific counterpart to `dingo/core/utils/plotting.py` (which holds
`plot_corner_multi`). It cannot live in `core/` because PPDs need the GW
likelihood, detector projection, and ASD whitening — `core/` stays
domain-agnostic. The module exposes:

```python
plot_ppd_td(wf_fd, data_fd, domain, map_fd=None, filename="ppd_td.png", zoom=None)
one_sided_fd_to_td(fd, domain)   # IFFT helper, also here
```

`wf_fd` is `{mode: {ifo: (n_waveforms, n_freq) complex}}` (whitened model
waveforms), `data_fd` is `{ifo: (n_freq,) complex}` (whitened data), and `map_fd`
is `{mode: {ifo: (n_freq,) complex}}` (the per-mode maximum-probability waveform)
— all from `_compute_ppd`. `plot_ppd_td` draws **one min/max envelope panel per
`(mode, ifo)`, stacked vertically** — `"dingo"` panels on top and, when present,
`"dingo-is"` below — each with the grey whitened-data trace beneath and the mode's
MAP waveform as a line on top. Each panel is labelled in-axes `"{mode} · {ifo}"`.

The GW `Result` (subclass of `CoreResult`) keeps a thin convenience method that
mirrors the `result.plot_corner` API and delegates to the module:

```python
result.plot_ppd_td(filename="ppd_td.png", credible_interval=0.9,
                   num_waveforms=1000, num_processes=1, zoom=None, ppd=None)
```

Both the **Dingo** posterior (credible set from the raw network samples) and —
when the result is importance-sampled (`"weights"` present) — the **Dingo-IS**
posterior get their own stacked panels (Dingo on top, Dingo-IS below).

The shared expensive computation stays a **private `Result` method**
`_compute_ppd(self, credible_interval, num_waveforms, num_processes, seed)`,
returning the tuple `(wf_fd, data_fd)`. It builds a credible set and generates
waveforms for **each** available posterior internally (no `weighting` argument):
`"dingo"` always, `"dingo-is"` iff importance-sampled. `plot_ppd_td` accepts a
precomputed `ppd=(wf_fd, data_fd)` to render without regenerating waveforms; the
`Result` method supplies `domain` from `self.domain`.

Robustness (prior filter): the normalizing-flow proposal `q(theta)` is a smooth
density over a box and leaks a small fraction of draws outside the prior — with
finite `log_prob` but unphysical (e.g. tiny negative spin magnitudes). These
account for most waveform-generation failures. `_compute_ppd` therefore restricts
to the prior support up front (`self.prior.ln_prob(...)`, keeping `log_prior`
finite) — the same in-prior restriction `importance_sample` applies — before
generating anything. This removes those failures *and* keeps unphysical waveforms
out of the envelope, and it is a strict improvement for the `"dingo"` mode (which
otherwise gives those out-of-prior draws equal mass `1/N`). Measured on the
GW150914 fixture: 43/5000 draws (0.86%) are out-of-prior; all failures there were
among them, zero in-prior draws failed.

Robustness (per-sample guard): the prior filter is **not** sufficient on its own.
A waveform model can fail on an *interior* parameter, and this bites even for
importance-sampled results whenever the waveform backend differs from the one used
at IS time — re-plotting a stored result with a newer/older `pyseobnr` /
`lalsimulation` is enough to make an IS-accepted draw fail. Observed on the
`S231001aq` (SEOBNRv5EHM) external result: 2/200 in-prior credible-set draws crash
inside pyseobnr's eccentric `CubicSpline` (`x must be strictly increasing`) under
`pyseobnr 0.3.6` / `lalsimulation 6.2.1`. So each draw is generated through the
module-level picklable `safe_signal` (`injection.py`): failures are dropped from
the `min`/`max` envelope with a warning giving the count, and only an all-failed
batch raises. (An earlier revision removed this guard on the mistaken assumption
that the prior filter left no in-prior failures for IS results; `S231001aq` is the
counterexample — the "IS already generated these" guarantee only holds for the
exact same waveform backend.)

Dev fixture: GW150914 dingo-ci result (IMRPhenomXPHM, no `pyseobnr` needed),
`.../dingo-ci/outdir_GW150914/result/
GW150914_data0_1126259462-4_importance_sampling_part1.hdf5`, rendered via the
throwaway `ppd_dev.py`.

## Pipeline (per detector)

1. `result = Result(file_name=...)`; `result._build_likelihood()`.
   `domain = result.domain.base_domain if hasattr(result.domain, "base_domain")
   else result.domain`.
2. **Sample selection.** Build the `credible_interval` (default 0.9) highest-
   posterior-density credible set using the per-sample probability **mass** that
   is correct for how the samples were drawn (no assumed reference density): for
   `"dingo"` the raw samples are equal-mass draws from q(θ) — mass `1/N`, ranked
   by `log_prob`; for `"dingo-is"` the target is the true posterior, so the mass
   is the self-normalized importance weight (the `weights` column, ∝ p/q), ranked
   by `log_likelihood + log_prior`. Sort by rank descending and keep the top
   samples whose cumulative mass ≤ `credible_interval` (standard HPD estimator —
   the same weighting `plot_corner` feeds to corner). Then draw up to
   `num_waveforms` rows **uniformly at random** from that set (seeded
   `np.random.default_rng`) so the envelope spans the whole interval rather than
   only the peak (the notebook took the top-N highest-weight rows, biasing toward
   the peak — we diverge here).
3. Generate whitened FD waveforms:
   `apply_func_with_multiprocessing(result.likelihood.signal, samples_df,
   num_processes)`. Each `wf["waveform"][ifo]` is **already whitened**
   (`/asd/noise_std`), the same convention applied to the data, so model and
   data share one y-scale.

### Time offset (hidden from the user)

The merger always lands at **t = 0**. Internally each waveform is phase-shifted
by a fixed offset derived from the data segment (`t0 = 1 / (2·domain.delta_f)`,
i.e. `T/2`, the segment **midpoint** where dingo centers the coalescence), via
`wf *= exp(2πi·f·t0)`, then the time axis has `t0` subtracted. `t0` is a private
detail — not a method argument. This generalizes the source notebook's hardcoded
`CENTRAL_TIME = 8` s (which was `T/2` for those `T = 16` s segments). Verified
empirically: for the GW150914 CI result (`T = 8` s, `t0 = 4`) the whitened-strain
peak lands at t ≈ +0.02 s in both H1 and L1.

## Time domain (`<event>_ppd_td.png`, one row per detector)

- Each draw: phase-shift by `t0`, `one_sided_fd_to_td`, keep real part.
- Band = pointwise **min/max envelope** across the selected draws
  (`td.min` / `td.max`), drawn with `fill_between`.
- **Nested credible levels:** `credible_interval` accepts a list (e.g. `[0.5, 0.9]`).
  Since the HPD set at level L is `{θ : density ≥ c_L}`, a tighter level is just a
  higher density threshold on the same ranked draws — so `_compute_ppd` returns the
  draws ordered by descending density plus a per-level count `k`, and the plot shades
  `td[:k].min/max` per level (tighter = darker).
- Overlay: whitened data trace (`√(4Δf)·context_waveform/asd`, same shift +
  IFFT, light sliding-window average), grey.
- x-axis: time to merger (s), linear, zoomed via `zoom=` (method default
  `(-1.0, 0.2)`; `(-0.7, 0.1)` works well for GW230709). y-axis: whitened strain
  (dimensionless, σ units).

### Alternative considered: shade by density p(s | t) (rejected)

We prototyped replacing the min/max band with the full per-time strain **density**
`p(s | t)` — the pushforward of the posterior through the waveform map, rendered as
an opacity/heatmap (mass-proportional draws, per-column-normalised + gamma, KDE
smoothing). Mathematically it is the "right" object (min/max is just its support),
and it looks great on the loud, well-localised GW150914. **But on the real
eccentric O4 events (e.g. S230709bi) it read as messy** — near merger the draws
decorrelate in phase, so the density fills a broad lens with fine internal banding
that is hard to parse, and finite-sample shot noise needs aggressive smoothing.
The min/max envelope + MAP line conveys the same "where does the signal live" story
far more legibly, so we kept it. (The density prototype was discarded, not merged; the
approach is recorded here in case it's worth revisiting with a cleaner rendering.)

Frequency-domain PPD plotting is deferred to a follow-up (a per-bin whitened FD
view buries the signal below the unit-variance noise floor; the useful FD view is
an amplitude-spectral-density / log-log plot, out of scope here).

## Reused logic

`one_sided_fd_to_td(fd, domain)` is reused verbatim from the notebook (a
module-level function in `dingo/gw/utils/plotting.py`): it builds the full Hermitian
two-sided spectrum (DC zeroed), `np.fft.ifft(...) * sqrt(N)` (normalization tied
to the whitening so noise has unit variance), `dt = 1/(2·f_max)`, output length
`2n-1`. Correct and whitening-consistent.

## Testing / determinism

- Seeded `np.random.default_rng` for resampling → deterministic figures.
- `tests/gw/test_ppd.py`: assert `one_sided_fd_to_td` (imported from
  `dingo.gw.utils.plotting`) round-trips — a one-sided spectrum with a single populated
  frequency bin returns a real cosine of the expected period (exercises the
  Hermitian-mirror + normalization logic, the one piece of non-trivial new math).

## Boundaries / conventions

- Rendering in `gw/utils/plotting.py`, thin `Result.plot_ppd_td` + `_compute_ppd` in
  `gw/result.py`; imports `gw/` + `core/` (allowed direction). No new classes,
  no new dependencies.
- Whitening conventions for data (`√(4Δf)/asd`) and model (likelihood's
  `/asd/noise_std`, `noise_std = 1/√(4Δf)`) are identical and must be preserved.
