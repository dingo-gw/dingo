# Waveform generator API migration

This page shows how the same tasks look under the **new** waveform-generator
API — the primary interface shipped in dingo-gw — versus the **legacy**
dict-based API, which is still importable as a thin, `@deprecated` wrapper
over the new interface. If you are updating scripts or notebooks written
against the old API, use this as a translation guide.

## What changed and why

Historically, dingo-gw shipped two classes side by side:

- `WaveformGenerator` — dict-based, LALSimulation backend, all approximants
  routed through one constructor.
- `NewInterfaceWaveformGenerator` — subclass switching to the newer
  `lalsimulation.gwsignal` backend, opted into via a `new_interface: True`
  flag in YAML settings.

Both accepted plain `dict` parameters (both for the WFG kwargs and for the
per-waveform theta), returned polarizations as `{"h_plus": array,
"h_cross": array}` dicts, and required callers to pick the right subclass by
name.

These historic names still resolve — they are now deprecated wrappers under
`dingo.gw.waveform_generator.legacy` — but every call emits a
`DeprecationWarning`. Use the natural API described below in new code.

The new API replaces this with:

- **One factory**, `build_waveform_generator`, that dispatches by approximant
  name to the right subclass (`LALSimWaveformGenerator`,
  `SEOBNRv4PHMWaveformGenerator`, `IMRPhenomXPHMWaveformGenerator`,
  `GWSignalWaveformGenerator`, `RandomWaveformGenerator`). No more
  `new_interface` flag.
- **Typed configuration** via `WaveformGeneratorParameters` (WFG-level
  config: approximant, f_ref, f_start, spin_conversion_phase, mode_list,
  transform, extra_kwargs).
- **Typed per-waveform inputs** via `BBHWaveformParameters` (mass_1,
  mass_2, spins, phase, theta_jn, luminosity_distance, …).
- **Typed outputs** via `Polarization` (`.h_plus`, `.h_cross`) and
  `BatchPolarizations` for batched arrays.
- **Extra gwsignal kwargs** (`lmax_nyquist`, `postadiabatic`,
  `enable_antisymmetric_modes`, …) go into a dedicated `extra_kwargs` dict
  instead of being mixed with core config.

The migration was purely internal: the LAL/GWSignal calls, the physics, and
the produced waveforms are unchanged. Only the Python surface differs.

## Constructing a waveform generator

**Legacy:**

```python
from dingo.gw.domains import build_domain
from dingo.gw.waveform_generator import (
    WaveformGenerator,
    NewInterfaceWaveformGenerator,
)

domain = build_domain(
    {"type": "UniformFrequencyDomain", "f_min": 20.0, "f_max": 1024.0, "delta_f": 0.125}
)

# LAL-based approximant
wfg = WaveformGenerator(
    approximant="IMRPhenomXPHM",
    domain=domain,
    f_ref=20.0,
    spin_conversion_phase=0.0,
)

# gwsignal-based approximant (had to pick the right class manually)
wfg_v5 = NewInterfaceWaveformGenerator(
    approximant="SEOBNRv5PHM",
    domain=domain,
    f_ref=20.0,
    lmax_nyquist=3,
    postadiabatic=True,
)
```

**New:**

```python
from dingo.gw.domains import build_domain
from dingo.gw.waveform_generator import build_waveform_generator

domain = build_domain(
    {"type": "UniformFrequencyDomain", "f_min": 20.0, "f_max": 1024.0, "delta_f": 0.125}
)

# Factory dispatches by approximant. No new_interface flag.
wfg = build_waveform_generator(
    {"approximant": "IMRPhenomXPHM", "f_ref": 20.0, "spin_conversion_phase": 0.0},
    domain,
)

# gwsignal-only options go under extra_kwargs.
wfg_v5 = build_waveform_generator(
    {
        "approximant": "SEOBNRv5PHM",
        "f_ref": 20.0,
        "extra_kwargs": {"lmax_nyquist": 3, "postadiabatic": True},
    },
    domain,
)
```

`build_waveform_generator` also accepts a JSON/YAML/TOML path in place of the
dict, or a two-level `{"domain": ..., "waveform_generator": ...}` dict where
the domain is built for you.

## Generating polarizations

**Legacy:**

```python
theta = {
    "mass_1": 30.0, "mass_2": 25.0, "luminosity_distance": 100.0,
    "phase": 0.5, "theta_jn": 1.0,
    "a_1": 0.3, "a_2": 0.2, "tilt_1": 0.5, "tilt_2": 0.3,
    "phi_12": 1.0, "phi_jl": 0.3, "geocent_time": 0.0,
}

pol = wfg.generate_hplus_hcross(theta, catch_waveform_errors=True)
h_plus, h_cross = pol["h_plus"], pol["h_cross"]  # arrays
```

**New:**

```python
from dingo.gw.waveform_generator import BBHWaveformParameters

params = BBHWaveformParameters(
    mass_1=30.0, mass_2=25.0, luminosity_distance=100.0,
    phase=0.5, theta_jn=1.0,
    a_1=0.3, a_2=0.2, tilt_1=0.5, tilt_2=0.3,
    phi_12=1.0, phi_jl=0.3, geocent_time=0.0,
)

pol = wfg.generate_hplus_hcross(params, catch_waveform_errors=True)
h_plus, h_cross = pol.h_plus, pol.h_cross  # attributes on Polarization
```

`catch_waveform_errors=True` behaves the same way in both APIs: LAL
`"Internal function call failed: Input domain error"` is trapped, a warning
is emitted, and NaN-filled arrays are returned instead of raising.

## Mode-separated generation

**Legacy:**

```python
pol_m = wfg.generate_hplus_hcross_m(theta)
# pol_m: {m_int: {"h_plus": array, "h_cross": array}}
for m, per_pol in pol_m.items():
    print(m, per_pol["h_plus"].shape)
```

**New:**

```python
pol_m = wfg.generate_hplus_hcross_m(params)
# pol_m: {Mode(m_int): Polarization}
for m, polarization in pol_m.items():
    print(m, polarization.h_plus.shape)
```

`sum_contributions_m` is still available for reconstructing the total
polarization with a phase shift; it now works on `Dict[Mode, Polarization]`
and returns a `Polarization`:

```python
from dingo.gw.waveform_generator import sum_contributions_m

pol_total = sum_contributions_m(pol_m, phase_shift=0.3)
```

## Batch / post-generation transforms

Both APIs expose a `.transform` slot for attaching per-waveform pipeline
transforms (e.g. whitening + SVD compression) that run after generation.
The slot works the same way; only the input/output types differ:

**Legacy:** `transform: dict -> dict`, called with `{"h_plus": ..., "h_cross": ...}`.

**New:** `transform: Polarization -> Polarization`.

```python
from dingo.gw.waveform_generator import Polarization

def double(pol: Polarization) -> Polarization:
    return Polarization(h_plus=pol.h_plus * 2.0, h_cross=pol.h_cross * 2.0)

wfg.transform = double
pol_scaled = wfg.generate_hplus_hcross(params)
```

Dataset generation attaches its whitening + SVD compression pipeline via this
slot automatically; see `dingo/gw/dataset/generate.py` for the
production example.

## Dataset generation

The CLI entry point is unchanged (`dingo_generate_dataset --settings_file
settings.yaml`); the module it invokes now lives in
`dingo.gw.dataset.cli`. In Python, use the new-API entry point directly:

**Legacy (removed):**

```python
from dingo.gw.dataset import generate_dataset  # gone
```

**New:**

```python
from dingo.gw.dataset import (
    DatasetSettings,
    generate_waveform_dataset,
    WaveformDataset,
)

settings = DatasetSettings.from_dict(yaml.safe_load(open("settings.yaml")))
dataset: WaveformDataset = generate_waveform_dataset(settings, num_processes=8)
dataset.save("waveform_dataset.hdf5")
```

`WaveformDataset` (the natural, dataclass-based container at
`dingo.gw.dataset.WaveformDataset`) stores `BatchPolarizations` for the
waveforms and a `pandas.DataFrame` for the parameters, plus the source
`DatasetSettings` for reproducibility.

The legacy `WaveformDataset` container is still importable from its fully
qualified path (`dingo.gw.dataset.waveform_dataset.WaveformDataset`) since
the training pipeline currently relies on it. It is expected to be phased
out as consumers migrate.

## YAML settings

The `new_interface: true` key that some legacy YAML configs used to select
the gwsignal backend has been removed — the factory dispatches based on
approximant name. Move any GWSignal-specific kwargs into `extra_kwargs`:

**Legacy:**

```yaml
waveform_generator:
  approximant: SEOBNRv5PHM
  f_ref: 20.0
  new_interface: true
  lmax_nyquist: 3
  postadiabatic: true
```

**New:**

```yaml
waveform_generator:
  approximant: SEOBNRv5PHM
  f_ref: 20.0
  extra_kwargs:
    lmax_nyquist: 3
    postadiabatic: true
```

## Injection

`dingo.gw.injection.GWSignal` / `Injection` keep the same constructor
signature (`wfg_kwargs`, `wfg_domain`, `data_domain`, `ifo_list`, `t_ref`)
and the same output shape from `.signal(theta)` and `.signal_m(theta)` —
still per-detector dicts of arrays. Internally they now build a new-API WFG
and convert `Polarization` back to `{"h_plus", "h_cross"}` at the boundary,
so downstream detector projection / whitening code was untouched.

Two small additions:

- If you previously reached for `injection.waveform_generator.approximant = "..."`
  or mutated `.domain`, that pattern no longer works — the new WFG is
  immutable. Instead:

  ```python
  injection.update_waveform_generator(approximant="SEOBNRv4PHM", f_ref=20.0)
  ```

- The legacy `new_interface: true` key in `wfg_kwargs` is silently dropped
  during construction; no config change is needed for existing model
  metadata, though it should be removed from new configs.

## Deprecated wrappers

The following legacy names still resolve, but each emits a
`DeprecationWarning` and delegates internally to the natural API. Prefer
the replacements in new code.

| Legacy import | Emits | Replacement |
| --- | --- | --- |
| `dingo.gw.waveform_generator.legacy.WaveformGenerator` | DeprecationWarning on `__init__` | `dingo.gw.waveform_generator.build_waveform_generator` + `BBHWaveformParameters` |
| `dingo.gw.waveform_generator.NewInterfaceWaveformGenerator` | DeprecationWarning on `__init__` | Same as above — the factory dispatches by approximant, no flag needed |
| `dingo.gw.waveform_generator.legacy.sum_contributions_m` | DeprecationWarning on call | `dingo.gw.waveform_generator.sum_contributions_m` (Polarization-typed) |
| `dingo.gw.waveform_generator.generate_waveforms_parallel` (legacy `pool=` signature) | DeprecationWarning on call | `dingo.gw.dataset.generate_waveforms_parallel(wfg, parameters, num_processes)` |
| `dingo.gw.dataset.generate_dataset.generate_dataset` | DeprecationWarning on call | `dingo.gw.dataset.generate_waveform_dataset(settings, num_processes)` |
| `dingo.gw.dataset.generate_dataset.generate_parameters_and_polarizations` | DeprecationWarning on call | `dingo.gw.dataset.generate_parameters_and_polarizations` (returns `BatchPolarizations`) |
| `dingo.gw.dataset.generate_dataset._generate_dataset_main` | DeprecationWarning on call | `dingo.gw.dataset.cli.generate_dataset_main` |
| `dingo.gw.prior.new_build_prior_with_defaults` | DeprecationWarning on call | `dingo.gw.prior.build_prior_with_defaults` (accepts both `IntrinsicPriors` and dict) |

To surface any lingering internal use of these paths, run the test suite
with warnings promoted to errors:

```bash
pytest -W error::DeprecationWarning tests/
```

## Reference

- Factory: `dingo.gw.waveform_generator.build_waveform_generator`
- Classes: `WaveformGenerator` (ABC),
  `LALSimWaveformGenerator`, `SEOBNRv4PHMWaveformGenerator`,
  `IMRPhenomXPHMWaveformGenerator`, `GWSignalWaveformGenerator`,
  `RandomWaveformGenerator`
- Types: `Polarization`, `BatchPolarizations`, `BBHWaveformParameters`,
  `WaveformGeneratorParameters`
- Dataset: `DatasetSettings`, `WaveformDataset`,
  `generate_waveform_dataset`
- CLI: `dingo_generate_dataset` (implemented in
  `dingo.gw.dataset.cli:main`)
