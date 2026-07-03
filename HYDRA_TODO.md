# Hydra Migration TODO

This document tracks a staged migration of Dingo from ad hoc YAML files,
`argparse` / `configargparse`, and hand-written config builders toward Hydra.

## Migration Philosophy

Do this in stages rather than trying to refactor the whole configuration model in
one pass.

1. **Stage 1: Hydra wrappers with old internal dictionaries**
   - Replace public CLI parsers with `@hydra.main(...)` entrypoints.
   - Replace old example/settings YAML files with Hydra configs that are mostly
     one-to-one copies of the old settings.
   - Keep these Stage 1 configs together in one examples-like folder under
     `configs/`, mirroring the current `examples/` organization as much as
     possible.
   - Convert `DictConfig` to plain dictionaries internally with
     `OmegaConf.to_container(cfg, resolve=True)`.
   - Let the existing implementation consume those dictionaries as before.
   - Preserve settings that are saved into datasets, models, and results.
   - Replace command-line `print(...)` status output with module loggers:
     `log = logging.getLogger(__name__)` and `log.info(...)`.
   - Remove the old argparse/configargparse command-line arguments for migrated
     commands, as was done for `dingo_generate_dataset`.

2. **Stage 2: Organize configs with Hydra groups and defaults**
   - Split the Stage 1 examples-like config folder into reusable groups.
   - Deduplicate examples and common settings.
   - Keep code mostly dictionary-based.

3. **Stage 3: Refactor internals to use Hydra-native construction**
   - Replace builder functions and string switches with `_target_` and
     `hydra.utils.instantiate(...)`.
   - Make transform/model/domain/prior construction more composable.
   - Move defaults that are currently hard-coded in Python into default Hydra
     configs/config groups.
   - Make configs complete: options that are currently omitted from example
     YAMLs because code supplies implicit defaults should become explicit
     defaults in Hydra configs.
   - Revisit saved metadata format and backward compatibility deliberately.

## Stage 1 Scope

Hydra Stage 1 covers scripts that consume substantial YAML/settings configs or
launch run-like workflows where a run directory, saved config, and Hydra
overrides are useful.

Tiny inspection, conversion, compatibility, debug/demo, and maintenance scripts
with only a few command-line arguments are intentionally allowed to stay as
argparse scripts. They should still use logging for user-facing progress where
we touch them, but they do not need Hydra configs.

## Stage 1 Definition Of Done

For each migrated CLI:

- [ ] Public entrypoint is wrapped by `@hydra.main(...)`.
- [ ] Old `argparse`, `configargparse`, or `BilbyArgParser` parsing is removed
      from that entrypoint.
- [ ] Config lives in one examples-like Stage 1 folder under `configs/`.
- [ ] Existing settings YAMLs/examples have equivalent Hydra configs.
- [ ] `.ini` files are explicitly out of scope for this migration pass.
- [ ] The Hydra config is converted to a plain dict before entering most
      existing business logic.
- [ ] Settings saved into HDF5/PT/result metadata remain loadable.
- [ ] User-facing progress messages use `logging` instead of `print`.
- [ ] Any `print(...)` statements reachable from the migrated CLI are replaced
      by logging calls, including prints in helper modules called by that CLI.
- [ ] Python warnings emitted by migrated CLIs are routed through logging, e.g.
      with `logging.captureWarnings(True)`, so they appear in both stdout and
      the Hydra log consistently.
- [ ] Smoke tests compare captured stdout with the Hydra log file; they should
      match exactly, confirming all progress output went through logging.
- [ ] A small smoke test exists or is documented.

Recommended shared pattern:

```python
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)
logging.captureWarnings(True)

@hydra.main(version_base="1.3", config_path="...", config_name="...")
def main(cfg: DictConfig) -> None:
    settings = OmegaConf.to_container(cfg, resolve=True)
    ...
```

Recommended default Hydra run-directory policy:

```yaml
hydra:
  run:
    dir: ${oc.env:DINGO_RUNS}/${now:%Y-%m-%d_%H-%M-%S}_${hydra.job.name}_${hydra.job.override_dirname}
  job:
    name: <job_name>
    chdir: true
```

Use `hydra.job.chdir: true` in Stage 1 so all relative outputs from a run land
in the Hydra run directory together with `generate_dataset.log` and `.hydra/`
metadata. The run root is controlled by the required `DINGO_RUNS` environment
variable.

When running local Stage 1 smoke tests, set `DINGO_RUNS` to the repo-local
`my_runs/` directory so the generated run directories are easy to inspect:

```bash
DINGO_RUNS=/home/hcampe/dingo/my_runs
```

For migrated CLIs, capture stdout during smoke tests and compare it against the
Hydra log file in the corresponding run directory. For example:

```bash
diff -u captured_stdout.txt my_runs/<run_dir>/<job_name>.log
```

If a user explicitly wants old current-working-directory behavior, use both:

```bash
hydra.job.chdir=false hydra.run.dir=.
```

Use `compression: null` rather than `compression: None`.

Do not migrate `.ini` files for now. In particular, leave Asimov/plugin-style
`.ini` configuration outside the Hydra migration TODO unless this is reopened
explicitly later.

## Known Follow-Ups

- [ ] Decide on the long-term config package layout. For Stage 1,
      `configs/` is made importable and included in setuptools package
      discovery so console scripts can use ordinary relative
      `@hydra.main(config_path=...)` values.
      In Stage 2/3, decide whether to keep this top-level `configs` package,
      move configs under `dingo/`, or introduce a shared config-search-path
      helper/plugin.

## Stage 1 Status

Stage 1 is complete for the scoped set of settings/config-driven workflows.
Remaining parser-based scripts are explicitly excluded below because they are
small utilities, compatibility helpers, debug/demo mains, `.ini`-based pipe
commands, or Condor/DAG submission paths deferred to Stage 3.

## Stage 2 Status

Stage 2 has started. The Stage 1 `configs/examples/` folder was renamed to
`configs/old_examples/`, and migrated command configs now live directly under
`configs/`.

Current group structure:

- `domain`
- `waveform_generator`
- `intrinsic_prior`
- `extrinsic_prior`
- `model`
- `optimizer`
- `scheduler`
- `hydra`
- `experiment`

The group configs contain default settings, including several defaults that were
previously implicit in examples or Python helper code. The `experiment` group
contains overrides for documented/example workflows such as toy NPE, NPE, GNPE,
GNPE initialization, FMPE, ASD variants, synthetic ASD generation, and
importance sampling.

The single `domain/uniform_frequency.yaml` entry is now the high-resolution
uniform-frequency variant (`delta_f: 0.125`). Workflows that intentionally use
the old coarser domain, such as the toy waveform dataset example, override
`domain.delta_f` downstream.

Posterior-model and embedding-model settings have been unified into a single
`model` group, with one file per model family. The toy-sized normalizing-flow
configuration from the documentation is named `model/toy_npe.yaml`; the
production NPE settings are named `model/npe.yaml`; the unconditional density
estimator used by density recovery/importance weighting is named
`model/unconditional_npe.yaml`.

Stage 2 audit notes:

- The old examples and documentation are now represented by experiment configs
  for toy NPE, production NPE, GNPE, GNPE initialization, FMPE, ASD variants,
  synthetic ASD generation, and importance sampling.
- The examples' explicit unconditional NDE settings are represented by
  `model/unconditional_npe.yaml` and used by `importance_weights.yaml` and
  `unconditional_density_estimation.yaml`.
- Training-stage `early_stopping: null` is explicit in the base stage and the
  documented experiment stages.
- Condor resource blocks from production training examples are kept only as
  settings data where they already existed in examples. Hydra-aware Condor
  submission remains Stage 3 work.

## Spotted Bugs

- [ ] `dingo_build_svd`: the old `num_train is None` branch used
      `len(WaveformDataset)`, i.e. the class object, which raises
      `TypeError: object of type 'type' has no len()`. This was changed to
      `len(dataset)`, the loaded instance length. Mention this deliberate bug
      fix in the PR even though the CLI itself is no longer Hydra-migrated.

## Entrypoint Inventory

Public console scripts are defined in `pyproject.toml`.

### Dataset Generation

- [x] `dingo_generate_dataset`
  - File: `dingo/gw/dataset/generate_dataset.py`
  - Stage 1 has started.
  - Uses Hydra and `instantiate(...)` for some objects already, so it is partly
    beyond a purely superficial Stage 1 pass.
  - Remaining concern: compression/SVD logic is still procedural and should be
    revisited in Stage 3.

- [ ] `dingo_generate_dataset_dag`
  - File: `dingo/gw/dataset/generate_dataset_dag.py`
  - Defer Condor/DAG integration to Stage 3.
  - Stage 3 goal: submit Condor jobs through a Hydra-aware launcher/workflow
    rather than hand-written temporary YAML chunks and manual submit files.
  - This may be analogous in spirit to existing Hydra Slurm launcher workflows,
    but Condor will likely need project-specific integration.
  - Estimated Stage 1 difficulty: out of scope.
  - Estimated Stage 3 difficulty: high.

- [x] `dingo_evaluate_multibanded_domain`
  - File: `dingo/gw/dataset/evaluate_multibanded_domain.py`
  - Stage 1 implemented: `main` is wrapped with Hydra and uses
    `configs/examples/evaluate_multibanded_domain.yaml`.
  - This remains in scope because it has a structured domain, waveform
    generator, prior, compression, and sample-count config.
  - Stage 3: replace `build_domain`, `build_prior_with_defaults`, and manual
    waveform-generator branching with Hydra targets.
  - Estimated Stage 1 difficulty: low.
  - Estimated Stage 3 difficulty: medium.

- [x] `dingo_build_svd`
  - File: `dingo/gw/dataset/utils.py`
  - Explicit Stage 1 exception: keep as a small argparse utility because it has
    only a few CLI arguments and does not consume a full settings config.
  - Still uses logging for user-facing progress.
  - Keeps the deliberate bug fix in the old `num_train is None` path: use
    `len(dataset)` rather than `len(WaveformDataset)`.
  - Stage 3: likely becomes part of a Hydra-instantiated compression/SVD
    workflow.
  - Estimated Stage 1 difficulty: low.
  - Estimated Stage 3 difficulty: medium.

- [x] `dingo_merge_datasets`
  - File: `dingo/gw/dataset/utils.py`
  - Explicit Stage 1 exception: keep as a small argparse utility. It has only a
    few CLI arguments; the optional `settings_file` is a metadata override, not
    a run-defining config.
  - Uses logging for user-facing progress.
  - Stage 3: probably little to do.
  - Estimated Stage 1 difficulty: low.
  - Estimated Stage 3 difficulty: low.

### Noise / ASD Generation

- [x] `dingo_generate_asd_dataset`
  - File: `dingo/gw/noise/generate_dataset.py`
  - Stage 1 implemented for local generation: `generate_dataset` is wrapped
    directly with Hydra and uses `configs/examples/generate_asd_dataset.yaml`.
  - Parser/settings-file handling was replaced by a shallow Hydra config that
    preserves the old settings dict shape internally.
  - Condor/DAG behavior is explicitly deferred to Stage 3 under Hydra.
  - Uses `logging.captureWarnings(True)` and logs user-facing progress instead
    of printing.
  - Smoke test output is in
    `my_runs/2026-07-01_14-43-30_generate_asd_dataset_dataset_settings.T=1.0,dataset_settings.detectors=[H1,L1],dataset_settings.f_s=16,dataset_settings.observing_run=O1,dataset_settings.time_psd=4`.
  - Estimated Stage 1 difficulty: medium.
  - Estimated Stage 3 difficulty: medium/high for Condor integration.

- [x] `dingo_estimate_psds`
  - File: `dingo/gw/noise/asd_estimation.py`
  - Stage 1 implemented: `download_and_estimate_cli` is wrapped directly with
    Hydra and uses `configs/examples/estimate_psds.yaml`.
  - The existing `download_and_estimate_psds(...)` dict-based implementation is
    unchanged.
  - Uses `logging.captureWarnings(True)`.
  - Smoke test output is in
    `my_runs/2026-07-01_14-55-15_estimate_psds_dataset_settings.T=1.0,dataset_settings.detectors=[H1,L1],dataset_settings.f_s=16,dataset_settings.observing_run=O1,dataset_settings.time_psd=4`.
  - Stage 3: domain construction and PSD-estimation settings could become more
    declarative.
  - Estimated Stage 1 difficulty: low/medium.
  - Estimated Stage 3 difficulty: medium.

- [x] `dingo_generate_synthetic_asd_dataset`
  - File: `dingo/gw/noise/synthetic/generate_dataset.py`
  - Stage 1 implemented: `main` is wrapped with Hydra and uses
    `configs/examples/generate_synthetic_asd_dataset.yaml`.
  - The existing `generate_dataset(...)` implementation remains dict-based.
  - Uses `logging.captureWarnings(True)`.
  - Smoke test output is in
    `my_runs/2026-07-01_14-44-49_generate_synthetic_asd_dataset_parameterization_settings.num_spectral_segments=2,parameterization_settings.num_spline_positions=3,sampling_settings=null`.
  - Stage 3: parameterization and sampling could become separate config groups.
  - Estimated Stage 1 difficulty: low.
  - Estimated Stage 3 difficulty: medium.

- [x] `dingo_merge_asd_datasets`
  - File: `dingo/gw/noise/utils.py`
  - Stage 1 implemented: `merge_datasets_cli` is wrapped directly with Hydra
    and uses `configs/examples/merge_asd_datasets.yaml`.
  - Uses `logging.captureWarnings(True)` and logs user-facing progress instead
    of printing.
  - Smoke test output is in
    `my_runs/2026-07-01_14-43-31_merge_asd_datasets_dataset_settings.T=1.0,dataset_settings.detectors=[H1,L1],dataset_settings.f_s=16,dataset_settings.observing_run=O1,dataset_settings.time_psd=4`.
  - Estimated Stage 1 difficulty: low.
  - Estimated Stage 3 difficulty: low.

### Training

- [x] `dingo_train`
  - File: `dingo/gw/training/train_pipeline.py`
  - Stage 1 implemented: `train_local` is wrapped directly with Hydra and uses
    `configs/examples/train.yaml`.
  - New training uses the Hydra config body as the old `train_settings` dict;
    checkpoint resume uses `checkpoint=<path>`.
  - `local` settings are still popped out and saved separately as
    `local_settings.yaml`.
  - Input dataset paths are resolved before training so `hydra.job.chdir=true`
    does not break file loading; these resolved paths are saved in model
    metadata.
  - Training progress reachable from this CLI now uses logging instead of
    printing, including `train_builders`, `BasePosteriorModel.train`, and
    `trainutils`.
  - Training omits `hydra.job.override_dirname` from its run directory template
    because real training overrides are commonly too large for filesystem-safe
    paths.
  - Smoke test output is in `my_runs/2026-07-01_15-26-09_train`.
  - Estimated Stage 1 difficulty: medium/high.
  - Estimated Stage 3 difficulty: high.

- [ ] `dingo_train_condor`
  - File: `dingo/gw/training/train_pipeline_condor.py`
  - Defer to Stage 3 with the rest of Condor integration.
  - Goal: a Hydra-aware Condor submission path, ideally with a clean
    single-command link from config composition to job submission.
  - Watch out:
    - Reads/writes `local_settings.yaml`.
    - Creates Condor submit files and mutates local settings.
  - Estimated Stage 1 difficulty: out of scope.
  - Estimated Stage 3 difficulty: high.

- [x] `dingo_append_training_stage`
  - File: `dingo/gw/training/utils.py`
  - Stage 1 implemented: `append_stage` is wrapped directly with Hydra and
    uses `configs/examples/append_training_stage.yaml`.
  - The new stage is provided directly in the Hydra config under `stage`.
  - Modifies model metadata/training-stage settings as before.
  - Smoke test output is in
    `my_runs/2026-07-01_15-26-55_append_training_stage_stage.batch_size=1,stage.epochs=2`.
  - Estimated Stage 1 difficulty: low.
  - Estimated Stage 3 difficulty: low/medium.

### Pipe / Inference

Explicitly out of scope for this Hydra migration for now. Keep the
`dingo_pipe*` family, `dingo_result`, and their `.ini` / bilby_pipe-style
configuration conventions as they are unless this decision is reopened later.

### Other Utilities

- [x] `dingo_pt_to_hdf5`
  - File: `dingo/core/utils/pt_to_hdf5.py`
  - Explicit Stage 1 exception: keep as a small argparse conversion utility
    because it has only `in_file`, `out_file`, and `model_version_number`.
  - Uses `logging.captureWarnings(True)` and stream-only logging instead of
    printing, without Hydra and without saving a log file.
  - Estimated Stage 1 difficulty: low.
  - Estimated Stage 3 difficulty: low.

- [x] `dingo_ls`
  - File: `dingo/gw/ls_cli.py`
  - Explicit Stage 1 exception: keep this as a small argparse inspection script
    because it has only one positional argument and is not a config-driven
    workflow.
  - Uses `logging.captureWarnings(True)` and stream-only logging instead of
    printing, without Hydra and without saving a log file. The stream formatter
    matches the standard Hydra log format.
  - `Result.print_summary()` was updated to use logging because it is reachable
    from this CLI.
  - Smoke test output was captured in
    `/tmp/dingo_ls_stdout_standard_format.out`.
  - Estimated Stage 1 difficulty: low.
  - Estimated Stage 3 difficulty: low.

- [x] Importance-weight generation
  - File: `dingo/gw/importance_sampling/importance_weights.py`
  - Stage 1 implemented: `main` is wrapped with Hydra and uses
    `configs/examples/importance_weights.yaml`.
  - The old `--settings` / YAML-loading path was removed; the Hydra config body
    is converted to a plain settings dict internally.
  - Input parameter-sample and calibration-envelope paths are resolved before
    use so `hydra.job.chdir=true` does not break file loading.
  - User-facing progress reachable from this CLI now uses logging instead of
    printing, including sampler progress, result summaries, and diagnostics.
  - Config-composition smoke was captured in
    `/tmp/dingo_importance_weights_cfg.out`. A full run still needs a
    physically meaningful GW result and likelihood setup.
  - Estimated Stage 1 difficulty: low/medium.
  - Estimated Stage 3 difficulty: medium.

- [x] Unconditional density estimation
  - File: `dingo/core/density/unconditional_density_estimation.py`
  - Stage 1 implemented: the stale `--settings` parser was replaced by a Hydra
    entry point using `configs/examples/unconditional_density_estimation.yaml`.
  - The Hydra config takes `result_file`, `train_dir`, and the old estimator
    settings; internally the estimator still receives a plain settings dict.
  - Smoke test output is in
    `my_runs/unconditional_density_estimation_stage1_smoke`.
  - Estimated Stage 1 difficulty: low.
  - Estimated Stage 3 difficulty: medium.

### Explicitly Not Ported To Hydra

These are intentionally excluded from Stage 1. Mention them in the PR so it is
clear this is a scoped migration rather than an incomplete sweep.

- Condor / DAG submission:
  - `dingo_generate_dataset_dag`
    (`dingo/gw/dataset/generate_dataset_dag.py`)
  - `dingo_train_condor`
    (`dingo/gw/training/train_pipeline_condor.py`)
  - Noise-generation DAG helper
    (`dingo/gw/noise/generate_dataset_dag.py`)
  - `dingo/core/utils/condor_utils.py`
  - Rationale: Condor integration belongs in Stage 3, ideally as a
    Hydra-aware submission path rather than shallow argparse replacement.
- Pipe / `.ini` workflows:
  - `dingo_pipe`
  - `dingo_pipe_generation`
  - `dingo_pipe_sampling`
  - `dingo_pipe_importance_sampling`
  - `dingo_pipe_plot`
  - `dingo_pipe_pp_test`
  - `dingo_result`
  - Rationale: keep bilby_pipe-style `.ini` conventions for now.
- Small public utilities kept as argparse:
  - `dingo_ls`
  - `dingo_pt_to_hdf5`
  - `dingo_build_svd`
  - `dingo_merge_datasets`
  - Rationale: only a few command-line arguments and no full run-defining
    settings config. These may still use logging for progress output.
- Compatibility scripts:
  - `compatibility/*.py`
  - Rationale: one-off saved-artifact migration helpers.
- Miscellaneous maintenance scripts:
  - `misc_scripts/*.py`
  - Rationale: ad hoc plotting, monitoring, evaluation, and maintenance tools.
    Some take a settings file, but they are not public run-launching workflows.
- Debug/demo module mains:
  - `dingo/gw/likelihood.py`
  - `dingo/gw/waveform_generator/waveform_generator.py`
  - `dingo/core/nn/enets.py`
  - `dingo/core/nn/nsf.py`
  - Rationale: local debug/demo entry points, not user-facing config workflows.

## Stage 2 Config Group Candidates

- `domain`
- `waveform_generator`
- `intrinsic_prior`
- `extrinsic_prior`
- `inference_parameters`
- `dataset`
- `asd_dataset`
- `noise`
- `model`
- `training/stage`
- `training/local`
- `condor` (Stage 3 only)
- `importance_sampling`
- `defaults`
- `defaults/prior`
- `defaults/recovery`
- `defaults/nde`

## Stage 3 Refactor Hotspots

- [ ] Move Python-defined defaults into Hydra default configs
  - Goal: make default choices visible, overridable, and composable from the
    config tree rather than hidden in module globals or helper functions.
  - Related goal: make configs complete. If an option affects behavior, it
    should appear in a default config somewhere, even if most examples inherit
    it unchanged.
  - Likely config groups:
    - `defaults/prior/intrinsic`
    - `defaults/prior/extrinsic`
    - `defaults/inference_parameters`
    - `defaults/density_recovery`
    - `defaults/importance_sampling`
    - `defaults/nde`
    - `defaults/training`
  - Keep a deliberate backward-compatibility story for old saved metadata that
    uses strings such as `default`.

- [ ] Audit implicit constructor/function defaults and expose them in config
  - Example configs currently omit many options because Python functions and
    constructors provide defaults.
  - Stage 3 should decide which defaults are part of the public configuration
    surface and place them in default configs.
  - This is different from Stage 1, where copied configs may remain incomplete
    as long as old code handles them.
  - Candidate areas:
    - waveform generator optional kwargs such as `f_start`, `mode_list`,
      `spin_conversion_phase`, `new_interface`, `lmax_nyquist`, and
      interface-specific options
    - dataset compression/SVD/whitening options
    - training local/runtime/checkpoint options
    - optimizer and scheduler defaults
    - transform defaults in the training data pipeline, including
      `zero_noise: false`, optional `random_strain_cropping`, derived
      `context_parameters`, and derived `standardization`
    - pipe/inference defaults currently supplied by parser defaults
    - ASD/noise generation defaults such as frequency range, channels, and
      Condor settings
    - dataloader defaults such as fixed split seed, `pin_memory`, worker seed
      initialization, and `persistent_workers`
    - early stopping defaults (`patience`, `verbose`, `delta`, `metric`)
    - runtime limit defaults (`max_time_per_run`, `max_epochs_per_run`,
      `max_epochs_total`, `epoch_start`)

- [ ] `dingo/core/density/nde_settings.py`
  - Move or retire `get_default_nde_settings_3d(...)` and
    `DEFAULT_NDE_SETTINGS_2D`.
  - These currently define additional unconditional NDE architectures:
    - 3D: 5 flow steps, hidden dimension 128, batch size 4096, 10 epochs,
      Adam learning rate 0.005, cosine `T_max: 10`.
    - 2D GNPE proxy recovery: 5 flow steps, hidden dimension 64, batch size
      4096, 10 epochs, Adam learning rate 0.001, cosine `T_max: 10`.
  - Decide whether these become separate `model/*` and training config groups,
    or whether they are superseded by `model/unconditional_npe.yaml`.

- [ ] `dingo/gw/domains/build_domain.py`
  - Replace `type`-based dispatch with Hydra instantiation.
  - Keep metadata/backward compatibility in mind.

- [ ] `dingo/gw/prior.py`
  - Replace `default` string handling and prior builders with explicit config
   targets or reusable config groups.
  - Current code-defined defaults:
    - `default_intrinsic_dict`
    - `default_extrinsic_dict`
    - `default_inference_parameters`

- [ ] Waveform generator construction
  - Replace `new_interface` flags and manual class branching with `_target_`.

- [ ] Compression/SVD workflow
  - Harder than plain instantiation because SVD may be loaded from file or
    trained from generated waveforms.
  - Possible long-term design: compression is an ordered transform list, with
    SVD handled by a project-specific factory target.

- [ ] `dingo/gw/training/train_builders.py`
  - Large procedural transform pipeline.
  - Prime candidate for Hydra targets, but many transforms need runtime objects:
    domain, detectors, ASD datasets, standardization, priors, reference time.

- [ ] `dingo/core/posterior_models/build_model.py`
  - Replace `posterior_model_type` dispatch with `_target_`, while preserving
    saved-model compatibility.

- [ ] `dingo/core/utils/torchutils.py`
  - Replace optimizer/scheduler `type` switches with `_target_` configs.

- [ ] Condor integration as a Hydra workflow/launcher
  - Keep Condor-specific migration out of Stage 1.
  - Stage 3 goal: compose the job config with Hydra and submit Condor jobs from
    that composed config.
  - Desired user experience: a single Hydra-driven command can configure and
    submit the Condor job.
  - Existing Hydra Slurm launchers may be useful as conceptual references, but
    HTCondor integration will likely require project-specific code.
  - Affected areas include:
    - `dingo/gw/dataset/generate_dataset_dag.py`
    - `dingo/gw/noise/generate_dataset.py`
    - `dingo/gw/training/train_pipeline_condor.py`

- [ ] `dingo/pipe/default_settings.py`
  - Move `DENSITY_RECOVERY_SETTINGS` and `IMPORTANCE_SAMPLING_SETTINGS` to
    Hydra default configs.
  - Current named defaults:
    - `ProxyRecoveryDefault`
    - `PhaseRecoveryDefault`
    - `MultibandingDefault`
  - These are intentionally deferred while `.ini`/pipe workflows remain outside
    the Stage 1/2 Hydra migration.

- [ ] `dingo/core/density/nde_settings.py`
  - Move `get_default_nde_settings_3d(...)` and `DEFAULT_NDE_SETTINGS_2D` into
    Hydra configs.
  - These can likely become reusable `defaults/nde` entries plus overrides for
    dimension/device/parameters.

- [ ] `dingo/pipe/parser.py`
  - Biggest migration knot. This is both a CLI parser and a compatibility layer
    with bilby_pipe-style settings.

## Suggested Order

1. Finish Stage 1 for standalone dataset/noise utilities.
2. Do Stage 1 for training.
3. Decide whether `dingo_pipe` should be a compatibility wrapper around Hydra
   configs or a full Hydra CLI.
4. Do Stage 2 grouping for dataset generation, ASD generation, and training
   before touching pipe internals.
5. Start Stage 3 with the small builder switches:
   domain, prior, waveform generator, optimizer, scheduler.
6. Design Condor integration as a Stage 3 Hydra workflow/launcher problem.
7. Tackle training transforms and model construction.
8. Leave pipe/parser migration until the rest of the config story is stable.
