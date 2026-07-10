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
  - Stage 3 progress: compression is now represented as an ordered list of
    Hydra-instantiated transform configs. The train-from-waveforms SVD path is
    a Hydra target that returns an `SVDBasis`, while the produced SVD arrays are
    still saved as dataset artifact data.

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

- [ ] Stage 3 implementation philosophy
  - Prefer moving logic into Hydra configs through defaults, interpolation, and
    `_target_` entries.
  - Source-code changes should usually delete old construction/defaulting code
    or replace it with direct `hydra.utils.instantiate(...)` calls.
  - Avoid adding new source-code logic. If a new helper seems necessary, discuss
    it first, except for the already-agreed generic
    `instantiate_with_runtime_dependencies(...)` helper.
  - Expect config files to grow and pure Python construction code to shrink.
  - Start implementation with easier direct-instantiation migrations to
    establish patterns before touching the transform/metadata-heavy pieces.

- [ ] Define the Stage 3 config-construction contract
  - Goal: replace old `from_config` / `build_*` / string-dispatch constructors
    with Hydra-native `_target_` configs and `hydra.utils.instantiate(...)`
    as the normal construction mechanism.
  - Target architecture: configs describe the object graph, code defines the
    objects/functions, and Hydra composes plus instantiates. Avoid config
    completion helpers, compatibility-shaped builders, and side-channel
    mutation of settings dictionaries.
  - Stage 3 implementation rule: only remove old config-construction code or
    replace it by `_target_` configs plus `hydra.utils.instantiate(...)`.
    Do not introduce new factories, adapters, compatibility builders, or
    config-normalization layers without first discussing the design and getting
    explicit agreement.
  - Approved helper exception: a small generic
    `instantiate_with_runtime_dependencies(...)` helper is acceptable for
    Hydra-instantiating transform lists whose entries sometimes need live
    runtime objects. This helper should remain generic and should not encode
    Dingo-specific config completion logic.
  - Runtime-dependent quantities should be handled by the instantiated objects
    or by explicit setup methods on those objects, not by patching config
    dictionaries before construction.
  - Avoid mixing two construction idioms for the same object family. Once a
    family is migrated, prefer `_target_` configs over parallel `type` strings
    plus builder switches.
  - Difficulty: medium/high because the easy cases are straightforward, but the
    boundary between declarative construction and normal object lifecycle must
    be chosen deliberately.

- [ ] Define the saved-metadata contract for Hydra configs
  - Current datasets, models, and results save settings dictionaries as
    metadata. Stage 3 must decide what gets saved once configs contain
    `_target_`, interpolations, defaults composition, and possibly instantiated
    Python objects.
  - Candidate policy:
    - Save the resolved, plain-container snapshot of the composed Hydra config
      as the authoritative run/model/dataset metadata.
    - Put as much as possible into the composed config before object
      construction starts: user choices, defaults, resolved paths, dimensions,
      standardization settings/statistics, and other values that can be known
      or computed ahead of time.
    - All paths saved in metadata should be absolute/resolved paths.
    - Save the exact config produced by Hydra, rather than translating it into
      the old `dataset_settings` / `train_settings` metadata shape.
    - Prefer loading saved config metadata back as an OmegaConf/DictConfig-style
      object so code can use dot access consistently instead of ordinary nested
      dictionary indexing.
    - For results, keep the original resolved model config separate from
      result-specific metadata. The model config records the proposal/model that
      produced the samples; event, inference, and importance-sampling metadata
      record what was done to/with those samples.
    - Compute parameter standardization before model construction and include
      the resulting standardization statistics in the resolved config.
    - Prefer Hydra/OmegaConf composition, interpolation, and resolvers for this
      ahead-of-time computation over code that mutates config dictionaries
      during construction.
    - For simple derived values, use OmegaConf interpolation, e.g. `${...}`, to
      reference values already present elsewhere in the config.
    - For heavier ahead-of-time computations, use `_target_` config blocks and
      `hydra.utils.instantiate(...)` on that config subtree. This is acceptable
      when the target is the actual computation/object being configured, not a
      new config-factory layer.
    - Do not save live instantiated objects; save importable targets,
      primitive parameters, paths, versions, and artifact provenance instead.
    - Keep produced arrays/tensors outside the config: model weights,
      optimizer/scheduler state, SVD basis arrays, result samples, context/event
      strain arrays, etc. These can remain stored in dedicated artifact/data
      fields broadly as they are today.
    - Keep the metadata structure close to the config structure so users can
      understand saved artifacts by looking at the configs.
    - Do not design the new metadata around legacy dictionary shapes unless a
      concrete loader requirement is reintroduced.
  - If old artifacts need support later, prefer a narrow explicit migration
    shim at the loading boundary rather than carrying legacy shape through the
    new config design.
  - Progress:
    - HDF5-backed `DingoDataset` settings are now saved as resolved YAML and
      loaded as OmegaConf/DictConfig metadata, with a fallback for old
      Python-literal attributes.
    - Posterior-model metadata is held as DictConfig in memory for dot access,
      but saved to checkpoints as resolved plain containers so `torch.load`
      remains compatible with safe loading.
    - Result-specific metadata remains separate from model metadata as before.
    - Training computes missing parameter standardizations before the
      configured training transforms are instantiated, writes the computed
      statistics back into the training settings, and syncs the resolved
      transform config that consumes them.
  - Difficulty: very high. This is a cross-cutting compatibility and
    reproducibility contract, even if we choose not to preserve old metadata
    formats.

- [ ] Replace manual config autocompletion/resolution with Hydra-era derived
      metadata handling
  - Current hotspot: `dingo/core/posterior_models/build_model.py` contains
    `autocomplete_model_kwargs(...)`, which mutates model settings in-place
    based on a runtime data sample.
  - It currently fills:
    - `embedding_kwargs.input_dims` from waveform/data shape
    - `posterior_kwargs.input_dim` from the parameter vector length
    - `embedding_kwargs.added_context` from whether GNPE proxy context exists
    - `posterior_kwargs.context_dim` from embedding output dimension plus
      optional GNPE proxy dimension
  - This is not plain static config composition: the values depend on the
    realized training dataset and transform pipeline.
  - Candidate policy:
    - Keep user-configurable architecture choices in Hydra configs.
    - Remove hidden in-place config mutation.
    - Move any dimensions or settings that can be known before construction into
      the composed Hydra config, using explicit config values or Hydra
      resolution mechanisms rather than `autocomplete_model_kwargs(...)`.
    - Only leave shape inference inside instantiated objects when the value
      genuinely cannot be known before the object is constructed.
    - Save the resolved config and any resulting learned/artifact state; do not
      save a separately patched config unless the object itself owns that state.
  - Related pipe-only helper: `dingo/pipe/main.py` has
    `write_complete_config_file(...)`, which writes a completed `.ini` file.
    Since `.ini` pipe workflows are out of scope for the current Hydra
    migration, keep this separate for now; if pipe migrates later, Hydra's
    composed config output should replace most of this behavior.
  - Difficulty: high. It sits at the intersection of model construction,
    transform realization, and saved metadata.
  - Design status: agreed. `autocomplete_model_kwargs(...)` should become
    superfluous. Ahead-of-time computable dimensions should live in the resolved
    Hydra config through interpolation or explicit ahead-of-time computation.

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
  - Replace old sentinel values such as `default` with explicit defaults-group
    choices where possible.

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
    - ahead-of-time computed model dimensions, replacing
      `autocomplete_model_kwargs(...)` where possible
    - pipe/inference defaults currently supplied by parser defaults
    - ASD/noise generation defaults such as frequency range, channels, and
      Condor settings
    - dataloader defaults such as fixed split seed, `pin_memory`, worker seed
      initialization, and `persistent_workers`
    - early stopping defaults (`patience`, `verbose`, `delta`, `metric`)
    - runtime limit defaults (`max_time_per_run`, `max_epochs_per_run`,
      `max_epochs_total`, `epoch_start`)

- [x] `dingo/core/density/nde_settings.py`
  - Move or retire `get_default_nde_settings_3d(...)` and
    `DEFAULT_NDE_SETTINGS_2D`.
  - These currently define additional unconditional NDE architectures:
    - 3D: 5 flow steps, hidden dimension 128, batch size 4096, 10 epochs,
      Adam learning rate 0.005, cosine `T_max: 10`.
    - 2D GNPE proxy recovery: 5 flow steps, hidden dimension 64, batch size
      4096, 10 epochs, Adam learning rate 0.001, cosine `T_max: 10`.
  - Retired in favor of Hydra configs, primarily `model/unconditional_npe.yaml`
    plus `unconditional_density_estimation.yaml`.

- [ ] `dingo/gw/domains/build_domain.py`
  - Replace `type`-based dispatch with Hydra instantiation.
  - New metadata should save the resolved Hydra domain config, not a parallel
    legacy domain dictionary.
  - Design status: agreed/easy. Use `_target_` domain configs directly.
    `build_domain(...)` and `build_domain_from_model_metadata(...)` should
    become unnecessary once downstream code reads the resolved Hydra config and
    calls `hydra.utils.instantiate(...)`.
  - Initial Stage 3 pass: default Hydra domain configs now use `_target_`, and
    `build_domain(...)` instantiates `_target_` configs while keeping the legacy
    `type` path for saved metadata and existing callers.
  - Progress: `dingo_generate_dataset` and
    `dingo_evaluate_multibanded_domain` now instantiate the Hydra domain config
    directly instead of calling `build_domain(...)`.

- [x] `dingo/gw/prior.py`
  - Replace `default` string handling and prior builders with explicit config
   targets or reusable config groups.
  - Current code-defined defaults:
    - `default_intrinsic_dict`
    - `default_extrinsic_dict`
    - `default_inference_parameters`
  - Agreed design:
    - Remove `default` sentinel strings from the Hydra-era configs.
    - The intrinsic prior config should target `BBHPriorDict` directly.
    - The `BBHPriorDict` config should contain explicit entries for the
      individual parameter priors/constraints.
    - Prefer making the prior dictionary target explicit over instantiating each
      Bilby prior object separately, unless a specific parameter requires a
      different treatment.
    - Retire `build_prior_with_defaults(...)` once all callers consume the
      instantiated/configured prior dictionary.
  - Progress: intrinsic-prior Hydra group configs now target
    `bilby.gw.prior.BBHPriorDict` directly, with explicit entries replacing
    `default` sentinels. Extrinsic-prior configs now target
    `dingo.gw.prior.BBHExtrinsicPriorDict` directly. Dataset generation,
    multibanded-domain evaluation, training, result, sampler, and injection
    paths consume these explicit configs. `build_prior_with_defaults(...)`,
    `get_extrinsic_prior_dict(...)`, and the Python-level default prior /
    inference-parameter globals were retired. Pipe remains out of scope.

- [ ] Waveform generator construction
  - Replace `new_interface` flags and manual class branching with `_target_`.
  - Design status: agreed/easy. Use `_target_` waveform-generator configs
    directly and remove manual branching once callers instantiate from config.
  - Progress: waveform-generator Hydra group configs now target
    `dingo.gw.waveform_generator.WaveformGenerator`, and the two dataset
    entrypoints instantiate them directly with the chosen domain.

- [ ] Compression/SVD workflow
  - Harder than plain instantiation because SVD may be loaded from file or
    trained from generated waveforms.
  - Possible long-term design: compression is an ordered Hydra-instantiated
    transform list. An SVD step can be configured as a normal object that knows
    whether to load an existing basis or train/build one from configured inputs.
  - Agreed design:
    - Represent compression as an ordered list of transform configs.
    - Use different Hydra targets for the different SVD construction paths,
      e.g. one target that loads an existing `SVDBasis` from file and another
      target that trains a new `SVDBasis`.
    - The train-SVD target should return only the `SVDBasis` object needed by
      `ApplySVD`. Auxiliary information such as actual train/validation counts
      or mismatch summaries belongs in metadata/provenance, not in the return
      value used by the transform.
    - The realized SVD basis should continue to be saved as a produced dataset
      artifact, embedded in the waveform dataset as today.
    - The config should save the requested SVD construction method and inputs,
      not the SVD arrays themselves.
  - Dataset metadata should save the resolved compression config/request. Any
    realized SVD arrays remain produced artifact data, saved in the waveform
    dataset as today.
  - Progress:
    - `train_svd_basis(...)` now returns only the `SVDBasis` used by
      `ApplySVD`.
    - Dataset-generation compression configs are now ordered lists of
      Hydra-instantiated transforms.
    - `train_svd_basis_from_waveforms(...)` is available as a Hydra target for
      training an `SVDBasis` from generated waveforms and passing it into
      `ApplySVD`.
    - The generated SVD arrays remain embedded in the waveform dataset as today.
    - `WaveformDataset` can reload list-based compression metadata and rebuild
      decompression transforms.
    - `load_svd_basis(...)` is available as a Hydra target for loading an
      existing `SVDBasis` from file and passing it into `ApplySVD`.
    - Small smoke: `my_runs/compression_svd_smoke` generates and reloads a
      compressed one-sample dataset with a two-sample SVD-training override.
  - Remaining work: decide how much SVD training provenance/mismatch
    information should be saved as metadata.

- [ ] `dingo/gw/training/train_builders.py`
  - Large procedural transform pipeline.
  - Prime candidate for Hydra targets, but many transforms need runtime objects:
    domain, detectors, ASD datasets, standardization, priors, reference time.
  - Long-term design option: configs define an ordered transform pipeline with
    `_target_` entries. Transform objects should receive their dependencies
    through Hydra-instantiated object graphs or through explicit object setup,
    not through config mutation.
  - Agreed runtime-dependency pattern:
    - Instantiate expensive/shared runtime objects once, e.g. effective domain,
      waveform dataset, ASD dataset, standardization, and optionally
      `InterferometerList`.
    - Package them in a `runtime_dependencies` dictionary.
    - Instantiate transform configs with a generic helper, conceptually:

      ```python
      transforms = [
          hydra.utils.instantiate(t_cfg)(**runtime_dependencies)
          if t_cfg.get("_partial_", False)
          else hydra.utils.instantiate(t_cfg)
          for t_cfg in cfg.transforms
      ]
      ```

    - Passing runtime objects this way is not a performance concern because
      Python passes references, and this happens at transform construction time,
      not for every sample.
    - Transforms that consume runtime dependencies should fail early if a
      required dependency is missing. Avoid silent swallowing of misspelled
      dependency names.
    - Do not use fake config interpolations such as `${runtime:asd_dataset}`
      unless a real resolver is explicitly designed later; Hydra does not
      provide such a runtime-object resolver by default.
  - Decide which transform state belongs in config, which belongs inside the
    instantiated transform/model objects, and which artifact provenance should
    be saved.
  - Difficulty: very high. This is probably the hardest non-Condor refactor
    because the pipeline mixes config, runtime datasets, derived state, and
    metadata persistence.
  - Progress: the default training transform order now lives in `configs/train.yaml`
    as `_target_` entries. `set_train_transforms(...)` instantiates that list via
    the generic `instantiate_with_runtime_dependencies(...)` helper. Cheap
    dependencies such as priors, interferometer lists, reference time,
    parameter dictionaries, standardization, selected output keys, and GNPE
    transforms are now config-owned and recursively instantiated. The live
    runtime dependency dictionary is reduced to the waveform domain and ASD
    dataset. `zero_noise` is handled through the ordinary omit-transform path,
    and `build_dataset(...)` was retired in favor of instantiating the
    configured `WaveformDataset`. The standardization-prefix transforms are
    explicit in config so the old standardization behavior is computed before
    the final transform list is instantiated.

- [x] `dingo/core/posterior_models/build_model.py`
  - Replace `posterior_model_type` dispatch with `_target_`.
  - Remove `autocomplete_model_kwargs(...)` as a config-mutation step. Its
    responsibilities should either become explicit config fields or normal
    shape-inference behavior of the instantiated model/embedding objects.
  - Design status: agreed. Model/checkpoint metadata should save the exact
    resolved Hydra config plus produced checkpoint state. Old
    `posterior_model_type` dispatch and config autocompletion should disappear
    in favor of `_target_` / `instantiate`.
  - Progress: model configs now carry posterior-model `_target_` entries, the
    `posterior_model_type` dispatch was removed from the Hydra model path, and
    `autocomplete_model_kwargs(...)` was retired. Model dimensions are computed
    before model construction in `prepare_training_new(...)` and saved into the
    config metadata.

- [x] `dingo/core/utils/torchutils.py`
  - Replace optimizer/scheduler `type` switches with `_target_` configs.
  - Design status: agreed/easy. Optimizers and schedulers should be direct
    `_target_` configs, removing the current string-switch helpers.
  - Initial Stage 3 pass: default optimizer/scheduler configs now use
    `_target_` partials, and the existing helper functions instantiate those
    configs while keeping legacy `type` settings for older stage/checkpoint
    metadata.
  - Progress: the string-switch fallbacks have been removed from the helper
    functions, and `BasePosteriorModel.initialize_optimizer_and_scheduler()`
    now calls `hydra.utils.instantiate(...)` directly.

- [ ] Condor integration as a Hydra workflow/launcher
  - Keep Condor-specific migration out of Stage 1.
  - Discuss later before implementation. This is intentionally not part of the
    first Stage 3 implementation pass.
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

- [ ] `dingo/pipe/parser.py`
  - Biggest migration knot. This is both a CLI parser and a compatibility layer
    with bilby_pipe-style settings.

## Stage 3 Testing Requirements

- [ ] Add extensive tests for Hydra configs and instantiated behavior.
  - Any option included in the configs should be covered by a test at the
    appropriate level.
  - At minimum, every config group/file should have composition tests.
  - Every `_target_` config should have an instantiation test, unless
    instantiation is intentionally too expensive; in that case add a cheaper
    test that validates the config shape and document the skipped expensive
    behavior.
  - Behavior-affecting options should have unit or smoke tests that exercise the
    resulting object behavior, not only config composition.
  - Important override combinations should be tested, especially for:
    - domain choices and domain updates
    - waveform generator variants
    - prior groups
    - compression/SVD load-vs-train choices
    - transform pipeline variants, including GNPE context, zero noise, random
      cropping, and stage-specific ASD/noise settings
    - model families and model dimensions
    - optimizer/scheduler configs
    - metadata saving/loading with dot-access config restoration
  - Tests should confirm that saved metadata contains the resolved Hydra config,
    absolute paths, ahead-of-time computed values such as standardization, and
    separate artifact/result state.
  - Progress: `tests/test_hydra_config_targets.py` covers Hydra target
    composition for domain, waveform generator, priors, models,
    optimizer/scheduler, SVD load-from-file, standardization-prefix transforms,
    and an in-memory training-transform smoke that verifies standardization is
    computed before final transform instantiation.

## Suggested Order

1. Start with easy direct `_target_` migrations:
   domain, waveform generator, optimizer, and scheduler.
2. Move Python-defined defaults into Hydra config groups and make configs
   complete. This should be mostly config edits plus deletion of old defaulting
   code.
3. Migrate priors: remove `default` sentinels, target `BBHPriorDict` directly,
   and retire `build_prior_with_defaults(...)`.
4. Migrate model construction: replace `posterior_model_type` dispatch with
   `_target_` configs and make `autocomplete_model_kwargs(...)` unnecessary
   through interpolation / ahead-of-time config computation.
5. Migrate compression/SVD as an ordered transform list with load-vs-train SVD
   targets and produced SVD arrays saved as dataset artifacts.
6. Migrate the training transform pipeline using explicit transform configs and
   the agreed generic `instantiate_with_runtime_dependencies(...)` helper.
7. Update metadata saving/loading to save the exact resolved Hydra config,
   restore dot access on load, keep result metadata separate, and keep produced
   artifacts in dedicated data/state fields.
8. Add or expand tests throughout each step; do not leave broad config options
   untested.
9. Discuss Condor integration later as a separate Stage 3 workflow/launcher
   design.
10. Leave pipe/parser migration until the rest of the config story is stable.

## Stage 3 Risk Notes

These are the areas that need the most care, even if implementation starts with
the easier direct-instantiation changes.

1. **Saved metadata contract** - very high difficulty.
   - Cross-cuts datasets, trained models, results, checkpoint resume, and
     reproducibility.
   - Needs a deliberate policy for saving resolved Hydra config snapshots that
     already include all ahead-of-time-computable values, plus artifact
     provenance and truly produced state without reintroducing parallel legacy
     metadata dictionaries.
2. **Training transform pipeline** - very high difficulty.
   - Many transforms are configured partly by settings and partly by runtime
     objects: waveform/domain metadata, detectors, ASD datasets, priors,
     standardization, reference time, and context parameters.
   - Target design should be a Hydra-instantiated transform/object graph, with
     runtime setup owned by the objects rather than by config patching.
3. **Compression/SVD workflow** - high difficulty.
   - Compression can mean loading an existing basis, training a basis from
     generated waveforms, applying transforms, and saving enough provenance to
     reproduce the dataset.
   - Target design should be a declarative compression-target list of
     Hydra-instantiated objects, including SVD steps.
4. **Model construction and checkpoint metadata** - high difficulty.
   - Replace `posterior_model_type` dispatch with `_target_`.
   - Model configs also interact strongly with saved metadata.
5. **Config autocompletion / derived model dimensions** - high difficulty.
   - `autocomplete_model_kwargs(...)` currently hard-codes model config
     resolution based on a runtime data sample.
   - Target design should remove hidden config mutation. Dimensions and related
     settings should be computed into the resolved Hydra config beforehand
     whenever possible; object-level inference is only the fallback for values
     that cannot exist before construction.
6. **Prior construction and defaults** - medium/high difficulty.
   - Moving `default` strings to explicit prior config groups is conceptually
     simple, but Bilby prior objects and constraints make it easy to
     accidentally change behavior.
7. **Domain and waveform generator construction** - medium difficulty.
   - Good candidates for early `_target_` migration once the saved-config
     policy is clear.
   - Main risks are waveform-generator interface flags and keeping the config
     readable.
8. **Optimizer/scheduler construction** - low/medium difficulty.
   - Natural `_target_` candidates with relatively small metadata surface.
   - Good later cleanup once the harder contracts are settled.
9. **Condor integration** - high difficulty, but separable and deferred.
   - Important for Stage 3, yet mostly an orchestration/launcher design problem
     rather than a core config-object construction problem.
   - Discuss later before implementation.
