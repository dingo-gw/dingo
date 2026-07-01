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
- [ ] A small smoke test exists or is documented.

Recommended shared pattern:

```python
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


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

If a user explicitly wants old current-working-directory behavior, use both:

```bash
hydra.job.chdir=false hydra.run.dir=.
```

Testing showed that `hydra.job.chdir=false` alone is not enough: application
outputs land in the current working directory, but Hydra logs and `.hydra/`
metadata still go to `hydra.run.dir`.

Use `compression: null` rather than `compression: None`.

Do not migrate `.ini` files for now. In particular, leave Asimov/plugin-style
`.ini` configuration outside the Hydra migration TODO unless this is reopened
explicitly later.

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

- [ ] `dingo_evaluate_multibanded_domain`
  - File: `dingo/gw/dataset/evaluate_multibanded_domain.py`
  - Stage 1: straightforward wrapper around current settings dict.
  - Stage 3: replace `build_domain`, `build_prior_with_defaults`, and manual
    waveform-generator branching with Hydra targets.
  - Estimated Stage 1 difficulty: low.
  - Estimated Stage 3 difficulty: medium.

- [ ] `dingo_build_svd`
  - File: `dingo/gw/dataset/utils.py`
  - Stage 1: small CLI migration.
  - Stage 3: likely becomes part of a Hydra-instantiated compression/SVD
    workflow.
  - Estimated Stage 1 difficulty: low.
  - Estimated Stage 3 difficulty: medium.

- [ ] `dingo_merge_datasets`
  - File: `dingo/gw/dataset/utils.py`
  - Stage 1: small CLI migration.
  - Stage 3: probably little to do.
  - Estimated Stage 1 difficulty: low.
  - Estimated Stage 3 difficulty: low.

### Noise / ASD Generation

- [ ] `dingo_generate_asd_dataset`
  - File: `dingo/gw/noise/generate_dataset.py`
  - Stage 1: replace parser/settings-file handling with Hydra.
  - Stage 1 should cover local generation only.
  - Condor/DAG behavior should remain as-is or be deferred.
  - Estimated Stage 1 difficulty: medium.
  - Estimated Stage 3 difficulty: medium/high for Condor integration.

- [ ] `dingo_estimate_psds`
  - File: `dingo/gw/noise/asd_estimation.py`
  - Stage 1: straightforward wrapper around current dict settings.
  - Stage 3: domain construction and PSD-estimation settings could become more
    declarative.
  - Estimated Stage 1 difficulty: low/medium.
  - Estimated Stage 3 difficulty: medium.

- [ ] `dingo_generate_synthetic_asd_dataset`
  - File: `dingo/gw/noise/synthetic/generate_dataset.py`
  - Stage 1: straightforward config migration.
  - Stage 3: parameterization and sampling could become separate config groups.
  - Estimated Stage 1 difficulty: low.
  - Estimated Stage 3 difficulty: medium.

- [ ] `dingo_merge_asd_datasets`
  - File: `dingo/gw/noise/utils.py`
  - Stage 1: small CLI migration.
  - Estimated Stage 1 difficulty: low.
  - Estimated Stage 3 difficulty: low.

### Training

- [ ] `dingo_train`
  - File: `dingo/gw/training/train_pipeline.py`
  - Stage 1: replace parser with Hydra, while still converting to the old
    `train_settings` and `local_settings` dicts.
  - Watch out:
    - New training versus checkpoint resume are currently mutually exclusive CLI
      modes.
    - `local` settings are popped out and saved separately as
      `local_settings.yaml`.
    - Train settings are saved into model metadata.
    - Several progress messages still use `print`.
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

- [ ] `dingo_append_training_stage`
  - File: `dingo/gw/training/utils.py`
  - Stage 1: small Hydra CLI.
  - Watch out: modifies model metadata/training-stage settings.
  - Estimated Stage 1 difficulty: low.
  - Estimated Stage 3 difficulty: low/medium.

### Pipe / Inference

The `dingo_pipe*` family is the largest migration area because it is based on
`BilbyArgParser` / `configargparse` and has bilby_pipe-style conventions.

- [ ] `dingo_pipe`
  - File: `dingo/pipe/main.py`
  - Uses parser from `dingo/pipe/parser.py`.
  - Estimated Stage 1 difficulty: high.
  - Estimated Stage 3 difficulty: very high.

- [ ] `dingo_pipe_generation`
  - File: `dingo/pipe/data_generation.py`
  - Estimated Stage 1 difficulty: high.
  - Estimated Stage 3 difficulty: high.

- [ ] `dingo_pipe_sampling`
  - File: `dingo/pipe/sampling.py`
  - Estimated Stage 1 difficulty: high.
  - Estimated Stage 3 difficulty: high.

- [ ] `dingo_pipe_importance_sampling`
  - File: `dingo/pipe/importance_sampling.py`
  - Estimated Stage 1 difficulty: high.
  - Estimated Stage 3 difficulty: high.

- [ ] `dingo_pipe_plot`
  - File: `dingo/pipe/plot.py`
  - Estimated Stage 1 difficulty: medium/high.
  - Estimated Stage 3 difficulty: medium.

- [ ] `dingo_pipe_pp_test`
  - File: `dingo/pipe/pp_test.py`
  - Estimated Stage 1 difficulty: low/medium.
  - Estimated Stage 3 difficulty: medium.

- [ ] `dingo_result`
  - File: `dingo/pipe/dingo_result.py`
  - Small CLI, but connected to pipe output conventions.
  - Estimated Stage 1 difficulty: low.
  - Estimated Stage 3 difficulty: low.

### Other Utilities

- [ ] `dingo_pt_to_hdf5`
  - File: `dingo/core/utils/pt_to_hdf5.py`
  - Small argparse CLI.
  - Estimated Stage 1 difficulty: low.
  - Estimated Stage 3 difficulty: low.

- [ ] `dingo_ls`
  - File: `dingo/gw/ls_cli.py`
  - Inspection tool rather than a config-driven workflow.
  - Could stay argparse for a while, but migrate for consistency if Stage 1 aims
    to eliminate all public parsers.
  - Estimated Stage 1 difficulty: low.
  - Estimated Stage 3 difficulty: low.

- [ ] Importance-weight generation
  - File: `dingo/gw/importance_sampling/importance_weights.py`
  - Not currently listed in `project.scripts`, but it is a config-driven CLI.
  - Estimated Stage 1 difficulty: low/medium.
  - Estimated Stage 3 difficulty: medium.

- [ ] Unconditional density estimation
  - File: `dingo/core/density/unconditional_density_estimation.py`
  - Not currently listed in `project.scripts`, but has an argparse settings CLI.
  - Estimated Stage 1 difficulty: low.
  - Estimated Stage 3 difficulty: medium.

## Stage 2 Config Group Candidates

- `domain`
- `waveform_generator`
- `intrinsic_prior`
- `extrinsic_prior`
- `inference_parameters`
- `dataset`
- `asd_dataset`
- `noise`
- `model/posterior`
- `model/embedding`
- `training/stage`
- `training/local`
- `condor` (Stage 3 only)
- `pipe/event`
- `pipe/sampler`
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
      `spin_conversion_phase`, and interface-specific options
    - dataset compression/SVD/whitening options
    - training local/runtime/checkpoint options
    - optimizer and scheduler defaults
    - transform defaults in the training data pipeline
    - pipe/inference defaults currently supplied by parser defaults
    - ASD/noise generation defaults such as frequency range, channels, and
      Condor settings

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
