# CLAUDE.md

## Project Overview
Dingo performs gravitational-wave parameter inference with neural posterior estimation. The
command-line pipeline (`dingo/pipe/`, built on bilby_pipe) orchestrates four stages:
**data generation → sampling → importance sampling → plotting**, typically as an HTCondor DAG.

## Development Commands
- `uv sync`; `uv run pytest [tests/...::test]`; `uv run black dingo tests`.
- `dingo_pipe <config.ini>` runs the pipeline; `dingo_ls <model.pt>` inspects checkpoint metadata.

## Architecture (orientation)
- `dingo/core/` — model-agnostic ML (posterior models, nets, datasets, results). Keep domain-agnostic.
- `dingo/gw/` — GW physics: `likelihood.py`, `domains/`, `waveform_generator/`, `transforms/`, `inference/`.
- `dingo/asimov/` — LVK Asimov automation.

## The pipeline in detail (`dingo/pipe/`)
- `parser.py` — defines all INI arguments (a bilby_pipe `ArgumentParser`). New config options are declared here.
- `main.py` — entry point `dingo_pipe`:
  - `fill_in_arguments_from_model(args)` loads the model **metadata** (via `build_model_from_kwargs(..., device="meta", load_training_info=False)`) and fills derived args (duration, frequencies, detectors, approximant, reference frequency, prior).
  - `write_complete_config_file(parser, args, inputs)` writes `{outdir}/{label}_config_complete.ini` — the *completed* config that every DAG node reads.
  - `generate_dag(inputs)` builds the HTCondor DAG.
- `nodes/` — one module per stage (`generation_node.py`, `sampling_node.py`, `importance_sampling_node.py`, `plot_node.py`). Each declares the files it transfers to the compute node.
- `data_generation.py` — the generation stage; currently loads the network to read `model.metadata` (e.g. `Injection.from_posterior_model_metadata`, `build_domain_from_model_metadata`).

## Config / metadata conventions
- Settings are plain nested dicts serialized into models/datasets, not dataclasses; they must round-trip.
- A checkpoint's `metadata` has `dataset_settings` (domain, intrinsic_prior, waveform_generator, compression) and `train_settings` (data: detectors, ref_time, extrinsic_prior, standardization; model; training).
- The completed config is the single source of truth passed between nodes. Prefer carrying information through the completed config rather than re-reading large artifacts on each node.

## `transfer_files`
- The `--transfer-files` option (default `True`) uses HTCondor file transfer to ship inputs (including the model `.pt`) to compute nodes. The network can be multiple GB; only stages that evaluate the network weights actually need it. Generation and plotting need only metadata.

## Conventions & boundaries
- Reuse existing functions/patterns; smallest correct change; no speculative abstractions or one-line helpers.
- Keep `core/` free of GW/bilby/LAL imports. Don't add dependencies or change the environment without approval. Don't commit/push.
