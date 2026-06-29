---
name: smoke-test
description: Run the toy-NPE end-to-end pipeline (generate dataset, train briefly, sample) to verify a change didn't break the dingo inference pipeline. Use for an end-to-end sanity check, not for unit testing.
---

# Toy-NPE end-to-end smoke test

Verifies the full pipeline runs: waveform dataset → ASD dataset → training → inference.
Mirrors `ci/dingo-ci`, using the configs in `examples/toy_npe_model/`.

**Run from a scratch copy, not inside the repo** (these commands write datasets, a
trained model, and inference outputs — don't pollute the working tree):

```bash
work=$(mktemp -d)
cp examples/toy_npe_model/* "$work"/
cd "$work"
mkdir -p training_data training

uv run dingo_generate_dataset --settings waveform_dataset_settings.yaml \
    --out_file training_data/waveform_dataset.hdf5
uv run dingo_generate_asd_dataset --settings_file asd_dataset_settings.yaml \
    --data_dir training_data/asd_dataset
uv run dingo_train --settings_file train_settings.yaml --train_dir training
uv run dingo_pipe GW150914.ini
```

(`uv run` from anywhere resolves the project venv since the CLI scripts are installed by
`uv sync`.) Success = all four commands exit 0. Inspect any produced file with
`dingo_ls <file>`. This is slower than the unit tests; use it to catch integration
breakage after non-trivial changes to datasets, training, waveforms, or the pipeline.
