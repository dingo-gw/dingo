# Dingo-T1 (Transformer model)

Dingo-T1 is the transformer-based variant of Dingo.{footcite:p}`Kofler:2026fgw`  Instead of
compressing strain data with an SVD projection and a dense residual network, it represents each
detector's data as a sequence of *tokens* — fixed-size frequency-domain chunks — and
processes them with a transformer encoder.  This architecture makes inference more
flexible: individual tokens can be masked at inference time to exclude specific
frequency bands without retraining the model.

The training and inference steps mirror the [NPE](example_npe_model.md) and
[GNPE](example_gnpe_model.md) examples; the main novelties are the different network
architecture, the masking options during training, and the set of
**inference-time frequency options** described at the end of this page.

## File structure

```
dingo_t1/

    #  config files
    waveform_dataset_settings.yaml
    asd_dataset_settings.yaml
    train_settings.yaml
    GW150914.ini

    training_data/
        waveform_dataset.hdf5
        asd_dataset/

    training/
        model_latest.pt
        model_stage_0.pt
        history.txt
        ...

    outdir_GW150914/
        #  dingo_pipe output
```


## Step 1: Generate a waveform dataset

```bash
cd dingo_t1
mkdir training_data training

dingo_generate_dataset \
    --settings waveform_dataset_settings.yaml \
    --out_file training_data/waveform_dataset.hdf5
```

The waveform dataset settings follow the same format as for the NPE model, but
use a multibanded domain up to 1810 Hz and a precessing-spin prior.  The
`compression` block controls how the stored dataset is compressed (waveforms are
decompressed when loaded); it is unrelated to the network, which has no SVD layer.


## Step 2: Generate an ASD dataset

```bash
dingo_generate_asd_dataset \
    --settings_file asd_dataset_settings.yaml \
    --data_dir training_data/asd_dataset \
    --out_name training_data/asd_dataset/asds.hdf5
```


## Step 3: Train the network

```bash
dingo_train --settings_file train_settings.yaml --train_dir training
```

The key difference from the NPE model is the `tokenization` block inside
`train_settings.yaml`:

```yaml
tokenization:
  token_size: 16                  # number of frequency bins per token
  mask_detectors:                 # enables subset-detector inference
    num_blocks: 3
    p_mask_012_detectors: [0.6, 0.3, 0.1]
    p_mask_hlv:
      H1: 0.3
      L1: 0.3
      V1: 0.4
  mask_frequency_range:           # enables inference-time f_min / f_max updates
    p_mask: 0.2
    f_min_upper: 100.0            # f_min can be raised up to this value
    f_max_lower: 800.0            # f_max can be lowered down to this value
    p_lower_upper_both: [0.4, 0.4, 0.2]
    p_same_all_detectors: 0.7
  mask_frequency_notches:         # enables inference-time interior masking (notching)
    p_per_detector: 0.3
    max_width: 10.0
```

`mask_frequency_range` trains the network to handle a variable lower and upper
frequency cutoff per detector.  `mask_frequency_notches` trains it to handle
masked interior intervals (used for PSD notching at inference time).
`mask_detectors` trains it to cope with missing detectors.  All three
augmentations are optional and independent of each other.

```{important}
The ranges set by `f_min_upper` and `f_max_lower` in `mask_frequency_range` define
the *in-distribution* envelope for inference-time frequency updates.  Requesting a
frequency range outside this envelope raises an error.
PSD notching via `mask_frequency_notches` does not produce a warning because the
masking is already encoded in the ASD.
```


## Step 4: Inference

```bash
dingo_pipe GW150914.ini
```

The `GW150914.ini` file in the `examples/dingo_t1/` directory shows all
inference-time options.  The sections below describe the new ones.


### Adjusting the frequency range

The frequency band used for inference can be restricted per detector at
inference time, without retraining.  This is useful when a detector has a
higher noise floor at low or high frequencies for a particular event, or when
the network was trained on a wider band than what a given event warrants.

```ini
# Single float applies the same limit to all detectors:
# minimum-frequency = 30.0

# Per-detector dict — detectors absent from the dict use the training default:
minimum-frequency = {H1: 30, L1: 40}
maximum-frequency = {H1: 1024, L1: 512}
```

The network must have been trained with `mask_frequency_range` for a non-default
value to be in-distribution.  If the requested range falls outside the training
envelope (set by `f_min_upper` / `f_max_lower`) an error is raised.


### PSD notching

Spectral artifacts such as power-line harmonics can be suppressed by *notching*
— setting the ASD to 1 in affected bins and masking the corresponding tokens.
Since ASD = 1 ≫ the typical noise level (~10⁻²³ 1/√Hz), those bins contribute
negligibly to the noise-weighted inner product used in importance sampling.
Dingo-T1 supports two equivalent paths to achieve this:

#### Standard path — set `psd-notch-dict` in the ini file

```ini
# Per-detector dict.  Each value is a single interval [f_lo, f_hi] or a list
# of intervals [[f_lo1, f_hi1], [f_lo2, f_hi2], ...].  Units: Hz.
psd-notch-dict = {H1: [[59.0, 61.0], [119.0, 121.0]], L1: [59.0, 61.0]}
```

During **data generation**, dingo_pipe sets the ASD to 1 in the specified bins,
saves the modified ASD to the event HDF5, and records the notched intervals in
the event metadata.  During **sampling**, the tokens overlapping these intervals
are masked before the forward pass.  During **importance sampling**, the ASD = 1
bins are already stored in the context, so no further action is needed.

#### Pre-notched path — no ini flag required

If the ASD has already been set to 1 in the notch regions before dingo_pipe
runs, leave `psd-notch-dict` commented out.  The notched intervals are detected
from the stored ASD at the end of data generation and recorded in the event
metadata, so sampling proceeds exactly as in the standard path.  (The Asimov
integration passes its notch dict through `psd-notch-dict`.)

```{note}
Both paths converge to the same outcome: notched bins have ASD = 1 in the event
HDF5, the overlapping tokens are masked at sampling time, and importance sampling
handles them correctly without any additional configuration.
```

```{note}
PSD notching is only supported for tokenized (transformer) models.  The token
masking has no effect on ResNet-based models.
```

```{eval-rst}
.. footbibliography::
```
