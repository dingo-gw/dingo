# Dingo-T1 (Transformer model)

Dingo-T1 is the transformer-based variant of Dingo.{footcite:p}`Kofler:2026fgw`  Instead of
compressing strain data with a convolutional embedding network, it represents each
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

The waveform dataset settings are the same as for the NPE model.  The
transformer does not use an SVD compression layer, so the `compression` block
in the settings file can be omitted.


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
  token_size: 16              # number of frequency bins per token
  mask_frequency_range:       # enables inference-time f_min / f_max updates
    p_mask: 0.2
    f_min_upper: 100.0        # tokens below this can be masked from the bottom
    f_max_lower: 800.0        # tokens above this can be masked from the top
  mask_frequency_notches:    # enables inference-time interior masking (notching)
    p_per_detector: 0.3
    f_min: 100.0
    f_max: 800.0
    max_width: 100.0
  mask_detectors:             # enables subset-detector inference
    num_blocks: 2
    p_mask_012_detectors: [0.6, 0.4]
    p_mask_hlv: {H1: 0.5, L1: 0.5, V1: 0.5}
```

`mask_frequency_range` trains the network to handle a variable lower and upper
frequency cutoff per detector.  `mask_frequency_notches` trains it to handle
masked interior intervals (used for PSD notching at inference time).
`mask_detectors` trains it to cope with missing detectors.  All three
augmentations are optional and independent of each other.

```{important}
The ranges set by `f_min_upper` and `f_max_lower` in `mask_frequency_range` define
the *in-distribution* envelope for inference-time frequency updates.  Requesting a
frequency range outside this envelope will raise an out-of-distribution warning.
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
minimum-frequency = {H1: 30, L1: 30, V1: 40}
maximum-frequency = {H1: 1024, L1: 1024, V1: 512}
```

The network must have been trained with `mask_frequency_range` for a non-default
value to be in-distribution.  If the requested range falls outside the training
envelope (set by `f_min_upper` / `f_max_lower`) a warning is printed but
inference proceeds.


### PSD notching

Spectral artefacts such as power-line harmonics can be suppressed by *notching*
— setting the ASD to 1 in affected bins and masking the corresponding tokens.
Since ASD = 1 ≫ the typical noise level (~10⁻²³ 1/√Hz), those bins contribute
negligibly to the noise-weighted inner product used in importance sampling.
Dingo-T1 supports two equivalent paths to achieve this:

#### Standard path — set `psd-notch-dict` in the ini file

```ini
# Per-detector dict.  Each value is a single interval [f_lo, f_hi] or a list
# of intervals [[f_lo1, f_hi1], [f_lo2, f_hi2], ...].  Units: Hz.
psd-notch-dict = {H1: [[59.0, 61.0], [119.0, 121.0]], L1: [59.0, 61.0], V1: [49.0, 51.0]}
```

During **data generation**, dingo_pipe sets the ASD to 1 in the specified bins
and saves the modified ASD to the event HDF5.  During **sampling**, it
auto-detects the notched regions from the stored ASD and masks the overlapping
tokens before the forward pass.  During **importance sampling**, the ASD = 1
bins are already stored in the context, so no further action is needed.

#### Asimov / pre-notched path — no ini flag required

In the Asimov pipeline, PSD notching is applied *before* dingo_pipe runs: an
external tool sets ASD = 1 in the notch regions and provides the modified PSD
dict.  In this case, leave `psd-notch-dict` commented out.  dingo_pipe will
auto-detect the notched regions from the stored ASD and mask the corresponding
tokens at sampling time, producing exactly the same result as the standard path.

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
