# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dingo (Deep Inference for Gravitational-wave Observations) is a Python framework for analyzing gravitational wave data using neural posterior estimation. It trains normalizing flows to represent Bayesian posteriors conditioned on detector data, enabling fast amortized inference.

## Development Commands

### Package Management (uv recommended)
```bash
uv sync                                    # Install all dependencies
uv sync --extra wandb --extra pyseobnr     # Include optional extras
```

### Testing
```bash
pytest tests/                   # Run all tests
pytest tests/core/              # Run core tests only
pytest tests/gw/                # Run GW tests only
pytest -m slow tests/           # Run only slow-marked tests
pytest tests/path/to/test.py   # Run single test file
pytest tests/path/to/test.py::test_function  # Run single test
```

### Code Formatting
```bash
black dingo/                    # Format code with Black
```

## Architecture

### Core Package Structure

**`dingo/core/`** - Neural network and inference foundations:
- `posterior_models/` - Flow-based posterior estimators (normalizing flows, flow matching, score diffusion)
- `nn/` - Neural network components: `enets.py` (embedding networks for GW data compression), `nsf.py` (neural spline flows), `cfnets.py` (coupling flows)
- `dataset.py` - Base DingoDataset with HDF5 serialization
- `result.py` - Posterior samples storage with importance sampling support

**`dingo/gw/`** - Gravitational wave-specific implementation:
- `likelihood.py` - StationaryGaussianGWLikelihood with time/phase/calibration marginalization
- `domains/` - Frequency/time domain definitions; `multibanded_frequency_domain.py` for frequency masking
- `waveform_generator/` - LALSuite waveform generation
- `transforms/` - Data transformation pipelines for waveforms, detectors, parameters
- `training/` - Model training with `train_pipeline.py`
- `inference/` - GWSampler for posterior sampling

**`dingo/pipe/`** - Command-line pipeline (bilby_pipe-based):
- Four-stage workflow: data generation → sampling → importance sampling → plotting
- `parser.py` - INI file configuration parsing
- `nodes/` - HTCondor DAG node definitions

### Key Data Flow

1. **Training**: WaveformDataset + ASDDataset → train posterior model → model.pt
2. **Inference**: EventDataset (real/simulated data) → GWSampler → Result (posterior samples)
3. **Validation**: Result → importance sampling → reweighted samples + evidence

### CLI Entry Points

Dataset generation: `dingo_generate_dataset`, `dingo_generate_asd_dataset`
Training: `dingo_train`, `dingo_train_condor`
Inference: `dingo_pipe`, `dingo_pipe_sampling`, `dingo_pipe_importance_sampling`
Utilities: `dingo_ls` (inspect HDF5), `dingo_result`

## Key Implementation Details

- Python 3.10-3.11 required (< 3.12)
- All datasets/results serialize to HDF5 with full metadata preservation
- Thread limiting: Set `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` in CI to avoid conflicts
- GPU selection via `CUDA_VISIBLE_DEVICES`
- Uses bilby for priors and bilby_pipe for workflow patterns
- Uses LALSuite for waveform models (SEOBNRv4, IMRPhenomD, etc.)

## Configuration

- Training/dataset settings: YAML files (see `examples/`)
- Inference pipeline: INI files for `dingo_pipe`

## Key Concepts

### GNPE (Group-equivariant Neural Posterior Estimation)
GNPE exploits physical symmetries to simplify learning. For GWs, it uses:
- **Exact symmetry**: Time translation (geocenter coalescence time)
- **Approximate symmetry**: Sky rotation (affects detector arrival times)

Requires training **two networks**: an initialization network for detector times, and a main network conditioned on proxy variables. Uses Gibbs sampling at inference time (~30 iterations).

### Training Stages
Training uses multi-stage approach:
- **Stage 0 (pre-training)**: Frozen RB (reduced basis/SVD) layer, fiducial ASD
- **Stage 1 (fine-tuning)**: Unfrozen RB layer, variable realistic ASDs

The RB layer projects data onto SVD basis vectors. Freezing during pre-training provides stability.

### Importance Sampling Workflow
After neural network sampling:
1. **Density recovery**: Train unconditional NDE on GNPE samples to get proposal q(θ)
2. **Synthetic phase**: Reconstruct phase parameter from phase-marginalized samples
3. **Reweighting**: w(θ) ∝ π(θ)L(θ)/q(θ) corrects network approximation errors
4. **Evidence**: Estimated via Monte Carlo integration

### Multibanded Frequency Domain
Non-uniform frequency grid that adapts to GW signal structure:
- Coarser resolution at higher frequencies (where waveforms oscillate slower)
- Significant data compression while preserving signal information
- Each band has 2× the bin-width of the previous band

## Practical Notes

### Pre-trained Models
Available on Zenodo: https://zenodo.org/communities/dingo-gw/records

### Model Types
- **NPE** (normalizing flows): Default choice. Fast log_prob evaluation for importance sampling.
- **FMPE** (flow matching): Often better performance, but slower log_prob evaluation.
- **Score diffusion**: Experimental. Flow matching typically outperforms it (arXiv:2305.17161).

### GNPE Trade-offs
GNPE provides better performance than plain NPE but adds complexity:
- Requires training two networks (init + main)
- Slower inference due to Gibbs iterations
- **Loses access to probability density** due to Gibbs sampling, requiring costly density recovery step before importance sampling
- Recent work (arXiv:2512.02968) shows transformer encoders can approach GNPE performance without the extra complexity (not yet in main branch).

### Waveform Models
- **BBH**: IMRPhenomXPHM, SEOBNRv5PHM, SEOBNRv4PHM (two waveform families: Phenom and SEOBNR)
- **BNS**: IMRPhenomPv2_NRTidal (includes tidal deformability parameters)

### Other Modules
- **`dingo/populations/`**: Hierarchical inference (population-level). Under development in separate branch. Uses transformer encoder with events as tokens to infer population distributions (e.g., mass distribution).
- **`dingo/asimov/`**: Integration with Asimov, an LVK tool for automating analyses.

## Resources

- **Documentation**: https://dingo-gw.readthedocs.io/ (source in `docs/source/`)
- **Examples**: `examples/` contains reference YAML/INI configs for all major workflows (dataset generation, training, inference)
- **Compatibility scripts**: `compatibility/` has scripts for updating old saved models when format changes
- **Inspecting files**: `dingo_ls <file>` shows stored settings and metadata for any HDF5/pt file

## Development Notes

### Performance Bottlenecks
- **Training data preparation is often the expensive part**, not just model training itself
- Waveform dataset is typically held in memory; **system RAM can become an issue** for large datasets
- Preprocessing transforms are costly

### Key File: `dingo/gw/training/train_builders.py`
Contains the transform structure for data preparation. This specifies how training data is processed and is often modified for different physical contexts. Improving this interface is an ongoing goal.

### Transform Sample Dict Convention
All transforms operate on "sample dicts": `{"waveform": {...}, "parameters": {...}, ...}`. Compression transforms (`WhitenFixedASD`, `ApplySVD`) were updated to this interface (on `bns` branch) so that parameter-dependent transforms like `HeterodynePhase` can participate in `Compose` chains. `WaveformDataset.__getitems__` and `WaveformGenerator.generate_hplus_hcross` wrap data as sample dicts before passing to transform chains.

### bilby_pipe Compatibility
`bilby_pipe` updates frequently break `dingo.pipe`. Expect maintenance when bilby_pipe is updated.

## BNS Development (In Progress)

Adding Binary Neutron Star inference capabilities based on Dax et al., Nature (2025) [arXiv:2407.09602]. Development happens on the `bns` feature branch with PR sub-branches (`bns-*`). See `memory/bns_merge_plan.md` for detailed plan.

### Key BNS Concepts

**Phase Heterodyning**: Multiply data by `exp(i*φ(f;M̃))` where `φ = (3/128)(πGM̃f/c³)^(-5/3)` to remove dominant chirp phase evolution. Dramatically reduces oscillations the network must learn. Implemented as single-step GNPE where chirp_mass_proxy is the GNPE proxy parameter.

**Prior Conditioning**: Train one network with hierarchical priors `q(θ|d,ρ)`, where `ρ` parameterizes narrow chirp mass prior bounds. At inference, set `ρ` to event-specific chirp mass estimate. Implemented via `fixed_context_parameters` in `GNPESampler`.

**Single-Step GNPE for Chirp Mass**: Unlike multi-iteration time GNPE, chirp mass GNPE uses just 1 iteration. This preserves log_prob access (no density recovery needed). The `GNPESampler` class with `num_iterations=1` and `fixed_context_parameters` handles this.

### Architecture Notes for BNS

- `GNPESampler._preprocess_data()` decouples one-time data preprocessing from init_sampler
- Heterodyning must happen BEFORE decimation (they don't commute)
- If chirp_mass_proxy is fixed, heterodyne once in `_preprocess_data` (efficient)
- The `Result` class (`gw/result.py`) is complex and handles many concerns (domain rebuilding, likelihood construction, synthetic phase, calibration). Improving this interface is desirable but must preserve backward compatibility.
- The `use_base_domain` / `frequency_update` logic in likelihood was a deliberate recent design for exact likelihoods on non-multibanded data - preserve it.
