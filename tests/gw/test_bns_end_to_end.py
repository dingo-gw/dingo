"""DINGO-BNS on a toy scale, end to end: a heterodyned multibanded dataset, training
with chirp-mass GNPE and sky-position conditioning, inference on an injection with the
context pinned, and importance sampling with the synthetic phase. The training-time
network input is also compared with the inference-time data preparation for the same
signal and proxy, which is the contract a trained network relies on."""

import copy

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from bilby.gw.detector import PowerSpectralDensity
from scipy.interpolate import interp1d

from dingo.core.posterior_models import build_model_from_kwargs
from dingo.gw.dataset import generate_dataset
from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.dataset.generate_multibanded_domain import (
    generate_multibanded_domain_settings,
)
from dingo.gw.domains import build_domain
from dingo.gw.inference.context import GWSamplerContext
from dingo.gw.inference.sampler import GWComposedSampler
from dingo.gw.injection import Injection
from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.training.train_builders import set_train_transforms
from dingo.gw.training.train_pipeline import run_training
from dingo.gw.waveform_generator import WaveformGenerator
from dingo.gw.transforms import (
    AddWhiteNoiseComplex,
    RepackageStrainsAndASDS,
    SelectStandardizeRepackageParameters,
    UnpackDict,
)

DETECTORS = ["H1", "L1"]
KERNEL_HALF_WIDTH = 0.01
ASD_FILE = "aLIGO_ZERO_DET_high_P_asd.txt"

WFD_SETTINGS = yaml.safe_load(
    f"""
domain:
  # The domain extends past the merger frequency: the NRTidalv3 PhenomX wrappers in
  # LALSimulation set the time origin at min(f_merger, last requested frequency), so
  # on a domain ending below the merger the pointwise multibanded evaluation (last
  # frequency = last bin center) and the base-domain evaluation (f_max) would differ
  # by a time shift.
  type: UniformFrequencyDomain
  f_min: 40.0
  f_max: 2048.0
  delta_f: 0.0625
waveform_generator:
  approximant: APPROXIMANT
  f_ref: 40.0
  spin_conversion_phase: 0.0
intrinsic_prior:
  mass_1: bilby.core.prior.Constraint(minimum=1.0, maximum=2.5)
  mass_2: bilby.core.prior.Constraint(minimum=1.0, maximum=2.5)
  chirp_mass: bilby.gw.prior.UniformInComponentsChirpMass(minimum=1.2, maximum=1.4)
  mass_ratio: bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.5, maximum=1.0)
  a_1: bilby.core.prior.Uniform(minimum=0.0, maximum=0.05)
  a_2: bilby.core.prior.Uniform(minimum=0.0, maximum=0.05)
  tilt_1: default
  tilt_2: default
  phi_12: default
  phi_jl: default
  theta_jn: default
  phase: default
  lambda_1: default
  lambda_2: default
  luminosity_distance: 100.0
  geocent_time: 0.0
num_samples: 320
compression:
  whitening: {ASD_FILE}
  phase_heterodyning:
    order: 0
  svd:
    size: 16
    num_training_samples: 40
    num_validation_samples: 8
"""
)

TRAIN_SETTINGS = yaml.safe_load(
    f"""
data:
  train_fraction: 0.9
  detectors: {DETECTORS}
  extrinsic_prior:
    dec: default
    ra: default
    geocent_time: bilby.core.prior.Uniform(minimum=-0.01, maximum=0.01)
    psi: default
    luminosity_distance: bilby.core.prior.Uniform(minimum=20.0, maximum=60.0)
  ref_time: 1187008882.42
  gnpe_chirp:
    kernel:
      chirp_mass: bilby.core.prior.Uniform(minimum=-{KERNEL_HALF_WIDTH}, maximum={KERNEL_HALF_WIDTH})
    order: 0
  context_parameters: [ra, dec]
  inference_parameters: [delta_chirp_mass, mass_ratio, a_1, a_2, tilt_1, tilt_2, phi_12,
    phi_jl, theta_jn, luminosity_distance, geocent_time, psi, lambda_1, lambda_2]
model:
  posterior_model_type: normalizing_flow
  posterior_kwargs:
    num_flow_steps: 2
    base_transform_kwargs:
      hidden_dim: 16
      num_transform_blocks: 1
      activation: elu
      dropout_probability: 0.0
      norm: BatchNorm
      num_bins: 4
      base_transform_type: rq-coupling
  embedding_type: resnet
  embedding_kwargs:
    output_dim: 16
    hidden_dims: [32, 16]
    activation: elu
    dropout: 0.0
    norm: BatchNorm
    svd:
      num_training_samples: 40
      num_validation_samples: 8
      size: 8
training:
  stage_0:
    epochs: 1
    freeze_rb_layer: True
    optimizer:
      type: adam
      lr: 1.0e-4
    scheduler:
      type: cosine
      T_max: 1
    batch_size: 32
"""
)
LOCAL_SETTINGS = {
    "device": "cpu",
    "num_workers": 0,
    "runtime_limits": {"max_time_per_run": 3600, "max_epochs_per_run": 10},
    "checkpoint_epochs": 100,
}


def _asd_file(path, ufd):
    """ASD dataset with one design-sensitivity ASD per detector on the base domain."""
    import h5py

    psd = PowerSpectralDensity(asd_file=ASD_FILE)
    asd = interp1d(
        psd.frequency_array, psd.asd_array, bounds_error=False, fill_value=1.0
    )(ufd())
    with h5py.File(path, "w") as f:
        for ifo in DETECTORS:
            f.create_dataset(f"asds/{ifo}", data=asd[None, :])
            f[f"gps_times/{ifo}"] = np.array([0])
        f.attrs["settings"] = str({"domain_dict": ufd.domain_dict})
    return str(path)


@pytest.fixture(
    scope="module", params=["IMRPhenomXP_NRTidalv3", "IMRPhenomPv2_NRTidal"]
)
def trained_bns_model(tmp_path_factory, request):
    """Dataset, ASDs, and a one-epoch model; returns (model path, paths dict)."""
    tmp_path = tmp_path_factory.mktemp("bns")
    ufd_path = tmp_path / "settings_ufd.yaml"
    wfd_settings = copy.deepcopy(WFD_SETTINGS)
    wfd_settings["waveform_generator"]["approximant"] = request.param
    with open(ufd_path, "w") as f:
        yaml.dump(wfd_settings, f)
    mfd_path = generate_multibanded_domain_settings(
        str(ufd_path),
        num_samples=8,
        target_median_mismatch=1e-3,
        chirp_mass_proxy_offset=KERNEL_HALF_WIDTH,
    )
    with open(mfd_path) as f:
        wfd_settings = yaml.safe_load(f)
    wfd = generate_dataset(wfd_settings, num_processes=1)
    wfd.to_file(tmp_path / "wfd.hdf5")
    asd_path = _asd_file(tmp_path / "asds.hdf5", build_domain(WFD_SETTINGS["domain"]))

    train_settings = copy.deepcopy(TRAIN_SETTINGS)
    train_settings["data"]["waveform_dataset_path"] = str(tmp_path / "wfd.hdf5")
    train_settings["training"]["stage_0"]["asd_dataset_path"] = asd_path
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    complete, _, epoch = run_training(
        train_settings, dict(LOCAL_SETTINGS), str(train_dir), None, False
    )
    assert complete and epoch == 1
    return str(train_dir / "model_latest.pt"), {
        "asd": asd_path,
        "wfd": str(tmp_path / "wfd.hdf5"),
    }


def test_trained_model_records_bns_settings(trained_bns_model):
    model_path, _ = trained_bns_model
    model = build_model_from_kwargs(
        filename=model_path, device="cpu", load_training_info=False
    )
    data_settings = model.metadata["train_settings"]["data"]
    assert data_settings["context_parameters"] == ["ra", "dec", "chirp_mass_proxy"]
    assert (
        data_settings["gnpe_chirp"]["kernel"]
        == TRAIN_SETTINGS["data"]["gnpe_chirp"]["kernel"]
    )
    assert set(data_settings["standardization"]["mean"]) == set(
        data_settings["inference_parameters"] + data_settings["context_parameters"]
    )
    assert "phase_heterodyning" in model.metadata["dataset_settings"]["compression"]
    assert (
        model.metadata["dataset_settings"]["domain"]["type"]
        == "MultibandedFrequencyDomain"
    )


def test_training_input_matches_inference_preparation(trained_bns_model):
    """For one signal and proxy, the network input built by the training transforms
    (pointwise on the multibanded domain) agrees with the sampler context's
    preparation of base-domain data (heterodyne, decimate, whiten) up to the
    decimation error. The training side uses exact polarizations rather than the
    SVD-compressed dataset, so that only the transform chains are compared."""
    model_path, paths = trained_bns_model
    model = build_model_from_kwargs(
        filename=model_path, device="cpu", load_training_info=False
    )
    data_settings = copy.deepcopy(model.metadata["train_settings"]["data"])
    extrinsic = {
        "ra": 1.0,
        "dec": 0.3,
        "geocent_time": 0.005,
        "psi": 0.4,
        "luminosity_distance": 40.0,
    }
    data_settings["extrinsic_prior"] = extrinsic
    data_settings["zero_noise"] = True

    dataset_settings = model.metadata["dataset_settings"]
    mfd = build_domain(dataset_settings["domain"])
    (
        mfd.update(data_settings["domain_update"])
        if "domain_update" in data_settings
        else None
    )
    wfg = WaveformGenerator(domain=mfd, **dataset_settings["waveform_generator"])
    intrinsic = {
        k: float(v)
        for k, v in build_prior_with_defaults(dataset_settings["intrinsic_prior"])
        .sample()
        .items()
    }
    wfd = WaveformDataset(
        dictionary={
            "settings": {"domain": mfd.domain_dict},
            "parameters": pd.DataFrame([intrinsic]),
            "polarizations": {
                k: v[None] for k, v in wfg.generate_hplus_hcross(intrinsic).items()
            },
        }
    )
    # Repackaging into an array is omitted (the dataset yields nested dicts) and
    # applied by hand below.
    set_train_transforms(
        wfd,
        data_settings,
        paths["asd"],
        omit_transforms=[
            AddWhiteNoiseComplex,
            RepackageStrainsAndASDS,
            SelectStandardizeRepackageParameters,
            UnpackDict,
        ],
        print_output=False,
    )
    sample = wfd[0]
    proxy = sample["extrinsic_parameters"]["chirp_mass_proxy"]
    training_input = RepackageStrainsAndASDS(DETECTORS, first_index=mfd.min_idx)(
        sample
    )["waveform"]

    injection_generator = Injection.from_posterior_model_metadata(model.metadata)
    injection_generator.use_base_domain = (
        True  # As in dingo_pipe: data on the base domain.
    )
    injection_generator.asd = ASDDataset(file_name=paths["asd"], ifos=DETECTORS)
    signal = injection_generator.signal({**intrinsic, **extrinsic})
    context = GWSamplerContext.from_model(
        model, {"waveform": signal["waveform"], "asds": signal["asds"]}
    )
    inference_input = context.prepared_data(
        {"chirp_mass_proxy": torch.tensor([proxy], dtype=torch.float64)}
    )[0].numpy()

    assert inference_input.shape == training_input.shape
    for i in range(len(DETECTORS)):
        a = training_input[i, 0] + 1j * training_input[i, 1]
        b = inference_input[i, 0] + 1j * inference_input[i, 1]
        mismatch = 1 - np.abs(np.vdot(a, b)) / np.sqrt(
            np.vdot(a, a).real * np.vdot(b, b).real
        )
        assert mismatch < 1e-2, mismatch
        np.testing.assert_allclose(
            inference_input[i, 2], training_input[i, 2], rtol=1e-3
        )


def test_inference_and_importance_sampling_on_injection(trained_bns_model):
    model_path, paths = trained_bns_model
    model = build_model_from_kwargs(
        filename=model_path, device="cpu", load_training_info=False
    )
    injection_generator = Injection.from_posterior_model_metadata(model.metadata)
    injection_generator.use_base_domain = True
    injection_generator.asd = ASDDataset(file_name=paths["asd"], ifos=DETECTORS)
    theta = injection_generator.prior.sample()
    theta = {k: float(v) for k, v in theta.items()}
    data = injection_generator.injection(theta)

    pins = {
        "chirp_mass_proxy": theta["chirp_mass"] + 0.3 * KERNEL_HALF_WIDTH,
        "ra": theta["ra"],
        "dec": theta["dec"],
    }
    sampler = GWComposedSampler.from_model(model, data, fixed_context_parameters=pins)
    sampler.run_sampler(num_samples=64, batch_size=64)
    result = sampler.to_result()
    samples = result.samples
    assert len(samples) == 64
    assert np.isfinite(samples["log_prob"]).all()
    # The chain reconstructs chirp_mass from the inferred offset and the pinned proxy.
    assert {"chirp_mass", "chirp_mass_proxy", "ra", "dec"} <= set(samples.columns)
    assert np.isfinite(samples["chirp_mass"]).all()
    np.testing.assert_allclose(samples["chirp_mass_proxy"], pins["chirp_mass_proxy"])
    assert (samples["ra"] == theta["ra"]).all()
    assert "phase" not in samples  # phase-marginalized network

    result.sample_synthetic_phase(
        {"co_rotate_spins": True, "n_grid": 101, "uniform_weight": 0.01}
    )
    result.importance_sample(num_processes=1)
    assert np.isfinite(result.samples["weights"]).all()
    assert result.samples["weights"].sum() > 0
    assert np.isfinite(result.log_evidence)
