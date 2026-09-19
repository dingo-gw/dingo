"""Phase-heterodyned dataset compression (DINGO-BNS): the compression transforms act on
sample dicts so that HeterodynePhase, which needs the chirp mass, can be composed with
whitening and SVD; decompression inverts the chain."""

import copy

import numpy as np
import pytest
import yaml

from dingo.gw.dataset import generate_dataset
from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.SVD import ApplySVD, SVDBasis
from dingo.gw.transforms import HeterodynePhase, WhitenFixedASD
from dingo.gw.waveform_generator import WaveformGenerator

SETTINGS = yaml.safe_load(
    """
domain:
  type: UniformFrequencyDomain
  f_min: 20.0
  f_max: 256.0
  delta_f: 0.25
waveform_generator:
  approximant: IMRPhenomXP_NRTidalv3
  f_ref: 20.0
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
num_samples: 6
compression:
  whitening: aLIGO_ZERO_DET_high_P_asd.txt
  phase_heterodyning:
    order: 0
"""
)


def _raw_polarizations(wfd):
    """Uncompressed polarizations regenerated from the dataset's parameters, zeroed
    below f_min as the dataset does on construction."""
    wfg = WaveformGenerator(domain=wfd.domain, **wfd.settings["waveform_generator"])
    return [
        {
            k: wfd.domain.update_data(v)
            for k, v in wfg.generate_hplus_hcross(dict(row)).items()
        }
        for _, row in wfd.parameters.iterrows()
    ]


def test_generator_transform_receives_sample_dict():
    """The waveform generator hands its transform the polarizations together with the
    parameters, and returns the transformed polarizations."""
    from dingo.gw.domains import build_domain

    wfg = WaveformGenerator(
        domain=build_domain(SETTINGS["domain"]), **SETTINGS["waveform_generator"]
    )
    parameters = {
        "chirp_mass": 1.3,
        "mass_ratio": 0.9,
        "a_1": 0.0,
        "a_2": 0.0,
        "tilt_1": 0.0,
        "tilt_2": 0.0,
        "phi_12": 0.0,
        "phi_jl": 0.0,
        "theta_jn": 0.5,
        "phase": 0.0,
        "lambda_1": 300.0,
        "lambda_2": 400.0,
        "luminosity_distance": 100.0,
        "geocent_time": 0.0,
    }
    raw = wfg.generate_hplus_hcross(parameters)
    received = {}

    def transform(sample):
        received.update(sample)
        return {**sample, "waveform": {k: 2 * v for k, v in sample["waveform"].items()}}

    wfg.transform = transform
    doubled = wfg.generate_hplus_hcross(parameters)
    assert set(received) == {"waveform", "parameters"}
    assert received["parameters"]["chirp_mass"] == parameters["chirp_mass"]
    for k in raw:
        np.testing.assert_array_equal(received["waveform"][k], raw[k])
        np.testing.assert_array_equal(doubled[k], 2 * raw[k])


def test_heterodyned_compression_roundtrip(tmp_path):
    """Stored polarizations are whitened and heterodyned at each sample's own chirp
    mass; decompression (also from file) restores the raw waveforms."""
    wfd = generate_dataset(copy.deepcopy(SETTINGS), num_processes=1)
    wfd.to_file(tmp_path / "wfd.hdf5")
    raw = _raw_polarizations(wfd)
    assert np.isfinite(raw[0]["h_plus"]).all()

    whiten = WhitenFixedASD(wfd.domain, asd_file=SETTINGS["compression"]["whitening"])
    heterodyne = HeterodynePhase(wfd.domain, order=0)
    for i, pols in enumerate(raw):
        expected = heterodyne(
            whiten({"waveform": pols, "parameters": wfd.parameters.iloc[i].to_dict()})
        )["waveform"]
        for k in pols:
            np.testing.assert_allclose(wfd.polarizations[k][i], expected[k], rtol=1e-10)

    for dataset in [wfd, WaveformDataset(file_name=str(tmp_path / "wfd.hdf5"))]:
        assert [type(t) for t in dataset.decompression_transform.transforms] == [
            HeterodynePhase,
            WhitenFixedASD,
        ]
        for i, pols in enumerate(raw):
            sample = dataset[i]
            for k in pols:
                np.testing.assert_allclose(sample["waveform"][k], pols[k], rtol=1e-10)
            assert sample["parameters"].keys() == set(dataset.parameters.columns)


def test_heterodyned_compression_with_svd(tmp_path):
    """The SVD basis is built on the heterodyned, whitened waveforms and decompression
    inverts the full chain."""
    settings = copy.deepcopy(SETTINGS)
    settings["compression"]["svd"] = {
        "size": 8,
        "num_training_samples": 6,
        "num_validation_samples": 2,
    }
    wfd = generate_dataset(settings, num_processes=1)
    assert [type(t) for t in wfd.decompression_transform.transforms] == [
        ApplySVD,
        HeterodynePhase,
        WhitenFixedASD,
    ]
    assert wfd.polarizations["h_plus"].shape == (6, 8)
    # The SVD truncation is the only loss: decompressing equals un-heterodyning and
    # un-whitening the SVD projection of the compressed waveform.
    basis = SVDBasis(dictionary=wfd.svd)
    whiten = WhitenFixedASD(wfd.domain, asd_file=SETTINGS["compression"]["whitening"])
    unwhiten = WhitenFixedASD(
        wfd.domain, asd_file=SETTINGS["compression"]["whitening"], inverse=True
    )
    heterodyne = HeterodynePhase(wfd.domain, order=0)
    unheterodyne = HeterodynePhase(wfd.domain, order=0, inverse=True)
    for i, pols in enumerate(_raw_polarizations(wfd)):
        compressed = heterodyne(
            whiten({"waveform": pols, "parameters": wfd.parameters.iloc[i].to_dict()})
        )
        compressed["waveform"] = {
            k: basis.decompress(basis.compress(v))
            for k, v in compressed["waveform"].items()
        }
        expected = unwhiten(unheterodyne(compressed))["waveform"]
        sample = wfd[i]
        for k in pols:
            np.testing.assert_allclose(sample["waveform"][k], expected[k], rtol=1e-8)

    wfd.to_file(tmp_path / "wfd.hdf5")
    single = WaveformDataset(file_name=str(tmp_path / "wfd.hdf5"), precision="single")
    assert single[0]["waveform"]["h_plus"].dtype == np.complex64


def test_heterodyning_requires_generation_on_dataset_domain():
    """Approximants that are generated on the base domain and decimated afterwards
    cannot be heterodyned as part of the compression (decimation must come last)."""
    settings = copy.deepcopy(SETTINGS)
    settings["domain"] = {
        "type": "MultibandedFrequencyDomain",
        "nodes": [20.0, 64.0, 256.0],
        "delta_f_initial": 0.25,
        "base_domain": SETTINGS["domain"],
    }
    settings["waveform_generator"]["approximant"] = "SEOBNRv4"
    with pytest.raises(NotImplementedError, match="decimated afterwards"):
        generate_dataset(settings, num_processes=1)
