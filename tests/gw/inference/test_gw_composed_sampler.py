"""Unit tests for ``GWComposedSampler`` assembly and export.

``test_bns_chain_builder`` covers the pinned (DINGO-BNS) ``from_model`` chain; this file
covers the two GW-builder concerns the branch's tests otherwise leave untested:

* the *plain-NPE* ``from_model`` chain, including the ``DeltaFactor`` that fills the
  network-independent delta-prior parameters (an aligned spin fixed to 0), and
* ``to_result`` / ``to_hdf5`` -- exporting the sampled DataFrame, the raw event data, and
  the chain provenance to a gw ``Result`` (and back through HDF5).

Both use metadata-only stubs and a network-free ``DeltaFactor`` chain, so no waveform
generation or trained network is needed.

(Adapts the ``run_sampler`` export coverage and the fixed-prior post-processing check
from the pre-sampler-revamp ``test_samplers.py`` / ``test_gw_samplers.py``, which
imported the now-deleted ``dingo.core.samplers`` and ``dingo.gw.inference.gw_samplers``.)
"""

import numpy as np
import pytest

from dingo.core.factors import ChainComposer, DeltaFactor
from dingo.gw.inference.context import GWSamplerContext
from dingo.gw.inference.sampler import GWComposedSampler
from dingo.gw.result import Result


DETECTORS = ["H1", "L1"]
REF_TIME = 1126259462.4
INFERENCE_PARAMETERS = ["chirp_mass", "mass_ratio", "ra", "dec"]

DOMAIN_SETTINGS = {
    "type": "UniformFrequencyDomain",
    "f_min": 20.0,
    "f_max": 256.0,  # small (T=2s) so any data view is cheap
    "delta_f": 0.5,
}

STANDARDIZATION = {
    "mean": {p: 0.0 for p in INFERENCE_PARAMETERS},
    "std": {p: 1.0 for p in INFERENCE_PARAMETERS},
}

INTRINSIC_PRIOR = {
    "mass_1": "bilby.core.prior.Constraint(minimum=10, maximum=80)",
    "mass_2": "bilby.core.prior.Constraint(minimum=10, maximum=80)",
    "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25, maximum=31)",
    "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio("
    "minimum=0.125, maximum=1)",
    "phase": "default",
    # Aligned spins pinned to 0: delta-prior parameters the network does not infer,
    # so the chain fills them with a DeltaFactor.
    "a_1": 0.0,
    "a_2": 0.0,
}
EXTRINSIC_PRIOR = {
    "dec": "default",
    "ra": "default",
    "geocent_time": "default",
    "luminosity_distance": "default",
    "psi": "default",
}


def _metadata():
    return {
        "dataset_settings": {
            "domain": DOMAIN_SETTINGS,
            "waveform_generator": {"approximant": "IMRPhenomD", "f_ref": 20.0},
            "intrinsic_prior": INTRINSIC_PRIOR,
        },
        "train_settings": {
            "data": {
                "detectors": DETECTORS,
                "ref_time": REF_TIME,
                "extrinsic_prior": EXTRINSIC_PRIOR,
                "inference_parameters": INFERENCE_PARAMETERS,
                "standardization": STANDARDIZATION,
            }
        },
    }


class _StubModel:
    metadata = _metadata()
    device = "cpu"


def _event_data():
    n_bins = int(DOMAIN_SETTINGS["f_max"] / DOMAIN_SETTINGS["delta_f"]) + 1
    strain = np.ones(n_bins, dtype=complex)
    asd = np.ones(n_bins)
    return {
        "waveform": {d: strain.copy() for d in DETECTORS},
        "asds": {d: asd.copy() for d in DETECTORS},
    }


# ---------------------------------------------------------------------------
# from_model: plain-NPE chain assembly + delta-prior filling
# ---------------------------------------------------------------------------


def test_from_model_plain_npe_chain_fills_delta_prior_parameters():
    sampler = GWComposedSampler.from_model(_StubModel(), _event_data())
    kinds = [type(s).__name__ for s in sampler.composer.steps]
    # The flow (exposing ra as ra@t_ref), the RA rotation to the event frame, and a
    # trailing DeltaFactor for the network-independent delta-prior parameters.
    assert kinds == ["FlowFactor", "RAToEventFrame", "DeltaFactor"]

    delta = sampler.composer.steps[-1]
    # a_1 and a_2 are DeltaFunctions (0.0) in the intrinsic prior and are not inference
    # parameters, so they are pinned as fixed constants; chirp_mass etc. are not.
    assert set(delta.parameters) == {"a_1", "a_2"}
    assert delta.values == {"a_1": 0.0, "a_2": 0.0}


# ---------------------------------------------------------------------------
# to_result / to_hdf5: export round trips
# ---------------------------------------------------------------------------


def _network_free_sampler():
    """A GWComposedSampler whose chain pins every inference parameter, so run_sampler
    produces a DataFrame with no network involved."""
    context = GWSamplerContext.from_model_metadata(_metadata(), _event_data())
    pins = {"chirp_mass": 30.0, "mass_ratio": 0.8, "ra": 1.0, "dec": 0.1}
    composer = ChainComposer([DeltaFactor(pins)])
    return GWComposedSampler(composer, context, _metadata(), INFERENCE_PARAMETERS)


def test_to_result_round_trips_samples_and_provenance():
    sampler = _network_free_sampler()
    sampler.run_sampler(num_samples=10)

    result = sampler.to_result()
    assert isinstance(result, Result)
    # The sampled DataFrame is exported unchanged.
    assert len(result.samples) == 10
    assert list(result.samples.columns) == list(sampler.samples.columns)
    np.testing.assert_allclose(
        result.samples[INFERENCE_PARAMETERS].to_numpy(),
        sampler.samples[INFERENCE_PARAMETERS].to_numpy(),
    )
    # Model metadata is carried through, and the chain provenance is recorded.
    assert result.settings["train_settings"] == sampler.metadata["train_settings"]
    provenance = result.settings["sampler"]
    assert provenance["implementation"] == "composed"
    assert [step["step"] for step in provenance["chain"]] == ["DeltaFactor"]


def test_to_hdf5_round_trips_samples(tmp_path):
    sampler = _network_free_sampler()
    sampler.run_sampler(num_samples=10)
    sampler.to_hdf5(label="result", outdir=str(tmp_path))

    reloaded = Result(file_name=str(tmp_path / "result.hdf5"))
    assert len(reloaded.samples) == 10
    np.testing.assert_allclose(
        reloaded.samples[INFERENCE_PARAMETERS].to_numpy(),
        sampler.samples[INFERENCE_PARAMETERS].to_numpy(),
    )
