import pytest
import os.path
import numpy as np
from bilby.gw.detector import InterferometerList

from dingo.gw.transforms import (
    GetDetectorTimes,
    ProjectOntoDetectors,
    SampleExtrinsicParameters,
    time_delay_from_geocenter,
)
from dingo.gw.prior import default_extrinsic_dict
from dingo.gw.domains import build_domain


@pytest.fixture
def reference_data_research_code():
    dir = os.path.dirname(os.path.realpath(__file__))
    ref_data = np.load(os.path.join(dir, "waveform_data.npy"), allow_pickle=True).item()
    sample_in = {
        "parameters": ref_data["intrinsic_parameters"],
        "waveform": {"h_cross": ref_data["hc"], "h_plus": ref_data["hp"]},
        "extrinsic_parameters": ref_data["extrinsic_parameters"],
    }
    parameters_ref = ref_data["all_parameters"]
    h_ref = ref_data["h_d_unwhitened"]
    return sample_in, parameters_ref, h_ref


@pytest.fixture
def setup_detector_projection():
    # setup arguments
    extrinsic_prior_dict = default_extrinsic_dict
    ref_time = 1126259462.391
    domain_dict = {
        "type": "UniformFrequencyDomain",
        "f_min": 10.0,
        "f_max": 1024.0,
        "delta_f": 0.125,
    }
    ifo_list = InterferometerList(["H1", "L1"])
    domain = build_domain(domain_dict)

    # build transformations
    sample_extrinsic_parameters = SampleExtrinsicParameters(extrinsic_prior_dict)
    get_detector_times = GetDetectorTimes(ifo_list, ref_time)
    project_onto_detectors = ProjectOntoDetectors(ifo_list, domain, ref_time)

    return sample_extrinsic_parameters, get_detector_times, project_onto_detectors


def test_detector_projection_against_research_code(
    reference_data_research_code, setup_detector_projection
):
    sample_in, parameters_ref, h_ref = reference_data_research_code
    _, get_detector_times, project_onto_detector = setup_detector_projection

    sample_out = get_detector_times(sample_in)
    sample_out["extrinsic_parameters"]["H1_time"] = parameters_ref["H1_time"]
    sample_out["extrinsic_parameters"]["L1_time"] = parameters_ref["L1_time"]
    sample_out = project_onto_detector(sample_out)

    for ifo_name in ["H1", "L1"]:
        strain = sample_out["waveform"][ifo_name]
        strain_ref = h_ref[ifo_name]
        deviation = np.abs(strain_ref - strain)
        assert np.max(deviation) / np.max(np.abs(strain)) < 5e-2


def test_project_onto_detectors_skip_time_shift(
    reference_data_research_code, setup_detector_projection
):
    """
    With apply_time_shift=False, ProjectOntoDetectors must:
      * leave the strain at t=0 (i.e., not apply the per-detector time shift),
      * still populate <ifo>_time in sample['parameters'] for downstream use.
    Concretely, applying the inverse time shift to the apply_time_shift=True
    output must yield the apply_time_shift=False output.
    """
    sample_in, parameters_ref, _ = reference_data_research_code
    _, get_detector_times, project_with_shift = setup_detector_projection
    ifo_list = project_with_shift.ifo_list
    domain = project_with_shift.domain
    ref_time = project_with_shift.ref_time

    project_no_shift = ProjectOntoDetectors(
        ifo_list, domain, ref_time, apply_time_shift=False
    )

    def _prep(sample):
        s = get_detector_times(sample)
        # The fixture's <ifo>_time values are precomputed; copy them in to match
        # the existing reference test.
        s["extrinsic_parameters"]["H1_time"] = parameters_ref["H1_time"]
        s["extrinsic_parameters"]["L1_time"] = parameters_ref["L1_time"]
        return s

    out_shift = project_with_shift(_prep(dict(sample_in)))
    out_no_shift = project_no_shift(_prep(dict(sample_in)))

    for ifo in ifo_list:
        # <ifo>_time still populated in parameters (independent of the flag).
        assert f"{ifo.name}_time" in out_no_shift["parameters"]

        ifo_time = out_no_shift["parameters"][f"{ifo.name}_time"]
        # Re-applying the forward shift to the unshifted strain must recover the
        # shifted strain (up to FFT round-trip precision).
        recovered = domain.time_translate_data(
            out_no_shift["waveform"][ifo.name], ifo_time
        )
        shifted = out_shift["waveform"][ifo.name]
        rel = np.max(np.abs(recovered - shifted)) / np.max(np.abs(shifted))
        assert rel < 1e-5, f"{ifo.name}: round-trip residual {rel:.2e} too large"


def test_time_delay_from_geocenter():
    ifo_list = InterferometerList(["H1", "L1", "V1"])
    for ifo in ifo_list:
        ra = np.random.uniform(0, 2 * np.pi, size=2)
        dec = np.random.uniform(-np.pi / 2, np.pi / 2, size=2)
        ref_time = 1126259462.4
        td = time_delay_from_geocenter(ifo, ra, dec, ref_time)
        # The vectorized version of time_delay_from_geocenter uses our own
        # implementation, whereas for floats it uses bilby. Below, we iterate through
        # the individual elements of the arrays, and compare the output of our
        # vectorized implementation to bilby outputs.
        for ra_idx, dec_idx, td_idx in zip(ra, dec, td):
            assert np.allclose(
                td_idx, time_delay_from_geocenter(ifo, ra_idx, dec_idx, ref_time)
            )
