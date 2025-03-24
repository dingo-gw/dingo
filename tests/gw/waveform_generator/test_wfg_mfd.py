import numpy as np
import pytest
from scipy.interpolate import interp1d

from dingo.gw.domains import MultibandedFrequencyDomain
from dingo.gw.gwutils import get_mismatch
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.waveform_generator import WaveformGenerator, NewInterfaceWaveformGenerator


@pytest.fixture
def mfd():
    domain_settings = {
        "nodes": [20.0, 26.0, 34.0, 46.0, 62.0, 78.0, 1038.0],
        "delta_f_initial": 0.0625,
        "base_domain": {
            "type": "UniformFrequencyDomain",
            "f_min": 20.0,
            "f_max": 1037.9375,
            "delta_f": 0.0625,
        },
    }
    domain = MultibandedFrequencyDomain(**domain_settings)
    return domain


@pytest.fixture(params=["IMRPhenomXPHM", "SEOBNRv4PHM", "SEOBNRv5PHM", "SEOBNRv5HM"])
def approximant(request):
    return request.param


@pytest.fixture
def intrinsic_prior(approximant):
    intrinsic_dict = {
        "mass_1": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)",
        "mass_2": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)",
        "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0)",
        "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25.0, maximum=100.0)",
        "luminosity_distance": 1000.0,
        "theta_jn": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
        "phase": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
        "a_1": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
        "a_2": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
        "tilt_1": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
        "tilt_2": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
        "phi_12": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
        "phi_jl": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
        "geocent_time": "bilby.core.prior.Uniform(minimum=-0.1, maximum=0.1)",
    }
    prior = build_prior_with_defaults(intrinsic_dict)
    return prior


@pytest.fixture
def wfg_mfd(mfd, approximant):
    if approximant in ["SEOBNRv5PHM", "SEOBNRv5HM"]:
        wfg_class = NewInterfaceWaveformGenerator
    else:
        wfg_class = WaveformGenerator
    return wfg_class(
        approximant=approximant,
        domain=mfd,
        f_ref=10.0,
        f_start=10.0,
        spin_conversion_phase=0.0,
    )


@pytest.fixture
def wfg_ufd(mfd, approximant):
    if approximant in ["SEOBNRv5PHM", "SEOBNRv5HM"]:
        wfg_class = NewInterfaceWaveformGenerator
    else:
        wfg_class = WaveformGenerator
    return wfg_class(
        approximant=approximant,
        domain=mfd.base_domain,
        f_ref=10.0,
        f_start=10.0,
        spin_conversion_phase=0.0,
    )


@pytest.fixture
def num_evaluations(approximant):
    if approximant == "SEOBNRv4PHM":
        return 1
    else:
        return 10


@pytest.fixture
def tolerances(approximant):
    # Return max mismatches in MFD, UFD.
    if approximant == "IMRPhenomXPHM":
        return 1e-4, 1e-3
    else:
        return 1e-9, 1e-3


# Uncomment to test only one approximant.
try:
    import pyseobnr

    approximant_list = ["IMRPhenomXPHM", "SEOBNRv4PHM", "SEOBNRv5PHM"]
except ImportError:
    approximant_list = ["IMRPhenomXPHM", "SEOBNRv4PHM"]


@pytest.mark.parametrize("approximant", approximant_list)
def test_decimation(
    intrinsic_prior, wfg_mfd, wfg_ufd, mfd, num_evaluations, tolerances
):
    mismatches_mfd = []
    mismatches_ufd = []
    for _ in range(num_evaluations):
        p = intrinsic_prior.sample()

        wf_mfd = wfg_mfd.generate_hplus_hcross(p)
        wf_ufd = wfg_ufd.generate_hplus_hcross(p)

        # Compare UFD waveforms decimated to MFD against waveforms generated in MFD.
        # FD-native waveforms (e.g., Phenom) can be generated directly in MFD, so this
        # comparison is non-exact. TD-native waveforms (e.g., EOB) are first generated
        # in FD and decimated to MFD within the WaveformGenerator class, so this should
        # be exact.

        wf_ufd_decimated = {k: mfd.decimate(v) for k, v in wf_ufd.items()}

        mismatches_mfd.append(
            [
                get_mismatch(
                    wf_mfd[pol],
                    wf_ufd_decimated[pol],
                    mfd,
                    asd_file="aLIGO_ZERO_DET_high_P_asd.txt",
                )
                for pol in ["h_plus", "h_cross"]
            ]
        )

        # Also compare UFD waveforms against MFD waveforms interpolated to UFD. This
        # comparison is always going to be non-exact.

        ufd = mfd.base_domain
        wf_mfd_interpolated = {
            k: ufd.update_data(interp1d(mfd(), v, fill_value="extrapolate")(ufd()))
            for k, v in wf_mfd.items()
        }

        mismatches_ufd.append(
            [
                get_mismatch(
                    wf_ufd[pol],
                    wf_mfd_interpolated[pol],
                    ufd,
                    asd_file="aLIGO_ZERO_DET_high_P_asd.txt",
                )
                for pol in ["h_plus", "h_cross"]
            ]
        )

    mismatches_mfd = np.array(mismatches_mfd)
    mismatches_ufd = np.array(mismatches_ufd)

    assert np.max(mismatches_mfd) < tolerances[0]
    assert np.max(mismatches_ufd) < tolerances[1]


@pytest.mark.parametrize("approximant", approximant_list)
def test_decimation_m(
    intrinsic_prior, wfg_mfd, wfg_ufd, mfd, num_evaluations, tolerances
):
    mismatches_mfd = []
    mismatches_ufd = []
    for _ in range(num_evaluations):
        p = intrinsic_prior.sample()

        wf_mfd = wfg_mfd.generate_hplus_hcross_m(p)
        wf_ufd = wfg_ufd.generate_hplus_hcross_m(p)
        modes = wf_mfd.keys()

        # Compare UFD waveforms decimated to MFD against waveforms generated in MFD.
        # FD-native waveforms (e.g., Phenom) can be generated directly in MFD, so this
        # comparison is non-exact. TD-native waveforms (e.g., EOB) are first generated
        # in FD and decimated to MFD within the WaveformGenerator class, so this should
        # be exact.

        wf_ufd_decimated = {
            k: {k2: mfd.decimate(v2) for k2, v2 in v.items()} for k, v in wf_ufd.items()
        }

        mismatches_mfd.append(
            [
                [
                    get_mismatch(
                        wf_mfd[m][pol],
                        wf_ufd_decimated[m][pol],
                        mfd,
                        asd_file="aLIGO_ZERO_DET_high_P_asd.txt",
                    )
                    for pol in ["h_plus", "h_cross"]
                ]
                for m in modes
            ]
        )

        # Also compare UFD waveforms against MFD waveforms interpolated to UFD. This
        # comparison is always going to be non-exact.

        ufd = mfd.base_domain
        wf_mfd_interpolated = {
            k: {
                k2: ufd.update_data(
                    interp1d(mfd(), v2, fill_value="extrapolate")(ufd())
                )
                for k2, v2 in v.items()
            }
            for k, v in wf_mfd.items()
        }

        mismatches_ufd.append(
            [
                [
                    get_mismatch(
                        wf_ufd[m][pol],
                        wf_mfd_interpolated[m][pol],
                        ufd,
                        asd_file="aLIGO_ZERO_DET_high_P_asd.txt",
                    )
                    for pol in ["h_plus", "h_cross"]
                ]
                for m in modes
            ]
        )

    mismatches_mfd = np.array(mismatches_mfd)
    mismatches_ufd = np.array(mismatches_ufd)

    assert np.max(mismatches_mfd) < tolerances[0]

    # Some of the negative m modes do not do well, so we exclude by taking the median.
    assert np.median(mismatches_ufd) < 10 * tolerances[1]
