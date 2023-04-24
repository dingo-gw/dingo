from dingo.gw.domains import MultibandedFrequencyDomain, build_domain
import numpy as np
import pytest


@pytest.fixture
def MFD_setup():
    base_domain = {
        "delta_f": 0.0078125,
        "f_max": 2048,
        "f_min": 20,
        "type": "FrequencyDomain",
        "window_factor": None,
    }
    delta_f_initial = 0.0078125
    nodes = [
        20.0,
        23.1875,
        30.734375,
        40.484375,
        52.359375,
        66.234375,
        82.234375,
        323.734375,
        665.734375,
        2047.734375,
    ]
    return MultibandedFrequencyDomain(nodes, delta_f_initial, base_domain)


def test_MFD_domain_update(MFD_setup):
    mfd = MFD_setup

    # check successful updates for valid settings
    for new_settings in [
        {"f_min": 21, "f_max": 1500},  # same number of bands
        {"f_min": 35, "f_max": 1500},  # truncate two bands from lower end
        {"f_min": 21, "f_max": 200},  # truncate two bands from upper end
        {"f_min": 21},  # no f_max, same number of bands
        {"f_min": 50},  # no f_min, truncate three bands from lower end
        {"f_max": 1500},  # no f_min, same number of bands
        {"f_max": 200},  # no f_min, truncate two bands from upper end
        {"f_max": 20 + mfd._delta_f_bands[0] + 1e-5},  # domain with only 1 bin
    ]:
        mfd_updated = build_domain(mfd.domain_dict)
        mfd_updated.update(new_settings)
        mfd_updated.update_data(mfd())
        assert (mfd_updated.update_data(mfd()) == mfd_updated()).all()
        assert ((mfd_updated.update_data(mfd_updated())) == mfd_updated()).all()
        assert len(mfd_updated.update_data(mfd())) == len(mfd_updated)
        with pytest.raises(ValueError):
            mfd_updated.update_data(np.ones(len(mfd_updated) + 1))

    # check that ValueErrors for invalid settings
    for new_settings in [
        {"f_min": 2049},  # f_min larger than maximum frequency
        {"f_max": 19},  # f_max smaller than minimum frequency
        {"f_max": 20 + mfd._delta_f_bands[0] - 1e-5},  # f_max too small
        {"f_min": 30, "f_max": 29},  # f_min > f_max
        {"f_min": 29, "f_max": 29},  # f_min == f_max
    ]:
        mfd_updated = build_domain(mfd.domain_dict)
        with pytest.raises(ValueError):
            mfd_updated.update(new_settings)

    # check that we can't update twice
    mfd_updated = build_domain(mfd.domain_dict)
    mfd_updated.update({"f_min": 30, "f_max": 40})
    with pytest.raises(ValueError):
        mfd_updated.update({"f_min": 32, "f_max": 34})
