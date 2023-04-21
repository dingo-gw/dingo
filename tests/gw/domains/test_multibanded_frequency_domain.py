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

    mfd_updated = build_domain(mfd.domain_dict)
    mfd_updated.update({"f_min": 23, "f_max": 350})
    mfd_updated.update_data(mfd())
    assert (mfd_updated.update_data(mfd()) == mfd_updated()).all()
    assert ((mfd_updated.update_data(mfd_updated())) == mfd_updated()).all()
    with pytest.raises(ValueError):
        mfd_updated.update(np.ones(len(mfd_updated) + 1))

    mfd_updated = build_domain(mfd.domain_dict)
    mfd_updated.update({"f_min": 50})
    mfd_updated.update_data(mfd())
    assert (mfd_updated.update_data(mfd()) == mfd_updated()).all()

    mfd_updated = build_domain(mfd.domain_dict)
    mfd_updated.update({"f_max": 100})
    mfd_updated.update_data(mfd())
    assert (mfd_updated.update_data(mfd()) == mfd_updated()).all()

    mfd_updated = build_domain(mfd.domain_dict)
    with pytest.raises(ValueError):
        mfd_updated.update({"f_max": 19})

    mfd_updated = build_domain(mfd.domain_dict)
    with pytest.raises(ValueError):
        mfd_updated.update({"f_min": 30, "f_max": 29})

    mfd_updated = build_domain(mfd.domain_dict)
    mfd_updated.update({"f_min": 30, "f_max": 40})
    with pytest.raises(ValueError):
        mfd_updated.update({"f_min": 32, "f_max": 34})
