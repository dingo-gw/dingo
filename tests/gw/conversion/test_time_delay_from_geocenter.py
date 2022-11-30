import pytest
import random
import numpy as np
import torch
from bilby.gw.detector import InterferometerList

from dingo.gw.transforms import time_delay_from_geocenter
from dingo.gw.prior import BBHExtrinsicPriorDict, default_extrinsic_dict


@pytest.fixture
def ifo_list_and_prior_and_tref():
    ifo_list = InterferometerList(["H1", "L1", "V1"])
    prior = BBHExtrinsicPriorDict({k: default_extrinsic_dict[k] for k in ["ra", "dec"]})
    tref = random.randint(
        946339215, 1577491218
    )  # random gps time between 2010 and 2030
    return ifo_list, prior, tref


def test_time_delay_from_geocenter_errors(ifo_list_and_prior_and_tref):
    ifo_list, prior, tref = ifo_list_and_prior_and_tref
    params = prior.sample(1000)
    with pytest.raises(ValueError):
        ra = params["ra"]
        dec = torch.Tensor(params["dec"].item())
        time_delay_from_geocenter(ifo_list[0], ra, dec, tref)
    with pytest.raises(ValueError):
        ra = params["ra"]
        dec = params["dec"][:100]
        time_delay_from_geocenter(ifo_list[0], ra, dec, tref)
    with pytest.raises(ValueError):
        ra = params["ra"][None,:]
        dec = params["dec"][None,:]
        time_delay_from_geocenter(ifo_list[0], ra, dec, tref)


def test_time_delay_from_geocenter_float(ifo_list_and_prior_and_tref):
    ifo_list, prior, tref = ifo_list_and_prior_and_tref
    params = prior.sample(1)
    ra = params["ra"].item()
    dec = params["dec"].item()
    for ifo in ifo_list:
        dt0 = ifo.time_delay_from_geocenter(ra, dec, tref)
        dt1 = time_delay_from_geocenter(ifo, ra, dec, tref)
        assert np.allclose(dt0, dt1, atol=1e-10)


def test_time_delay_from_geocenter_array(ifo_list_and_prior_and_tref):
    ifo_list, prior, tref = ifo_list_and_prior_and_tref
    params = prior.sample(1000)
    # check that our vectorized function returns the same result as the bilby method
    ra = params["ra"]
    dec = params["dec"]
    for ifo in ifo_list:
        dt0 = [
            ifo.time_delay_from_geocenter(ra_idx, dec_idx, tref)
            for ra_idx, dec_idx in zip(ra, dec)
        ]
        dt0 = np.array(dt0)
        dt1 = time_delay_from_geocenter(ifo, ra, dec, tref)
        assert np.allclose(dt0, dt1, atol=1e-10)
    # check that this also works for arrays of length 1, for which the bilby method is
    # called since it is faster
    ra = params["ra"][:1]
    dec = params["dec"][:1]
    for ifo in ifo_list:
        dt0 = [
            ifo.time_delay_from_geocenter(ra_idx, dec_idx, tref)
            for ra_idx, dec_idx in zip(ra, dec)
        ]
        dt0 = np.array(dt0)
        dt1 = time_delay_from_geocenter(ifo, ra, dec, tref)
        assert dt0 == dt1


def test_time_delay_from_geocenter_tensor(ifo_list_and_prior_and_tref):
    ifo_list, prior, tref = ifo_list_and_prior_and_tref
    params = prior.sample(1000)
    # check that our vectorized function returns the same result as the bilby method
    ra = torch.Tensor(params["ra"])
    dec = torch.Tensor(params["dec"])
    for ifo in ifo_list:
        dt0 = [
            ifo.time_delay_from_geocenter(ra_idx, dec_idx, tref)
            for ra_idx, dec_idx in zip(ra, dec)
        ]
        dt0 = np.array(dt0)
        dt1 = time_delay_from_geocenter(ifo, ra, dec, tref)
        assert np.allclose(dt0, dt1, atol=1e-6) # larger atol due to single precision
    # check that this also works for arrays of length 1, for which the bilby method is
    # called since it is faster
    ra = torch.Tensor(params["ra"])[:1]
    dec = torch.Tensor(params["dec"])[:1]
    for ifo in ifo_list:
        dt0 = [
            ifo.time_delay_from_geocenter(ra_idx, dec_idx, tref)
            for ra_idx, dec_idx in zip(ra, dec)
        ]
        dt0 = np.array(dt0)
        dt1 = time_delay_from_geocenter(ifo, ra, dec, tref)
        assert dt0 == dt1
