from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.waveform_generator import WaveformGenerator
from dingo.gw.waveform_dataset import WaveformDataset
from bilby.gw.prior import BBHPriorDict
from dingo.gw.prior_split import default_intrinsic_dict

import pytest
import numpy as np


@pytest.fixture
def setup_waveform_generator():
    domain_kwargs = {'f_min': 20.0, 'f_max': 4096.0, 'delta_f': 1.0 / 4.0, 'window_factor': 1.0}
    domain = UniformFrequencyDomain(**domain_kwargs)
    priors = BBHPriorDict(default_intrinsic_dict)
    approximant = 'IMRPhenomXPHM'
    waveform_generator = WaveformGenerator(approximant, domain, reference_frequency=20.0)
    return waveform_generator, priors


# def test_load_waveform_dataset(setup_waveform_generator):
#     # FIXME: call generation script and activate test
#     waveform_generator, priors = setup_waveform_generator
#     root = '/Users/mpuer/Documents/NDE/dingo-devel/'
#     path = f'{root}/tutorials/02_gwpe/datasets/waveforms/waveform_dataset.hdf5'
#     wd = WaveformDataset(prior=priors, waveform_generator=waveform_generator)
#     wd.load(path)
#
#     assert len(wd) > 0
#     el = wd[0]
#     assert isinstance(el, dict)
#     assert list(el.keys()) == ['parameters', 'waveform']
#     assert isinstance(el['parameters'], dict)
#     assert isinstance(el['waveform'], dict)
#     assert isinstance(el['waveform']['h_plus'], np.ndarray)
#     assert isinstance(el['waveform']['h_cross'], np.ndarray)
