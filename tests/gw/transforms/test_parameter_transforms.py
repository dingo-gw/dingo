import pytest
import numpy as np

from dingo.gw.transforms import SelectStandardizeRepackageParameters

def test_SelectStandardizeRepackageParameters():
    standardization_dict = {'mean': {'par2': np.random.rand(),
                                     'par1': np.random.rand()},
                            'std': {'par2': np.random.rand(),
                                    'par1': np.random.rand()}}
    select_standardize_repackage_params = \
        SelectStandardizeRepackageParameters(standardization_dict)
    sample = {'a': None, 'b': np.random.rand(100),
              'parameters': {'par0': np.random.rand(),
                             'par1': np.random.rand(),
                             'par2': np.random.rand()}
              }
    sample_out = select_standardize_repackage_params(sample)
    # check that sample keys are not modified
    assert sample.keys() == sample_out.keys()
    # check that sample elements (except sample['parameters']) are not modified
    for k, v in sample_out.items():
        if k != 'parameters':
            assert id(v) == id(sample[k])
    # check that correct number of parameters is selected
    assert len(sample_out['parameters']) == len(standardization_dict['mean'])
    # check that parameter array contains correct elements, in correct order,
    # correctly normalized
    for idx, k in enumerate(standardization_dict['mean'].keys()):
        m, std = standardization_dict['mean'][k], standardization_dict['std'][k]
        par_in = sample['parameters'][k]
        par_out = sample_out['parameters'][idx]
        # standardization changes dtype to float32
        assert par_out == np.float32((par_in - m) / std)