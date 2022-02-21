import pytest
import numpy as np

from dingo.gw.transforms import SelectStandardizeRepackageParameters


def test_SelectStandardizeRepackageParameters():
    standardization_dict = {
        "mean": {"par2": np.random.rand(), "par1": np.random.rand()},
        "std": {"par2": np.random.rand(), "par1": np.random.rand()},
    }
    parameters_dict = {"inference_parameters": ["par1", "par2"]}
    select_standardize_repackage_params = SelectStandardizeRepackageParameters(
        parameters_dict, standardization_dict
    )
    sample = {
        "a": None,
        "b": np.random.rand(100),
        "parameters": {
            "par0": np.random.rand(),
            "par1": np.random.rand(),
            "par2": np.random.rand(),
        },
    }
    sample_out = select_standardize_repackage_params(sample)
    # Check that correct new key has been added.
    assert list(sample.keys()) + ["inference_parameters"] == list(sample_out.keys())
    # check that pre-existing sample elements are not modified
    for k, v in sample.items():
        assert id(v) == id(sample[k])
    # check that correct number of parameters is selected
    assert len(sample_out["inference_parameters"]) == len(
        parameters_dict["inference_parameters"]
    )
    # check that parameter array contains correct elements, in correct order,
    # correctly normalized
    for idx, k in enumerate(parameters_dict["inference_parameters"]):
        m, std = standardization_dict["mean"][k], standardization_dict["std"][k]
        par_in = sample["parameters"][k]
        par_out = sample_out["inference_parameters"][idx]
        # standardization changes dtype to float32
        assert par_out == np.float32((par_in - m) / std)
