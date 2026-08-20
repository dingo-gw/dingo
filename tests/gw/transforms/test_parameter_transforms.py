import pytest
import numpy as np
import torch
import torch.distributions

from dingo.gw.transforms import SelectStandardizeRepackageParameters
from dingo.gw.transforms.parameter_transforms import StandardizeParameters


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


def test_standardize_parameters_on_distribution():
    """Check standardization of samples from a multi-normal distribution."""
    mean_ = torch.tensor([3.0, 2.0, 8.0])
    std_ = torch.tensor([2.0, 4.0, 7.0])
    n_samples = 100000
    parameters = torch.distributions.Normal(mean_, std_).sample((n_samples,)).numpy()
    samples = {"parameters": {"x": parameters}, "waveform": None}
    tr = StandardizeParameters({"x": mean_.numpy()}, {"x": std_.numpy()})
    samples_tr = tr(samples)
    parameters_tr = samples_tr["parameters"]["x"]

    tol = 0.01
    assert np.all(np.abs(np.mean(parameters_tr, axis=0)) < tol)
    assert np.all(np.abs(np.std(parameters_tr, axis=0)) - np.ones(3) < tol)
