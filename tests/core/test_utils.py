from pathlib import Path
from typing import get_args

import pytest
import torch

from dingo.core.utils.backward_compatibility import (
    Device,
    torch_available_devices,
    torch_load_with_fallback,
)


class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)


@pytest.fixture
def model_path(tmp_path) -> str:
    model = TestModel()
    model.linear.weight.data.fill_(1.0)
    model.linear.bias.data.fill_(2.0)
    model_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), model_path)
    return str(model_path)


def test_torch_available_device() -> None:
    """
    checks torch_available_devices do not
    raise an error and lists at least cpu
    """
    devices = torch_available_devices()
    assert len(devices) != 0
    assert "cpu" in devices


def test_load_torch_with_fallback(model_path) -> None:
    """
    checks torch_load_with_fallback do not
    raise an error and map at least correctly to cpu
    """
    devices = get_args(Device)
    model, device = torch_load_with_fallback(
        model_path, preferred_map_location="cpu"
    )
    assert device == torch.device("cpu")
    for device in devices:
        # just checking no error is raised.
        # Not checking if it is mapped to the proper device, as we do not
        # want to assume which device is available on the test machine.
        torch_load_with_fallback(model_path, preferred_map_location=device)
