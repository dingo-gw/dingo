"""Test if asimov.py can correctly choose which DINGO network to use."""
import os
import copy
import pytest

import torch

from asimov.pipelines import known_pipelines
from asimov.pipeline import PipelineException


class DummyEvent:

    def __init__(self):
        self.name = "GW150914"


class DummyProduction:

    def __init__(self, available_networks=None):
        self.pipeline = self.name = "DINGO"
        self.meta = {
            "available networks": available_networks,
            "data": {"segment length" : 4},
            "interferometers": ["H1", "L1"],
            "quality": {
                "minimum frequency": {"H1": 20, "L1": 20},
                "maximum frequency": {"H1": 1024, "L1": 1024},
            },
            "priors": {"luminosity distance": {"maximum": 1000}},
        }
        self.category = None
        self.event = DummyEvent()


def apply_modifications(modifications, base_meta, test_dir, prefix=""):
    """
    Creates .pt files for each modification in the list.

    Args:
        modifications (list): List of dictionaries with "path" and "value" keys.
        base_meta (dict): Base metadata to modify.
        test_dir (str): Directory to save the .pt files.

    Returns:
        list: List of paths to the created .pt files.
    """
    paths = []
    os.makedirs(test_dir, exist_ok=True)

    for i, mod in enumerate(modifications):
        modified_meta = copy.deepcopy(base_meta)
        current = modified_meta
        if "path" in mod:
            for key in mod["path"][:-1]:
                current = current[key]
            current[mod["path"][-1]] = mod["value"]

        path = {"model": os.path.join(test_dir, f"{prefix}network{i + 1}.pt")}
        paths.append(path)
        torch.save(modified_meta, path["model"])

    return paths


def create_test_network_files(test_dir):
    """
    Creates test network files for valid and invalid DINGO network configurations.

    Args:
        test_dir (str):
            Directory where the `.pt` files will be saved.
            If the directory does not exist, it will be created.

    Returns:
        tuple[list[dict], list[dict]]:
            A tuple containing two lists:
            - valid_paths: List of dictionaries, each containing the path to a valid `.pt` file.
            - invalid_paths: List of dictionaries, each containing the path to an invalid `.pt` file.

            Each dictionary has the form:
            ```python
            {"model": "/path/to/network.pt"}
            ```
    """
    reference_meta = {
        "metadata": {
            "dataset_settings": {
                "domain": {
                    "base_domain": {
                        "f_min": 10,
                        "f_max": 1024,
                        "delta_f": 0.25
                    }
                }
            },
            "train_settings": {
                "data": {
                    "detectors": ["H1", "L1"],
                    "extrinsic_prior": {"luminosity_distance": "maximum=1000"},
                    "domain_update": {},
                    "random_strain_cropping": {"f_min_upper": 20},
                }
            }
        }
    }

    valid_modifications = [
        {},
        {
            "path": ["metadata", "train_settings", "data", "extrinsic_prior", "luminosity_distance"],
            "value": "maximum=2000"
        }
    ]

    invalid_modifications = [
        {
            "path": ["metadata", "dataset_settings", "domain", "base_domain", "f_min"],
            "value": 30
        },
        {
            "path": ["metadata", "dataset_settings", "domain", "base_domain", "f_max"],
            "value": 512
        },
        {
            "path": ["metadata", "train_settings", "data", "detectors"],
            "value": ["H1"]
        }
    ]

    valid_paths = apply_modifications(valid_modifications, reference_meta,
                                      test_dir, prefix="valid")
    invalid_paths = apply_modifications(invalid_modifications, reference_meta,
                                        test_dir, prefix="invalid")
    return valid_paths, invalid_paths


@pytest.fixture
def valid_and_invalid_paths():
    valid_paths, invalid_paths = create_test_network_files(".")
    return valid_paths, invalid_paths


def test_valid_networks(valid_and_invalid_paths):
    valid_paths, _ = valid_and_invalid_paths

    for path in valid_paths:
        pipe = known_pipelines["dingo"](DummyProduction(available_networks=[path]))
        pipe.before_config()
        assert path["model"] == pipe.production.meta["networks"]["model"], \
            f"Network {path['model']} should be valid and have 'network' key"

def test_invalid_networks(valid_and_invalid_paths):
    _, invalid_paths = valid_and_invalid_paths

    for path in invalid_paths:
        pipe = known_pipelines["dingo"](DummyProduction(available_networks=[path]))
        with pytest.raises(PipelineException):
            pipe.before_config()

def test_network_preference(valid_and_invalid_paths):
    valid_paths, _ = valid_and_invalid_paths
    pipe = known_pipelines["dingo"](DummyProduction(available_networks=valid_paths))
    pipe.before_config()
    assert valid_paths[0]["model"] == pipe.production.meta["networks"]["model"]
