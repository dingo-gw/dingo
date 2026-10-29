"""Test-only Dingo workers for the local Asimov/HTCondor integration test.

The real parser and DAG builder are retained. Worker commands exchange real
HDF5 files with synthetic contents; no network, training, or likelihood runs.
These executables are installed only in the disposable test project.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from bilby_pipe.utils import (convert_string_to_dict,
                              get_command_line_arguments, parse_args,
                              resolve_filename_with_transfer_fallback)

from dingo.gw.data.event_dataset import EventDataset
from dingo.gw.domains import build_domain_from_model_metadata
from dingo.gw.result import Result
from dingo.pipe.parser import create_parser


def install_mock_executables(directory: Path) -> Path:
    """Expose the mock workers and the real PESummary in a private bin directory."""
    bin_directory = directory / "bin"
    bin_directory.mkdir(parents=True)
    stages = {
        "dingo_pipe": "build",
        "dingo_pipe_generation": "generation",
        "dingo_pipe_sampling": "sampling",
        "dingo_pipe_importance_sampling": "importance_sampling",
    }
    for executable, stage in stages.items():
        path = bin_directory / executable
        path.write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
            "from mock_dingo import main\n"
            f"main({stage!r})\n"
        )
        path.chmod(0o755)
    (bin_directory / "summarypages").symlink_to(Path(sys.prefix) / "bin/summarypages")
    os.environ["PATH"] = f"{bin_directory}{os.pathsep}{os.environ['PATH']}"
    return directory


def load_metadata_model(filename, **kwargs):
    """Supply the DAG builder's metadata interface without a posterior model."""
    path = resolve_filename_with_transfer_fallback(filename)
    metadata = torch.load(path, map_location="cpu", weights_only=True)["metadata"]
    return SimpleNamespace(metadata=metadata)


def generate_event(args):
    metadata = load_metadata_model(args.model).metadata
    domain = build_domain_from_model_metadata(metadata, base=True)
    detectors = metadata["train_settings"]["data"]["detectors"]
    psds = convert_string_to_dict(args.psd_dict)
    data = {"waveform": {}, "asds": {}}
    for detector in detectors:
        psd = np.loadtxt(resolve_filename_with_transfer_fallback(psds[detector]))
        data["asds"][detector] = np.sqrt(np.interp(domain(), psd[:, 0], psd[:, 1]))
        data["waveform"][detector] = np.zeros(len(domain), dtype=complex)
    event = EventDataset(
        dictionary={
            "data": data,
            "settings": {
                "time_event": float(args.trigger_time),
                "detectors": detectors,
                "domain": domain.domain_dict,
                "minimum_frequency": convert_string_to_dict(args.minimum_frequency),
                "maximum_frequency": convert_string_to_dict(args.maximum_frequency),
                "mock": True,
            },
        }
    )
    directory = Path(args.outdir) / "data"
    directory.mkdir(parents=True, exist_ok=True)
    event.to_file(directory / f"{args.label}_event_data.hdf5")


def generate_samples(args):
    metadata = load_metadata_model(args.model).metadata
    models = {"model": args.model}
    if args.model_init is not None:
        # Exercise the configured initializer path, without constructing a GNPE chain.
        load_metadata_model(args.model_init)
        models["model_init"] = args.model_init
    data = metadata["train_settings"]["data"]
    rng = np.random.default_rng(args.sampling_seed)
    scales = data["standardization"]
    samples = pd.DataFrame(
        {
            parameter: scales["mean"][parameter]
            + scales["std"][parameter] * rng.uniform(-1, 1, args.num_samples)
            for parameter in data["inference_parameters"]
        }
    )
    samples["log_prob"] = 0.0
    # Keep nested sampler provenance to exercise Asimov's PESummary export.
    metadata["sampler"] = {
        "chain": [{"type": "MockSamples", "parameters": data["inference_parameters"]}],
        "models": models,
        "sampling_seed": args.sampling_seed,
    }
    event = EventDataset(
        file_name=resolve_filename_with_transfer_fallback(args.event_data_file)
    )
    result = Result(
        dictionary={
            "settings": metadata,
            "samples": samples,
            "context": event.data,
            "event_metadata": event.settings,
        }
    )
    for parameter in result.fixed_parameter_keys:
        result.samples[parameter] = result.prior[parameter].peak
    return result


def importance_sample(args):
    result = Result(
        file_name=resolve_filename_with_transfer_fallback(args.proposal_samples_file)
    )
    event = EventDataset(
        file_name=resolve_filename_with_transfer_fallback(args.event_data_file)
    )
    result.reset_event(event)
    # Synthetic statistics keep every sample for PESummary's resampling step.
    result.samples["weights"] = 1.0
    result.samples["log_likelihood"] = np.linspace(-2.0, -1.0, len(result.samples))
    result.samples["log_prior"] = 0.0
    result.log_evidence = -1.5
    result.log_noise_evidence = -2.0
    return result


def main(stage):
    if stage == "build":
        from dingo.pipe import main as pipe

        with patch.object(pipe, "build_model_from_kwargs", load_metadata_model):
            pipe.main()
        return

    args, _ = parse_args(get_command_line_arguments(), create_parser(top_level=False))
    if stage == "generation":
        generate_event(args)
        return
    result = generate_samples(args) if stage == "sampling" else importance_sample(args)
    directory = Path(args.outdir) / "result"
    directory.mkdir(parents=True, exist_ok=True)
    result.to_file(directory / f"{args.label}.hdf5")
