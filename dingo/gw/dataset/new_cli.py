#!/usr/bin/env python3
"""
Command-line interface for generating waveform datasets with the new API.

This tool generates training datasets of gravitational waveform polarizations
based on settings specified in a YAML configuration file.
"""

import argparse
import logging
import sys
import textwrap
from pathlib import Path
from typing import Dict, Any

import yaml

from .dataset_settings import DatasetSettings
from .new_generate import new_generate_waveform_dataset
from dingo.gw.logs import set_logging

_logger = logging.getLogger(__name__)


def load_settings(settings_file: str) -> Dict[str, Any]:
    """
    Load dataset generation settings from YAML file.

    Parameters
    ----------
    settings_file : str
        Path to YAML configuration file.

    Returns
    -------
    dict
        Settings dictionary loaded from YAML.
    """
    settings_path = Path(settings_file)
    if not settings_path.is_file():
        raise FileNotFoundError(f"Settings file not found: {settings_file}")

    with open(settings_path, "r") as f:
        try:
            settings = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML file {settings_file}: {e}")

    if settings is None:
        raise ValueError(f"Settings file is empty: {settings_file}")

    return settings


def validate_output_path(out_file: str) -> None:
    """Validate that output file path is writable."""
    out_path = Path(out_file)
    if not out_path.parent.exists():
        raise FileNotFoundError(
            f"Cannot create output file {out_file}: "
            f"parent directory {out_path.parent} does not exist"
        )


def generate_dataset_main(
    settings_file: str, out_file: str, num_processes: int
) -> None:
    """
    Main function for dataset generation CLI.

    Parameters
    ----------
    settings_file : str
        Path to YAML settings file.
    out_file : str
        Path to output HDF5 file.
    num_processes : int
        Number of parallel processes for waveform generation.
    """
    validate_output_path(out_file)

    _logger.info(f"Loading settings from: {settings_file}")
    settings_dict = load_settings(settings_file)

    _logger.info("Parsing dataset settings...")
    try:
        settings = DatasetSettings.from_dict(settings_dict)
    except Exception as e:
        _logger.error(f"Failed to parse settings: {e}")
        raise ValueError(f"Invalid settings in {settings_file}: {e}")

    _logger.info("Dataset generation configuration:")
    _logger.info(f"  Domain type: {type(settings.domain).__name__}")
    _logger.info(f"  Approximant: {settings.waveform_generator.approximant}")
    _logger.info(f"  Number of samples: {settings.num_samples}")
    _logger.info(f"  Parallel processes: {num_processes}")
    _logger.info(f"  Output file: {out_file}")

    _logger.info("Starting dataset generation...")
    try:
        dataset = new_generate_waveform_dataset(settings, num_processes=num_processes)
    except Exception as e:
        _logger.error(f"Dataset generation failed: {e}")
        raise

    _logger.info(f"Saving dataset to: {out_file}")
    try:
        dataset.save(out_file)
    except Exception as e:
        _logger.error(f"Failed to save dataset: {e}")
        raise

    _logger.info("Dataset generation completed successfully!")
    _logger.info(f"Generated {len(dataset.parameters)} waveforms")
    _logger.info(f"Saved to: {out_file}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Generate a waveform dataset based on a settings file.

            This tool generates training datasets of gravitational waveform polarizations
            for use in neural posterior estimation. Parameters are sampled from the
            specified prior distribution, and waveforms are generated using LALSimulation.

            Example:
                dingo_new_generate_dataset --settings_file config.yaml --num_processes 8
            """
        ),
    )
    parser.add_argument(
        "--settings_file",
        type=str,
        required=True,
        help="YAML file containing dataset generation settings",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes to use for parallel waveform generation (default: 1)",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="waveform_dataset.hdf5",
        help="Name of output HDF5 file for storing dataset (default: waveform_dataset.hdf5)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging output",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for dingo_new_generate_dataset command-line tool."""
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    set_logging(level=log_level)

    try:
        generate_dataset_main(args.settings_file, args.out_file, args.num_processes)
    except Exception as e:
        _logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
