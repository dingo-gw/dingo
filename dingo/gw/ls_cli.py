import argparse
import json
import logging
from pathlib import Path
from pprint import pformat

import h5py
import yaml

from dingo.core.dataset import DingoDataset
from dingo.core.result import Result
from dingo.core.utils.backward_compatibility import torch_load_with_fallback
from dingo.core.utils.logging_utils import setup_logger
from dingo.gw.dataset import WaveformDataset
from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.SVD import SVDBasis


logger = logging.getLogger(__name__)
logging.captureWarnings(True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str)
    return parser.parse_args()


def ls():
    setup_logger(use_bilby=False)
    args = parse_args()
    file_name = args.file_name

    path = Path(file_name)
    if path.suffix == ".pt":
        logger.info("Extracting information about torch model.\n")
        d, _ = torch_load_with_fallback(path, preferred_map_location="meta")
        logger.info(f"Version: {d.get('version')}\n")
        logger.info(f"Model epoch: {d['epoch']}\n")
        logger.info("Model metadata:")
        logger.info(
            yaml.dump(
                d["metadata"],
                default_flow_style=False,
                sort_keys=False,
            )
        )

    elif path.suffix == ".hdf5":

        dataset_type = determine_dataset_type(file_name)

        if dataset_type == "gw_result" or dataset_type == "core_result":
            result = Result(file_name=file_name)
            logger.info(f"Version: {result.version}")
            logger.info("\nDingo Result\n" + "============\n")

            logger.info(
                "Metadata\n"
                + "--------\n"
                + yaml.dump(
                    result.settings,
                    default_flow_style=False,
                    sort_keys=False,
                )
            )
            if result.event_metadata:
                logger.info(
                    "Event information:\n"
                    + "------------------\n"
                    + yaml.dump(
                        result.event_metadata,
                        default_flow_style=False,
                        sort_keys=False,
                    ),
                )
            if result.importance_sampling_metadata is not None:
                logger.info(
                    "Importance sampling:\n"
                    + "--------------------\n"
                    + yaml.dump(
                        result.importance_sampling_metadata,
                        default_flow_style=False,
                        sort_keys=False,
                    ),
                )
            if result.log_evidence:
                logger.info("Summary:\n" + "--------")
                result.print_summary()

        elif dataset_type == "svd_basis":
            svd = SVDBasis(file_name=file_name)
            logger.info(f"Dingo version: {svd.version}")
            logger.info("\nSVD Basis\n" + "=========\n")

            logger.info(f"Basis size: {svd.n}.")
            logger.info("\nValidation summary:\n" + "-------------------")
            svd.print_validation_summary()

        elif dataset_type == "waveform_dataset":
            waveform_dataset = WaveformDataset(
                file_name=file_name, leave_waveforms_on_disk=True
            )
            logger.info(f"Dingo version: {waveform_dataset.version}")
            logger.info("\nWaveform dataset\n" + "================\n")

            logger.info(f"Dataset size: {len(waveform_dataset)}")

            logger.info(
                "\nSettings\n"
                + "--------\n"
                + yaml.dump(
                    waveform_dataset.settings,
                    default_flow_style=False,
                    sort_keys=False,
                )
            )

            if waveform_dataset.svd:
                svd = SVDBasis(dictionary=waveform_dataset.svd)
                logger.info("\nSVD validation summary:\n" + "---------------------------")
                svd.print_validation_summary()

        elif dataset_type == "asd_dataset":
            asd_dataset = ASDDataset(file_name=file_name)
            logger.info(f"Dingo version: {asd_dataset.version}")
            logger.info("\nASD dataset\n" + "================\n")

            logger.info(f"Dataset size: {asd_dataset.length_info}\n")
            logger.info(f"GPS times (min/max): {asd_dataset.gps_info}")

            logger.info(
                "\nSettings\n"
                + "--------\n"
                + yaml.dump(
                    asd_dataset.settings,
                    default_flow_style=False,
                    sort_keys=False,
                )
            )

        elif dataset_type == "trained_model":
            with h5py.File(file_name, "r") as f:
                logger.info("Extracting information about torch model.\n")
                logger.info(f"Version: {f.attrs['version']}")
                logger.info(f"Model epoch: {f.attrs['epoch']}")
                logger.info("Model metadata:")

                for d in ["model_kwargs", "metadata"]:
                    json_data = json.loads(f["serialized_dicts"][d][()])
                    logger.info(f"\n{d}:\n" + "-" * (len(d) + 1))
                    logger.info(pformat(json_data))

        else:
            # Legacy (before dataset_type identifier).
            try:
                svd = SVDBasis(file_name=file_name)
                logger.info(f"SVD dataset of size n={svd.n}.")
                logger.info("Validation summary:")
                svd.print_validation_summary()

            except KeyError:

                dataset = DingoDataset(
                    file_name=file_name,
                    data_keys=[
                        "svd",
                    ],
                )
                if dataset.settings is not None:
                    logger.info(
                        yaml.dump(
                            dataset.settings,
                            default_flow_style=False,
                            sort_keys=False,
                        )
                    )
                if dataset.svd is not None:
                    svd = SVDBasis(dictionary=dataset.svd)
                    logger.info("SVD validation summary:")
                    svd.print_validation_summary()

    elif path.suffix == ".yaml":
        with open(path, "r") as f:
            settings = yaml.safe_load(f)
        logger.info(
            yaml.dump(
                settings,
                default_flow_style=False,
                sort_keys=False,
            )
        )

    else:
        logger.info("File type unrecognized.")


def determine_dataset_type(file_name):
    with h5py.File(file_name, "r") as f:
        if "dataset_type" in f.attrs:
            return f.attrs["dataset_type"]
        else:
            return None
