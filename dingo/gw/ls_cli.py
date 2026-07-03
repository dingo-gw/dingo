import argparse
import json
import logging
import sys
from pathlib import Path
from pprint import pformat

import h5py
import yaml

from dingo.core.dataset import DingoDataset
from dingo.core.result import Result
from dingo.core.utils.backward_compatibility import torch_load_with_fallback
from dingo.gw.dataset import WaveformDataset
from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.SVD import SVDBasis


log = logging.getLogger(__name__)
logging.captureWarnings(True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str)
    return parser.parse_args()


def ls():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        stream=sys.stdout,
        force=True,
    )
    args = parse_args()
    file_name = args.file_name

    path = Path(file_name)
    if path.suffix == ".pt":
        log.info("Extracting information about torch model.\n")
        d, _ = torch_load_with_fallback(path, preferred_map_location="meta")
        log.info(f"Version: {d.get('version')}\n")
        log.info(f"Model epoch: {d['epoch']}\n")
        log.info("Model metadata:")
        log.info(
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
            log.info(f"Version: {result.version}")
            log.info("\nDingo Result\n" + "============\n")

            log.info(
                "Metadata\n"
                + "--------\n"
                + yaml.dump(
                    result.settings,
                    default_flow_style=False,
                    sort_keys=False,
                )
            )
            if result.event_metadata:
                log.info(
                    "Event information:\n"
                    + "------------------\n"
                    + yaml.dump(
                        result.event_metadata,
                        default_flow_style=False,
                        sort_keys=False,
                    ),
                )
            if result.importance_sampling_metadata is not None:
                log.info(
                    "Importance sampling:\n"
                    + "--------------------\n"
                    + yaml.dump(
                        result.importance_sampling_metadata,
                        default_flow_style=False,
                        sort_keys=False,
                    ),
                )
            if result.log_evidence:
                log.info("Summary:\n" + "--------")
                result.print_summary()

        elif dataset_type == "svd_basis":
            svd = SVDBasis(file_name=file_name)
            log.info(f"Dingo version: {svd.version}")
            log.info("\nSVD Basis\n" + "=========\n")

            log.info(f"Basis size: {svd.n}.")
            log.info("\nValidation summary:\n" + "-------------------")
            svd.print_validation_summary()

        elif dataset_type == "waveform_dataset":
            waveform_dataset = WaveformDataset(
                file_name=file_name, leave_waveforms_on_disk=True
            )
            log.info(f"Dingo version: {waveform_dataset.version}")
            log.info("\nWaveform dataset\n" + "================\n")

            log.info(f"Dataset size: {len(waveform_dataset)}")

            log.info(
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
                log.info("\nSVD validation summary:\n" + "---------------------------")
                svd.print_validation_summary()

        elif dataset_type == "asd_dataset":
            asd_dataset = ASDDataset(file_name=file_name)
            log.info(f"Dingo version: {asd_dataset.version}")
            log.info("\nASD dataset\n" + "================\n")

            log.info(f"Dataset size: {asd_dataset.length_info}\n")
            log.info(f"GPS times (min/max): {asd_dataset.gps_info}")

            log.info(
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
                log.info("Extracting information about torch model.\n")
                log.info(f"Version: {f.attrs['version']}")
                log.info(f"Model epoch: {f.attrs['epoch']}")
                log.info("Model metadata:")

                for d in ["model_kwargs", "metadata"]:
                    json_data = json.loads(f["serialized_dicts"][d][()])
                    log.info(f"\n{d}:\n" + "-" * (len(d) + 1))
                    log.info(pformat(json_data))

        else:
            # Legacy (before dataset_type identifier).
            try:
                svd = SVDBasis(file_name=file_name)
                log.info(f"SVD dataset of size n={svd.n}.")
                log.info("Validation summary:")
                svd.print_validation_summary()

            except KeyError:

                dataset = DingoDataset(
                    file_name=file_name,
                    data_keys=[
                        "svd",
                    ],
                )
                if dataset.settings is not None:
                    log.info(
                        yaml.dump(
                            dataset.settings,
                            default_flow_style=False,
                            sort_keys=False,
                        )
                    )
                if dataset.svd is not None:
                    svd = SVDBasis(dictionary=dataset.svd)
                    log.info("SVD validation summary:")
                    svd.print_validation_summary()

    elif path.suffix == ".yaml":
        with open(path, "r") as f:
            settings = yaml.safe_load(f)
        log.info(
            yaml.dump(
                settings,
                default_flow_style=False,
                sort_keys=False,
            )
        )

    else:
        log.info("File type unrecognized.")


def determine_dataset_type(file_name):
    with h5py.File(file_name, "r") as f:
        if "dataset_type" in f.attrs:
            return f.attrs["dataset_type"]
        else:
            return None
