import argparse
from pathlib import Path

import h5py
import torch
import yaml
import json
from pprint import pprint

from dingo.core.dataset import DingoDataset
from dingo.gw.SVD import SVDBasis
from dingo.core.result import Result
from dingo.gw.dataset import WaveformDataset
from dingo.gw.noise.asd_dataset import ASDDataset


def ls():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str)
    args = parser.parse_args()

    path = Path(args.file_name)
    if path.suffix == ".pt":
        print("Extracting information about torch model.\n")
        d = torch.load(path, map_location=torch.device("cpu"))
        print(f"Version: {d.get('version')}\n")
        print(f"Model epoch: {d['epoch']}\n")
        print("Model metadata:")
        print(
            yaml.dump(
                d["metadata"],
                default_flow_style=False,
                sort_keys=False,
            )
        )

    elif path.suffix == ".hdf5":

        dataset_type = determine_dataset_type(args.file_name)

        if dataset_type == "gw_result" or dataset_type == "core_result":
            result = Result(file_name=args.file_name)
            print(f"Version: {result.version}")
            print("\nDingo Result\n" + "============\n")

            print(
                "Metadata\n"
                + "--------\n"
                + yaml.dump(
                    result.settings,
                    default_flow_style=False,
                    sort_keys=False,
                )
            )
            if result.event_metadata:
                print(
                    "Event information:\n"
                    + "------------------\n"
                    + yaml.dump(
                        result.event_metadata,
                        default_flow_style=False,
                        sort_keys=False,
                    ),
                )
            if result.importance_sampling_metadata is not None:
                print(
                    "Importance sampling:\n"
                    + "--------------------\n"
                    + yaml.dump(
                        result.importance_sampling_metadata,
                        default_flow_style=False,
                        sort_keys=False,
                    ),
                )
            if result.log_evidence:
                print("Summary:\n" + "--------")
                result.print_summary()

        elif dataset_type == "svd_basis":
            svd = SVDBasis(file_name=args.file_name)
            print(f"Dingo version: {svd.version}")
            print("\nSVD Basis\n" + "=========\n")

            print(f"Basis size: {svd.n}.")
            print("\nValidation summary:\n" + "-------------------")
            svd.print_validation_summary()

        elif dataset_type == "waveform_dataset":
            waveform_dataset = WaveformDataset(file_name=args.file_name)
            print(f"Dingo version: {waveform_dataset.version}")
            print("\nWaveform dataset\n" + "================\n")

            print(f"Dataset size: {len(waveform_dataset)}")

            print(
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
                print("\nSVD validation summary:\n" + "---------------------------")
                svd.print_validation_summary()

        elif dataset_type == "asd_dataset":
            asd_dataset = ASDDataset(file_name=args.file_name)
            print(f"Dingo version: {asd_dataset.version}")
            print("\nASD dataset\n" + "================\n")

            print(f"Dataset size: {asd_dataset.length_info}\n")
            print(f"GPS times (min/max): {asd_dataset.gps_info}")

            print(
                "\nSettings\n"
                + "--------\n"
                + yaml.dump(
                    asd_dataset.settings,
                    default_flow_style=False,
                    sort_keys=False,
                )
            )

        elif dataset_type == "trained_model":
            with h5py.File(args.file_name, "r") as f:
                print("Extracting information about torch model.\n")
                print(f"Version: {f.attrs['version']}")
                print(f"Model epoch: {f.attrs['epoch']}")
                print("Model metadata:")

                for d in ['model_kwargs', 'metadata']:
                    json_data = json.loads(f['serialized_dicts'][d][()])
                    print(f"\n{d}:\n" + "-"*(len(d)+1))
                    pprint(json_data)

        else:
            # Legacy (before dataset_type identifier).
            try:
                svd = SVDBasis(file_name=args.file_name)
                print(f"SVD dataset of size n={svd.n}.")
                print("Validation summary:")
                svd.print_validation_summary()

            except KeyError:

                dataset = DingoDataset(
                    file_name=args.file_name,
                    data_keys=[
                        "svd",
                    ],
                )
                if dataset.settings is not None:
                    print(
                        yaml.dump(
                            dataset.settings,
                            default_flow_style=False,
                            sort_keys=False,
                        )
                    )
                if dataset.svd is not None:
                    svd = SVDBasis(dictionary=dataset.svd)
                    print("SVD validation summary:")
                    svd.print_validation_summary()

    elif path.suffix == ".yaml":
        with open(path, "r") as f:
            settings = yaml.safe_load(f)
        print(
            yaml.dump(
                settings,
                default_flow_style=False,
                sort_keys=False,
            )
        )

    else:
        print("File type unrecognized.")


def determine_dataset_type(file_name):
    with h5py.File(file_name, "r") as f:
        if "dataset_type" in f.attrs:
            return f.attrs["dataset_type"]
        else:
            return None
