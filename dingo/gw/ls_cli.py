import argparse
from pathlib import Path

import h5py
import torch
import yaml

from dingo.core.dataset import DingoDataset
from dingo.gw.SVD import SVDBasis
from dingo.core.result import Result


def ls():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str)
    args = parser.parse_args()

    path = Path(args.file_name)
    if path.suffix == ".pt":
        print("Extracting information about torch model.\n")
        d = torch.load(path, map_location=torch.device("cpu"))
        print(f"Dingo version: {d.get('version')}\n")
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
        try:
            svd = SVDBasis(file_name=args.file_name)
            print(f"SVD dataset of size n={svd.n}.")
            print("Validation summary:")
            svd.print_validation_summary()

        except KeyError:
            dataset_type = determine_dataset_type(args.file_name)

            if dataset_type == "gw_result" or dataset_type == "core_result":
                print("Dingo Result\n" + "============")

                result = Result(file_name=args.file_name)
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

            else:
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
