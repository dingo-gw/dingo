import argparse
from pathlib import Path

import torch
import yaml

from dingo.core.dataset import DingoDataset
from dingo.gw.SVD import SVDBasis


def ls():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str)
    args = parser.parse_args()

    path = Path(args.file_name)
    if path.suffix == ".pt":
        print("Extracting information about torch model.\n")
        if torch.cuda.is_available():
            d = torch.load(path)
        else:
            d = torch.load(path, map_location=torch.device("cpu"))
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
        if path.stem.startswith("svd"):
            svd = SVDBasis(file_name=args.file_name)
            print(f"SVD dataset of size n={svd.n}.")
            print("Validation summary:")
            svd.print_validation_summary()

        else:
            dataset = DingoDataset(file_name=args.file_name, data_keys=["svd"])
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
