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
                    "importance_sampling_metadata",
                    "event_metadata",
                    "log_evidence",
                    "log_evidence_std",
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
            if dataset.importance_sampling_metadata is not None:
                print(
                    "Importance sampling:\n"
                    + "====================\n"
                    + yaml.dump(
                        dataset.importance_sampling_metadata,
                        default_flow_style=False,
                        sort_keys=False,
                    ),
                )
            if dataset.event_metadata:
                print(
                    "Event information:\n"
                    + "==================\n"
                    + yaml.dump(
                        dataset.event_metadata,
                        default_flow_style=False,
                        sort_keys=False,
                    ),
                )
            if (
                dataset.log_evidence is not None
                and dataset.log_evidence_std is not None
            ):
                # TODO: Better to load the Result class so this code does not have to
                #  be copied. To do this we have to encode the type of DingoDataset in
                #  the HDF5 file. The current approach is a temporary workaround.
                print(
                    f"Log(evidence): {dataset.log_evidence:.3f} +- "
                    f"{dataset.log_evidence_std:.3f}"
                )

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
