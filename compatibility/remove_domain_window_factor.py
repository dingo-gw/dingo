import argparse
import ast
import textwrap
from pathlib import Path

import h5py
import torch

from dingo.core.utils.backward_compatibility import torch_load_with_fallback


def main():

    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Removes any domain window_factor from dataset or model settings. This 
        modifies the file in-place."""
        )
    )
    parser.add_argument(
        "file_name", type=str, help="Dataset or model file to be modified."
    )
    args = parser.parse_args()

    path = Path(args.file_name)

    if path.suffix == ".hdf5":
        with h5py.File(path, "r+") as f:
            settings = ast.literal_eval(f.attrs["settings"])
            try:
                if "domain" in settings:
                    del settings["domain"]["window_factor"]
                elif "domain_dict" in settings:
                    del settings["domain_dict"]["window_factor"]
                f.attrs["settings"] = str(settings)
            except KeyError:
                print("Dataset is already in correct format.")

    elif path.suffix == ".pt":
        d, _ = torch_load_with_fallback(args.checkpoint)

        try:
            del d["metadata"]["dataset_settings"]["domain"]["window_factor"]
            torch.save(d, path)
        except KeyError:
            print("Dataset is already in correct format.")


if __name__ == "__main__":
    main()
