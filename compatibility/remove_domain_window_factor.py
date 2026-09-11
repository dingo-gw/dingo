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
            domain_settings = settings.get("domain", settings.get("domain_dict"))
            if domain_settings is not None and _remove_window_factor(domain_settings):
                f.attrs["settings"] = str(settings)
            else:
                print("Dataset is already in correct format.")

    elif path.suffix == ".pt":
        d, _ = torch_load_with_fallback(path)

        domain_settings = d["metadata"]["dataset_settings"]["domain"]
        if _remove_window_factor(domain_settings):
            torch.save(d, path)
        else:
            print("Dataset is already in correct format.")


def _remove_window_factor(domain_settings: dict) -> bool:
    """Delete `window_factor` from the domain settings, including a multibanded
    domain's nested base domain. Returns whether anything was removed."""
    removed = False
    for settings in (domain_settings, domain_settings.get("base_domain", {})):
        if "window_factor" in settings:
            del settings["window_factor"]
            removed = True
    return removed


if __name__ == "__main__":
    main()
