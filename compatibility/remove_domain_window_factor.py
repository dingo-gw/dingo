import argparse
import ast
from pathlib import Path
import h5py
import torch


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str)
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
        if torch.cuda.is_available():
            d = torch.load(args.checkpoint)
        else:
            d = torch.load(args.checkpoint, map_location=torch.device("cpu"))

        try:
            del d["metadata"]["dataset_settings"]["domain"]["window_factor"]
            torch.save(d, path)
        except KeyError:
            print("Dataset is already in correct format.")


if __name__ == "__main__":
    main()
