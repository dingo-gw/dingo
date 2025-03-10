import argparse

import torch

from dingo.core.utils.backward_compatibility import torch_load_with_fallback


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    d, _ = torch_load_with_fallback(args.checkpoint)

    data_settings = d["metadata"]["train_settings"]["data"]
    f_min, f_max = data_settings["conditioning"]["frequency_range"]
    del data_settings["conditioning"]["frequency_range"]

    data_settings["domain_update"] = {"f_min": f_min, "f_max": f_max}

    torch.save(d, args.out_file)


if __name__ == "__main__":
    main()
