import argparse

import torch

from dingo.core.utils.backward_compatibility import torch_load_with_fallback


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--key", type=str, required=True, nargs="+")
    parser.add_argument("--value", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    d, _ = torch_load_with_fallback(args.checkpoint)

    # Figure out type of value
    try:
        value = int(args.value)
    except ValueError:
        try:
            value = float(args.value)
        except ValueError:
            value = args.value

    settings = d["metadata"]
    for i in args.key[:-1]:
        settings = settings[i]
    settings[args.key[-1]] = value

    torch.save(d, args.checkpoint)


if __name__ == "__main__":
    main()
