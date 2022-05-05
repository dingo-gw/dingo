import torch
import argparse


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--key', type=str, required=True, nargs='+')
    parser.add_argument('--value', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    # Typically training is done on the GPU, so the model could be saved on a GPU
    # device. Since this routine may be run on a CPU machine, allow for a remap of the
    # torch tensors.
    if torch.cuda.is_available():
        d = torch.load(args.checkpoint)
    else:
        d = torch.load(args.checkpoint, map_location=torch.device('cpu'))

    # Figure out type of value
    try:
        value = int(args.value)
    except ValueError:
        try:
            value = float(args.value)
        except ValueError:
            value = args.value

    settings = d['metadata']
    for i in args.key[:-1]:
        settings = settings[i]
    settings[args.key[-1]] = value

    torch.save(d, args.checkpoint)


if __name__ == "__main__":
    main()