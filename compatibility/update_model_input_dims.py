import torch
import argparse


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
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

    d["model_kwargs"]["embedding_net_kwargs"]["input_dims"] = list(d["model_kwargs"]["embedding_net_kwargs"]["input_dims"])

    torch.save(d, args.checkpoint)


if __name__ == "__main__":
    main()