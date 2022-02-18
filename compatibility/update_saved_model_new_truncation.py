import torch
import argparse


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
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

    data_settings = d['metadata']['train_settings']['data']
    f_min, f_max = data_settings['conditioning']['frequency_range']
    del data_settings['conditioning']['frequency_range']

    data_settings['domain_update'] = {'f_min': f_min, 'f_max': f_max}

    torch.save(d, args.out_file)


if __name__ == "__main__":
    main()
