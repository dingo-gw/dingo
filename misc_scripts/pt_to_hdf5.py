import os
import torch
import numpy as np
import h5py
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_file", type=str, required=True, help='Input model weights file .pt')
    parser.add_argument("-o", "--out_file", type=str, required=True, help='Output model weights file .hdf5')
    return parser.parse_args()


def main():
    args = parse_args()
    if os.path.splitext(args.in_file)[-1] != '.pt':
        raise ValueError('Expected a .pt input file')
    if os.path.splitext(args.out_file)[-1] != '.hdf5':
        raise ValueError('Expected a .hdf5 output file')

    # Load data into CPU memory since we'll be saving it using CPU libraries
    d = torch.load(args.in_file, map_location=torch.device("cpu"))

    # Collect the names of the dicts that can be serialized to JSON; model_state_dict and
    # optimizer_state_dict contain torch.tensors and cannot be JSON serialized
    dicts_to_serialize = ['model_kwargs', 'epoch', 'metadata', 'optimizer_kwargs',
            'scheduler_kwargs', 'scheduler_state_dict']


    with h5py.File('model_weights.h5', 'w') as f:
        # Save small nested dicts as json
        grp = f.create_group('serialized_dicts')
        for k in dicts_to_serialize:
            dict_str = json.dumps(d[k])
            grp.create_dataset(k, data=dict_str)

        # Save the OrderedDict containing the model weights
        # The keys are ordered alphanumerically as well.
        grp_model = f.create_group('model_weights')
        for k, v in d['model_state_dict'].items():
            grp_model.create_dataset(k, data=v.numpy())

        # FIXME: do we need to save optimizer_state_dict?
        # If we don't we could omit it and save some space.


if __name__ == "__main__":
    main()

