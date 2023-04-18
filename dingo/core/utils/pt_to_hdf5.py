import os
import torch
import numpy as np
import h5py
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the weights of a trained Dingo model from a PyTorch pickle .pt file to HDF5,"
        " for distribution in the LVK's CVMFS.",
        epilog="Training history (optimizer_state_dict) is discarded.")
    parser.add_argument("-i", "--in_file", type=str, required=True,
            help='Input model ".pt" weights file')
    parser.add_argument("-o", "--out_file", type=str, required=True,
            help='Output model ".hdf5" weights file')
    parser.add_argument("-n", "--model_version_number", type=int, required=True,
            help="Model version number (integer). "
            "Will be included in the output filename and metadata.")
    return parser.parse_args()


def main():
    args = parse_args()
    if os.path.splitext(args.in_file)[-1] != '.pt':
        raise ValueError('Expected a .pt input file')
    if os.path.splitext(args.out_file)[-1] != '.hdf5':
        raise ValueError('Expected a .hdf5 output file')

    # Build output filename with the version number for this network
    # This is required for use on CVMFS
    root, ext = os.path.splitext(args.out_file)
    out_file_name = f'{root}_v{args.model_version_number}{ext}'
    print('Output will be written to', out_file_name)

    # Load data into CPU memory since we'll be saving it using CPU libraries
    d = torch.load(args.in_file, map_location=torch.device("cpu"))

    # Collect the names of the dicts that can be serialized to JSON; model_state_dict and
    # optimizer_state_dict contain torch.tensors and cannot be JSON serialized
    # In addition, we drop dicts related to training information that is not needed at inference time
    dicts_to_serialize = ['model_kwargs', 'epoch', 'metadata']


    with h5py.File(out_file_name, 'w') as f:
        # Save small nested dicts as json
        grp = f.create_group('serialized_dicts')
        for k in dicts_to_serialize:
            dict_str = json.dumps(d[k])
            grp.create_dataset(k, data=dict_str)

        # Save the OrderedDict containing the model weights
        # The keys are ordered alphanumerically as well.
        grp_model = f.create_group('model_weights')
        for k, v in d['model_state_dict'].items():
            if len(v.size()) > 0:
                grp_model.create_dataset(k, data=v.numpy(), fletcher32=True)
            else:
                # fletcher32 is not available for scalars
                grp_model.create_dataset(k, data=v.numpy())

        # Note we do not save optimizer_state_dict which is
        # not needed at inference time and saves a significant
        # amount of disk space.

        # Metadata for CVMFS LVK distribution
        # This needs to be exactly the same as the "basename" of the hdf5 file
        f.attrs['CANONICAL_FILE_BASENAME'] = os.path.basename(out_file_name)

        # Add a few metadata entries as attributes
        f.attrs['approximant'] = d['metadata']['dataset_settings']['waveform_generator']['approximant']
        f.attrs['epoch'] = d['epoch']
        # Add the dingo version used for training
        f.attrs['version'] = str(d.get('version'))
        # Make it clear to dingo_ls that this is file contains model weights
        f.attrs['dataset_type'] = 'trained_model' 


if __name__ == "__main__":
    main()

