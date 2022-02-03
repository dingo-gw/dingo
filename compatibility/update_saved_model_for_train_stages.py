import os

import torch
import argparse
import yaml
import h5py
import ast


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--settings_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--train_dir', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.settings_file, 'r') as f:
        train_settings = yaml.safe_load(f)

    # Typically training is done on the GPU, so the model could be saved on a GPU
    # device. Since this routine may be run on a CPU machine, allow for a remap of the
    # torch tensors.
    if torch.cuda.is_available():
        d = torch.load(args.checkpoint)
    else:
        d = torch.load(args.checkpoint, map_location=torch.device('cpu'))

    data = {
        'waveform_dataset_path': train_settings['waveform_dataset_path'],
        'train_fraction': train_settings['train_settings']['train_fraction'],
        'conditioning': train_settings['data_conditioning'],
    }
    data.update(train_settings['transform_settings'])

    model = d['model_kwargs']
    model['type'] = 'nsf+embedding'
    model['embedding_net_kwargs']['svd'] = {
        'size': model['embedding_net_kwargs']['n_rb'],
    },
    del model['embedding_net_kwargs']['n_rb']
    del model['embedding_net_kwargs']['V_rb_list']

    training = {'stage_0': {
        'epochs': train_settings['train_settings']['runtime_limits']['max_epochs_total'],
        'asd_dataset_path': train_settings['asd_dataset_path'],
        'freeze_rb_layer': train_settings['train_settings']['freeze_rb_layer'],
        'optimizer': train_settings['train_settings']['optimizer_kwargs'],
        'scheduler': train_settings['train_settings']['scheduler_kwargs'],
        'batch_size': train_settings['train_settings']['batch_size'],
    }
    }

    local = {
        'device': train_settings['train_settings']['device'],
        'num_workers': train_settings['train_settings']['num_workers'],
        'runtime_limits': train_settings['train_settings']['runtime_limits'],
        'checkpoint_epochs': train_settings['train_settings']['checkpoint_epochs'],
        'condor': train_settings['condor_settings'],
    }
    del local['runtime_limits']['max_epochs_total']

    d['metadata'] = {'train_settings': {
        'data': data,
        'model': model,
        'training': training,
    }}

    # Save (for posterity) the waveform dataset settings
    f = h5py.File(d['metadata']['train_settings']['data']['waveform_dataset_path'], 'r')
    settings = ast.literal_eval(f.attrs['settings'])
    d['metadata']['dataset_settings'] = settings
    f.close()

    torch.save(d, args.out_file)

    # Save local settings
    with open(os.path.join(args.train_dir, 'local_settings.yaml'), 'w') as f:
        yaml.dump(local, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
