import yaml
from os.path import join, abspath, dirname

from dingo.gw.waveform_dataset import generate_and_save_reduced_basis
from dingo.gw.transforms import AddWhiteNoiseComplex,  \
    RepackageStrainsAndASDS, SelectStandardizeRepackageParameters
from dingo.api import build_dataset
from dingo.core.utils import *

import argparse

parser = argparse.ArgumentParser(description='Train Dingo.')
parser.add_argument('--train_dir', required=True,
                    help='Train directory for Dingo training. Contains'
                         'train_settings.yaml file, used for logging.')
args = parser.parse_args()

with open(join(args.train_dir, 'train_settings.yaml'), 'r') as fp:
    train_settings = yaml.safe_load(fp)

# build dataset with fixed luminosity distance of 100
train_settings['transform_settings']['extrinsic_prior']['luminosity_distance'] \
    = '100.0'
wfd = build_dataset(train_settings)

# set suffix according to gnpe settings
wfd_dir = dirname(abspath(train_settings['waveform_dataset_path']))
suffix = '_'
with open(join(wfd_dir, 'settings.yaml'), 'r') as fp:
    suffix += yaml.safe_load(fp)['waveform_generator_settings']['approximant']
if 'gnpe_time_shifts' in train_settings['transform_settings']:
    dt = train_settings['transform_settings']['gnpe_time_shifts'][
        'kernel_kwargs']['high']
    suffix += f'_gnpe-timeshift-{dt:.4f}'
if 'gnpe_chirp_mass' in train_settings['transform_settings']:
    dt = train_settings['transform_settings']['gnpe_chirp_mass'][
        'kernel_kwargs']['high']
    suffix += f'_gnpe-chirpmass-{dt:.2f}'

generate_and_save_reduced_basis(
    wfd,
    omitted_transforms = [AddWhiteNoiseComplex, RepackageStrainsAndASDS,
                          SelectStandardizeRepackageParameters],
    N = 50000,
    num_workers = train_settings['train_settings']['num_workers'],
    n_rb = 1000,
    out_dir = dirname(abspath(train_settings['asd_dataset_path'])),
    suffix = suffix)