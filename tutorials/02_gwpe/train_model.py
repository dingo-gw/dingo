import yaml
from os.path import join, isfile, abspath, dirname
import torchvision
from torch.utils.data import DataLoader

from bilby.gw.detector import InterferometerList
from dingo.gw.waveform_dataset import WaveformDataset, \
    generate_and_save_reduced_basis
from dingo.gw.domains import build_domain
from dingo.gw.transforms import SampleExtrinsicParameters,\
    GetDetectorTimes, ProjectOntoDetectors, SampleNoiseASD, \
    WhitenAndScaleStrain, AddWhiteNoiseComplex, \
    SelectStandardizeRepackageParameters, RepackageStrainsAndASDS, \
    UnpackDict, GNPEDetectorTimes
from dingo.gw.noise_dataset import ASDDataset
from dingo.gw.prior_split import default_params
from dingo.gw.gwutils import *

from dingo.core.nn.nsf import create_nsf_with_rb_projection_embedding_net, \
    autocomplete_model_kwargs_nsf
from dingo.core.models.posterior_model import PosteriorModel, train_epoch, \
    test_epoch
from dingo.core.utils import *

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Train Dingo.')
parser.add_argument('--log_dir', required=True,
                    help='Log directory for Dingo training. Contains'
                         'train_settings.yaml file, used for logging.')
args = parser.parse_args()

with open(join(args.log_dir, 'train_settings.yaml'), 'r') as fp:
    train_settings = yaml.safe_load(fp)

# build datasets
wfd = WaveformDataset(train_settings['waveform_dataset_path'])
asd_dataset = ASDDataset(
    train_settings['asd_dataset_path'],
    ifos=train_settings['transform_settings']['detectors'])
# truncate datasets
wfd.truncate_dataset_domain(
    train_settings['data_conditioning']['frequency_range'])
asd_dataset.truncate_dataset_domain(
    train_settings['data_conditioning']['frequency_range'])
# check compatibility of datasets
assert wfd.domain.domain_dict == asd_dataset.domain.domain_dict
# add window factor to domain
domain = build_domain(wfd.domain.domain_dict)
domain.window_factor = get_window_factor(
    train_settings['data_conditioning']['window_kwargs'])

extrinsic_prior_dict = get_extrinsic_prior_dict(
    train_settings['transform_settings']['extrinsic_prior'])
if train_settings['transform_settings']['selected_parameters'] == 'default':
    train_settings['transform_settings']['selected_parameters'] = default_params
standardization_dict = get_standardization_dict(
    extrinsic_prior_dict, wfd,
    train_settings['transform_settings']['selected_parameters'])
ref_time = train_settings['transform_settings']['ref_time']
window_factor = get_window_factor(
    train_settings['data_conditioning']['window_kwargs'])
# build objects
ifo_list = InterferometerList(
    train_settings['transform_settings']['detectors'])

# build transforms
gnpe_proxy_dim = 0
transforms = []
transforms.append(SampleExtrinsicParameters(extrinsic_prior_dict))
transforms.append(GetDetectorTimes(ifo_list, ref_time))
if 'gnpe_time_shifts' in train_settings['transform_settings']:
    d = train_settings['transform_settings']['gnpe_time_shifts']
    transforms.append(GNPEDetectorTimes(
        ifo_list, d['kernel_kwargs'], d['exact_equiv'],
        std=standardization_dict['std']['geocent_time']))
    gnpe_proxy_dim += transforms[-1].gnpe_proxy_dim
transforms.append(ProjectOntoDetectors(ifo_list, domain, ref_time))
transforms.append(SampleNoiseASD(asd_dataset))
transforms.append(WhitenAndScaleStrain(domain.noise_std))
transforms.append(AddWhiteNoiseComplex())
transforms.append(SelectStandardizeRepackageParameters(standardization_dict))
transforms.append(RepackageStrainsAndASDS(
    train_settings['transform_settings']['detectors']))
if gnpe_proxy_dim == 0:
    selected_keys = ['parameters', 'waveform']
else:
    selected_keys = ['parameters', 'waveform', 'gnpe_proxies']
transforms.append(UnpackDict(selected_keys=selected_keys))


# generate reduced basis
# get rb transforms
omitted_transforms = [AddWhiteNoiseComplex, RepackageStrainsAndASDS]
transforms_rb = [tr for tr in transforms if type(tr) not in omitted_transforms]
# set suffix according to gnpe settings
suffix = ''
if 'gnpe_time_shifts' in train_settings['transform_settings']:
    dt = train_settings['transform_settings']['gnpe_time_shifts'][
        'kernel_kwargs']['high']
    suffix += f'_gnpe-timeshift-{dt:.4f}'

generate_and_save_reduced_basis(
    wfd,
    torchvision.transforms.Compose(transforms_rb),
    N = 50,
    batch_size = 10,
    num_workers = train_settings['train_settings']['num_workers'],
    n_rb = 200,
    out_dir = dirname(abspath(train_settings['asd_dataset_path'])),
    suffix = suffix)

# set wfd transforms to the composition of the above transforms
wfd.transform = torchvision.transforms.Compose(transforms)

train_dataset, test_dataset = split_dataset_into_train_and_test(
    wfd, train_settings['train_settings']['train_fraction'])

# build dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=train_settings['train_settings']['batch_size'],
    shuffle=True,
    pin_memory=True,
    num_workers=train_settings['train_settings']['num_workers'],
    worker_init_fn=lambda _:np.random.seed(int(torch.initial_seed())%(2**32-1)))
test_loader = DataLoader(
    test_dataset,
    batch_size=train_settings['train_settings']['batch_size'],
    shuffle=False,
    pin_memory=True,
    num_workers=train_settings['train_settings']['num_workers'],
    worker_init_fn=lambda _:np.random.seed(int(torch.initial_seed())%(2**32-1)))


# build model
if not isfile(join(args.log_dir, 'model_latest.pt')):
    # autocomplete model kwargs is train settings (e.g., input dim from domain)
    model_kwargs = autocomplete_model_kwargs_nsf(
        train_settings, ifo_list, domain, gnpe_proxy_dim)
    # initialize posterior model
    pm = PosteriorModel(
        model_builder=create_nsf_with_rb_projection_embedding_net,
        model_kwargs=model_kwargs,
        init_for_training=True,
        optimizer_kwargs=train_settings['train_settings']['optimizer_kwargs'],
        scheduler_kwargs=train_settings['train_settings']['scheduler_kwargs'],
        device='cpu')
else:
    pm = PosteriorModel(
        model_builder=create_nsf_with_rb_projection_embedding_net,
        model_filename=join(args.log_dir, 'model_latest.pt'),
        init_for_training=True)
# assert get_number_of_model_parameters(pm.model) == 131448775

device = 'cpu'

pm.train(
    train_loader,
    test_loader,
    log_dir=args.log_dir,
    runtime_limits_kwargs=train_settings['train_settings']['runtime_limits'],
    checkpoint_epochs=train_settings['train_settings']['checkpoint_epochs'],
)

train_epoch(pm, train_loader)
test_epoch(pm, test_loader)



for idx, data in enumerate(train_loader):
    print(data)


test_loader = DataLoader(
    wfd_test, batch_size=batch_size, shuffle=False, pin_memory=True,
    num_workers=num_workers, worker_init_fn=lambda _: np.random.seed(
        int(torch.initial_seed()) % (2 ** 32 - 1)))





# wrap dataset for torch
class DatasetWrapper(Dataset):
    """Wrapper for a dataset to use with PyTorch DataLoader. Its purposes are
    the split into train and validation sets, and application of transforms.

    Parameters
    ----------
    dataset : Dataset
        for GW inference, this is the WaveformDataset
    indices: np.array = None
        indices used for split (e.g., train); entire dataset is used if None
    transforms: transforms = None
        transforms applied to the dataset sample; no transform applied if None
    """

    def __init__(self, dataset, indices=None, transforms=None):
        self.dataset = dataset
        self.indices = indices
        self.transforms = transforms

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        else:
            return len(self.dataset)

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = self.indices[idx]
        sample = self.dataset[idx]
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

