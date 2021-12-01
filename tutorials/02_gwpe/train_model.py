import yaml
import torchvision
from torch.utils.data import DataLoader

from bilby.gw.detector import InterferometerList
from dingo.gw.waveform_dataset import WaveformDataset
from dingo.gw.domains import build_domain
from dingo.gw.transforms import SampleExtrinsicParameters,\
    GetDetectorTimes, ProjectOntoDetectors, SampleNoiseASD, \
    WhitenAndScaleStrain, AddWhiteNoiseComplex, \
    SelectStandardizeRepackageParameters, RepackageStrainsAndASDS, UnpackDict
from dingo.gw.noise_dataset import ASDDataset
from dingo.gw.gwutils import *

from dingo.core.nn.nsf import create_nsf_with_rb_projection_embedding_net
from dingo.core.utils.torchutils import *

import numpy as np
import time

with open('./train_dir/train_settings.yaml', 'r') as fp:
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
transforms = []
transforms.append(SampleExtrinsicParameters(extrinsic_prior_dict))
transforms.append(GetDetectorTimes(ifo_list, ref_time))
transforms.append(ProjectOntoDetectors(ifo_list, domain, ref_time))
transforms.append(SampleNoiseASD(asd_dataset))
transforms.append(WhitenAndScaleStrain(domain.noise_std))
transforms.append(AddWhiteNoiseComplex())
transforms.append(SelectStandardizeRepackageParameters(
    standardization_dict))
transforms.append(RepackageStrainsAndASDS(
    train_settings['transform_settings']['detectors']))
transforms.append(UnpackDict(selected_keys=['parameters', 'waveform']))
# set wfd transforms to the composition of the above transforms
wfd.transform = torchvision.transforms.Compose(transforms)

train_dataset, test_dataset = split_dataset_into_train_and_test(
    wfd, train_settings['train_settings']['train_fraction'])

# build dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=train_settings['train_settings']['batch_size'],
    shuffle=True,
    pin_memory=True,
    num_workers=train_settings['train_settings']['num_workers'],
    worker_init_fn=lambda _:np.random.seed(int(torch.initial_seed())%(2**32-1)))

# build model
model = create_nsf_with_rb_projection_embedding_net(
    **train_settings['model_arch']['model_kwargs'])
assert get_number_of_model_parameters(model) == 131448775



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

