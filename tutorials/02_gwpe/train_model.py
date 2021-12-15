import yaml

from dingo.api import build_dataset, build_train_and_test_loaders
from dingo.core.nn.nsf import create_nsf_with_rb_projection_embedding_net, \
    autocomplete_model_kwargs_nsf # move to api, since it contains train settings?
from dingo.core.models.posterior_model import PosteriorModel, train_epoch, \
    test_epoch
from dingo.core.utils import *

import argparse

parser = argparse.ArgumentParser(description='Train Dingo.')
parser.add_argument('--log_dir', required=True,
                    help='Log directory for Dingo training. Contains'
                         'train_settings.yaml file, used for logging.')
args = parser.parse_args()

with open(join(args.log_dir, 'train_settings.yaml'), 'r') as fp:
    train_settings = yaml.safe_load(fp)

# build WaveformDataset used for training
wfd = build_dataset(train_settings)

# build DataLoader objects for training and testing
train_loader, test_loader = build_train_and_test_loaders(train_settings, wfd)

# build model
if not isfile(join(args.log_dir, 'model_latest.pt')):
    pm_kwargs = {
        # autocomplete model kwargs in train settings
        'model_kwargs': autocomplete_model_kwargs_nsf(train_settings, wfd[0]),
        'optimizer_kwargs': train_settings['train_settings'][
            'optimizer_kwargs'],
        'scheduler_kwargs': train_settings['train_settings'][
            'scheduler_kwargs'],
    }
else:
    pm_kwargs = {'model_filename':join(args.log_dir, 'model_latest.pt')}

pm = PosteriorModel(model_builder=create_nsf_with_rb_projection_embedding_net,
                    init_for_training=True, device='cpu', **pm_kwargs)
# assert get_number_of_model_parameters(pm.model) == 131448775

# train model
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

