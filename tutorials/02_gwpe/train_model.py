import yaml
import argparse
import os

os.environ['OMP_NUM_THREADS'] = str(1)
os.environ['MKL_NUM_THREADS'] = str(1)

from dingo.api import build_dataset, build_train_and_test_loaders, \
    build_posterior_model, resubmit_condor_job
from dingo.core.utils import *

parser = argparse.ArgumentParser(description='Train Dingo.')
parser.add_argument('--train_dir', required=True,
                    help='Train directory for Dingo training. Contains'
                         'train_settings.yaml file, used for logging.')
args = parser.parse_args()

with open(join(args.train_dir, 'train_settings.yaml'), 'r') as fp:
    train_settings = yaml.safe_load(fp)

# build WaveformDataset used for training
wfd = build_dataset(train_settings)

# build DataLoader objects for training and testing
train_loader, test_loader = build_train_and_test_loaders(train_settings, wfd)

# build posterior model; initialize a new one, or load existing model
pm = build_posterior_model(args.train_dir, train_settings, wfd[0])

# train model
pm.train(
    train_loader,
    test_loader,
    train_dir=args.train_dir,
    runtime_limits_kwargs=train_settings['train_settings']['runtime_limits'],
    checkpoint_epochs=train_settings['train_settings']['checkpoint_epochs'],
)

# resubmit condor job
resubmit_condor_job(
    train_dir=args.train_dir,
    train_settings=train_settings,
    epoch=pm.epoch
)







# train_epoch(pm, train_loader)
# test_epoch(pm, test_loader)
#
#
#
# for idx, data in enumerate(train_loader):
#     print(data)
#
#
# test_loader = DataLoader(
#     wfd_test, batch_size=batch_size, shuffle=False, pin_memory=True,
#     num_workers=num_workers, worker_init_fn=lambda _: np.random.seed(
#         int(torch.initial_seed()) % (2 ** 32 - 1)))

#
#
# # wrap dataset for torch
# class DatasetWrapper(Dataset):
#     """Wrapper for a dataset to use with PyTorch DataLoader. Its purposes are
#     the split into train and validation sets, and application of transforms.
#
#     Parameters
#     ----------
#     dataset : Dataset
#         for GW inference, this is the WaveformDataset
#     indices: np.array = None
#         indices used for split (e.g., train); entire dataset is used if None
#     transforms: transforms = None
#         transforms applied to the dataset sample; no transform applied if None
#     """
#
#     def __init__(self, dataset, indices=None, transforms=None):
#         self.dataset = dataset
#         self.indices = indices
#         self.transforms = transforms
#
#     def __len__(self):
#         if self.indices is not None:
#             return len(self.indices)
#         else:
#             return len(self.dataset)
#
#     def __getitem__(self, idx):
#         if self.indices is not None:
#             idx = self.indices[idx]
#         sample = self.dataset[idx]
#         if self.transforms is not None:
#             sample = self.transforms(sample)
#         return sample

