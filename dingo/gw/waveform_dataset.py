from torch.utils.data import Dataset


# TODO:
#  * These will be multi-process classes
#   Only use pytorch rng generators; for numpy rng the seed would be the same --
#   this currently hacked around by setting seeds differently; using pytorch should be easier

class WaveformDataset(Dataset):
    """This class generates, saves, and loads a dataset of simulated waveforms (plus and cross
    polarizations, as well as associated parameter values.

    There are two ways of calling this -- should this be split into two classes?
    1. call with wf generator and extrinsic parameters
    2. call with dataset file and a transform (or composition of transforms -- there is a pytorch way of doing this)
    """

    def __init__(self, dataset_file=None, prior=None, waveform_generator=None, transform=None):
        self.prior = prior

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def generate_dataset(self, size):

        self.parameter_samples = self.prior.sample(size)

        # Look at WaveformDataset.generate_dataset()

        # convert to torch.tensors here?

    def load(self):
        # use SVD to compress more than just HDF5 compression?
        pass

    def save(self):
        pass

    def split_into_train_test(self, train_fraction):
        # of type WaveformDataset
        pass


