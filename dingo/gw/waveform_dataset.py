from torch.utils.data import Dataset


class WaveformDataset(Dataset):
    """This class generates, saves, and loads a dataset of simulated waveforms (plus and cross
    polarizations, as well as associated parameter values."""

    def __init__(self, dataset_file=None, prior=None, waveform_generator=None, transform=None):
        self.prior = prior

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def generate_dataset(self, size):

        self.parameter_samples = self.prior.sample(size)
