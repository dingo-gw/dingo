from torch.utils.data import Dataset

from dingo.gw.parameters import GWPriorDict
from dingo.gw.domains import UniformFrequencyDomain, TimeDomain
from dingo.gw.waveform_generator import WaveformGenerator, StandardizedDistribution



# TODO:
#  * here we will want to use a 2D array or pd.DataFrame of the waveform parameter samples
#    since they will be repackaged into torch Tensors and this class will be repeatedly feeding the training loop.
#  * These will be multi-process classes
#   Only use pytorch rng generators; for numpy rng the seed would be the same --
#   this currently hacked around by setting seeds differently; using pytorch should be easier

# At which point do we actually sample?
#  We can posit a joint prior distribution (currently in terms fo bilby priors)
#  and later wrap it in a StandardizedDistribution.
#  When we generate h+/hx we need to already have sampled from the intrinsic parameters
#  Afterwards we want to standardize these parameters: this means that StandardizedDistribution
#  should just operates on numerical values. This does not work with the current implementation
#  since StandardizedDistribution will call .sample() on the base distribution.
# At a later point (in WaveformDataset) we sample over extrinsic parameters and call
#  RandomProjectToDetectors() to compute the strains in the detector network.


class WaveformDataset(Dataset):
    """This class generates, saves, and loads a dataset of simulated waveforms (plus and cross
    polarizations, as well as associated parameter values.

    There are two ways of calling this -- should this be split into two classes?
    1. call with wf generator and extrinsic parameters
    2. call with dataset file and a transform (or composition of transforms -- there is a pytorch way of doing this)
        - example of which transforms should be performed?
    """

    def __init__(self, dataset_file=None,
                 prior: GWPriorDict = None,
                 waveform_generator: WaveformGenerator = None,
                 transform=None):
        self.prior = prior # needs to contain the extrinsic parameters we still need to sample in

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


