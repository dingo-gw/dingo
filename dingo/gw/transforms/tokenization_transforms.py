import torch
import numpy as np


class StrainTokenization(object):
    """
        Divide frequency bins into frequency segments of equal length and add frequency information to
        sample['tokenization_parameters'].
    """
    def __init__(self, num_tokens: int, f_min: float, f_max: float, df: float):
        """
        Parameters
        ----------
        num_tokens : int
            Number of tokens into which the frequency bins should be divided.
        f_min : float
            Minimal frequency value.
        f_max : float
            Maximal frequency value.
        df : float
            Frequency interval between bins.

        """
        num_f = torch.tensor((f_max - f_min)/df)
        self.num_bins_per_token = torch.ceil(num_f / num_tokens).to(int)
        f_token_width = self.num_bins_per_token * df
        self.token_indices = torch.arange(0, num_tokens*self.num_bins_per_token, self.num_bins_per_token, dtype=torch.int)
        assert num_tokens == len(self.token_indices)
        self.f_min_per_token = torch.arange(f_min, f_max, f_token_width)
        self.f_max_per_token = self.f_min_per_token + f_token_width - df
        self.num_padded_f_bins = int(num_tokens * self.num_bins_per_token - num_f)
        self.num_tokens = num_tokens

    def __call__(self, input_sample):
        """
        Parameters
        ----------
        input_sample : Dict
            Value for key "waveform":
            Sample of shape [num_blocks, num_channels, num_bins] where num_blocks=num_ifos (number of interferometers),
            num_channels>=3 (real, imag, auxiliary channels, e.g. asd), and num_bins=number of frequency bins.
        """
        sample = input_sample.copy()

        # pad last dimension
        strain = np.pad(sample["waveform"], ((0, 0), (0, 0), (0, self.num_padded_f_bins)), "constant")
        strain = strain.reshape(strain.shape[0], strain.shape[1], self.num_tokens, self.num_bins_per_token)

        sample["waveform"] = strain
        sample["tokenization_parameters"] = {
            "f_min_per_token": self.f_min_per_token,
            "f_max_per_token": self.f_max_per_token,
            "num_bins_per_token": torch.diff(self.token_indices)
        }

        return sample
