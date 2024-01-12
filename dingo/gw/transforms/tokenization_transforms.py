import torch
import numpy as np


class StrainTokenization(object):
    """
    Divide frequency bins into frequency segments of equal length and add frequency information and
    encoding of blocks (i.e. interferometers in GW use case) to sample. It is assumed that f_min
    and f_max are the same for all blocks, that all waveforms contain the same number of blocks
    and that the ordering of the blocks within 'waveform' is fixed.
    """

    def __init__(self, num_tokens: int, f_min: float, f_max: float, df: float):
        """
        Parameters
        ----------
        num_tokens: int
            Number of tokens into which the frequency bins should be divided.
        f_min: float
            Minimal frequency value.
        f_max: float
            Maximal frequency value.
        df: float
            Frequency interval between bins.

        """
        num_f = torch.tensor((f_max - f_min) / df) + 1
        self.num_bins_per_token = torch.ceil(num_f / num_tokens).to(int)
        f_token_width = self.num_bins_per_token * df
        self.token_indices = torch.arange(
            0,
            num_tokens * self.num_bins_per_token,
            self.num_bins_per_token,
            dtype=torch.int,
        )
        assert num_tokens == len(self.token_indices)
        self.f_min_per_token = torch.arange(f_min, f_max, f_token_width)
        self.f_max_per_token = self.f_min_per_token + f_token_width - df
        self.num_padded_f_bins = int(num_tokens * self.num_bins_per_token - num_f)
        self.num_tokens = num_tokens

    def __call__(self, input_sample):
        """
        Parameters
        ----------
        input_sample: Dict
            Value for key 'waveform':
            Sample of shape [num_blocks, num_channels, num_bins]
            where num_blocks = number of detectors in GW use case,
            num_channels>=3 (real, imag, auxiliary channels, e.g. asd),
            and num_bins=number of frequency bins.

        Returns
        ----------
        sample: Dict
            input_sample with modified value for key
            - 'waveform', shape [num_blocks, num_channels, num_tokens, num_bins_per_token]
            and additional keys
            - 'blocks', shape [num_blocks]
            - 'f_min_per_token', shape [num_tokens]
            - 'f_max_per_token', shape [num_tokens]
        """
        sample = input_sample.copy()

        # pad last dimension
        strain = np.pad(
            sample["waveform"],
            ((0, 0), (0, 0), (0, self.num_padded_f_bins)),
            "constant",
        )
        strain = strain.reshape(
            strain.shape[0], strain.shape[1], self.num_tokens, self.num_bins_per_token
        )

        sample["waveform"] = strain
        detector_dict = {"H1": 0, "L1": 1, "V1": 2}
        detectors = [detector_dict[key] for key in input_sample["asds"]]
        sample["blocks"] = torch.Tensor(detectors)
        sample["f_min_per_token"] = self.f_min_per_token
        sample["f_max_per_token"] = self.f_max_per_token

        return sample
