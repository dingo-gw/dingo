import numpy as np
import torch
from bilby.gw.detector import PowerSpectralDensity
from scipy.interpolate import interp1d

from dingo.gw.domains import FrequencyDomain


class SampleNoiseASD(object):
    """
    Sample a random asds for each detector and add them to sample['asds'].
    """

    def __init__(self, asd_dataset):
        self.asd_dataset = asd_dataset

    def __call__(self, input_sample):
        sample = input_sample.copy()
        sample["asds"] = self.asd_dataset.sample_random_asds()
        return sample


class WhitenStrain(object):
    """
    Whiten the strain data by dividing w.r.t. the corresponding asds.
    """

    def __init__(self):
        pass

    def __call__(self, input_sample):
        sample = input_sample.copy()
        ifos = sample["waveform"].keys()
        if ifos != sample["asds"].keys():
            raise ValueError(
                f"Detectors of strain data, {ifos}, do not match "
                f'those of asds, {sample["asds"].keys()}.'
            )
        whitened_strains = {
            ifo: sample["waveform"][ifo] / sample["asds"][ifo] for ifo in ifos
        }
        sample["waveform"] = whitened_strains
        return sample


class WhitenFixedASD(object):
    """
    Whiten frequency-series data according to an ASD specified in a file. This uses the
    ASD files contained in Bilby.
    """

    def __init__(
        self,
        domain: FrequencyDomain,
        asd_file: str = None,
        inverse: bool = False,
        precision=None,
    ):
        """
        Parameters
        ----------
        domain : FrequencyDomain
            ASD is interpolated to the associated frequency grid.
        asd_file : str
            Name of the ASD file. If None, use the aligo ASD.
            [Default: None]
        inverse : bool
            Whether to apply the inverse whitening transform, to un-whiten data.
            [Default: False]
        precision : str ("single", "double")
            If not None, sets precision of ASD to specified precision.

        """
        if asd_file is not None:
            psd = PowerSpectralDensity(asd_file=asd_file)
        else:
            psd = PowerSpectralDensity.from_aligo()

        if psd.frequency_array[-1] < domain.f_max:
            raise ValueError(
                f"ASD in {asd_file} has f_max={psd.frequency_array[-1]}, "
                f"which is lower than domain f_max={domain.f_max}."
            )
        asd_interp = interp1d(
            psd.frequency_array, psd.asd_array, bounds_error=False, fill_value=np.inf
        )
        self.asd_array = asd_interp(domain.sample_frequencies)
        self.asd_array = domain.update_data(self.asd_array, low_value=1e-22)

        if precision is not None:
            if precision == "single":
                self.asd_array = self.asd_array.astype(np.float32)
            elif precision == "double":
                self.asd_array = self.asd_array.astype(np.float64)
            else:
                raise TypeError(
                    'precision can only be changed to "single" or "double".'
                )

        self.inverse = inverse

    def __call__(self, sample):
        """
        Parameters
        ----------
        sample : dict
            Dictionary of numpy arrays, e.g., with keys corresponding to polarizations.
            Method whitens each array with the same ASD.

        Returns
        -------
        dict of the same form as sample, but with whitened / un-whitened data.
        """
        result = {}
        for k, v in sample.items():
            if self.inverse:
                result[k] = v * self.asd_array
            else:
                result[k] = v / self.asd_array
        return result


class WhitenAndScaleStrain(object):
    """
    Whiten the strain data by dividing w.r.t. the corresponding asds,
    and scale it with 1/scale_factor.

    In uniform frequency domain the scale factor should be
    np.sqrt(window_factor) / np.sqrt(4.0 * delta_f).
    It has two purposes:
        (*) the denominator accounts for frequency binning
        (*) dividing by window factor accounts for windowing of strain data
    """

    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, input_sample):
        sample = input_sample.copy()
        ifos = sample["waveform"].keys()
        if ifos != sample["asds"].keys():
            raise ValueError(
                f"Detectors of strain data, {ifos}, do not match "
                f'those of asds, {sample["asds"].keys()}.'
            )
        whitened_strains = {
            ifo: sample["waveform"][ifo] / (sample["asds"][ifo] * self.scale_factor)
            for ifo in ifos
        }
        sample["waveform"] = whitened_strains
        return sample


class AddWhiteNoiseComplex(object):
    """
    Adds white noise with a standard deviation determined by self.scale to the
    complex strain data.
    """

    def __init__(self):
        pass

    def __call__(self, input_sample):
        sample = input_sample.copy()
        noisy_strains = {}
        for ifo, pure_strain in sample["waveform"].items():
            # Use torch rng and convert to numpy, which is slightly faster than using
            # numpy directly. Using torch.randn gives single-precision floats by default
            # (which we want)  whereas np.random.random gives double precision (and
            # must subsequently  be cast to single precision).
            # np.random.default_rng().standard_normal() can be set to output single
            # precision, but in testing this is slightly slower than the torch call.
            noise = (
                torch.randn(len(pure_strain), device=torch.device("cpu"))
                + torch.randn(len(pure_strain), device=torch.device("cpu")) * 1j
            )
            noise = noise.numpy()
            noisy_strains[ifo] = pure_strain + noise
        sample["waveform"] = noisy_strains
        return sample


class RepackageStrainsAndASDS(object):
    """
    Repackage the strains and the asds into an [num_ifos, 3, num_bins]
    dimensional tensor. Order of ifos is provided by self.ifos. By
    convention, [:,i,:] is used for:
        i = 0: strain.real
        i = 1: strain.imag
        i = 2: 1 / (asd * 1e23)
    """

    def __init__(self, ifos, first_index=0):
        self.ifos = ifos
        self.first_index = first_index

    def __call__(self, input_sample):
        sample = input_sample.copy()
        strains = np.empty(
            (len(self.ifos), 3, len(sample["asds"][self.ifos[0]]) - self.first_index),
            dtype=np.float32,
        )
        for idx_ifo, ifo in enumerate(self.ifos):
            strains[idx_ifo, 0] = sample["waveform"][ifo][self.first_index :].real
            strains[idx_ifo, 1] = sample["waveform"][ifo][self.first_index :].imag
            strains[idx_ifo, 2] = 1 / (sample["asds"][ifo][self.first_index :] * 1e23)
        sample["waveform"] = strains
        return sample


# if __name__ == "__main__":
#     AD = ASDDataset("../../../data/PSDs/asds_O1.hdf5")
#     asd_samples = AD.sample_random_asds()
