from typing import Dict

from functools import lru_cache
from abc import ABC, abstractmethod

import torch

from dingo.gw.gwutils import *
from dingo.core.models import PosteriorModel


class Domain(ABC):
    """Defines the physical domain on which the data of interest live.

    This includes a specification of the bins or points,
    and a few additional properties associated with the data.
    """

    @abstractmethod
    def __len__(self):
        """Number of bins or points in the domain"""
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> np.ndarray:
        """Array of bins in the domain"""
        pass

    @abstractmethod
    def time_translate_data(self, data, dt) -> np.ndarray:
        """Time translate strain data by dt seconds."""
        pass

    @property
    @abstractmethod
    def noise_std(self) -> float:
        """Standard deviation of the whitened noise distribution"""
        # FIXME: For this to make sense, it assumes knowledge about how the domain is used in conjunction
        #  with (waveform) data, whitening and adding noise. Is this the best place to define this?
        pass

    @property
    @abstractmethod
    def sampling_rate(self) -> float:
        """The sampling rate of the data [Hz]."""
        pass

    @property
    @abstractmethod
    def f_max(self) -> float:
        """The maximum frequency [Hz] is set to half the sampling rate."""

    @property
    @abstractmethod
    def duration(self) -> float:
        """Waveform duration in seconds."""
        pass

    @property
    @abstractmethod
    def min_idx(self) -> int:
        pass

    @property
    @abstractmethod
    def max_idx(self) -> int:
        pass

    @property
    @abstractmethod
    def domain_dict(self):
        """Enables to rebuild the domain via calling build_domain(domain_dict)."""
        pass


class FrequencyDomain(Domain):
    """Defines the physical domain on which the data of interest live.

    The frequency bins are assumed to be uniform between [0, f_max]
    with spacing delta_f.
    Given a finite length of time domain data, the Fourier domain data
    starts at a frequency f_min and is zero below this frequency.
    window_kwargs specify windowing used for FFT to obtain FD data from TD
    data in practice.
    """

    def __init__(
        self, f_min: float, f_max: float, delta_f: float, window_factor: float = 1.0
    ):
        self._f_min = f_min
        self._f_max = f_max
        self._delta_f = delta_f
        self._window_factor = window_factor

    @staticmethod
    def clear_cache_for_all_instances():
        """
        Whenever self._f_min and self._f_max are modified, this method needs to
        be the called to clear the cached properties such as
        self.sample_frequencies.

        This clears the cache for the corresponding properties for *all*
        class instances.
        """
        FrequencyDomain.sample_frequencies.fget.cache_clear()
        FrequencyDomain.sample_frequencies_torch.fget.cache_clear()
        FrequencyDomain.sample_frequencies_torch_cuda.fget.cache_clear()
        FrequencyDomain.frequency_mask.fget.cache_clear()
        FrequencyDomain.noise_std.fget.cache_clear()

    def update(self, new_settings: dict):
        """
        Update the domain with new settings. This is only allowed if the new settings
        are "compatible" with the old ones. E.g., f_min should be larger than the
        existing f_min.

        Parameters
        ----------
        new_settings : dict
            Settings dictionary. Must contain a subset of the keys contained in
            domain_dict.
        """
        new_settings = new_settings.copy()
        if "type" in new_settings and new_settings.pop("type") not in [
            "FrequencyDomain",
            "FD",
        ]:
            raise ValueError("Cannot update domain to type other than FrequencyDomain.")
        for k, v in new_settings.items():
            if k not in ["f_min", "f_max", "delta_f", "window_factor"]:
                raise KeyError(f"Invalid key for domain update: {k}.")
            if k == "window_factor" and v != self._window_factor:
                raise ValueError("Cannot update window_factor.")
            if k == "delta_f" and v != self._delta_f:
                raise ValueError("Cannot update delta_f.")
        self.set_new_range(
            f_min=new_settings.get("f_min", None), f_max=new_settings.get("f_max", None)
        )

    def set_new_range(self, f_min: float = None, f_max: float = None):
        """
        Set a new range for the domain. This changes the range of the domain to
        [0, f_max], and the truncation range to [f_min, f_max].
        """
        if f_min is not None and f_max is not None and f_min >= f_max:
            raise ValueError("f_min must not be larger than f_max.")
        if f_min is not None:
            if self._f_min <= f_min <= self._f_max:
                self._f_min = f_min
            else:
                raise ValueError(
                    f"f_min = {f_min} is not in expected range "
                    f"[{self._f_min,self._f_max}]."
                )
        if f_max is not None:
            if self._f_min <= f_max <= self._f_max:
                self._f_max = f_max
            else:
                raise ValueError(
                    f"f_max = {f_max} is not in expected range "
                    f"[{self._f_min, self._f_max}]."
                )
        # clear cached properties, such that they are recomputed when needed
        # instead of using the old (incorrect) ones.
        self.clear_cache_for_all_instances()

    def update_data(self, data: np.ndarray, axis: int = -1, low_value: float = 0.0):
        """
        Adjusts data to be compatible with the domain:

            * Below f_min, it sets the data to low_value (typically 0.0 for a waveform,
            but for a PSD this might be a large value).
            * Above f_max, it truncates the data array.

        Parameters
        ----------
        data : np.ndarray
            Data array
        axis : int
            Which data axis to apply the adjustment along.
        low_value : float
            Below f_min, set the data to this value.

        Returns
        -------
        np.ndarray
            The new data array.
        """
        sl = [slice(None)] * data.ndim

        # First truncate beyond f_max.
        sl[axis] = slice(0, self.max_idx + 1)
        data = data[tuple(sl)]

        # Set data value below f_min to low_value.
        sl[axis] = slice(0, self.min_idx)
        data[tuple(sl)] = low_value

        return data

    def time_translate_data(self, data, dt):
        """Time translate complex frequency domain data by dt [in seconds]."""
        if isinstance(data, np.ndarray) and np.iscomplexobj(data):
            f = self.sample_frequencies
            return data * np.exp(-2j * np.pi * dt * f)

        elif isinstance(data, torch.Tensor) and not torch.is_complex(data):
            # add batch dimension if not present
            omit_batch_dimension = False
            if len(data.shape) == 3:
                data = data[None, ...]
                omit_batch_dimension = True
            # expected shape: (batch_size, num_detectors, num_channels, num_fbins).
            # The third axis contains strain.real and strain.imag in channel 0 and 1,
            # and optionally additional channels (e.g., ASD).
            batch_size, Nd, Nc, Nf = data.shape
            cos_txf = torch.empty((batch_size, Nd, Nf), device=data.device)
            sin_txf = torch.empty((batch_size, Nd, Nf), device=data.device)
            if data.is_cuda:
                f = self.sample_frequencies_torch_cuda[self.min_idx :]
            else:
                f = self.sample_frequencies_torch[self.min_idx :]
            assert Nd == len(dt), "Number of detectors does not match."
            assert len(f) == Nf, "Number of frequency bins does not match"
            for idx in range(Nd):
                # get local phases
                txf_det = torch.outer(dt[idx], f)
                cos_txf_det = torch.cos(-2 * np.pi * txf_det)
                sin_txf_det = torch.sin(-2 * np.pi * txf_det)
                cos_txf[:, idx, ...] = cos_txf_det[...]
                sin_txf[:, idx, ...] = sin_txf_det[...]

            x = torch.empty(*data.shape, device=data.device)
            x[:, :, 0, :] = cos_txf * data[:, :, 0, :] - sin_txf * data[:, :, 1, :]
            x[:, :, 1, :] = sin_txf * data[:, :, 0, :] + cos_txf * data[:, :, 1, :]
            x[:, :, 2:, :] = data[:, :, 2:, :]

            if omit_batch_dimension:
                assert x.shape[0] == 1
                x = x[0]

            return x

        else:
            raise NotImplementedError()

    # def time_translate_batch(self, data, dt, axis=None):
    #     # h_d * np.exp(- 2j * np.pi * time_shift * self.sample_frequencies)
    #     if isinstance(data, np.ndarray):
    #         if np.iscomplexobj(data):
    #             pass
    #         else:
    #             pass
    #     elif isinstance(data, torch.Tensor):
    #         pass
    #     else:
    #         raise NotImplementedError(f'Method only implemented for np arrays '
    #                                   f'and torch tensors, got {type(data)}')

    def __len__(self):
        """Number of frequency bins in the domain [0, f_max]"""
        return int(self._f_max / self._delta_f) + 1

    def __call__(self) -> np.ndarray:
        """Array of uniform frequency bins in the domain [0, f_max]"""
        return self.sample_frequencies

    def __getitem__(self, idx):
        """Slice of uniform frequency grid."""
        sample_frequencies = self.__call__()
        return sample_frequencies[idx]

    @property
    @lru_cache()
    def sample_frequencies(self):
        # print('Computing sample_frequencies.') # To understand caching
        num_bins = self.__len__()
        return np.linspace(
            0.0, self._f_max, num=num_bins, endpoint=True, dtype=np.float32
        )

    @property
    @lru_cache()
    def sample_frequencies_torch(self):
        num_bins = self.__len__()
        return torch.linspace(0.0, self._f_max, steps=num_bins, dtype=torch.float32)

    @property
    @lru_cache()
    def sample_frequencies_torch_cuda(self):
        return self.sample_frequencies_torch.to("cuda")

    @property
    @lru_cache()
    def frequency_mask(self) -> np.ndarray:
        """Mask which selects frequency bins greater than or equal to the
        starting frequency"""
        return self.sample_frequencies >= self._f_min

    @property
    def frequency_mask_length(self) -> int:
        """Number of samples in the subdomain domain[frequency_mask]."""
        mask = self.frequency_mask
        return len(np.flatnonzero(np.asarray(mask)))

    @property
    def min_idx(self):
        return round(self._f_min / self._delta_f)

    @property
    def max_idx(self):
        return round(self._f_max / self._delta_f)

    @property
    def window_factor(self):
        return self._window_factor

    @window_factor.setter
    def window_factor(self, value):
        """Set self._window_factor and clear cache of self.noise_std."""
        self._window_factor = value
        FrequencyDomain.noise_std.fget.cache_clear()

    @property
    @lru_cache()
    def noise_std(self) -> float:
        """Standard deviation of the whitened noise distribution.

        To have noise that comes from a multivariate *unit* normal
        distribution, you must divide by this factor. In practice, this means
        dividing the whitened waveforms by this.

        TODO: This description makes some assumptions that need to be clarified.
        Windowing of TD data; tapering window has a slope -> reduces power only for noise,
        but not for the signal which is in the main part unaffected by the taper
        """
        if self._window_factor is None:
            raise ValueError("Window factor needs to be set for noise_std.")
        return np.sqrt(self._window_factor) / np.sqrt(4.0 * self._delta_f)

    @property
    def f_max(self) -> float:
        """The maximum frequency [Hz] is typically set to half the sampling
        rate."""
        return self._f_max

    @property
    def f_min(self) -> float:
        """The minimum frequency [Hz]."""
        return self._f_min

    @property
    def delta_f(self) -> float:
        """The frequency spacing of the uniform grid [Hz]."""
        return self._delta_f

    @property
    def duration(self) -> float:
        """Waveform duration in seconds."""
        return 1.0 / self._delta_f

    @property
    def sampling_rate(self) -> float:
        return 2.0 * self._f_max

    @property
    def domain_dict(self):
        """Enables to rebuild the domain via calling build_domain(domain_dict)."""
        return {
            "type": "FrequencyDomain",
            "f_min": self._f_min,
            "f_max": self._f_max,
            "delta_f": self._delta_f,
            "window_factor": self._window_factor,
        }


class TimeDomain(Domain):
    """Defines the physical time domain on which the data of interest live.

    The time bins are assumed to be uniform between [0, duration]
    with spacing 1 / sampling_rate.
    window_factor is used to compute noise_std().
    """

    def __init__(self, time_duration: float, sampling_rate: float):
        self._time_duration = time_duration
        self._sampling_rate = sampling_rate

    @lru_cache()
    def __len__(self):
        """Number of time bins given duration and sampling rate"""
        return int(self._time_duration * self._sampling_rate)

    @lru_cache()
    def __call__(self) -> np.ndarray:
        """Array of uniform times at which data is sampled"""
        num_bins = self.__len__()
        return np.linspace(
            0.0, self._time_duration, num=num_bins, endpoint=False, dtype=np.float32
        )

    @property
    def delta_t(self) -> float:
        """The size of the time bins"""
        return 1.0 / self._sampling_rate

    @delta_t.setter
    def delta_t(self, delta_t: float):
        self._sampling_rate = 1.0 / delta_t

    @property
    def noise_std(self) -> float:
        """Standard deviation of the whitened noise distribution.

        To have noise that comes from a multivariate *unit* normal
        distribution, you must divide by this factor. In practice, this means
        dividing the whitened waveforms by this.

        In the continuum limit in time domain, the standard deviation of white
        noise would at each point go to infinity, hence the delta_t factor.
        """
        return 1.0 / np.sqrt(2.0 * self.delta_t)

    def time_translate_data(self, data, dt) -> np.ndarray:
        raise NotImplementedError

    @property
    def f_max(self) -> float:
        """The maximum frequency [Hz] is typically set to half the sampling
        rate."""
        return self._sampling_rate / 2.0

    @property
    def duration(self) -> float:
        """Waveform duration in seconds."""
        return self._time_duration

    @property
    def sampling_rate(self) -> float:
        return self._sampling_rate

    @property
    def min_idx(self) -> int:
        return 0

    @property
    def max_idx(self) -> int:
        return round(self._time_duration * self._sampling_rate)

    @property
    def domain_dict(self):
        """Enables to rebuild the domain via calling build_domain(domain_dict)."""
        return {
            "type": "TimeDomain",
            "time_duration": self._time_duration,
            "sampling_rate": self._sampling_rate,
        }


class PCADomain(Domain):
    """TODO"""

    # Not super important right now
    # FIXME: Should this be defined for FD or TD bases or both?
    # Nrb instead of Nf

    @property
    def noise_std(self) -> float:
        """Standard deviation of the whitened noise distribution.

        To have noise that comes from a multivariate *unit* normal
        distribution, you must divide by this factor. In practice, this means
        dividing the whitened waveforms by this.

        In the continuum limit in time domain, the standard deviation of white
        noise would at each point go to infinity, hence the delta_t factor.
        """
        # FIXME
        return np.sqrt(self.window_factor) / np.sqrt(4.0 * self.delta_f)


def build_domain(settings: Dict) -> Domain:
    """
    Instantiate a domain class from settings.

    Parameters
    ----------
    settings : dict
        Dicionary with 'type' key denoting the type of domain, and keys corresponding
        to the kwargs needed to construct the Domain.

    Returns
    -------
    A Domain instance of the correct type.
    """
    if "type" not in settings:
        raise ValueError(
            f'Domain settings must include a "type" key. Settings included '
            f"the keys {settings.keys()}."
        )

    # The settings other than 'type' correspond to the kwargs of the Domain constructor.
    kwargs = {k: v for k, v in settings.items() if k != "type"}
    if settings["type"] in ["FrequencyDomain", "FD"]:
        return FrequencyDomain(**kwargs)
    elif settings["type"] == ["TimeDomain", "TD"]:
        return TimeDomain(**kwargs)
    else:
        raise NotImplementedError(f'Domain {settings["name"]} not implemented.')


def build_domain_for_model(model: PosteriorModel) -> Domain:
    """
    Instantiate a domain class from settings of model.

    Parameters
    ----------
    model: PosteriorModel
        model containing metadata to build the domain

    Returns
    -------
    A Domain instance of the correct type.
    """
    domain = build_domain(model.metadata["dataset_settings"]["domain"])
    if "domain_update" in model.metadata["train_settings"]["data"]:
        domain.update(model.metadata["train_settings"]["data"]["domain_update"])
    domain.window_factor = get_window_factor(
        model.metadata["train_settings"]["data"]["window"]
    )
    return domain


if __name__ == "__main__":
    kwargs = {"f_min": 20, "f_max": 2048, "delta_f": 0.125}
    domain = FrequencyDomain(**kwargs)

    d1 = domain()
    d2 = domain()
    print("Clearing cache.", end=" ")
    domain.clear_cache_for_all_instances()
    print("Done.")
    d3 = domain()

    print("Changing domain range.", end=" ")
    domain.set_new_range(20, 100)
    print("Done.")

    d4 = domain()
    d5 = domain()

    print(len(d1), len(d4))
