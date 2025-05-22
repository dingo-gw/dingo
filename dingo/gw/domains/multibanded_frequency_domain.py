from typing import Iterable, Union, Optional
import numpy as np
import torch
from copy import copy

from .base_frequency_domain import BaseFrequencyDomain
from .uniform_frequency_domain import UniformFrequencyDomain


class MultibandedFrequencyDomain(BaseFrequencyDomain):
    r"""
    Defines a non-uniform frequency domain that is made up of a sequence of
    uniform-frequency domain bands. Each subsequent band in the sequence has double the
    bin-width of the previous one, i.e., delta_f is doubled each band as one moves up
    the bands. This is intended to allow for efficient representation of gravitational
    waveforms, which generally have slower oscillations at higher frequencies. Indeed,
    the leading order chirp has phase evolution [see
    https://doi.org/10.1103/PhysRevD.49.2658],
    $$
    \Psi(f) = \frac{3}{4}(8 \pi \mathcal{M} f)^{-5/3},
    $$
    hence a coarser grid can be used at higher f.

    The domain is partitioned into bands via a sequence of nodes that are specified at
    initialization.

    In comparison to the UniformFrequencyDomain, the MultibandedFrequencyDomain has the
    following key differences:

    * The sample frequencies start at the first node, rather than f = 0.0 Hz.

    * Quantities such as delta_f, noise_std, etc., are represented as arrays rather than
    scalars, as they vary depending on f.

    The MultibandedFrequencyDomain furthermore has an attribute base_domain,
    which holds an underlying UniformFrequencyDomain object. The decimate() method
    decimates data in the base_domain to the multi-banded domain.
    """

    def __init__(
        self,
        nodes: Iterable[float],
        delta_f_initial: float,
        base_domain: Union[UniformFrequencyDomain, dict],
    ):
        """
        Parameters
        ----------
        nodes: Iterable[float]
            Defines the partitioning of the underlying frequency domain into bands. In
            total, there are len(nodes) - 1 frequency bands. Band j consists of
            decimated data from the base domain in the range [nodes[j]:nodes[j+1]).
        delta_f_initial: float
            delta_f of band 0. The decimation factor doubles between adjacent bands,
            so delta_f is doubled as well.
        base_domain: Union[UniformFrequencyDomain, dict]
            Original (uniform frequency) domain of data, which is the starting point
            for the decimation. This determines the decimation details and the noise_std.
            Either provided as dict for build_domain, or as domain_object.
        """
        super().__init__()
        if isinstance(base_domain, dict):
            from .build_domain import build_domain

            base_domain = build_domain(base_domain)

        self.nodes = np.array(nodes, dtype=np.float32)
        self.base_domain = base_domain
        self._initialize_bands(delta_f_initial)
        if not isinstance(self.base_domain, UniformFrequencyDomain):
            raise ValueError(
                f"Expected domain type UniformFrequencyDomain, got {type(base_domain)}."
            )
        # truncation indices for domain update
        self._range_update_idx_lower = None
        self._range_update_idx_upper = None
        self._range_update_initial_length = None

    def _initialize_bands(self, delta_f_initial: float):
        if len(self.nodes.shape) != 1:
            raise ValueError(
                f"Expected format [num_bands + 1] for nodes, "
                f"got {self.nodes.shape}."
            )
        self.num_bands = len(self.nodes) - 1
        self._nodes_indices = (self.nodes / self.base_domain.delta_f).astype(int)

        self._delta_f_bands = (
            delta_f_initial * (2 ** np.arange(self.num_bands))
        ).astype(np.float32)
        self._decimation_factors_bands = (
            self._delta_f_bands / self.base_domain.delta_f
        ).astype(int)
        self._num_bins_bands = (
            (self._nodes_indices[1:] - self._nodes_indices[:-1])
            / self._decimation_factors_bands
        ).astype(int)

        self._band_assignment = np.concatenate(
            [
                np.ones(num_bins_band, dtype=int) * idx
                for idx, num_bins_band in enumerate(self._num_bins_bands)
            ]
        )
        self._delta_f = self._delta_f_bands[self._band_assignment]

        # For each bin, [self._f_base_lower, self._f_base_upper] describes the
        # frequency range in the base domain which is used for truncation.
        self._f_base_lower = np.concatenate(
            (self.nodes[:1], self.nodes[0] + np.cumsum(self._delta_f[:-1]))
        )
        self._f_base_upper = (
            self.nodes[0] + np.cumsum(self._delta_f) - self.base_domain.delta_f
        )

        # Set sample frequencies as mean of decimation range.
        self._sample_frequencies = (self._f_base_upper + self._f_base_lower) / 2
        self._sample_frequencies_torch = None
        self._sample_frequencies_torch_cuda = None
        # sample_frequencies should always be the decimation of the base domain
        # frequencies.

        if self.f_min not in self.base_domain() or self.f_max not in self.base_domain():
            raise ValueError(
                f"Endpoints ({self.f_min}, {self.f_max}) not in base "
                f"domain, {self.base_domain.domain_dict}"
            )

        # Note that f_max from the base domain can differ from that of the multi-banded
        # domain. This occurs as a boundary effect, since a fixed number of bins in the
        # base domain are averaged to obtain a bin in the multi-banded domain. When
        # decimating, any extra bins in the base domain are dropped.

    def decimate(self, data: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Decimate data from the base_domain to the multi-banded domain.

        Parameters
        ----------
        data : array-like (np.ndarray or torch.Tensor)
            Decimation is done along the trailing dimension of this array. This
            dimension should therefore be compatible with the base frequency domain,
            i.e., running from 0.0 Hz or f_min up to f_max, with uniform delta_f.

        Returns
        -------
        Decimated array of the same type as the input.
        """
        if data.shape[-1] == len(self.base_domain):
            offset_idx = 0
        elif data.shape[-1] == len(self.base_domain) - self.base_domain.min_idx:
            offset_idx = -self.base_domain.min_idx
        else:
            raise ValueError(
                f"Provided data has {data.shape[-1]} bins, which is incompatible with "
                f"the expected domain of length {len(self.base_domain)}"
            )
        if isinstance(data, np.ndarray):
            data_decimated = np.empty((*data.shape[:-1], len(self)), dtype=data.dtype)
        elif isinstance(data, torch.Tensor):
            data_decimated = torch.empty(
                (*data.shape[:-1], len(self)), dtype=data.dtype
            )
        else:
            raise NotImplementedError(
                f"Decimation not implemented for data of type {data}."
            )

        lower_out = 0  # running index for decimated band data
        for idx_band in range(self.num_bands):
            lower_in = self._nodes_indices[idx_band] + offset_idx
            upper_in = self._nodes_indices[idx_band + 1] + offset_idx
            decimation_factor = self._decimation_factors_bands[idx_band]
            num_bins = self._num_bins_bands[idx_band]

            data_decimated[..., lower_out : lower_out + num_bins] = decimate_uniform(
                data[..., lower_in:upper_in], decimation_factor
            )
            lower_out += num_bins

        assert lower_out == len(self)

        return data_decimated

    def update(self, new_settings: dict):
        """
        Update the domain by truncating the frequency range (by specifying new f_min,
        f_max).

        After calling this function, data from the original domain can be truncated to
        the new domain using self.update_data(). For simplicity, we do not allow for
        multiple updates of the domain.

        Parameters
        ----------
        new_settings : dict
            Settings dictionary. Keys must either be the keys contained in domain_dict, or
            a subset of ["f_min", "f_max"].
        """
        if set(new_settings.keys()).issubset(["f_min", "f_max"]):
            self._set_new_range(**new_settings)
        elif set(new_settings.keys()) == self.domain_dict.keys():
            if new_settings == self.domain_dict:
                return
            self._set_new_range(
                f_min=new_settings["base_domain"]["f_min"],
                f_max=new_settings["base_domain"]["f_max"],
            )
            if self.domain_dict != new_settings:
                raise ValueError(
                    f"Update settings {new_settings} are incompatible with "
                    f"domain settings {self.domain_dict}."
                )
        else:
            raise ValueError(
                f"Invalid argument for domain update {new_settings}. Must either be "
                f'{list(self.domain_dict.keys())} or a subset of ["f_min, f_max"].'
            )

    def _set_new_range(
        self, f_min: Optional[float] = None, f_max: Optional[float] = None
    ):
        """
        Set a new range [f_min, f_max] for the domain. This operation is only allowed
        if the new range is contained within the old one.

        Note: f_min, f_max correspond to the range in the *base_domain*.

        Parameters
        ----------
        f_min : float
            New minimum frequency (optional).
        f_max : float
            New maximum frequency (optional).
        """
        if f_min is None and f_max is None:
            return
        if self._range_update_initial_length is not None:
            raise ValueError(f"Can't update domain of type {type(self)} a second time.")
        if f_min is not None and f_max is not None and f_min >= f_max:
            raise ValueError("f_min must not be larger than f_max.")

        lower_bin, upper_bin = 0, len(self) - 1

        if f_min is not None:
            if self._f_base_lower[0] <= f_min <= self._f_base_lower[-1]:
                # find new starting bin (first element with f >= f_min)
                lower_bin = np.where(self._f_base_lower >= f_min)[0][0]
            else:
                raise ValueError(
                    f"f_min = {f_min} is not in expected range "
                    f"[{self._f_base_lower[-1], self._f_base_lower[-1]}]."
                )

        if f_max is not None:
            if self._f_base_upper[0] <= f_max <= self._f_base_upper[-1]:
                # find new final bin (last element where f <= f_max)
                upper_bin = np.where(self._f_base_upper <= f_max)[0][-1]
            else:
                raise ValueError(
                    f"f_min = {f_min} is not in expected range "
                    f"[{self._f_base_lower[-1], self._f_base_lower[-1]}]."
                )

        lower_band = self._band_assignment[lower_bin]
        upper_band = self._band_assignment[upper_bin]
        # new nodes extend to upper_band + 2: we have +1 from the exclusive end index
        # and +1, as we have num_bands + 1 elements in nodes
        nodes_new = copy(self.nodes)[lower_band : upper_band + 2]
        nodes_new[0] = self._f_base_lower[lower_bin]
        nodes_new[-1] = self._f_base_upper[upper_bin] + self.base_domain.delta_f

        # Update base_domain f_min and f_max. These values might differ slightly from
        # the final values for the multi-banded domain due to edge effects. Note that
        # we do this update *after* all of our validation checks but *before* changing
        # the state of the class.
        self.base_domain.update({"f_min": f_min, "f_max": f_max})

        self._range_update_initial_length = len(self)
        self._range_update_idx_lower = lower_bin
        self._range_update_idx_upper = upper_bin

        self.nodes = nodes_new
        self._initialize_bands(self._delta_f_bands[lower_band])
        assert self._range_update_idx_upper - self._range_update_idx_lower + 1 == len(
            self
        )

        assert self.base_domain.f_min <= self.f_min
        assert self.base_domain.f_max >= self.f_max

    def update_data(
        self, data: np.ndarray | torch.Tensor, axis: int = -1, **kwargs
    ) -> np.ndarray | torch.Tensor:
        """
        Truncates the data array to be compatible with the domain. This is used when
        changing f_min or f_max.

        update_data() will only have an effect after updating the domain to have a new
        frequency range using self.update().

        Parameters
        ----------
        data : array-like (np.ndarray or torch.Tensor)
            Array should be compatible with either the original or updated
            MultibandedFrequencyDomain along the specified axis. In the latter
            case, nothing is done. In the former, data are truncated appropriately.
        axis: int
            Axis along which to operate.

        Returns
        -------
        Updated data of the same type as input.
        """
        if data.shape[axis] == len(self):
            return data
        elif (
            self._range_update_initial_length is not None
            and data.shape[axis] == self._range_update_initial_length
        ):
            sl = [slice(None)] * data.ndim
            # First truncate beyond f_max.
            sl[axis] = slice(
                self._range_update_idx_lower, self._range_update_idx_upper + 1
            )
            data = data[tuple(sl)]
            return data
        else:
            raise ValueError(
                f"Data (shape {data.shape}) incompatible with the domain "
                f"(length {len(self)})."
            )

    @property
    def sample_frequencies(self) -> np.ndarray:
        return self._sample_frequencies

    @property
    def frequency_mask(self) -> np.ndarray:
        """Array of len(self) consisting of ones.

        As the MultibandedFrequencyDomain starts from f_min, no masking is generally
        required."""
        return np.ones_like(self.sample_frequencies)

    @property
    def frequency_mask_length(self) -> int:
        return len(self.frequency_mask)

    @property
    def min_idx(self):
        return 0

    @property
    def max_idx(self):
        return len(self) - 1

    @property
    def window_factor(self):
        return self.base_domain.window_factor

    @window_factor.setter
    def window_factor(self, value):
        """Set window factor of base domain."""
        self.base_domain.window_factor = float(value)

    @property
    def f_max(self) -> float:
        return self._f_base_upper[-1]

    @property
    def f_min(self) -> float:
        return self._f_base_lower[0]

    @property
    def duration(self) -> float:
        raise NotImplementedError()

    @property
    def sampling_rate(self) -> float:
        raise NotImplementedError()

    @property
    def domain_dict(self) -> dict:
        """Enables to rebuild the domain via calling build_domain(domain_dict)."""
        # Call tolist() on self.bands, such that it can be saved as str for metadata.
        return {
            "type": "MultibandedFrequencyDomain",
            "nodes": self.nodes.tolist(),
            "delta_f_initial": self._delta_f_bands[0].item(),
            "base_domain": self.base_domain.domain_dict,
        }


######################
### util functions ###
######################


def decimate_uniform(data, decimation_factor: int):
    """
    Reduce dimension of data by decimation_factor along last axis, by uniformly
    averaging sets of decimation_factor neighbouring bins.

    Parameters
    ----------
    data
        Array or tensor to be decimated.
    decimation_factor
        Factor by how much to compress. Needs to divide data.shape[-1].
    Returns
    -------
    data_decimated
        Uniformly decimated data, as array or tensor.
        Shape (*data.shape[:-1], data.shape[-1]/decimation_factor).
    """
    if data.shape[-1] % decimation_factor != 0:
        raise ValueError(
            f"data.shape[-1] ({data.shape[-1]} is not a multiple of decimation_factor "
            f"({decimation_factor})."
        )
    if isinstance(data, np.ndarray):
        return (
            np.sum(np.reshape(data, (*data.shape[:-1], -1, decimation_factor)), axis=-1)
            / decimation_factor
        )
    elif isinstance(data, torch.Tensor):
        return (
            torch.sum(
                torch.reshape(data, (*data.shape[:-1], -1, decimation_factor)), dim=-1
            )
            / decimation_factor
        )
    else:
        raise NotImplementedError(
            f"Decimation not implemented for data of type {data}."
        )
