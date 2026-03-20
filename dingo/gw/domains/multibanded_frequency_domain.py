from typing import Iterable, Union, Optional
import numpy as np
import torch
from copy import copy

from .base import DomainParameters
from .base_frequency_domain import BaseFrequencyDomain
from .uniform_frequency_domain import UniformFrequencyDomain
from .binning.adaptive_binning import (
    BinningParameters,
    Band,
    compute_adaptive_binning,
    decimate_uniform as _new_decimate_uniform,
)

_module_import_path = "dingo.gw.domains.multibanded_frequency_domain"


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

    The MultibandedFrequencyDomain can be constructed either with:
    - base_domain: a UniformFrequencyDomain object (or dict) — legacy API
    - base_delta_f: a float scalar — new API from dingo-waveform

    The MultibandedFrequencyDomain furthermore has an attribute base_domain,
    which holds an underlying UniformFrequencyDomain object. The decimate() method
    decimates data in the base_domain to the multi-banded domain.
    """

    def __init__(
        self,
        nodes: Iterable[float],
        delta_f_initial: float,
        base_domain: Union[UniformFrequencyDomain, dict, None] = None,
        base_delta_f: Optional[float] = None,
        window_factor: Optional[float] = None,
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
        base_domain: Union[UniformFrequencyDomain, dict, None]
            Original (uniform frequency) domain of data, which is the starting point
            for the decimation. Either provided as dict for build_domain, or as domain object.
        base_delta_f: Optional[float]
            Base uniform frequency spacing. Alternative to base_domain — if provided
            without base_domain, a base_domain will not be created.
        window_factor: Optional[float]
            Window factor, preserved for narrowing and parameter serialization.
        """
        super().__init__()

        # Resolve base_delta_f from either base_domain or base_delta_f parameter
        if base_domain is not None and base_delta_f is not None:
            raise ValueError(
                "Provide either base_domain or base_delta_f, not both."
            )
        if base_domain is None and base_delta_f is None:
            raise ValueError(
                "Must provide either base_domain or base_delta_f."
            )

        if base_domain is not None:
            if isinstance(base_domain, dict):
                from .build_domain import build_domain
                base_domain = build_domain(base_domain)
            if not isinstance(base_domain, UniformFrequencyDomain):
                raise ValueError(
                    f"Expected domain type UniformFrequencyDomain, got {type(base_domain)}."
                )
            self.base_domain = base_domain
            self._base_delta_f = float(base_domain.delta_f)
        else:
            self._base_delta_f = float(base_delta_f)
            self.base_domain = None

        self.window_factor = window_factor
        self.nodes = np.array(nodes, dtype=np.float32)

        # Build binning parameters using the new adaptive binning system
        self._binning: BinningParameters = compute_adaptive_binning(
            nodes=list(self.nodes),
            delta_f_initial=float(delta_f_initial),
            base_delta_f=self._base_delta_f,
        )
        self.num_bands = int(self._binning.num_bands)

        # Legacy arrays (kept for backward compatibility with existing code that
        # accesses these directly)
        self._nodes_indices = self._binning.nodes_indices
        self._delta_f_bands = self._binning.delta_f_bands
        self._decimation_factors_bands = self._binning.decimation_factors_bands
        self._num_bins_bands = self._binning.num_bins_bands
        self._band_assignment = self._binning.band_assignment
        self._delta_f = self._binning.delta_f
        self._f_base_lower = self._binning.f_base_lower
        self._f_base_upper = self._binning.f_base_upper

        # Set sample frequencies as mean of decimation range.
        self._sample_frequencies = (
            (self._f_base_upper + self._f_base_lower) / 2
        ).astype(np.float32)
        self._sample_frequencies_torch = None
        self._sample_frequencies_torch_cuda = None

        # Validate endpoints against base_domain if it exists
        if self.base_domain is not None:
            if self.f_min not in self.base_domain() or self.f_max not in self.base_domain():
                raise ValueError(
                    f"Endpoints ({self.f_min}, {self.f_max}) not in base "
                    f"domain, {self.base_domain.domain_dict}"
                )

        # truncation indices for domain update
        self._range_update_idx_lower = None
        self._range_update_idx_upper = None
        self._range_update_initial_length = None

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
        if self.base_domain is not None:
            # Legacy behavior: detect offset from base_domain
            if data.shape[-1] == len(self.base_domain):
                offset_idx = 0
            elif data.shape[-1] == len(self.base_domain) - self.base_domain.min_idx:
                offset_idx = -self.base_domain.min_idx
            else:
                raise ValueError(
                    f"Provided data has {data.shape[-1]} bins, which is incompatible with "
                    f"the expected domain of length {len(self.base_domain)}"
                )
            return self._binning.decimate(
                data, base_offset_idx=offset_idx, mode="explicit", policy="mean"
            )
        else:
            # New API: use auto mode to infer offset
            return self._binning.decimate(
                data, mode="auto", base_offset_idx=0, policy="pick"
            )

    def narrowed(
        self,
        f_min: Optional[float] = None,
        f_max: Optional[float] = None,
    ) -> "MultibandedFrequencyDomain":
        """
        Return a new MultibandedFrequencyDomain narrowed to [f_min, f_max] within this domain.
        The current instance is not modified.

        Parameters
        ----------
        f_min : Optional[float]
            New minimum frequency (None to keep current).
        f_max : Optional[float]
            New maximum frequency (None to keep current).

        Returns
        -------
        MultibandedFrequencyDomain
            A new narrowed instance.
        """
        if self._binning.total_bins == 0:
            raise ValueError("Domain has no bins; cannot narrow.")

        f_min_req = self.f_min if f_min is None else float(f_min)
        f_max_req = self.f_max if f_max is None else float(f_max)
        if f_min_req >= f_max_req:
            raise ValueError("f_min must be strictly smaller than f_max.")

        if not (self.f_min <= f_min_req <= self.f_max):
            raise ValueError(f"Requested f_min={f_min_req} not in [{self.f_min}, {self.f_max}].")
        if not (self.f_min <= f_max_req <= self.f_max):
            raise ValueError(f"Requested f_max={f_max_req} not in [{self.f_min}, {self.f_max}].")

        # Identify surviving bin range in the old domain
        try:
            lower_idx = int(np.where(self._binning.f_base_lower >= f_min_req)[0][0])
            upper_idx = int(np.where(self._binning.f_base_upper <= f_max_req)[0][-1])
        except IndexError:
            raise ValueError("Requested range does not align with bin boundaries.")

        lower_band = int(self._binning.band_assignment[lower_idx])
        upper_band = int(self._binning.band_assignment[upper_idx])

        # Build new nodes covering [lower_band : upper_band], adjust endpoints
        nodes_old = self.nodes.copy()
        nodes_new = nodes_old[lower_band : upper_band + 2]
        nodes_new[0] = float(self._binning.f_base_lower[lower_idx])
        nodes_new[-1] = float(self._binning.f_base_upper[upper_idx] + self._base_delta_f)

        # New delta_f_initial derived from the band that becomes the new first band
        new_delta_f_initial = float(self._binning.delta_f_bands[lower_band])

        return MultibandedFrequencyDomain(
            nodes=nodes_new,
            delta_f_initial=new_delta_f_initial,
            base_delta_f=self._base_delta_f,
            window_factor=self.window_factor,
        )

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
        nodes_new[-1] = self._f_base_upper[upper_bin] + self._base_delta_f

        # Update base_domain f_min and f_max if it exists.
        if self.base_domain is not None:
            self.base_domain.update({"f_min": f_min, "f_max": f_max})

        self._range_update_initial_length = len(self)
        self._range_update_idx_lower = lower_bin
        self._range_update_idx_upper = upper_bin

        self.nodes = nodes_new
        self._reinitialize_bands(self._delta_f_bands[lower_band])
        assert self._range_update_idx_upper - self._range_update_idx_lower + 1 == len(
            self
        )

        if self.base_domain is not None:
            assert self.base_domain.f_min <= self.f_min
            assert self.base_domain.f_max >= self.f_max

    def _reinitialize_bands(self, delta_f_initial: float):
        """Reinitialize binning after a range update."""
        self._binning = compute_adaptive_binning(
            nodes=list(self.nodes),
            delta_f_initial=float(delta_f_initial),
            base_delta_f=self._base_delta_f,
        )
        self.num_bands = int(self._binning.num_bands)

        # Update legacy arrays
        self._nodes_indices = self._binning.nodes_indices
        self._delta_f_bands = self._binning.delta_f_bands
        self._decimation_factors_bands = self._binning.decimation_factors_bands
        self._num_bins_bands = self._binning.num_bins_bands
        self._band_assignment = self._binning.band_assignment
        self._delta_f = self._binning.delta_f
        self._f_base_lower = self._binning.f_base_lower
        self._f_base_upper = self._binning.f_base_upper

        # Recalculate sample frequencies
        self._sample_frequencies = (
            (self._f_base_upper + self._f_base_lower) / 2
        ).astype(np.float32)
        self._sample_frequencies_torch = None
        self._sample_frequencies_torch_cuda = None

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
    def base_delta_f(self) -> float:
        """Base uniform frequency spacing."""
        return self._base_delta_f

    @property
    def fbase(self):
        """Tuple of (f_base_lower, f_base_upper) arrays from binning."""
        return self._f_base_lower, self._f_base_upper

    @property
    def bands(self):
        """Tuple of per-band metadata objects from binning."""
        return self._binning.bands

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
    def f_max(self) -> float:
        if self._f_base_upper.size == 0:
            return 0.0
        return float(self._f_base_upper[-1])

    @property
    def f_min(self) -> float:
        if self._f_base_lower.size == 0:
            return 0.0
        return float(self._f_base_lower[0])

    @property
    def duration(self) -> float:
        raise NotImplementedError()

    @property
    def sampling_rate(self) -> float:
        raise NotImplementedError()

    @property
    def domain_dict(self) -> dict:
        """Enables to rebuild the domain via calling build_domain(domain_dict)."""
        # Call tolist() on self.nodes, such that it can be saved as str for metadata.
        d = {
            "type": "MultibandedFrequencyDomain",
            "nodes": self.nodes.tolist(),
            "delta_f_initial": self._delta_f_bands[0].item(),
        }
        if self.base_domain is not None:
            d["base_domain"] = self.base_domain.domain_dict
        else:
            d["base_delta_f"] = self._base_delta_f
        return d

    def get_parameters(self) -> DomainParameters:
        """
        Get the parameters of the multibanded frequency domain.

        Returns
        -------
        DomainParameters
        """
        return DomainParameters(
            f_max=self.f_max,
            f_min=self.f_min,
            delta_t=0.5 / self.f_max if self.f_max > 0 else None,
            nodes=self.nodes.tolist(),
            delta_f_initial=float(self._delta_f_bands[0]) if self.num_bands > 0 else None,
            base_delta_f=self._base_delta_f,
            window_factor=self.window_factor,
            type=f"{_module_import_path}.MultibandedFrequencyDomain",
        )

    @classmethod
    def from_parameters(
        cls, domain_parameters: DomainParameters
    ) -> "MultibandedFrequencyDomain":
        """
        Create a MultibandedFrequencyDomain from DomainParameters.
        """
        for attr in ("nodes", "delta_f_initial", "base_delta_f"):
            if getattr(domain_parameters, attr) is None:
                raise ValueError(
                    "Can not construct MultibandedFrequencyDomain from "
                    f"{domain_parameters}: {attr} should not be None"
                )

        return cls(
            nodes=domain_parameters.nodes,
            delta_f_initial=domain_parameters.delta_f_initial,
            base_delta_f=domain_parameters.base_delta_f,
            window_factor=domain_parameters.window_factor,
        )


def adapt_data(
    former_domain: MultibandedFrequencyDomain,
    new_domain: MultibandedFrequencyDomain,
    data: Union[np.ndarray, torch.Tensor],
    axis: int = -1,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Slice data from a former MultibandedFrequencyDomain to a narrowed one.

    Parameters
    ----------
    former_domain : MultibandedFrequencyDomain
        The original domain that the data was generated on.
    new_domain : MultibandedFrequencyDomain
        The narrowed domain (must be a subrange of former_domain).
    data : np.ndarray | torch.Tensor
        Data array compatible with former_domain along the specified axis.
    axis : int
        Axis along which to slice.

    Returns
    -------
    Sliced data matching new_domain.
    """
    if data.shape[axis] != len(former_domain):
        raise ValueError(
            f"Data trailing length {data.shape[axis]} does not match source domain length {len(former_domain)}."
        )

    # Map new domain's [f_min, f_max] into indices on the old domain
    try:
        lower_idx = int(np.where(former_domain.fbase[0] >= new_domain.f_min)[0][0])
        upper_idx = int(np.where(former_domain.fbase[1] <= new_domain.f_max)[0][-1])
    except IndexError:
        raise ValueError("New domain range is not a subrange of the old domain.")

    # Safety: ensure number of bins aligns exactly
    if (upper_idx - lower_idx + 1) != len(new_domain):
        raise ValueError(
            "Inconsistent bin mapping between domains; cannot slice data safely."
        )

    sl = [slice(None)] * data.ndim
    sl[axis] = slice(lower_idx, upper_idx + 1)
    return data[tuple(sl)]


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
