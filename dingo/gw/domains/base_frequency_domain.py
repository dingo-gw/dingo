from abc import ABC, abstractmethod

import numpy as np
import torch

from dingo.gw.domains import Domain


class BaseFrequencyDomain(Domain, ABC):
    def __init__(self):
        self._sample_frequencies = None
        self._sample_frequencies_torch = None
        self._sample_frequencies_torch_cuda = None
        self._delta_f = None

    def __len__(self) -> int:
        """Number of frequency bins in the domain."""
        return len(self.sample_frequencies)

    def __call__(self) -> np.ndarray:
        """Return the frequency array."""
        return self.sample_frequencies

    def __getitem__(self, idx: int) -> float:
        """Index into the frequency array."""
        return self.sample_frequencies[idx]

    @property
    @abstractmethod
    def sample_frequencies(self) -> np.ndarray:
        pass

    @property
    def sample_frequencies_torch(self) -> torch.Tensor:
        if self._sample_frequencies_torch is None:
            self._sample_frequencies_torch = torch.tensor(
                self.sample_frequencies, dtype=torch.float32
            )
        return self._sample_frequencies_torch

    @property
    def sample_frequencies_torch_cuda(self) -> torch.Tensor:
        if self._sample_frequencies_torch_cuda is None:
            self._sample_frequencies_torch_cuda = self.sample_frequencies_torch.to(
                "cuda"
            )
        return self._sample_frequencies_torch_cuda

    def get_sample_frequencies_astype(
        self, data: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """
        Returns a 1D frequency array compatible with the last index of data array.

        Decides whether array is numpy or torch tensor (and cuda vs cpu).

        Parameters
        ----------
        data : Union[np.array, torch.Tensor]
            Sample data

        Returns
        -------
        Frequency array compatible with last index, of the same type as input
        """
        # Type
        if isinstance(data, np.ndarray):
            f = self.sample_frequencies
        elif isinstance(data, torch.Tensor):
            if data.is_cuda:
                f = self.sample_frequencies_torch_cuda
            else:
                f = self.sample_frequencies_torch
        else:
            raise TypeError("Invalid data type. Should be np.array or torch.Tensor.")

        return f

    @property
    @abstractmethod
    def f_min(self) -> float:
        pass

    @property
    @abstractmethod
    def frequency_mask(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def frequency_mask_length(self) -> int:
        pass

    @property
    @abstractmethod
    def window_factor(self) -> float:
        pass

    @window_factor.setter
    @abstractmethod
    def window_factor(self, value: float):
        pass

    @property
    def delta_f(self) -> float | np.ndarray:
        return self._delta_f

    def time_translate_data(
        self, data: np.ndarray | torch.Tensor, dt: float | np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        r"""
        Time translate frequency-domain data by dt. Time translation corresponds (in
        frequency domain) to multiplication by
        $$
        \exp(-2 \pi i f \cdot dt).
        $$

        This method allows for multiple batch dimensions. For torch.Tensor data,
        allow for either a complex or a (real, imag) representation.

        Parameters
        ----------
        data : array-like (numpy, torch)
            Shape (B, C, N), where

                - B corresponds to any dimension >= 0,
                - C is either absent (for complex data) or has dimension >= 2 (for data
                  represented as real and imaginary parts), and
                - N is either len(self) or len(self)-self.min_idx (for truncated data).

        dt : torch tensor, or scalar (if data is numpy)
            Shape (B)

        Returns
        -------
        Array-like of the same form as data.
        """
        f = self.get_sample_frequencies_astype(data)
        if isinstance(data, np.ndarray):
            phase_shift = 2 * np.pi * np.einsum("...,i", dt, f)
        elif isinstance(data, torch.Tensor):
            # Allow for possible multiple "batch" dimensions (e.g., batch + detector,
            # which might have independent time shifts).
            phase_shift = 2 * np.pi * torch.einsum("...,i", dt, f)
        else:
            raise NotImplementedError(
                f"Time translation not implemented for data of type {data}."
            )
        return self.add_phase(data, phase_shift)

    @staticmethod
    def add_phase(
        data: np.ndarray | torch.Tensor, phase: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """
        Add a (frequency-dependent) phase to a frequency series. Allows for batching,
        as well as additional channels (such as detectors). Accounts for the fact that
        the data could be a complex frequency series or real and imaginary parts.

        Convention: the phase $\phi(f)$ is defined via $\exp[- i \phi(f)]$.

        Parameters
        ----------
        data : Union[np.array, torch.Tensor]
        phase : Union[np.array, torch.Tensor]

        Returns
        -------
        New array or tensor of the same shape as data.
        """
        if isinstance(data, np.ndarray) and np.iscomplexobj(data):
            return data * np.exp(-1j * phase)

        elif isinstance(data, torch.Tensor):
            if torch.is_complex(data):
                # Expand the trailing batch dimensions to allow for broadcasting.
                while phase.dim() < data.dim():
                    phase = phase[..., None, :]
                return data * torch.exp(-1j * phase)
            else:
                # The first two components of the second last index should be the real
                # and imaginary parts of the data. Remaining components correspond to,
                # e.g., the ASD. The "-1" below accounts for this extra dimension when
                # broadcasting.
                while phase.dim() < data.dim() - 1:
                    phase = phase[..., None, :]

                cos_phase = torch.cos(phase)
                sin_phase = torch.sin(phase)
                result = torch.empty_like(data)
                result[..., 0, :] = (
                    data[..., 0, :] * cos_phase + data[..., 1, :] * sin_phase
                )
                result[..., 1, :] = (
                    data[..., 1, :] * cos_phase - data[..., 0, :] * sin_phase
                )
                if data.shape[-2] > 2:
                    result[..., 2:, :] = data[..., 2:, :]
                return result

        else:
            raise TypeError(f"Invalid data type {type(data)}.")

    @property
    def noise_std(self) -> float:
        r"""
        Standard deviation per bin for white noise,
        $$
        \sigma_{\mathrm{noise}} = \sqrt{\frac{w}{4 \delta f}}
        $$
        where $w$ is the window factor.

        The window factor arises because of the way frequency domain data is
        constructed from observed time domain data. Generally a window is applied to
        the time domain data before taking the Fourier transform. This reduces the
        power in the noise by $w^2$. However, the signal is assumed to be unaffected by
        the window, which tapers near the edges of the domain. We keep track of this in
        DINGO by generating noise with standard deviation as above.

        To scale noise such that it is consistent with a multivariate *unit* normal
        distribution, you must divide whitened data by the noise_std. For the
        UniformFrequencyDomain, noise_std is a number, as delta_f is constant across
        the domain. For the MultibandedFrequencyDomain, it is an array.
        """
        if self.window_factor is None:
            raise ValueError("Window factor needs to be set for noise_std.")
        return np.sqrt(self.window_factor) / np.sqrt(4.0 * self.delta_f)

    def check_data_compatibility(self, data: np.ndarray) -> bool:
        """
        Check that the trailing dimension of data is compatible with the domain, i.e.,
        compare against the domain length.

        Parameters
        ----------
        data : np.ndarray

        Returns
        -------
        bool
            Whether the data are compatible with domain.
        """
        if data.shape[-1] == len(self.sample_frequencies):
            return True
        else:
            return False
