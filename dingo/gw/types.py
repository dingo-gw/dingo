from __future__ import annotations

from typing import TYPE_CHECKING, NewType, Tuple, TypeAlias, Union

from nptyping import Complex128, Float64, NDArray, Shape

Iota = NewType("Iota", float)
"""
Type for iota (inclination angle between the binary's orbital angular momentum and the line of sight)
"""

F_ref = NewType("F_ref", float)
"""
Type for frequency reference
"""

FrequencySeries: TypeAlias = NDArray[Shape["*"], Complex128]
"""
Waveform frequency series, i.e. a one dimentional numpy array of complex type.
"""

BatchFrequencySeries: TypeAlias = NDArray[Shape["*, *"], Complex128]
"""
Batched waveform frequency series, i.e. a two dimensional numpy array of complex type
with shape (num_waveforms, frequency_bins).
"""

Mode = NewType("Mode", int)
"""
Gravitational wave mode
"""

Modes: TypeAlias = Tuple[Mode, Mode]
"""
Tuple of two modes (the degree of the spherical harmonic mode and the order of the spherical harmonic mode)
"""

# SVD Compression Type Aliases
# -----------------------------
# These types are used for SVD-based waveform compression in dingo.gw.compression.svd

SVDBasisMatrix: TypeAlias = NDArray[Shape["*, *"], Complex128]
"""
SVD basis matrix (V) for waveform compression.

Shape: (n_features, n_components) where:
  - n_features: dimensionality of each waveform (number of frequency/time bins)
  - n_components: number of basis vectors kept (≤ min(n_samples, n_features))

This is the right singular vector matrix V from the SVD decomposition:
    training_data = U @ diag(s) @ Vh
where V = Vh.T.conj()

Each column of V is one complex-valued basis vector that represents a fundamental
pattern in the waveform dataset. Used in SVDBasis.V attribute.
"""

SVDBasisMatrixH: TypeAlias = NDArray[Shape["*, *"], Complex128]
"""
Hermitian conjugate of SVD basis matrix (Vh) for waveform compression.

Shape: (n_components, n_features) where:
  - n_components: number of basis vectors kept
  - n_features: dimensionality of each waveform

This is the Vh matrix from the SVD decomposition: training_data = U @ diag(s) @ Vh
Relationship to V: Vh = V.T.conj()

Each row of Vh is one complex-conjugate-transposed basis vector. This matrix is used
for decompression: waveform ≈ coefficients @ Vh. Used in SVDBasis.Vh attribute.
"""

SingularValues: TypeAlias = NDArray[Shape["*"], Float64]
"""
Singular values from SVD decomposition.

Shape: (n_components,) - 1D array of non-negative real numbers

These are the singular values s from SVD: training_data = U @ diag(s) @ Vh
Ordered from largest to smallest, they represent the "importance" of each basis vector.
Larger values indicate basis vectors that capture more variance in the training data.

Used in SVDBasis.s attribute to track relative importance of basis components.
"""

WaveformTrainingData: TypeAlias = NDArray[Shape["*, *"], Complex128]
"""
Training data for SVD basis generation.

Shape: (n_samples, n_features) where:
  - n_samples: number of training waveforms
  - n_features: dimensionality of each waveform (number of frequency/time bins)

Each row is one complex-valued waveform. This is the input to SVDBasis.generate_basis()
which performs SVD to find a compressed representation of the waveform space.
"""

SVDCoefficients: TypeAlias = NDArray[Shape["*, *"], Complex128]
"""
Compressed SVD coefficients for waveforms.

Shape: (n_samples, n_components) or (n_components,) for single waveform

These are the compressed representations obtained by projecting waveforms onto the SVD
basis: coefficients = waveform @ V. The waveform can be approximately reconstructed
as: waveform ≈ coefficients @ Vh.

The compression reduces dimensionality from n_features to n_components while
preserving the most important information.
"""

if TYPE_CHECKING:
    from lalsimulation.gwsignal.core import waveform
    from lalsimulation.gwsignal.models import pyseobnr_model

    GWSignalGenerators = Union[
        pyseobnr_model.SEOBNRv5HM,
        pyseobnr_model.SEOBNRv5EHM,
        pyseobnr_model.SEOBNRv5PHM,
        waveform.LALCompactBinaryCoalescenceGenerator,
    ]
    """
    Return type of the lalsimulation method gwsignal_get_waveform_generator
    """


class WaveformGenerationError(Exception):
    """
    To be raised when generation of gravitational waveform fails.
    """

    ...
