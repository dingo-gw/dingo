from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    import torch  # for typing without hard runtime dep

# 1D float32 vector (e.g., per-band or per-bin frequency arrays)
F32Vec = np.ndarray[tuple[int], np.dtype[np.float32]]
# 1D integer vector (indices, counts)
IntVec = np.ndarray[tuple[int], np.dtype[np.int_]]
# 2D integer matrix with exactly two columns (e.g., [start, end) ranges)
IntMat2 = np.ndarray[tuple[int, Literal[2]], np.dtype[np.int_]]


@dataclass(frozen=True)
class Band:
    """
    Immutable, per-band metadata and bin-range view.

    Fields
    ------
    - node_lower, node_upper: frequency boundaries of the band (float32).
    - node_lower_idx (inclusive), node_upper_idx_exclusive: corresponding base-grid index boundaries (int).
    - delta_f_band (float32): per-bin spacing for this band.
    - decimation_factor_band (int): integer decimation relative to base_delta_f.
    - num_bins (int): number of output bins in this band (remainder dropped).
    - remainder (int): leftover base samples not covered by bins in this band.
    - bin_start, bin_end (int): half-open [start, end) slice into global per-bin arrays.
    """

    index: int

    # Band boundaries in frequency space
    node_lower: float
    node_upper: float

    # Band boundaries in base-grid index space
    node_lower_idx: int  # inclusive index
    node_upper_idx_exclusive: int  # exclusive index

    # Per-band numerics
    delta_f_band: float
    decimation_factor_band: int
    num_bins: int
    remainder: int

    # Bin range into the global per-bin arrays (half-open)
    bin_start: int
    bin_end: int

    @property
    def bin_slice(self) -> slice:
        """Slice to select this band's bins from global per-bin arrays."""
        return slice(self.bin_start, self.bin_end)

    @property
    def band_width_indices(self) -> int:
        """Width of the band in base-grid indices."""
        return self.node_upper_idx_exclusive - self.node_lower_idx

    @property
    def covered_base_samples(self) -> int:
        """Number of base-grid samples covered by bins in this band."""
        return self.num_bins * self.decimation_factor_band

    @property
    def coverage_ratio(self) -> float:
        """Fraction of the band's base indices covered by the bins (drops remainder)."""
        width = self.band_width_indices
        return 0.0 if width == 0 else self.covered_base_samples / float(width)


@dataclass(frozen=True)
class BinningParameters:
    """
    Stores the calculated binning parameters for adaptive binning.

    Conventions
    -----------
    - f_base_upper and base_upper_idx are inclusive with respect to the base grid.
    - Remainders (base samples that don't fit an integer number of decimated bins
      within a band) are dropped.
    - Bands are provided as immutable views holding per-band metadata and a slice
      into the global per-bin arrays.

    Shapes
    ------
    - nodes, nodes_indices: shape (num_bands + 1,)
    - Per-band arrays (delta_f_bands, decimation_factors_bands, num_bins_bands, remainder_per_band):
      shape (num_bands,)
    - band_bin_ranges: shape (num_bands, 2)
    - Per-bin arrays (band_assignment, delta_f, f_base_lower, f_base_upper, base_lower_idx, base_upper_idx):
      shape (total_bins,)
    """

    # Inputs and global metadata
    nodes: F32Vec  # shape: (num_bands + 1,), dtype: float32 (band boundaries)
    base_delta_f: float  # scalar, base uniform spacing
    delta_f_initial: float  # scalar, first-band spacing
    dtype: np.dtype[np.float32]  # scalar dtype for frequency arrays

    # Derived metadata
    nodes_indices: (
        IntVec  # shape: (num_bands + 1,), dtype: int (base-grid indices of nodes)
    )
    num_bands: int  # scalar, equals len(nodes) - 1
    total_bins: int  # scalar, equals sum(num_bins_bands)
    inclusive_upper: bool  # scalar, per-bin upper bound is inclusive

    # Per-band arrays (shape: (num_bands,))
    delta_f_bands: (
        F32Vec  # shape: (num_bands,), dtype: float32 (per-band dyadic spacing)
    )
    decimation_factors_bands: (
        IntVec  # shape: (num_bands,), dtype: int (per-band decimation vs base)
    )
    num_bins_bands: IntVec  # shape: (num_bands,), dtype: int (bins per band)
    remainder_per_band: (
        IntVec  # shape: (num_bands,), dtype: int (dropped base samples per band)
    )

    # Per-band bin ranges
    band_bin_ranges: (
        IntMat2  # shape: (num_bands, 2), dtype: int, [start, end) in per-bin arrays
    )

    # Per-bin arrays (shape: (total_bins,))
    band_assignment: IntVec  # shape: (total_bins,), dtype: int (band index per bin)
    delta_f: F32Vec  # shape: (total_bins,), dtype: float32 (per-bin spacing)
    f_base_lower: (
        F32Vec  # shape: (total_bins,), dtype: float32 (per-bin lower freq bound)
    )
    f_base_upper: F32Vec  # shape: (total_bins,), dtype: float32 (per-bin upper freq bound, inclusive)
    base_lower_idx: (
        IntVec  # shape: (total_bins,), dtype: int (per-bin lower base index, inclusive)
    )
    base_upper_idx: (
        IntVec  # shape: (total_bins,), dtype: int (per-bin upper base index, inclusive)
    )

    # Band objects for ergonomic access
    bands: Tuple[Band, ...]  # length: num_bands (immutable per-band views)

    def __eq__(self, other):
        if not isinstance(other, BinningParameters):
            return NotImplemented
        # Compare scalar fields
        scalar_fields = (
            "base_delta_f", "delta_f_initial", "num_bands", "total_bins", "inclusive_upper"
        )
        for f in scalar_fields:
            if getattr(self, f) != getattr(other, f):
                return False
        # Compare array fields
        array_fields = (
            "nodes", "nodes_indices", "delta_f_bands", "decimation_factors_bands",
            "num_bins_bands", "remainder_per_band", "band_bin_ranges",
            "band_assignment", "delta_f", "f_base_lower", "f_base_upper",
            "base_lower_idx", "base_upper_idx",
        )
        for f in array_fields:
            if not np.array_equal(getattr(self, f), getattr(other, f)):
                return False
        return self.bands == other.bands

    def __hash__(self):
        return id(self)

    # Convenience instance method that forwards to the free function
    def decimate(
        self,
        data: Union[np.ndarray, "torch.Tensor"],
        base_offset_idx: int = 0,
        mode: Literal["explicit", "auto"] = "explicit",
        policy: Literal["pick", "mean"] = "pick",
        check: bool = True,
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Decimate data from the base uniform grid to the multi-banded domain using
        these BinningParameters.

        Parameters
        ----------
        data : np.ndarray | torch.Tensor
            Array/tensor whose trailing dimension is along the base frequency grid.
        base_offset_idx : int, default 0
            Offset to apply to base indices when slicing into `data`.
        mode : {"explicit", "auto"}, default "explicit"
            When "auto", infer a reasonable base_offset_idx from `data.shape[-1]` and
            `nodes_indices`. When "explicit", use the provided base_offset_idx.
        policy : {"pick", "mean"}, default "pick"
            Downsampling policy per decimation block.
        check : bool, default True
            When True, perform lightweight validations per band.

        Returns
        -------
        np.ndarray | torch.Tensor
            Decimated array/tensor with the same leading dims and trailing dim == total_bins.
        """
        return decimate(
            data=data,
            params=self,
            base_offset_idx=base_offset_idx,
            mode=mode,
            policy=policy,
            check=check,
        )


def plan_bands(
    nodes: Sequence[float],
    base_delta_f: float,
    delta_f_initial: float,
) -> List[Band]:
    """
    Plan per-band parameters from nodes and base grid spacing, producing Band instances.

    Notes
    -----
    - Uses float32 internally for frequency-space arrays.
    - Base-index alignment is via integer casts (floor for positive frequencies).
    - Per-band remainder base samples are dropped (not binned).
    """
    # Validate scalar inputs
    base_delta_f = float(base_delta_f)
    if base_delta_f <= 0:
        raise ValueError(f"base_delta_f must be positive, got {base_delta_f}.")
    if delta_f_initial <= 0:
        raise ValueError(f"delta_f_initial must be positive, got {delta_f_initial}.")

    # Prepare and validate nodes (float32)
    nodes_arr: F32Vec = np.asarray(nodes, dtype=np.float32)
    if nodes_arr.ndim != 1:
        raise ValueError(
            f"Expected 1D nodes array of shape [num_bands + 1], got shape {nodes_arr.shape}."
        )
    if nodes_arr.size < 2:
        raise ValueError(
            "nodes must contain at least two elements (defining one band)."
        )
    if not np.all(np.diff(nodes_arr) > 0):
        raise ValueError(
            "nodes must be strictly increasing; zero or negative band widths are invalid."
        )

    num_bands = int(nodes_arr.size - 1)

    # Map nodes to base-grid indices (floor via integer cast)
    nodes_indices: IntVec = (nodes_arr / base_delta_f).astype(int)

    # Per-band computations (float32 for frequencies)
    delta_f_bands: F32Vec = (delta_f_initial * (2 ** np.arange(num_bands))).astype(
        np.float32
    )
    decimation_factors_bands: IntVec = (delta_f_bands / base_delta_f).astype(int)

    if np.any(decimation_factors_bands < 1):
        raise ValueError(
            "Invalid decimation factors (< 1). Ensure delta_f_initial >= base_delta_f "
            "so that each band's decimation factor is at least 1."
        )

    # Band widths in base indices and per-band bin counts (drop remainder)
    band_width_indices: IntVec = (nodes_indices[1:] - nodes_indices[:-1]).astype(int)
    num_bins_bands: IntVec = (band_width_indices / decimation_factors_bands).astype(int)

    # Diagnostics: remainder per band (leftover base samples not binned)
    remainder_per_band: IntVec = band_width_indices - (
        num_bins_bands * decimation_factors_bands
    )

    # Assign [bin_start, bin_end) ranges via cumulative sums of num_bins
    if num_bands > 0:
        bin_starts: IntVec = np.concatenate(
            ([0], np.cumsum(num_bins_bands)[:-1])
        ).astype(int)
        bin_ends: IntVec = np.cumsum(num_bins_bands).astype(int)
    else:
        bin_starts = np.empty((0,), dtype=int)
        bin_ends = np.empty((0,), dtype=int)

    # Construct Band instances
    bands: List[Band] = []
    for i in range(num_bands):
        bands.append(
            Band(
                index=i,
                node_lower=float(nodes_arr[i]),
                node_upper=float(nodes_arr[i + 1]),
                node_lower_idx=int(nodes_indices[i]),
                node_upper_idx_exclusive=int(nodes_indices[i + 1]),
                delta_f_band=float(delta_f_bands[i]),
                decimation_factor_band=int(decimation_factors_bands[i]),
                num_bins=int(num_bins_bands[i]),
                remainder=int(remainder_per_band[i]),
                bin_start=int(bin_starts[i]),
                bin_end=int(bin_ends[i]),
            )
        )

    return bands


def compile_binning_from_bands(
    bands: Sequence[Band],
    base_delta_f: float,
    delta_f_initial: float,
    dtype: np.dtype[np.float32] = np.dtype(np.float32),
) -> BinningParameters:
    """
    Compile vectorized per-bin arrays and metadata from a list of Band instances.

    Parameters
    ----------
    bands : Sequence[Band]
        Planned bands (e.g., from plan_bands). Must be contiguous and ordered by index.
    base_delta_f : float
        Base uniform frequency spacing. Must be > 0.
    delta_f_initial : float
        Initial spacing of the first band. Included for metadata completeness.
    dtype : np.dtype[np.float32]
        Floating dtype for frequency arrays (defaults to float32).

    Returns
    -------
    BinningParameters
        Immutable structure with inputs, metadata, per-band/per-bin arrays, diagnostics,
        and Band instances for per-band access.
    """
    base_delta_f = float(base_delta_f)
    if base_delta_f <= 0:
        raise ValueError(f"base_delta_f must be positive, got {base_delta_f}.")
    if delta_f_initial <= 0:
        raise ValueError(f"delta_f_initial must be positive, got {delta_f_initial}.")

    num_bands = len(bands)
    if num_bands == 0:
        # Empty case: no bands, create empty structures
        nodes: F32Vec = np.empty((0,), dtype=dtype)  # (0,)
        nodes_indices: IntVec = np.empty((0,), dtype=int)  # (0,)
        empty_f32: F32Vec = np.empty((0,), dtype=dtype)  # (0,)
        empty_i: IntVec = np.empty((0,), dtype=int)  # (0,)
        band_bin_ranges: IntMat2 = np.empty((0, 2), dtype=int)  # (0, 2)
        return BinningParameters(
            nodes=nodes,
            base_delta_f=base_delta_f,
            delta_f_initial=float(delta_f_initial),
            dtype=dtype,
            nodes_indices=nodes_indices,
            num_bands=0,
            total_bins=0,
            inclusive_upper=True,
            delta_f_bands=empty_f32,
            decimation_factors_bands=empty_i,
            num_bins_bands=empty_i,
            remainder_per_band=empty_i,
            band_bin_ranges=band_bin_ranges,
            band_assignment=empty_i,
            delta_f=empty_f32,
            f_base_lower=empty_f32,
            f_base_upper=empty_f32,
            base_lower_idx=empty_i,
            base_upper_idx=empty_i,
            bands=tuple(),
        )

    # Validate ordering and contiguity; reconstruct nodes and nodes_indices
    bands_sorted = sorted(bands, key=lambda b: b.index)
    if any(b.index != i for i, b in enumerate(bands_sorted)):
        raise ValueError("Bands must be indexed 0..num_bands-1 without gaps.")

    # Reconstruct nodes and indices; check continuity
    nodes: F32Vec = np.empty((num_bands + 1,), dtype=dtype)  # (num_bands + 1,)
    nodes_indices: IntVec = np.empty((num_bands + 1,), dtype=int)  # (num_bands + 1,)

    nodes[0] = bands_sorted[0].node_lower
    nodes_indices[0] = bands_sorted[0].node_lower_idx

    for i in range(num_bands):
        b = bands_sorted[i]
        if i > 0 and not np.isclose(b.node_lower, bands_sorted[i - 1].node_upper):
            raise ValueError(
                f"Non-contiguous band boundaries between band {i-1} and {i}."
            )
        if i > 0 and b.node_lower_idx != bands_sorted[i - 1].node_upper_idx_exclusive:
            raise ValueError(
                f"Non-contiguous base index boundaries between band {i-1} and {i}."
            )
        nodes[i + 1] = b.node_upper
        nodes_indices[i + 1] = b.node_upper_idx_exclusive

    # Per-band arrays derived from Band (each of length num_bands)
    delta_f_bands: F32Vec = np.array(
        [b.delta_f_band for b in bands_sorted], dtype=dtype
    )  # (num_bands,)
    decimation_factors_bands: IntVec = np.array(
        [b.decimation_factor_band for b in bands_sorted], dtype=int
    )  # (num_bands,)
    num_bins_bands: IntVec = np.array(
        [b.num_bins for b in bands_sorted], dtype=int
    )  # (num_bands,)
    remainder_per_band: IntVec = np.array(
        [b.remainder for b in bands_sorted], dtype=int
    )  # (num_bands,)

    # Bin ranges (num_bands, 2), each row is [start, end) into per-bin arrays
    band_bin_ranges: IntMat2 = np.empty((num_bands, 2), dtype=int)  # (num_bands, 2)
    for i, b in enumerate(bands_sorted):
        band_bin_ranges[i, 0] = b.bin_start
        band_bin_ranges[i, 1] = b.bin_end

    total_bins = int(num_bins_bands.sum())

    # Per-bin arrays (each of length total_bins)
    if total_bins > 0:
        band_assignment: IntVec = np.repeat(
            np.arange(num_bands, dtype=int), num_bins_bands
        )  # (total_bins,)

        # Per-bin delta_f and decimation
        delta_f_per_bin: F32Vec = delta_f_bands[band_assignment].astype(
            dtype
        )  # (total_bins,)

        # f_base_lower/upper: start at nodes[0], cumulative per-bin widths; inclusive upper = - base_delta_f
        f_base_lower: F32Vec = np.concatenate(
            (nodes[:1], nodes[0] + np.cumsum(delta_f_per_bin[:-1]))
        ).astype(
            dtype
        )  # (total_bins,)
        f_base_upper: F32Vec = (
            nodes[0] + np.cumsum(delta_f_per_bin) - base_delta_f
        ).astype(
            dtype
        )  # (total_bins,)

        # Base index lower/upper (inclusive) via decimation factors too
        decimation_per_bin: IntVec = decimation_factors_bands[band_assignment].astype(
            int
        )  # (total_bins,)
        base_lower_idx: IntVec = np.concatenate(
            (nodes_indices[:1], nodes_indices[0] + np.cumsum(decimation_per_bin[:-1]))
        ).astype(
            int
        )  # (total_bins,)
        base_upper_idx: IntVec = (base_lower_idx + decimation_per_bin - 1).astype(
            int
        )  # (total_bins,)
    else:
        band_assignment = np.empty((0,), dtype=int)  # (0,)
        delta_f_per_bin = np.empty((0,), dtype=dtype)  # (0,)
        f_base_lower = np.empty((0,), dtype=dtype)  # (0,)
        f_base_upper = np.empty((0,), dtype=dtype)  # (0,)
        base_lower_idx = np.empty((0,), dtype=int)  # (0,)
        base_upper_idx = np.empty((0,), dtype=int)  # (0,)

    return BinningParameters(
        nodes=nodes,
        base_delta_f=base_delta_f,
        delta_f_initial=float(delta_f_initial),
        dtype=dtype,
        nodes_indices=nodes_indices,
        num_bands=num_bands,
        total_bins=total_bins,
        inclusive_upper=True,
        delta_f_bands=delta_f_bands,
        decimation_factors_bands=decimation_factors_bands,
        num_bins_bands=num_bins_bands,
        remainder_per_band=remainder_per_band,
        band_bin_ranges=band_bin_ranges,
        band_assignment=band_assignment,
        delta_f=delta_f_per_bin,
        f_base_lower=f_base_lower,
        f_base_upper=f_base_upper,
        base_lower_idx=base_lower_idx,
        base_upper_idx=base_upper_idx,
        bands=tuple(bands_sorted),
    )


def compute_adaptive_binning(
    nodes: Sequence[float],
    delta_f_initial: float,
    base_delta_f: float,
    dtype: np.dtype[np.float32] = np.dtype(np.float32),
) -> BinningParameters:
    """
    Convenience function that runs the two-step process:
      1) plan_bands to produce Band instances, and
      2) compile_binning_from_bands to produce BinningParameters.
    """
    bands = plan_bands(
        nodes=nodes,
        base_delta_f=base_delta_f,
        delta_f_initial=delta_f_initial,
    )
    return compile_binning_from_bands(
        bands=bands,
        base_delta_f=base_delta_f,
        delta_f_initial=delta_f_initial,
        dtype=dtype,
    )


def _infer_base_offset_idx_auto(params: BinningParameters, data_len: int) -> int:
    """
    Infer a reasonable base_offset_idx given data length and nodes_indices.

    Heuristics:
    - If data_len == nodes_indices[-1] - nodes_indices[0]: assume windowed coverage
      starting exactly at nodes_indices[0] (offset = -nodes_indices[0]).
    - Else if data_len >= nodes_indices[-1]: assume full grid starting at 0 (offset = 0),
      regardless of whether nodes_indices[0] == 0. This handles cases where waveforms
      are generated from f=0 but the MFD starts at a higher frequency.
    - Else: no unambiguous inference; ask the caller to pass an explicit offset.
    """
    lo = int(params.nodes_indices[0])
    hi_ex = int(params.nodes_indices[-1])  # exclusive high
    coverage = hi_ex - lo

    if data_len == coverage:
        return -lo
    if data_len >= hi_ex:
        return 0
    raise ValueError(
        f"Cannot auto-infer base_offset_idx: data length {data_len} does not match "
        f"band coverage {coverage} nor full-grid length {hi_ex}."
    )


def decimate_uniform(
    x: Union[np.ndarray, "torch.Tensor"],
    decimation_factor: int,
    policy: Literal["pick", "mean"] = "pick",
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Downsample the trailing dimension of x by an integer factor, dropping remainder.

    - policy="pick": pick first sample of each block (fast, deterministic).
    - policy="mean": average within each block.

    Returns same type as x with trailing dimension floor(L / d).
    """
    if decimation_factor < 1:
        raise ValueError(f"decimation_factor must be >= 1, got {decimation_factor}")

    L = x.shape[-1]
    if L == 0:
        return x[..., :0]

    d = decimation_factor
    n_bins = L // d
    if n_bins == 0:
        # Not enough samples for even one bin; return empty slice of same type
        return x[..., :0]

    trim_len = n_bins * d
    x_trim = x[..., :trim_len]

    # Use stride for "pick" as it's efficient and identical to taking the first of each block
    if policy == "pick":
        return x_trim[..., ::d]

    # Mean over each block
    if isinstance(x_trim, np.ndarray):
        x_reshaped = x_trim.reshape(*x_trim.shape[:-1], n_bins, d)
        return x_reshaped.mean(axis=-1)
    else:
        # Lazy import torch to avoid hard dependency
        try:
            import torch  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Torch is required for tensor decimation but is not available."
            ) from e
        x_reshaped = x_trim.reshape(*x_trim.shape[:-1], n_bins, d)
        return x_reshaped.mean(dim=-1)


def decimate(
    data: Union[np.ndarray, "torch.Tensor"],
    params: BinningParameters,
    base_offset_idx: int = 0,
    mode: Literal["explicit", "auto"] = "explicit",
    policy: Literal["pick", "mean"] = "pick",
    check: bool = True,
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Decimate data from the base uniform grid to the multi-banded domain using params.

    Parameters
    ----------
    data : np.ndarray | torch.Tensor
        Input array/tensor. Decimation is applied along the trailing dimension.
    params : BinningParameters
        Precomputed binning parameters that define bands, decimation factors,
        and output layout.
    base_offset_idx : int, default 0
        Offset to apply to base indices when slicing into `data`. Positive values
        shift slices to the right; negative shift to the left.
    mode : {"explicit", "auto"}, default "explicit"
        - "explicit": use the provided base_offset_idx as-is.
        - "auto": infer base_offset_idx from data length and params.nodes_indices.
    policy : {"pick", "mean"}, default "pick"
        Downsampling policy per decimation block.
    check : bool, default True
        When True, perform lightweight validations per band (slice bounds, lengths).

    Returns
    -------
    np.ndarray | torch.Tensor
        Decimated output with trailing dimension equal to params.total_bins and
        the same leading dims, dtype, and device as the input.
    """
    # Lazy torch import for isinstance checks without hard dependency
    torch_mod = None
    try:
        import torch as torch_mod  # type: ignore
    except Exception:
        torch_mod = None

    trailing = data.shape[-1]
    if mode == "auto":
        base_offset_idx = _infer_base_offset_idx_auto(params, trailing)

    # Allocate output matching input type/device/dtype
    out_shape = (*data.shape[:-1], params.total_bins)
    if isinstance(data, np.ndarray):
        out: Union[np.ndarray, "torch.Tensor"] = np.empty(out_shape, dtype=data.dtype)
    elif (torch_mod is not None) and isinstance(data, torch_mod.Tensor):  # type: ignore[truthy-bool]
        out = torch_mod.empty(out_shape, dtype=data.dtype, device=data.device)  # type: ignore[name-defined]
    else:
        raise NotImplementedError(
            f"Decimation not implemented for data of type {type(data)}."
        )

    # Process each band
    for i in range(params.num_bands):
        in_lo = int(params.nodes_indices[i]) + base_offset_idx
        in_hi = int(params.nodes_indices[i + 1]) + base_offset_idx
        out_lo = int(params.band_bin_ranges[i, 0])
        out_hi = int(params.band_bin_ranges[i, 1])
        factor = int(params.decimation_factors_bands[i])
        expected_bins = int(params.num_bins_bands[i])

        if check:
            if not (0 <= in_lo <= in_hi <= trailing):
                raise IndexError(
                    f"Input slice out of bounds for band {i}: "
                    f"[{in_lo}:{in_hi}) with data length {trailing}."
                )
            if (out_hi - out_lo) != expected_bins:
                raise AssertionError(
                    f"Output slice length mismatch for band {i}: "
                    f"{out_hi - out_lo} vs expected {expected_bins}."
                )
            available = in_hi - in_lo
            if available < factor * expected_bins:
                raise ValueError(
                    f"Band {i}: insufficient input samples {available} for "
                    f"{expected_bins} bins at factor {factor} (need >= {factor * expected_bins})."
                )

        # Decimate the base-grid slice and write to the band's output slice
        band_in = data[..., in_lo:in_hi]
        band_out = decimate_uniform(band_in, factor, policy=policy)
        if check and band_out.shape[-1] != expected_bins:
            raise AssertionError(
                f"Band {i}: decimator returned {band_out.shape[-1]} bins, expected {expected_bins}."
            )

        out[..., out_lo:out_hi] = band_out  # type: ignore[assignment,index]

    return out
