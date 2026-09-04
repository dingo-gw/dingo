from typing import Optional
import warnings
import numpy as np
from copy import deepcopy
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d

from bilby.gw.detector import PowerSpectralDensity

from dingo.gw.prior import default_extrinsic_dict
from dingo.gw.prior import BBHExtrinsicPriorDict


def get_window(window_kwargs):
    """Compute window from window_kwargs."""
    type = window_kwargs["type"]
    if type == "tukey":
        roll_off, T, f_s = (
            window_kwargs["roll_off"],
            window_kwargs["T"],
            window_kwargs["f_s"],
        )
        alpha = 2 * roll_off / T
        w = tukey(int(T * f_s), alpha)
        return w
    else:
        raise NotImplementedError(f"Unknown window type {type}.")


def get_extrinsic_prior_dict(extrinsic_prior):
    """Build dict for extrinsic prior by starting with
    default_extrinsic_dict, and overwriting every element for which
    extrinsic_prior is not default.
    TODO: Move to dingo.gw.prior.py?"""
    extrinsic_prior_dict = default_extrinsic_dict.copy()
    for k, v in extrinsic_prior.items():
        if not isinstance(v, str) or v.lower() != "default":
            extrinsic_prior_dict[k] = v
    return extrinsic_prior_dict


def get_mismatch(a, b, domain, asd_file=None):
    """
    Mistmatch is 1 - overlap, where overlap is defined by
    inner(a, b) / sqrt(inner(a, a) * inner(b, b)).
    See e.g. Eq. (44) in https://arxiv.org/pdf/1106.1021.pdf.

    Parameters
    ----------
    a
    b
    domain
    asd_file

    Returns
    -------

    """
    if asd_file is not None:
        # whiten a and b, such that we can use flat-spectrum inner products below
        psd = PowerSpectralDensity(asd_file=asd_file)
        asd_interp = interp1d(
            psd.frequency_array, psd.asd_array, bounds_error=False, fill_value=np.inf
        )
        asd_array = asd_interp(domain.sample_frequencies)
        a = a / asd_array
        b = b / asd_array
    min_idx = domain.min_idx
    inner_ab = np.sum((a.conj() * b)[..., min_idx:], axis=-1).real
    inner_aa = np.sum((a.conj() * a)[..., min_idx:], axis=-1).real
    inner_bb = np.sum((b.conj() * b)[..., min_idx:], axis=-1).real
    overlap = inner_ab / np.sqrt(inner_aa * inner_bb)
    return 1 - overlap


def get_standardization_dict(
    extrinsic_prior_dict, wfd, selected_parameters, transform=None
):
    """
    Calculates the mean and standard deviation of parameters. This is needed for
    standardizing neural-network input and output.

    Parameters
    ----------
    extrinsic_prior_dict : dict
    wfd : WaveformDataset
    selected_parameters : list[str]
        List of parameters for which to estimate standardization factors.
    transform : Transform
        Operator that will generate samples for parameters contained in
        selected_parameters that are not contained in the intrinsic or extrinsic prior.
        (E.g., H1_time, L1_time_proxy)

    Returns
    -------

    """
    # The intrinsic standardization is estimated based on the entire dataset.
    mean_intrinsic, std_intrinsic = wfd.parameter_mean_std()

    # Some of the extrinsic prior parameters have analytic means and standard
    # deviations. If possible, this will either get these, or else it will estimate
    # them numerically.
    ext_prior = BBHExtrinsicPriorDict(extrinsic_prior_dict)
    mean_extrinsic, std_extrinsic = ext_prior.mean_std(ext_prior.keys())

    # Check that overlap between intrinsic and extrinsic parameters is only
    # due to fiducial values (-> std 0)
    for k in std_intrinsic.keys() & std_extrinsic.keys():
        if std_intrinsic[k] != 0:
            raise ValueError(
                f"Expected intrinsic prior for {k} to be a fixed value in the waveform dataset, "
                f"since {k} is specified as an extrinsic prior in the train settings and will be sampled"
                f"during training. However, the standard deviation of {k} is non-zero: {std_intrinsic[k]}"
                f"Please re-generate the waveform dataset with a fixed value for {k}."
            )

    # Merge dicts, overwriting fiducial values for parameters (e.g.,
    # luminosity_distance) in intrinsic parameters by the extrinsic ones
    mean = {**mean_intrinsic, **mean_extrinsic}
    std = {**std_intrinsic, **std_extrinsic}

    # For all remaining parameters that require standardization, we use the transform
    # to sample these and estimate the mean and standard deviation numerically.
    additional_parameters = [p for p in selected_parameters if p not in mean]
    if additional_parameters:
        num_samples = min(100_000, len(wfd.parameters))
        samples = {p: np.empty(num_samples) for p in additional_parameters}
        for n in range(num_samples):
            sample = {"parameters": wfd.parameters.iloc[n].to_dict()}
            sample = transform(sample)
            for p in additional_parameters:
                # This assumes all of the additional parameters are contained within
                # extrinsic_parameters. We have set it up so this is the case for the
                # GNPE proxies and the detector coalescence times.
                samples[p][n] = sample["extrinsic_parameters"][p]
        mean_additional = {p: np.mean(samples[p]).item() for p in additional_parameters}
        std_additional = {p: np.std(samples[p]).item() for p in additional_parameters}

        mean.update(mean_additional)
        std.update(std_additional)

    standardization_dict = {
        "mean": {k: mean[k] for k in selected_parameters},
        "std": {k: std[k] for k in selected_parameters},
    }
    return standardization_dict


def add_defaults_for_missing_detectors(
    object_to_update: Optional[float | dict],
    update_value: float,
    detectors: list[str],
) -> Optional[float | dict]:
    """Fill in a default frequency value for any detector missing from a per-detector dict.

    If `object_to_update` is a dict, any detector in `detectors` not present in the dict
    gets `update_value` inserted. Floats and None are returned unchanged.
    """
    object_to_update = deepcopy(object_to_update)
    if isinstance(object_to_update, dict) and detectors is not None:
        for det in detectors:
            if det not in object_to_update:
                object_to_update[det] = update_value
    return object_to_update


def parse_psd_notch_dict(raw: dict) -> dict:
    """Convert string-valued intervals from convert_string_to_dict to floats.

    ``convert_string_to_dict`` returns numeric values as strings.  This helper
    normalises ``{det: [f_lo, f_hi]}`` and ``{det: [[f_lo1, f_hi1], ...]}``
    entries so all boundaries are Python floats.
    """
    result = {}
    for det, notch in raw.items():
        if isinstance(notch[0], (list, tuple)):
            result[det] = [[float(a), float(b)] for a, b in notch]
        else:
            result[det] = [float(notch[0]), float(notch[1])]
    return result


def detect_asd_notches(asd_dict: dict, domain) -> dict | None:
    """
    Detect contiguous interior frequency intervals where ASD >= 0.5 * HIGH_ASD_VALUE.

    Scans per-detector ASDs (as stored in the event HDF5) and returns the
    frequency ranges of any suppressed regions.  The threshold is
    ``0.5 * HIGH_ASD_VALUE`` (= 0.5 for the current sentinel value of 1.0),
    which sits midway between real noise levels (~10⁻²³ 1/√Hz) and the notch
    sentinel, giving a robust detection margin in case of floating-point rounding.
    Runs touching the first or last valid bin (f_min, f_max) are skipped as edge
    padding rather than notches; when such a run is wider than one bin the PSD does
    not cover the model band, and a warning is issued.

    Parameters
    ----------
    asd_dict:
        Per-detector ASD arrays of length ``domain.max_idx + 1``.
    domain:
        UniformFrequencyDomain corresponding to the data.

    Returns
    -------
    dict or None
        ``{det: [[f_lo, f_hi], ...]}`` for each detector that has at least one
        notch.  Returns ``None`` when no notches are found.
    """
    from dingo.gw.noise.asd_dataset import HIGH_ASD_VALUE

    # Stored ASDs live on the base (uniform) domain; for a multibanded domain the
    # indices below must therefore refer to the base grid.
    domain = getattr(domain, "base_domain", domain)
    sample_freqs = domain.sample_frequencies  # length max_idx + 1
    min_idx = domain.min_idx
    notch_dict = {}

    for ifo, asd in asd_dict.items():
        valid_asd = asd[min_idx:]
        valid_freqs = sample_freqs[min_idx:]

        is_notched = valid_asd >= HIGH_ASD_VALUE * 0.5

        if not np.any(is_notched):
            continue

        changes = np.diff(is_notched.astype(int))
        starts = list(np.where(changes == 1)[0] + 1)
        ends = list(np.where(changes == -1)[0] + 1)

        if is_notched[0]:
            starts = [0] + starts
        if is_notched[-1]:
            ends = ends + [len(is_notched)]

        intervals = []
        for s, e in zip(starts, ends):
            if s == 0 or e == len(is_notched):
                # A high run touching f_min or f_max is edge padding, not a notch:
                # ASDs built from an ASDDataset carry HIGH_ASD_VALUE below f_min,
                # and bilby fills frequencies beyond a PSD file's range with inf
                # (LVK release PSDs end one bin short of f_max). A genuine notch at
                # an edge is indistinguishable from this; move the frequency bound
                # instead. Anything wider than one bin means the PSD does not
                # cover the model band, which the user should hear about.
                if e - s > 1:
                    warnings.warn(
                        f"ASD for {ifo} is non-physical over "
                        f"[{valid_freqs[s]}, {valid_freqs[e - 1]}] Hz at the edge of "
                        f"the model band [{domain.f_min}, {domain.f_max}] Hz; the PSD "
                        f"probably does not cover it. Set minimum_frequency / "
                        f"maximum_frequency to match."
                    )
                continue
            f_lo = float(valid_freqs[s])
            f_hi = float(valid_freqs[e - 1])
            intervals.append([f_lo, f_hi])

        if intervals:
            notch_dict[ifo] = intervals

    return notch_dict if notch_dict else None
