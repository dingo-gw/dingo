"""
Per-event analysis settings validated against a model's training settings.

A frequency range in the event metadata (`minimum_frequency` / `maximum_frequency`)
is applied to the network input, so a range narrower than the network's domain is
only allowed when the network was trained with a matching license: random strain
cropping (`random_strain_cropping` in the training data settings) or, for tokenized
(transformer) networks, token masking (`tokenization.mask_frequency_range`;
`mask_random_tokens` alone passes with a warning). Likewise, PSD notches
(`psd_notch_dict`) are checked against `tokenization.mask_frequency_notches`, and a
detector subset against `tokenization.mask_detectors`. `check_frequency_updates`,
`check_psd_notches` and `check_detector_update` check these when `dingo_pipe` parses
its INI file, and the sampler context checks them again before preparing network
input. A range given under `importance-sampling-updates` applies to the likelihood
only and needs no such check. The likelihood masks the ASDs outside each detector's
range and places the calibration spline nodes across it; `resolve_frequency_bounds`
gives those per-detector bounds.
"""

import warnings

import numpy as np

from dingo.gw.domains import (
    MultibandedFrequencyDomain,
    UniformFrequencyDomain,
    build_domain_from_model_metadata,
)


def _validate_frequency_bound(
    value: dict[str, float] | float,
    bound: str,
    domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
    data_settings: dict,
    detectors: list[str] | None = None,
):
    """
    Validate a requested minimum or maximum frequency against the model's training
    settings.

    ``value`` may be a float (applying to all analyzed detectors) or a per-detector
    dict constraining only the detectors it names; keys must be analyzed detectors.
    Values equal to the domain bound are always allowed. A changed
    value requires frequency flexibility from training: ``random_strain_cropping``
    and/or ``tokenization.mask_frequency_range`` are validated against their
    envelopes; a model with only ``tokenization.mask_random_tokens`` passes with a
    warning, since the contiguous masking pattern differs from the random training
    distribution.

    Parameters
    ----------
    value : dict[str, float] or float
        Requested frequency bound.
    bound : str
        "minimum_frequency" or "maximum_frequency".
    domain : UniformFrequencyDomain or MultibandedFrequencyDomain
        The model's base (uniform) domain.
    data_settings : dict
        ``train_settings["data"]`` of the model.
    detectors : list[str], optional
        The analyzed detectors, a subset of the training detectors (see
        ``check_detector_update``); defaults to the training list.

    Raises
    ------
    ValueError
        If the request is incompatible with the training settings.
    """
    minimum = bound == "minimum_frequency"
    domain_value = domain.f_min if minimum else domain.f_max
    model_detectors = data_settings["detectors"]
    detectors = model_detectors if detectors is None else list(detectors)

    if isinstance(value, dict):
        unknown = set(value) - set(detectors)
        if unknown:
            raise ValueError(
                f"{bound} names detectors {sorted(unknown)} that are not analyzed "
                f"(detectors: {detectors}; the model was trained with "
                f"{model_detectors})."
            )
        values = dict(value)
    else:
        values = {d: value for d in detectors}

    # Hard domain bounds.
    for det, v in values.items():
        if minimum and v < domain.f_min:
            raise ValueError(f"f_min {values} < domain.f_min = {domain.f_min}.")
        if not minimum and v > domain.f_max:
            raise ValueError(f"f_max {values} > domain.f_max = {domain.f_max}.")

    changed = {d: v for d, v in values.items() if v != domain_value}
    if not changed:
        return

    crop_settings = data_settings.get("random_strain_cropping")
    tok = data_settings.get("tokenization") or {}
    range_settings = tok.get("mask_frequency_range")

    if crop_settings is None and range_settings is None:
        if "mask_random_tokens" in tok:
            warnings.warn(
                f"Updating {bound} relies on mask_random_tokens training only; the "
                f"contiguous masking pattern differs from the random training "
                f"distribution. Expect reduced importance-sampling efficiency and "
                f"check the effective sample size."
            )
            return
        raise ValueError(
            f"Model was not trained with variable frequency ranges "
            f"(no random_strain_cropping, mask_frequency_range, or "
            f"mask_random_tokens). Cannot update {bound}."
        )

    if crop_settings is not None:
        if crop_settings.get("cropping_probability", 0.0) == 0.0:
            raise ValueError(f"Cropping disabled; cannot update {bound} to {value}.")
        if not crop_settings.get("independent_detectors", True):
            effective = {d: values.get(d, domain_value) for d in detectors}
            if len(set(effective.values())) > 1:
                raise ValueError(
                    f"Independent frequencies per detector not enabled. All "
                    f"frequencies must match, got {bound} = {value}."
                )

    # Training envelopes, in shared vocabulary: f_min may be raised up to
    # f_min_upper, f_max lowered down to f_max_lower; an absent key means that
    # side was never cropped / cut in training.
    key = "f_min_upper" if minimum else "f_max_lower"
    for settings, source in (
        (crop_settings, "random_strain_cropping"),
        (range_settings, "tokenization.mask_frequency_range"),
    ):
        if settings is None:
            continue
        cap = settings.get(key, domain_value)
        caps = cap if isinstance(cap, dict) else {d: cap for d in model_detectors}
        for det, v in changed.items():
            if (minimum and v > caps[det]) or (not minimum and v < caps[det]):
                raise ValueError(
                    f"Requested {bound} for {det} ({v} Hz) is outside the "
                    f"training envelope ({key}={cap} Hz from {source})."
                )


def check_frequency_updates(
    model_metadata: dict,
    f_min: dict[str, float] | float | None = None,
    f_max: dict[str, float] | float | None = None,
    detectors: list[str] | None = None,
):
    """
    Validate requested minimum / maximum frequencies against a model's metadata.

    Thin metadata-level wrapper around ``_validate_frequency_bound``, used by
    dingo_pipe at DAG-build time; see there for the accepted forms and semantics.
    ``detectors`` are the analyzed detectors (default: the training list).
    """
    domain = build_domain_from_model_metadata(model_metadata, base=True)
    if not isinstance(domain, (UniformFrequencyDomain, MultibandedFrequencyDomain)):
        raise ValueError("Frequency updates only possible for frequency domains.")
    data_settings = model_metadata["train_settings"]["data"]
    for bound, value in (("minimum_frequency", f_min), ("maximum_frequency", f_max)):
        if value is not None:
            _validate_frequency_bound(value, bound, domain, data_settings, detectors)


def _validate_psd_notches(
    psd_notch_dict: dict,
    domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
    data_settings: dict,
):
    """
    Validate PSD notch intervals against the domain and the model's training settings.

    ``psd_notch_dict`` maps detectors to one ``[f_lo, f_hi]`` interval or a list of
    them. Configuration errors raise: a detector the model was not trained with, an
    empty interval, or an interval touching the domain bounds (at data generation a
    high-ASD run at an edge is taken for PSD padding, see ``detect_asd_notches``, so
    the frequency bound must be moved instead). A mismatch with the training
    distribution only warns, since the likelihood stays exact and the network is
    merely a worse proposal: no notch training (including non-tokenized models),
    ``mask_random_tokens`` only, or an interval outside the
    ``tokenization.mask_frequency_notches`` envelope (range and ``max_width``).

    Parameters
    ----------
    psd_notch_dict : dict
        ``{det: [f_lo, f_hi]}`` or ``{det: [[f_lo, f_hi], ...]}``.
    domain : UniformFrequencyDomain or MultibandedFrequencyDomain
        The model's base (uniform) domain.
    data_settings : dict
        ``train_settings["data"]`` of the model.

    Raises
    ------
    ValueError
        If the notches are incompatible with the model or the domain.
    """
    model_detectors = data_settings["detectors"]
    unknown = set(psd_notch_dict) - set(model_detectors)
    if unknown:
        raise ValueError(
            f"psd_notch_dict names detectors {sorted(unknown)} the model was not "
            f"trained with (detectors: {model_detectors})."
        )
    intervals = []
    for det, notch in psd_notch_dict.items():
        ranges = [notch] if not isinstance(notch[0], (list, tuple)) else notch
        for f_lo, f_hi in ranges:
            if not f_lo <= f_hi:
                raise ValueError(
                    f"psd_notch_dict interval [{f_lo}, {f_hi}] for {det} is empty."
                )
            if f_lo <= domain.f_min or f_hi >= domain.f_max:
                raise ValueError(
                    f"psd_notch_dict interval [{f_lo}, {f_hi}] for {det} touches the "
                    f"domain bounds [{domain.f_min}, {domain.f_max}]; move "
                    f"minimum_frequency / maximum_frequency instead of notching an edge."
                )
            intervals.append((det, f_lo, f_hi))

    tok = data_settings.get("tokenization") or {}
    notch_settings = tok.get("mask_frequency_notches")
    if notch_settings is None:
        if "mask_random_tokens" in tok:
            warnings.warn(
                "psd_notch_dict relies on mask_random_tokens training only; the "
                "contiguous masking pattern differs from the random training "
                "distribution. Expect reduced importance-sampling efficiency and "
                "check the effective sample size."
            )
        else:
            warnings.warn(
                "Model was not trained with mask_frequency_notches; the notched bins "
                "are out of distribution for the network. The likelihood is exact, "
                "so check the importance-sampling efficiency."
            )
        return

    # Training envelope as MaskFrequencyNotches resolves it: an explicit range is
    # clamped to the domain, and the width is capped by the range.
    f_min = notch_settings.get("f_min")
    f_max = notch_settings.get("f_max")
    notch_f_min = domain.f_min if f_min is None else max(f_min, domain.f_min)
    notch_f_max = domain.f_max if f_max is None else min(f_max, domain.f_max)
    max_width = min(notch_settings["max_width"], notch_f_max - notch_f_min)
    for det, f_lo, f_hi in intervals:
        if f_lo < notch_f_min or f_hi > notch_f_max or f_hi - f_lo > max_width + 1e-9:
            warnings.warn(
                f"psd_notch_dict interval [{f_lo}, {f_hi}] for {det} is outside the "
                f"training envelope (mask_frequency_notches: range "
                f"[{notch_f_min}, {notch_f_max}] Hz, max_width {max_width} Hz). "
                f"Expect reduced importance-sampling efficiency."
            )


def check_psd_notches(model_metadata: dict, psd_notch_dict: dict):
    """
    Validate PSD notch intervals against a model's metadata.

    Thin metadata-level wrapper around ``_validate_psd_notches``, used by dingo_pipe
    at DAG-build time; see there for the accepted forms and semantics.
    """
    domain = build_domain_from_model_metadata(model_metadata, base=True)
    if not isinstance(domain, (UniformFrequencyDomain, MultibandedFrequencyDomain)):
        raise ValueError("psd_notch_dict requires a frequency domain.")
    _validate_psd_notches(
        psd_notch_dict, domain, model_metadata["train_settings"]["data"]
    )


def _validate_detectors_transformer(
    detectors_event: list[str],
    detectors_network: list[str],
    mask_detector_settings: dict,
):
    """
    Validate that the event detectors are compatible with a transformer network
    trained with detector masking.

    The event detectors must be a subset of the training detectors, and every
    *absent* training detector must have been maskable in training. Keys missing
    from ``mask_detector_settings`` impose no constraint, since ``MaskDetectors``
    then defaulted to uniform probabilities.

    Parameters
    ----------
    detectors_event : list[str]
        Detectors present in the event data.
    detectors_network : list[str]
        Detectors the network was trained with.
    mask_detector_settings : dict
        The ``tokenization.mask_detectors`` sub-dict from the train settings.

    Raises
    ------
    ValueError
        If the detector configuration is incompatible with the network.
    """
    if not set(detectors_event).issubset(set(detectors_network)):
        raise ValueError(
            f"Event has detectors {detectors_event} but model was only trained "
            f"with detectors {detectors_network}."
        )
    absent = set(detectors_network) - set(detectors_event)

    p_num_masked = mask_detector_settings.get("p_num_masked")
    # p_num_masked[k] = probability of masking k detectors during training.
    if p_num_masked is not None and (
        len(absent) >= len(p_num_masked) or p_num_masked[len(absent)] == 0.0
    ):
        raise ValueError(
            f"Event has detectors {detectors_event}, but model was trained with "
            f"p_num_masked={p_num_masked}, not allowing "
            f"{len(detectors_event)} active detectors."
        )

    p_detector = mask_detector_settings.get("p_detector")
    # p_detector[det] = probability of drawing det to be masked; zero means det was
    # always present in training, so it must also be present in the event.
    if p_detector is not None:
        for det in absent:
            if p_detector.get(det, 0.0) == 0.0:
                raise ValueError(
                    f"Detector {det} was never masked in training "
                    f"(p_detector={p_detector}); cannot drop it at inference."
                )


def check_detector_update(
    model_metadata: dict,
    detectors: list[str],
):
    """
    Validate that a given set of detectors is compatible with the network.

    For transformer networks trained with ``tokenization.mask_detectors``, the event
    detectors must be a subset of the training detectors and must be allowed by the
    masking probabilities.  For networks trained with ``tokenization.mask_random_tokens``
    only the subset check is performed.  For non-tokenization networks the event detectors
    must exactly match the training detectors.

    Parameters
    ----------
    model_metadata : dict
        Dictionary containing the network's training settings and data.
    detectors : list[str]
        Detectors present in the event data.

    Raises
    ------
    ValueError
        If the detector configuration is incompatible with the model.
    """
    detectors_network = model_metadata["train_settings"]["data"]["detectors"]
    if not set(detectors).issubset(set(detectors_network)):
        raise ValueError(
            f"Event has detectors {detectors} but model was only trained with "
            f"detectors {detectors_network}."
        )
    tok = model_metadata["train_settings"]["data"].get("tokenization", {})
    if "mask_detectors" in tok:
        _validate_detectors_transformer(
            detectors_event=detectors,
            detectors_network=detectors_network,
            mask_detector_settings=tok["mask_detectors"],
        )
    elif "mask_random_tokens" in tok:
        # Token-level masking does not constrain which detectors are present.
        pass
    elif set(detectors) != set(detectors_network):
        # Without detector masking (tokenized or not), an exact match is required.
        raise ValueError(
            f"Detectors {detectors} of event do not match detectors "
            f"{detectors_network} from model."
        )


def resolve_frequency_bounds(
    detectors: list[str],
    domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
    minimum_frequency: dict[str, float] | float | None = None,
    maximum_frequency: dict[str, float] | float | None = None,
) -> dict[str, tuple[float, float]]:
    """Return `(f_min, f_max)` for each detector from an event's frequency range,
    given as one float for all detectors or as a dict naming any of them. Missing
    values default to the domain bounds."""

    def expand(value, default):
        if isinstance(value, dict):
            unknown = set(value) - set(detectors)
            if unknown:
                raise ValueError(
                    f"Frequency bounds name detectors {sorted(unknown)} that are not "
                    f"analyzed (detectors: {detectors})."
                )
            return {d: float(value.get(d, default)) for d in detectors}
        return {d: float(default if value is None else value) for d in detectors}

    f_min = expand(minimum_frequency, domain.f_min)
    f_max = expand(maximum_frequency, domain.f_max)
    return {d: (f_min[d], f_max[d]) for d in detectors}


def check_importance_sampling_frequency_range(
    model_metadata: dict,
    minimum_frequency: dict[str, float] | float | None = None,
    maximum_frequency: dict[str, float] | float | None = None,
    sampling_frequency: float | None = None,
    detectors: list[str] | None = None,
):
    """Check a frequency range given under `importance-sampling-updates`. It changes
    the likelihood only, never the network input, so the strain-cropping rules do
    not apply. Each analyzed detector's (default: the training list) range must be
    positive, non-empty, and at most the Nyquist frequency of the data (when
    `sampling_frequency` is given)."""
    if minimum_frequency is None and maximum_frequency is None:
        return
    if detectors is None:
        detectors = model_metadata["train_settings"]["data"]["detectors"]
    network = build_domain_from_model_metadata(model_metadata, base=True)
    f_nyquist = sampling_frequency / 2 if sampling_frequency else np.inf
    bounds = resolve_frequency_bounds(
        detectors, network, minimum_frequency, maximum_frequency
    )
    for d, (f_min, f_max) in bounds.items():
        if not 0 < f_min < f_max <= f_nyquist:
            raise ValueError(
                f"Importance-sampling frequency range [{f_min}, {f_max}] Hz for {d} "
                f"must be positive, non-empty, and at most the Nyquist frequency "
                f"{f_nyquist} Hz."
            )
