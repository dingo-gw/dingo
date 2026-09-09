"""
Frequency-range updates against a model's frequency domain.

A frequency range in the event metadata (`minimum_frequency` / `maximum_frequency`)
is applied to the network input, so a range narrower than the network's domain is
only allowed when the network was trained with random strain cropping
(`random_strain_cropping` in the training data settings) that covers it.
`check_frequency_updates` checks this when `dingo_pipe` parses its INI file, and
the sampler context checks it again before preparing network input. A range given
under `importance-sampling-updates` applies to the likelihood only and needs no
such check. The likelihood masks the ASDs outside each detector's range and places
the calibration spline nodes across it; `resolve_frequency_bounds` gives those
per-detector bounds.
"""

import numpy as np

from dingo.gw.domains import (
    MultibandedFrequencyDomain,
    UniformFrequencyDomain,
    build_domain_from_model_metadata,
)


def _validate_maximum_frequency(
    f_max: dict[str, float] | float,
    detectors: list[str],
    domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
    crop_settings: dict | None,
):
    if isinstance(f_max, (int, float)):
        f_max = {d: f_max for d in detectors}
    if set(f_max) != set(detectors):
        raise ValueError(
            f"f_max must have exactly detectors {detectors}, got " f"{list(f_max)}."
        )
    f_max_vals = np.array([f_max[d] for d in detectors])

    # Hard upper bound
    if np.any(f_max_vals > domain.f_max):
        raise ValueError(f"f_max {f_max} > domain.f_max = {domain.f_max}.")

    # Nothing changed
    if np.all(f_max_vals == domain.f_max):
        return

    # Cropping must be on
    if not crop_settings or crop_settings.get("cropping_probability", 0.0) == 0.0:
        raise ValueError(
            f"Cropping disabled; cannot lower maximum frequency to {f_max}."
        )

    # Extract lower bounds
    floors = crop_settings.get("f_max_lower")
    if floors is None:
        floors = domain.f_max
    if not isinstance(floors, dict):
        floors = {d: floors for d in detectors}

    # Check lower bound.
    if not crop_settings.get("independent_detectors", True):
        if len(set(f_max_vals)) > 1:
            raise ValueError(
                f"Independent max frequencies per detector not enabled. "
                f"All frequencies must match, got f_max = {f_max}."
            )
        # TODO: Risk of non-constant floors with non-independent detectors.
        assert len(set(floors.values())) == 1
    for d in detectors:
        if f_max[d] < floors[d]:
            raise ValueError(
                f"Maximum frequency requested for {d} ({f_max[d]} Hz) "
                f"less than lower bound of {floors[d]} Hz."
            )


def _validate_minimum_frequency(
    f_min: dict[str, float] | float,
    detectors: list[str],
    domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
    crop_settings: dict | None,
):
    if isinstance(f_min, (int, float)):
        f_min = {d: f_min for d in detectors}
    if set(f_min) != set(detectors):
        raise ValueError(
            f"f_min must have exactly detectors {detectors}, got {list(f_min)}."
        )
    f_min_vals = np.array([f_min[d] for d in detectors])

    # Hard lower bound
    if np.any(f_min_vals < domain.f_min):
        raise ValueError(f"f_min {f_min} < domain.f_min = {domain.f_min}.")

    # Nothing changed
    if np.all(f_min_vals == domain.f_min):
        return

    # Cropping must be on
    if not crop_settings or crop_settings.get("cropping_probability", 0.0) == 0.0:
        raise ValueError(
            f"Cropping disabled; cannot raise minimum frequency to {f_min}."
        )

    # Extract upper bounds
    caps = crop_settings.get("f_min_upper")
    if caps is None:
        caps = domain.f_min
    if not isinstance(caps, dict):
        caps = {d: caps for d in detectors}

    # Check upper bound.
    if not crop_settings.get("independent_detectors", True):
        if len(set(f_min_vals)) > 1:
            raise ValueError(
                f"Independent min frequencies per detector not enabled. "
                f"All frequencies must match, got f_min = {f_min}."
            )
        # TODO: Risk of non-constant caps with non-independent detectors.
        assert len(set(caps.values())) == 1
    for d in detectors:
        if f_min[d] > caps[d]:
            raise ValueError(
                f"Minimum frequency requested for {d} ({f_min[d]} Hz) "
                f"greater than upper bound of {caps[d]} Hz."
            )


def check_frequency_updates(
    model_metadata: dict,
    f_min: dict[str, float] | float | None = None,
    f_max: dict[str, float] | float | None = None,
):
    """
    Validate optional minimum and maximum frequency updates against a model's
    frequency domain.

    `f_min` / `f_max` may be a single float, applied to all detectors, or a dict
    mapping each detector to its own value. The update must:

    - match exactly the set of detectors in the model metadata,
    - respect the hard bounds of the domain (`domain.f_min` / `domain.f_max`),
    - comply with the training-time random-strain-cropping settings (probability,
      independent vs. joint detectors, and per-detector caps and floors).

    Parameters
    ----------
    model_metadata : dict
        The model's training settings and data; the detector list and the
        optional `random_strain_cropping` settings are read from
        `["train_settings"]["data"]`.
    f_min : dict[str, float], float, or None, optional
        Single float or per-detector dict of minimum frequencies to enforce.
        If a float is provided, it is applied to all detectors. Each value
        must be ≥ `domain.f_min`. If `None`, no minimum-frequency
        validation is performed.
    f_max : dict[str, float], float, or None, optional
        Single float or per-detector dict of maximum frequencies to enforce.
        If a float is provided, it is applied to all detectors. Each value
        must be ≤ `domain.f_max`. If `None`, no maximum-frequency
        validation is performed.

    Raises
    ------
    ValueError
        - If `model_metadata` does not describe a `UniformFrequencyDomain`
          or `MultibandedFrequencyDomain`.
        - If `f_min`/`f_max` keys don’t exactly match the detector list.
        - If any requested frequency lies outside the hard domain bounds.
        - If cropping is disabled but a change in frequency is requested.
        - If per-detector constraints (independent vs. joint) or
          cropping caps/floors are violated.

    Returns
    -------
    None
    """
    crop_settings = model_metadata["train_settings"]["data"].get(
        "random_strain_cropping"
    )
    detectors = model_metadata["train_settings"]["data"]["detectors"]
    domain = build_domain_from_model_metadata(model_metadata, base=True)
    if not isinstance(domain, (UniformFrequencyDomain, MultibandedFrequencyDomain)):
        raise ValueError("Frequency updates only possible for frequency domains.")

    if f_min is not None:
        _validate_minimum_frequency(f_min, detectors, domain, crop_settings)
    if f_max is not None:
        _validate_maximum_frequency(f_max, detectors, domain, crop_settings)


def resolve_frequency_bounds(
    detectors: list[str],
    domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
    minimum_frequency: dict[str, float] | float | None = None,
    maximum_frequency: dict[str, float] | float | None = None,
) -> dict[str, tuple[float, float]]:
    """Return `(f_min, f_max)` for each detector from an event's frequency range,
    given as one float for all detectors or as one value per detector. Missing
    values default to the domain bounds."""

    def expand(value, default):
        if value is None:
            return {d: float(default) for d in detectors}
        if isinstance(value, dict):
            if set(value) != set(detectors):
                raise ValueError(
                    f"Frequency bounds must have exactly detectors {detectors}, got "
                    f"{sorted(value)}."
                )
            return {d: float(value[d]) for d in detectors}
        return {d: float(value) for d in detectors}

    f_min = expand(minimum_frequency, domain.f_min)
    f_max = expand(maximum_frequency, domain.f_max)
    return {d: (f_min[d], f_max[d]) for d in detectors}


def check_importance_sampling_frequency_range(
    model_metadata: dict,
    minimum_frequency: dict[str, float] | float | None = None,
    maximum_frequency: dict[str, float] | float | None = None,
    sampling_frequency: float | None = None,
):
    """Check a frequency range given under `importance-sampling-updates`. It changes
    the likelihood only, never the network input, so the strain-cropping rules do
    not apply. Each detector's range must be positive, non-empty, and at most the
    Nyquist frequency of the data (when `sampling_frequency` is given)."""
    if minimum_frequency is None and maximum_frequency is None:
        return
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
