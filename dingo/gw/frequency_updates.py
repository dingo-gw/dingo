"""
Validation of event- and importance-sampling-time frequency-range updates against a
model's frequency domain.

A frequency range narrower than the network's domain is only permitted when the
network was trained with random strain cropping (`random_strain_cropping` in the
training data settings), within the bounds that the cropping covered. These
functions are called at INI-parse time (`dingo_pipe`, on the model metadata) and by
the samplers when event metadata carries `minimum_frequency` / `maximum_frequency`.
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
    if isinstance(f_max, float):
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
    if isinstance(f_min, float):
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
    Validate and apply optional minimum and maximum frequency constraints
    for a model’s frequency domain.

    This function checks that any provided per-detector minimum (`f_min`)
    or maximum (`f_max`) frequencies—either as a single float applied to
    all detectors or as a dict mapping each detector to its own value—:
      - Match exactly the set of detectors in the model metadata.
      - Respect the hard bounds defined by the domain (`domain.f_min` /
        `domain.f_max`).
      - Comply with optional random-strain-cropping settings (probability,
        independent vs. joint detectors, and per-detector caps/floors).

    Parameters
    ----------
    model_metadata : dict
        Dictionary containing the model’s training settings and data.
        Must include:
          - `["train_settings"]["data"]["detectors"]`: list of detector names.
          - `["train_settings"]["data"]["random_strain_cropping"]`: optional
            dict of cropping parameters.
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
