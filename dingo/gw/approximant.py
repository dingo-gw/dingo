from typing import NewType

import lalsimulation as LS

Approximant = NewType("Approximant", str)


def get_approximant(approximant: Approximant) -> int:
    """
    Converts a string representation of an approximant to its integer value.
    Note: alias for lalsimulation.GetApproximantFromString

    Parameters
    ----------
    approximant
        The string representation of the approximant.

    Returns
    -------
    The integer value of the approximant.
    """
    return LS.GetApproximantFromString(approximant)


def get_approximant_description(approximant: int) -> str:
    """
    Converts an integer value of an approximant to its string description.
    Note: alias for lalsimulation.GetStringFromApproximant

    Parameters
    ----------
    approximant
        The integer value of the approximant.

    Returns
    -------
        The string description of the approximant.
    """
    return LS.GetStringFromApproximant(int(approximant))
