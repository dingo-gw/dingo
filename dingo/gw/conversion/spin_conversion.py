# Transformations between PE spins and Cartesian spins
import pandas as pd
import lal
import lalsimulation as LS

DINGO_PE_SPIN_PARAMETERS = (
    "theta_jn",
    "phi_jl",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "a_1",
    "a_2",
)
DINGO_CARTESIAN_SPIN_PARAMETERS = (
    "iota",
    "spin_1x",
    "spin_1y",
    "spin_1z",
    "spin_2x",
    "spin_2y",
    "spin_2z",
)


def component_masses(p):
    if "m1" in p and "m2" in p:
        return p["m1"], p["m2"]

    Mc = p["chirp_mass"]
    q = p["mass_ratio"]
    eta = q / (1 + q) ** 2
    M = Mc * eta ** (-3 / 5)

    m1 = M / (1 + q)
    m2 = q * m1
    return m1, m2


def cartesian_spins(p, f_ref):
    """
    Transform PE spins to cartesian spins.

    Parameters
    ----------
    p: dict
        contains parameters, including PE spins
    f_ref: float
        reference frequency for definition of spins

    Returns
    -------
    result: dict
        parameters, including cartesian spins
    """
    m1, m2 = component_masses(p)
    m1 *= lal.MSUN_SI
    m2 *= lal.MSUN_SI

    # Get pe spin params. Output is order identically to DINGO_CARTESIAN_SPIN_PARAMETERS,
    # iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z.
    cart_spin_params = LS.SimInspiralTransformPrecessingNewInitialConditions(
        p["theta_jn"],
        p["phi_jl"],
        p["tilt_1"],
        p["tilt_2"],
        p["phi_12"],
        p["a_1"],
        p["a_2"],
        m1,
        m2,
        f_ref,
        p["phase"],
    )

    # remove PE spin parameters
    result = {k: v for k, v in p.items() if k not in DINGO_PE_SPIN_PARAMETERS}
    # add cartesian spin parameters
    for param_name, param_value in zip(
        DINGO_CARTESIAN_SPIN_PARAMETERS, cart_spin_params
    ):
        result[param_name] = param_value

    return result


def pe_spins(p, f_ref):
    """
    Transform cartesian spins to PE spins.

    Parameters
    ----------
    p: dict
        contains parameters, including cartesian spins
    f_ref: float
        reference frequency for definition of spins

    Returns
    -------
    result: dict
        parameters, including PE spins
    """
    m1, m2 = component_masses(p)

    # Get pe spin params. Output is order identically to DINGO_PE_SPIN_PARAMETERS,
    # theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2.
    pe_spin_params = LS.SimInspiralTransformPrecessingWvf2PE(
        p["iota"],
        p["spin_1x"],
        p["spin_1y"],
        p["spin_1z"],
        p["spin_2x"],
        p["spin_2y"],
        p["spin_2z"],
        m1,
        m2,
        f_ref,
        p["phase"],
    )

    # remove cartesian spin parameters
    result = {k: v for k, v in p.items() if k not in DINGO_CARTESIAN_SPIN_PARAMETERS}
    # add PE spin parameters
    for param_name, param_value in zip(DINGO_PE_SPIN_PARAMETERS, pe_spin_params):
        result[param_name] = param_value

    return result


def change_spin_conversion_phase(samples, f_ref, sc_phase_old, sc_phase_new):
    """
    Change the phase used to convert cartesian spins to PE spins. The cartesian spins
    are independent of the spin conversion phase. When converting from cartesian spins
    to PE spins, the phase value has an impact on theta_jn and phi_jl.

    The usual convention for the PE spins is to use the phase parameter for the conversion
    (cart. spins <--> PE spins), but for dingo-IS with the synthetic phase extension we
    need to use another convention, where the PE spins are defined with spin conversion
    phase 0. This function transforms between the different conventions.

    Parameters
    ----------
    samples: pd.Dataframe
        Parameters.
    f_ref: float
        Reference frequency for definition of spins.
    sc_phase_old: float or None
        Spin conversion phase used for input parameters. If None, use the phase parameter.
    sc_phase_new: float or None
        Spin conversion phase used for output parameters. If None, use the phase
        parameter.

    Returns
    -------
    p_new:
        parameters with changed spin conversion phase
    """
    samples = samples.astype(float)
    if sc_phase_old == sc_phase_new:
        return samples

    # check validity of phases
    for sc_phase in [sc_phase_old, sc_phase_new]:
        if sc_phase is None:
            if "phase" not in samples:
                raise ValueError(
                    "Using phase parameter for spin conversion, but phase not in "
                    "parameters."
                )
        elif type(sc_phase) not in (int, float):
            raise ValueError(
                f"Phase for spin conversion needs to be either a number or None; got"
                f"{sc_phase}."
            )

    phase_old = sc_phase_old
    phase_new = sc_phase_new
    samples_new = {}
    for idx, sample in samples.to_dict(orient="index").items():
        try:
            if sc_phase_old is None:
                phase_old = sample["phase"]
            if sc_phase_new is None:
                phase_new = sample["phase"]

            # transform to cartesian spins (which is the fundamental basis, independent of
            # the phase) with the old spin conversion phase, and back to PE spins with the
            # new spin conversion phase
            sample_cartesian = cartesian_spins({**sample, "phase": phase_old}, f_ref)
            sample_pe_new = pe_spins({**sample_cartesian, "phase": phase_new}, f_ref)

            # The conversions above will set the phase parameter to sc_phase_new, however,
            # this is only the phase used for spin conversion, *not* the actual phase for
            # the parameters. Below, we set the phase to its correct (i.e., input) value.
            if "phase" in sample:
                sample_pe_new["phase"] = sample["phase"]
            else:
                sample_pe_new.pop("phase")

            samples_new[idx] = sample_pe_new
        except RuntimeError:
            print("Failed to convert spins. Saving sample unchanged.")
            samples_new[idx] = sample

    return pd.DataFrame.from_dict(samples_new, orient="index")
