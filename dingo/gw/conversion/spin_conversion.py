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


def change_spin_conversion_phase(p, f_ref, sc_phase_old, sc_phase_new):
    """
    Change the phase used to convert cartesian spins to PE spins. The cartesian spins
    are independent of the spin conversion phase. When converting from cartesian spins
    to PE spins, the phase value has an impact on theta_jn and phi_jl.

    The usual convention for the PE spins is to use the real phase for the conversion
    (cart. spins <--> PE spins), but for dingo-IS with the synthetic phase extension we
    need to use another convention, where the PE spins are defined with spin conversion
    phase 0. This function transforms between the different conventions.

    Parameters
    ----------
    p: pd.Dataframe
        parameters
    f_ref: float
        reference frequency for definition of spins
    sc_phase_old: float or str
        spin conversion phase used for input parameters
    sc_phase_new: float or str
        spin conversion phase used for output parameters

    Returns
    -------
    p_new:
        parameters with changed spin conversion phase
    """
    if sc_phase_old == sc_phase_new:
        return p

    # check validity of phases
    for sc_phase in [sc_phase_old, sc_phase_new]:
        if isinstance(sc_phase, str):
            if not sc_phase == "real_phase":
                raise ValueError(
                    f"Phase for spin conversion needs to be either a number, "
                    f'or the string "real_phase", got {sc_phase}.'
                )
            if not "phase" in p:
                raise ValueError(
                    "Using real phase for spin conversion, but phase not in parameters."
                )
        elif not type(sc_phase) in (int, float):
            raise ValueError(
                f"Phase for spin conversion needs to be either a number, "
                f'or the string "real_phase", got {sc_phase}.'
            )

    phase_old = sc_phase_old
    phase_new = sc_phase_new
    p_new = {k: [] for k in p.keys()}
    for idx in range(len(p)):
        if sc_phase_old == "real_phase":
            phase_old = p["phase"][idx]
        if sc_phase_new == "real_phase":
            phase_new = p["phase"][idx]

        params_idx = {k: v[idx] for k, v in p.items()}

        # transform to cartesian spins (which is the fundamental basis, independent of
        # the phase) with the old spin conversion phase, and back to PE spins with the
        # new spin conversion phase
        params_idx_cart = cartesian_spins({**params_idx, "phase": phase_old}, f_ref)
        params_idx_pe_new = pe_spins({**params_idx_cart, "phase": phase_new}, f_ref)

        # The conversions above will set the phase parameter to sc_phase_new, however,
        # this is only the phase used for spin conversion, *not* the actual phase for
        # the parameters. Below, we set the phase to its correct (i.e., input) value.
        if "phase" in p:
            params_idx_pe_new["phase"] = p["phase"][idx]
        else:
            params_idx_pe_new.pop("phase")

        for k, v in p_new.items():
            v.append(params_idx_pe_new[k])

    return pd.DataFrame(p_new)
