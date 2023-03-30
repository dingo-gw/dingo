import lal
import numpy as np
import torch

from dingo.gw.domains import FrequencyDomain
from dingo.gw.domains.multibanded_frequency_domain import MultibandedFrequencyDomain


def heterodyne_LO(data, domain, chirp_mass):
    mc_f = chirp_mass * domain()
    if mc_f[0] == 0.0:
        mc_f[0] = 1.0
    pi_mc_f_SI = np.pi * (lal.GMSUN_SI / lal.C_SI ** 3) * mc_f
    fiducial_phase = (3 / 128) * (pi_mc_f_SI) ** (-5 / 3)
    return domain.add_phase(data, -fiducial_phase)


def change_heterodyning(data, domain, old_kwargs, new_kwargs):
    data_unhet = factor_fiducial_waveform(data, domain, **old_kwargs, inverse=True)
    data_het_new = factor_fiducial_waveform(data_unhet, domain, **new_kwargs)
    return data_het_new


def factor_fiducial_waveform(
    data, domain, chirp_mass, mass_ratio=None, order=0, inverse=False
):
    """
    Relative binning / heterodyning. Divides the data by a fiducial waveform defined by
    the chirp mass and (optionally) mass ratio. Allows for batching.

    At leading order, this factors out the overall chirp by dividing the data by a
    fiducial waveform of the form
        exp( - 1j * (3/128) * (pi G chirp_mass f / c**3)**(-5/3) ) ;
    see 2001.11412, eq. (7.2). This is the leading order chirp due to the emission of
    quadrupole radiation.

    At next-to-leading order, this implements 1PN corrections involving the mass ratio.

    We do not include any amplitude in the fiducial waveform, since at inference time
    this transformation will be applied to noisy data. Multiplying the frequency-domain
    noise by a complex number of unit norm is allowed because it only changes the
    phase, not the overall amplitude, which would change the noise PSD.

    Parameters
    ----------
    data : Union[dict, torch.Tensor]
        If a dict, the keys would correspond to different detectors or
        polarizations. For a Tensor, these would be within different components.
        This method uses the same fiducial waveform for each detector.
    domain : Domain
        Only works for a FrequencyDomain or MultibandedFrequencyDomain at present.
    chirp_mass : Union[np.array, torch.Tensor, float]
        Chirp mass parameter(s).
    mass_ratio : Union[np.array, torch.Tensor, float]
        Mass ratio parameter(s).
    order : int
        Twice the post-Newtonian order for the expansion. Valid orders are 0 and 2.

    Returns
    -------
    data: Union[dict, torch.Tensor]
        Transformed data, of the same form as data.
    """
    if order not in (0, 2):
        raise ValueError(f"Order {order} invalid. Acceptable values are 0 and 2.")
    if order == 2:
        if mass_ratio is None:
            raise ValueError(f"Mass ratio required for 2nd order heterodyning.")
        elif (
            isinstance(chirp_mass, (np.ndarray, torch.Tensor))
            and chirp_mass.shape != mass_ratio.shape
        ):
            raise ValueError(
                f"Shape of chirp_mass ({chirp_mass.shape}) and mass_ratio "
                f"({mass_ratio.shape}) don't match"
            )
    if not np.isfinite(chirp_mass).all():
        raise ValueError("Got nan or inf elements in chirp_mass.")

    if isinstance(domain, (FrequencyDomain, MultibandedFrequencyDomain)):
        if type(data) == dict:
            f = domain.get_sample_frequencies_astype(list(data.values())[0])
        else:
            f = domain.get_sample_frequencies_astype(data)

        # Expand across possible batch dimension.
        if type(chirp_mass) == np.float64 or type(chirp_mass) == float:
            mc_f = chirp_mass * f
            if f[0] == 0.0:
                mc_f[0] = 1.0
        elif type(chirp_mass) == np.ndarray:
            mc_f = np.outer(chirp_mass, f)
            if mass_ratio is not None:
                mass_ratio = mass_ratio[:, None]
        elif type(chirp_mass) == torch.Tensor:
            mc_f = torch.outer(chirp_mass, f)
            if mass_ratio is not None:
                mass_ratio = mass_ratio[:, None]
        else:
            raise TypeError(
                f"Invalid type {type(chirp_mass)}. "
                f"Only implemented for floats, arrays and tensors"
            )

        # Avoid taking a negative power of 0 in the first index. This will get
        # chopped off or multiplied by 0 later anyway.
        if f[0] == 0.0:
            mc_f[..., 0] = 1.0

        # Leading (0PN) phase
        pi_mc_f_SI = np.pi * (lal.GMSUN_SI / lal.C_SI ** 3) * mc_f
        fiducial_phase = (3 / 128) * (pi_mc_f_SI) ** (-5 / 3)

        # 1PN correction
        if order >= 2:
            assert mass_ratio is not None
            symmetric_mass_ratio = mass_ratio / (1 + mass_ratio) ** 2
            pi_m_f_SI = pi_mc_f_SI / symmetric_mass_ratio ** (3 / 5)
            correction = 1 + (55 * symmetric_mass_ratio / 9 + 3715 / 756) * (
                pi_m_f_SI
            ) ** (2 / 3)

            fiducial_phase *= correction

        if inverse:
            fiducial_phase *= -1

        if type(data) == dict:
            result = {}
            for k, v in data.items():
                result[k] = domain.add_phase(v, -fiducial_phase)
        else:
            result = domain.add_phase(data, -fiducial_phase)

        return result

    else:
        raise NotImplementedError("Can only use GNPEChirp in frequency domain.")
