import lal
import numpy as np
import torch

from dingo.gw.domains import Domain, MultibandedFrequencyDomain, FrequencyDomain


class HeterodynePhase(object):
    """
    Transform operator for applying phase heterodyning. See docstring of
    factor_fiducial_waveform for details.
    """

    def __init__(self, domain: Domain, order: int = 0, inverse: bool = False):
        """
        Parameters
        ----------
        domain: Domain
            Domain of the waveform data.
        order : int
            Order for phase heterodyning. Either 0 or 2.
        inverse : bool
            Whether to apply for the forward or inverse transform. Default: False.
        """
        self.domain = domain
        self.order = order
        self.inverse = inverse

    def __call__(self, input_sample: dict):
        """
        Parameters
        ----------
        input_sample : dict
            Values of sample["waveform"] should be arrays containing waveforms to be
            transformed.
            sample["parameters"] contains parameters of the binary system. Required for
            chirp mass (and mass ratio if self.order == 2).

        Returns
        -------
        dict of the same form as the input, but with transformed (heterodyned) waveforms.
        """
        sample = input_sample.copy()
        waveform = sample["waveform"]
        parameters = sample["parameters"]
        sample["waveform"] = {
            k: factor_fiducial_waveform(
                v,
                self.domain,
                parameters["chirp_mass"],
                mass_ratio=parameters.get("mass_ratio"),
                order=self.order,
                inverse=self.inverse,
            )
            for k, v in waveform.items()
        }

        return sample


class Decimate(object):
    """Transform operator for decimation to multibanded frequency domain."""

    def __init__(
        self,
        multibanded_frequency_domain: MultibandedFrequencyDomain,
        decimation_keys,
    ):
        """
        Parameters
        ----------
        multibanded_frequency_domain: MultibandedFrequencyDomain
            New domain of the decimated data. Original data must be in
            multibanded_frequency_domain.base_domain
        decimation_keys: list
            Keys for decimation, e.g., "waveform". If sample[decimation_key] is a dict,
            the decimation is applied to every element of the dict.
        """
        self.multibanded_frequency_domain = multibanded_frequency_domain
        self.decimation_keys = decimation_keys

    def __call__(self, input_sample: dict):
        """
        Parameters
        ----------
        input_sample : dict
            For each decimation_key in self.decimation_keys, sample[decimation_key]
            should be (1) a dict with arrays containing data to be transformed,
            or (2) an array with data to be transformed.

        Returns
        -------
        dict of the same form as the input, but with transformed (decimated) data.
        """
        sample = input_sample.copy()

        for decimation_key in self.decimation_keys:
            if isinstance(sample[decimation_key], dict):
                sample[decimation_key] = {
                    k: self.multibanded_frequency_domain.decimate(v)
                    for k, v in sample[decimation_key].items()
                }
            else:
                sample[decimation_key] = self.multibanded_frequency_domain.decimate(
                    sample[decimation_key]
                )

        return sample


class DecimateWaveformsAndASDS(object):
    """Transform operator for decimation of unwhitened waveforms and corresponding ASDS
    to multibanded frequency domain (MFD).


    For decimation, we have two options.


    1) decimation_mode = whitened
    In this case, the GW data is whitened first,

            dw = d / ASD,

    and then decimated to the MFD,

            dw_mfd = decimate(dw).

    In this case, the effective ASD in the MFD is given by

            ASD_mfd = 1 / decimate(1 / ASD).

    See [1] for details. ASD_mfd can then be provided to the inference network.


    2) decimation_mode = unwhitened
    In this case, the GW data is first decimated,

            d_mfd = decimate(d)

    and then whitened.

            dw_mfd = d_mfd / ASD_mfd.

    In this case, the ASD_mfd that whitens the data d_mfd is given by

            ASD_mfd = decimate(ASD ** 2) ** 0.5.

    In other words, in this case we need to decimate the *PSD*. See [1] for details.


    Method 1) better preserves the signal.

    [1] https://github.com/dingo-gw/dingo-development/blob/binary_neutron_stars/
    binary_neutron_stars/prototyping/psd_decimation.ipynb
    """

    def __init__(
        self,
        multibanded_frequency_domain: MultibandedFrequencyDomain,
        decimation_mode: str,
    ):
        """
        Parameters
        ----------
        multibanded_frequency_domain: MultibandedFrequencyDomain
            New domain of the decimated data. Original data must be in
            multibanded_frequency_domain.base_domain
        decimation_mode: str
            One of ["whitened", "unwhitened"]. Determines whether decimation is
            performed on whitened data or on unwhitened data. See class docstring for
            details.
        """
        self.multibanded_frequency_domain = multibanded_frequency_domain
        if decimation_mode not in ["whitened", "unwhitened"]:
            raise ValueError(
                f"Unsupported decimation mode {decimation_mode}, needs to be one of "
                f'["whitened", "unwhitened"].'
            )
        self.decimation_mode = decimation_mode

    def __call__(self, input_sample: dict):
        """
        Parameters
        ----------
        input_sample : dict
            Values of sample["waveform"] should be arrays containing waveforms to be
            transformed, Values of sample["asds"] should be arrays containing asds
            to be transformed.

        Returns
        -------
        dict of the same form as the input, but with transformed (decimated) waveforms
        and asds.
        """
        sample = input_sample.copy()

        if self.decimation_mode == "whitened":
            whitened_waveforms = {
                k: v / sample["asds"][k] for k, v in sample["waveform"].items()
            }
            whitened_waveforms_dec = {
                k: self.multibanded_frequency_domain.decimate(v)
                for k, v in whitened_waveforms.items()
            }
            asds_dec = {
                k: 1 / self.multibanded_frequency_domain.decimate(1 / v)
                for k, v in sample["asds"].items()
            }
            # color the whitened waveforms with the effective asd
            waveform_dec = {
                k: v * asds_dec[k] for k, v in whitened_waveforms_dec.items()
            }
            sample["waveform"] = waveform_dec
            sample["asds"] = asds_dec

        elif self.decimation_mode == "unwhitened":
            sample["waveform"] = {
                k: self.multibanded_frequency_domain.decimate(v)
                for k, v in sample["waveform"].items()
            }
            sample["asds"] = {
                k: self.multibanded_frequency_domain.decimate(v ** 2) ** 0.5
                for k, v in sample["asds"].items()
            }

        else:
            raise NotImplementedError()

        return sample


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
            # Use np.outer.squeeze instead of chirp_mass * f, as the latter has a
            # different precision, and we don't want the behaviour of this function to
            # depend on whether chirp_mass is an array or float.
            mc_f = np.outer(chirp_mass, f).squeeze()
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
        if (f[..., 0] == 0.0).all():
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
        raise NotImplementedError("Can only use phase heterodyning frequency domains.")


def change_phase_heterodyning(data, domain, old_kwargs, new_kwargs):
    data_unhet = factor_fiducial_waveform(data, domain, **old_kwargs, inverse=True)
    data_het_new = factor_fiducial_waveform(data_unhet, domain, **new_kwargs)
    return data_het_new
