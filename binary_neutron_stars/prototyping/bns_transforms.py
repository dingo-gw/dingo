from dingo.gw.domains import Domain
from multibanded_frequency_domain import MultibandedFrequencyDomain
from heterodyning import factor_fiducial_waveform


class ApplyHeterodyning(object):
    """Transform operator for applying phase heterodyning."""

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


class ApplyDecimation(object):
    """Transform operator for decimation to multibanded frequency domain."""

    def __init__(self, multibanded_frequency_domain: MultibandedFrequencyDomain):
        """
        Parameters
        ----------
        multibanded_frequency_domain: MultibandedFrequencyDomain
            New domain of the decimated waveform data. Original waveform data must be in
            multibanded_frequency_domain.original_domain
        """
        self.multibanded_frequency_domain = multibanded_frequency_domain

    def __call__(self, input_sample: dict):
        """
        Parameters
        ----------
        input_sample : dict
            Values of sample["waveform"] should be arrays containing waveforms to be
            transformed.

        Returns
        -------
        dict of the same form as the input, but with transformed (decimated) waveforms.
        """
        sample = input_sample.copy()

        waveform_dec = {
            k: self.multibanded_frequency_domain.decimate(v)
            for k, v in sample["waveform"].items()
        }
        sample["waveform"] = waveform_dec

        return sample
