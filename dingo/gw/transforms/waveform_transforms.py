import numpy as np

from dingo.gw.domains import MultibandedFrequencyDomain, UniformFrequencyDomain


class DecimateAll(object):
    """Transform operator for decimation to multibanded frequency domain."""

    def __init__(
        self,
        multibanded_frequency_domain: MultibandedFrequencyDomain,
    ):
        """
        Parameters
        ----------
        multibanded_frequency_domain: MultibandedFrequencyDomain
            New domain of the decimated data. Original data must be in
            multibanded_frequency_domain.base_domain
        """
        self.multibanded_frequency_domain = multibanded_frequency_domain

    def __call__(self, input_sample: dict) -> dict:
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
        decimate_recursive(sample, self.multibanded_frequency_domain)
        return sample


def decimate_recursive(d: dict, mfd: MultibandedFrequencyDomain):
    """
    In-place decimation of nested dicts of arrays.

    Parameters
    ----------
    d : dict
        Nested dictionary to decimate.
    mfd : MultibandedFrequencyDomain
    """
    for k, v in d.items():
        if isinstance(v, dict):
            decimate_recursive(v, mfd)
        elif isinstance(v, np.ndarray):
            if v.shape[-1] == len(mfd.base_domain):
                d[k] = mfd.decimate(v)
        else:
            raise ValueError(f"Cannot decimate item of type {type(v)}.")


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

    [1] https://github.com/dingo-gw/dingo/blob/fede5c01524f3e205acf5750c0a0f101ff17e331/binary_neutron_stars/prototyping/psd_decimation.ipynb
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

    def __call__(self, input_sample: dict) -> dict:
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

        # Only decimate the data if it is in the base domain. If it has already been
        # decimated, do not change it.

        if check_sample_in_domain(
            sample, self.multibanded_frequency_domain.base_domain
        ):
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


def check_sample_in_domain(sample, domain: UniformFrequencyDomain) -> bool:
    lengths = []
    base_domain_length = len(domain)
    for k in ["waveform", "asds"]:
        lengths += [d.shape[-1] for d in sample[k].values()]
    if all(l == base_domain_length for l in lengths):
        return True
    else:
        return False
