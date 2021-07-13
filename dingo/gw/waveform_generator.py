import numpy as np
import math
import numbers
from typing import Dict, List, Tuple, Union
import warnings

import torch
from torch.distributions.transforms import Transform
from torch.distributions import constraints
from torch.distributions.transformed_distribution import TransformedDistribution

import lal
import lalsimulation as LS

from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters, bilby_to_lalsimulation_spins

from dingo.gw.domains import Domain


# TODO:
#  * Check FD and TD waveforms against pycbc
#  * check EOB precessing model calls
#  * add unit test for TD

#  * GPNPE needs to be put in: need to carry around detector coalescence times
#  * Need to transform into input data for NN. This will require an extra class, not in this module
#  * whitening will be done later as a transform: ASD
#    - Rather treat the noise as an additional transform: add noise and whiten
#    similar for noise transients / realistic noise



class WaveformGenerator:
    """Generate polarizations in the specified domain for a
    single GW coalescence given a set of waveform parameters.
    """

    def __init__(self,
                 approximant: str,
                 domain: Domain,
                 mode_list: List[Tuple] = None):
        """
        Parameters
        ----------
        approximant : str
            Waveform "approximant" string understood by lalsimulation
            This is defines which waveform model is used.
        domain : Domain
            Domain object that specifies on which physical domain the
            waveform polarizations will be generated, e.g. Fourier
            domain, time domain.
        mode_list : List[Tuple]
            A list of waveform (ell, m) modes to include when generating
            the polarizations.
        """
        if not isinstance(approximant, str):
            raise ValueError('approximant should be a string, but got', approximant)
        else:
            self.approximant_str = approximant
            self.approximant = LS.GetApproximantFromString(approximant)

        if not issubclass(type(domain), Domain):
            raise ValueError('domain should be an instance of a subclass of Domain, but got', type(domain))
        else:
            self.domain = domain

        self.lal_params = None
        if mode_list is not None:
            self.lal_params = self.setup_mode_array(mode_list)


    def generate_hplus_hcross(self, parameters: Dict[str, float],
                              catch_waveform_errors=True) -> Dict[str, np.ndarray]:
        """Generate GW polarizations (h_plus, h_cross).

        Parameters
        ----------
        parameters: Dict[str, float]
            A dictionary of parameter names and scalar values.
            The parameter dictionary must include the following keys.
            For masses, spins, and distance there are multiple options.

            Mass: (mass_1, mass_2) or a pair of quantities from
                ((chirp_mass, total_mass), (mass_ratio, symmetric_mass_ratio))
            Spin:
                (a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl) if precessing binary or
                (chi_1, chi_2) if the binary has aligned spins
            Reference frequency: f_ref at which spin vectors are defined
            Extrinsic:
                Distance: one of (luminosity_distance, redshift, comoving_distance)
                Inclination: theta_jn
                Reference phase: phase
                Geocentric time: geocent_time (GPS time)
            The following parameters are not required:
                Sky location: ra, dec,
                Polarization angle: psi
            Units:
                Masses should be given in units of solar masses.
                Distance should be given in megaparsecs (Mpc).
                Frequencies should be given in Hz and time in seconds.
                Spins should be dimensionless.
                Angles should be in radians.

        catch_waveform_errors: bool
            Whether to catch lalsimulation errors

        Use the domain, approximant, and mode_list specified in the constructor
        along with the waveform parameters to generate the waveform polarizations.
        """
        if not isinstance(parameters, dict):
            raise ValueError('parameters should be a dictionary, but got', parameters)
        elif not isinstance(list(parameters.values())[0], float):
            raise ValueError('parameters dictionary must contain floats', parameters)

        # Convert to lalsimulation parameters according to the specified domain
        parameters_lal = self._convert_parameters_to_lal_frame(parameters, self.lal_params)

        # Generate GW polarizations
        if self.domain.domain_type == 'uFD':
            wf_generator = self.generate_FD_waveform
        elif domain.domain_type == 'TD':
            wf_generator = self.generate_TD_waveform
        else:
            raise ValueError(f'Unsupported domain type {domain.domain_type}.')

        try:
            wf_dict = wf_generator(parameters_lal)
        except Exception as e:
            if not catch_waveform_errors:
                raise
            else:
                EDOM = (e.args[0] == 'Internal function call failed: Input domain error')
                if EDOM:
                    warnings.warn(f"Evaluating the waveform failed with error: {e}\n"
                                  f"The parameters were {parameters_lal}\n")
                    return None
                else:
                    raise

        return wf_dict

    def _convert_to_scalar(self, x: Union[np.ndarray, float]):
        """Convert a single element array to a number."""
        if isinstance(x, np.ndarray):
            if x.shape == () or x.shape == (1, ):
                return x.item()
            else:
                raise ValueError(f'Expected an array of length one, but shape = {x.shape}')
        else:
            return x

    def _convert_parameters_to_lal_frame(self, parameter_dict: Dict, lal_params=None):
        """Convert to lal source frame parameters

        Parameters
        ----------
        parameter_dict : Dict
            A dictionary of parameter names and 1-dimensional prior distribution
            objects. If None, we use a default binary black hole prior.
        lal_params : (None, or Swig Object of type 'tagLALDict *')
            Extra parameters which can be passed to lalsimulation calls.
        """
        # Transform mass, spin, and distance parameters
        p, _ = convert_to_lal_binary_black_hole_parameters(parameter_dict)

        # Transform to lal source frame: iota and Cartesian spin components
        param_keys_in = ('theta_jn', 'phi_jl', 'tilt_1', 'tilt_2', 'phi_12',
                         'a_1', 'a_2', 'mass_1', 'mass_2', 'f_ref', 'phase')
        param_values_in = [p[k] for k in param_keys_in]
        iota_and_cart_spins = bilby_to_lalsimulation_spins(*param_values_in)
        iota, s1x, s1y, s1z, s2x, s2y, s2z = [self._convert_to_scalar(x) for x in iota_and_cart_spins]

        # Convert to SI units
        p['mass_1'] *= lal.MSUN_SI
        p['mass_2'] *= lal.MSUN_SI
        p['luminosity_distance'] *= 1e6 * lal.PC_SI

        # Construct argument list for FD and TD lal waveform generator wrappers
        spins_cartesian = s1x, s1y, s1z, s2x, s2y, s2z
        masses = (p['mass_1'], p['mass_2'])
        extra_params = (p['luminosity_distance'], p['theta_jn'], p['phase'])
        ecc_params = (0.0, 0.0, 0.0)  # longAscNodes, eccentricity, meanPerAno

        D = self.domain
        if D.domain_type == 'uFD':
            domain_pars = (D.delta_f, D.f_min, D.f_max, p['f_ref'])
        elif D.domain_type == 'TD':
            # FIXME: compute f_min from duration or specify it if SimInspiralTD
            #  is used for a native FD waveform
            f_min = 20.0
            domain_pars = (D.delta_t, f_min, p['f_ref'])
        else:
            raise ValueError(f'Unsupported domain type {D.domain_type}.')

        lal_parameter_tuple = masses + spins_cartesian + extra_params + ecc_params + \
                              domain_pars + (lal_params, self.approximant)
        return lal_parameter_tuple


    def setup_mode_array(self, mode_list : List[Tuple]):
        """Define a mode array to select waveform modes
        to include in the polarizations from a list of modes.

        Parameters
        ----------
        mode_list : a list of (ell, m) modes
        """
        lal_params = lal.CreateDict()
        ma = LS.SimInspiralCreateModeArray()
        for (ell, m) in mode_list:
            LS.SimInspiralModeArrayActivateMode(ma, ell, m)
            LS.SimInspiralModeArrayActivateMode(ma, ell, -m)
        LS.SimInspiralWaveformParamsInsertModeArray(lal_params, ma)
        return lal_params


    def generate_FD_waveform(self, parameters_lal) -> Dict[str, np.ndarray]:
        """Generate Fourier domain GW polarizations (h_plus, h_cross)."""
        # TODO:
        #  * put preferred way of calling precessing EOB models with spins
        #   defined at specified reference frequency
        #  * also check against ChooseFD
        #
        # LS.SimInspiralFD takes parameters:
        #   m1, m2, S1x, S1y, S1z, S2x, S2y, S2z,
        #   distance, inclination, phiRef,
        #   longAscNodes, eccentricity, meanPerAno,
        #   deltaF, f_min, f_max, f_ref,
        #   lal_params, approximant

        # Depending on whether the domain is uniform or non-uniform call the appropriate wf generator
        hp, hc = LS.SimInspiralFD(*parameters_lal)

        # Ensure that the waveform agrees with the frequency grid defined in the domain.
        frequency_array = self.domain()
        h_plus = np.zeros_like(frequency_array, dtype=complex)
        h_cross = np.zeros_like(frequency_array, dtype=complex)
        if len(hp.data.data) > len(frequency_array):
            warnings.warn("LALsimulation waveform longer than domain's `frequency_array`"
                          f"({len(hp.data.data)} vs {len(frequency_array)}). Truncating lalsim array.")
            h_plus = hp.data.data[:len(h_plus)]
            h_cross = hc.data.data[:len(h_cross)]
        else:
            h_plus[:len(hp.data.data)] = hp.data.data
            h_cross[:len(hc.data.data)] = hc.data.data

        # Undo the time shift done in SimInspiralFD to the waveform
        dt = 1 / hp.deltaF + (hp.epoch.gpsSeconds + hp.epoch.gpsNanoSeconds * 1e-9)
        time_shift = np.exp(-1j * 2 * np.pi * dt * frequency_array)
        h_plus *= time_shift
        h_cross *= time_shift
        return {'h_plus': h_plus, 'h_cross': h_cross}


    def generate_TD_waveform(self, parameters_lal) -> Dict[str, np.ndarray]:
        """Generate time domain GW polarizations (h_plus, h_cross)"""
        # LS.SimInspiralTD takes parameters:
        #   m1, m2, S1x, S1y, S1z, S2x, S2y, S2z,
        #   distance, inclination, phiRef,
        #   longAscNodes, eccentricity, meanPerAno,
        #   deltaT, f_min, f_ref
        #   lal_params, approximant
        hp, hc = LS.SimInspiralTD(*parameters_lal)
        # TODO:
        #  * do we need to do anything else
        #  * put preferred way of calling EOB precessing models
        h_plus = hp.data.data,
        h_cross = hc.data.data
        return {'h_plus': h_plus, 'h_cross': h_cross}



# TODO:
#  * waveform_generator.WaveformDataset._compute_parameter_statistics()
#    computes mean and stdev analytically for known distributions which
#    is more accurate than computing sample means and stdevs

class StandardizeParameters:
    """
    Standardize parameters according to the transform (x - mu) / std.
    """
    def __init__(self, mu: np.ndarray, std: np.ndarray):
        """
        Initialize the standardization transform with means
        and standard deviations for each parameter

        Parameters
        ----------
        mu : torch.tensor()
            The (estimated) 1D tensor of the means
        std : torch.tensor()
            The (estimated) 1D tensor of the standard deviations
        """
        self.mu = mu
        self.std = std

    def __call__(self, samples: Dict[str, np.ndarray]):
        """Standardize the parameter array according to the
        specified means and standard deviations.

        Parameters
        ----------
        samples: Dict[str, np.ndarray]
            A dictionary with keys 'parameters', 'waveform'.

        FIXME: Check: Only the parameters get transformed.
        """
        x = samples['parameters']
        y = (x - self.mu) / self.std
        return {'parameters': y, 'waveform': samples['waveform']}

    def inverse(self, samples: Dict[str, torch.tensor]):
        """De-standardize the parameter array according to the
        specified means and standard deviations.

        Parameters
        ----------
        samples: Dict[str, np.ndarray]
            A dictionary with keys 'parameters', 'waveform'.

        FIXME: Check: Only the parameters get transformed.
        """
        y = samples['parameters']
        x = self.mu + y * self.std
        return {'parameters': x, 'waveform': samples['waveform']}



if __name__ == "__main__":
    """A visual test."""
    from dingo.gw.domains import Domain, UniformFrequencyDomain
    import matplotlib.pyplot as plt

    # approximant = 'IMRPhenomPv2'
    # f_min = 20.0
    # f_max = 512.0
    # domain = UniformFrequencyDomain(f_min=f_min, f_max=f_max, delta_f=1.0/4.0, window_factor=1.0)
    # parameters = {'chirp_mass': 34.0, 'mass_ratio': 0.35, 'chi_1': 0.2, 'chi_2': 0.1, 'theta_jn': 1.57, 'f_ref': 20.0, 'phase': 0.0, 'luminosity_distance': 1.0}
    # WG = WaveformGenerator(approximant, domain)
    # waveform_polarizations = WG.generate_hplus_hcross(parameters)
    # print(waveform_polarizations['h_plus'])
    #
    # plt.loglog(domain(), np.abs(waveform_polarizations['h_plus']))
    # plt.xlim([f_min/2, 2048.0])
    # plt.axvline(f_min, c='gray', ls='--')
    # plt.show()
