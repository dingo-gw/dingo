# Import all of the prior classes needed to define the default BBH prior by hand.
# We're not using all of them. We could make the default prior more explicit
# by defining it ourselves in the constructor.
from bilby.gw.prior import BBHPriorDict, UniformSourceFrame
from bilby.core.prior import Uniform, Constraint, Sine, Cosine
from astropy.cosmology import Planck15

import numpy as np
from typing import Dict, Set
import warnings


# Silence INFO and WARNING messages from bilby
import logging
logging.getLogger('bilby').setLevel("ERROR")


# This is subclassing the Bilby BBHPriorDict.
#
# TODO:
#  * For some extrinsic parameters (e.g., distance, time of coalescence) we need to also
#  set reference values to be able to generate waveforms and position the detectors.
#  I am wondering whether this class should also take care of that.
#    - We could set delta function priors for some parameters, but this would mean
#      that we can't sample from their usual distributions. Perhaps it is better to
#      do this elsewhere?

#  * [x] Check that the user has provided a "complete" set of prior parameters
#    - How do you want to define complete? I.e. which parameters should always be present and which are optional?
#  * Maybe there should be a default prior dict that gets used for parameters the user doesn't specify? I'm not sure how to deal with that. It looks like you are adding in a geocent_time prior if it's absent.
#     - That can be done; I'm adding the time prior since bilby does not include it in the default BBH prior.

#  * Allow for a variety of parametrizations (e.g., also zenith, azimuth would work for sky position)
#    - That is possible, but ties in with checking for completeness
#  * Let's leave the parameter conversion for the waveform generator
#     - Agreed, but allowing for a variety of parametrizations implies that some conversion needs to be done
#       There needs to be some underlying representation we can convert to in order to check completeness.
#  * Provide "reference" values for certain extrinsic parameters (like distance, time) needed to generate waveforms. Should these be user-specified?
#    - Should these really live in a prior class? A fixed value in terms of a distribution would be a delta function prior
#      But don't we need to still sample from a distribution for e.g. distance? If we don't then having the reference values here is certainly fine.
#  * For the geocent_time reference value, we need to decide where to keep track of that. It will be used in postprocessing as well to correct the right ascension.


# Note:
#  * We add bilby as a dependency here which requires further packages :
#    https://git.ligo.org/lscsoft/bilby/blob/master/requirements.txt
#  * bilby uses logger to log messages when using bilby classes; should
#  we turn this off or reconfigure it somehow?

# Note: We keep the detectors somewhat fixed at some reference time -- we assume
# that Detectors don't move between events
# in post-processing fix that by looking at how much the earth has rotated
# They are not totally fixed (plug in reference time + dt)

# Q: Do we need parameter conversion / transformation somewhere? Perhaps only
# for transforming parameters for the posterior PDF.
# Look at bilby.gw.conversion


class GWPriorDict(BBHPriorDict):
    """
    Collect the prior distributions of parameters the data generating model
    depends on, distinguish intrinsic and extrinsic parameters (see below),
    and provide methods to sample from subsets of parameters.

    *Intrinsic parameters* are parameters that the data model depends on in a
    complicated way and, therefore, there parameters need to be sampled when
    generating the dataset. In contrast, *extrinsic parameters* are parameters
    on which the model depends in a simple way. Therefore, intrinsic samples
    from the data model can be augmented during training to include samples
    of extrinsic parameters by using simple transformations.

    In the GW inference case, the data (generating) model is a GW waveform model,
    This is the list of parameters in the BBH default prior:

        ['mass_1', 'mass_2', 'mass_ratio', 'chirp_mass',
        'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
        'theta_jn', 'phase',
        'luminosity_distance', 'dec', 'ra', 'psi']

    Note that this list does not explicitly include a time parameter, and that
    the mass parameters are over-specified in order to add constraints in the
    component mass plane on top of the chirp-mass and mass-ratio priors.
    """

    def __init__(self,
                 parameter_prior_dict: Dict = None,
                 geocent_time: float = 1126259642.413):
        """
        Parameters
        ----------
        parameter_prior_dict : Dict
            A dictionary of parameter names and 1-dimensional prior distribution
            objects. If None, we use a default binary black hole prior.
        geocent_time : float
            The geocentric GPS time used to determine the time prior.
        """

        if parameter_prior_dict is None:
            warnings.warn('No priors specified. Using default 15D priors.')
        else:
            if not isinstance(parameter_prior_dict, dict):
                raise ValueError('Expected dictionary of parameters and priors, '
                                 'but got:', parameter_prior_dict)

        # Build this prior dict
        super().__init__(dictionary=parameter_prior_dict)
        self._check_prior_completeness()

        # TODO: Which time parameter should be added?
        if not ('geocent_time' in self):
            self['geocent_time'] = Uniform(
                minimum=geocent_time - 0.1, maximum=geocent_time + 0.1,
                name='geocent_time', latex_label='$t_c$', unit='$s$')

        # TODO: Please check whether this is what we want
        self._intrinsic_parameters = ['mass_1', 'mass_2', 'mass_ratio',
                                      'chirp_mass', 'phase', 'theta_jn',
                                      'a_1', 'a_2', 'tilt_1', 'tilt_2',
                                      'phi_12', 'phi_jl']
        self._extrinsic_parameters = ['luminosity_distance', 'dec', 'ra', 'psi',
                                      'geocent_time']


    def _check_mass_parameters(self, key_set):
        """
        Check the presence of mass parameters in the prior.

        Either the component masses or mass-ratio and chirp-mass
        must be specified. If both are specified then the component mass
        priors must impose constraints on the mass-ratio and chirp-mass prior.
        """
        masses_present = {'mass_1', 'mass_2'} <= key_set
        mc_q_present = {'mass_ratio', 'chirp_mass'} <= key_set

        if masses_present:
            comp_constraints = [isinstance(self[k], Constraint)
                               for k in ['mass_1', 'mass_2']]
            if not mc_q_present:
                # This is OK as long as the masses are not just constraint priors
                if any(comp_constraints):
                    raise ValueError('Mass priors cannot be Constraint priors when'
                                     'mass-ratio and chirp mass priors are not set.')
            else:
                if not all(comp_constraints):
                    raise ValueError('When both component mass priors and chirp-mass and mass-ratio priors'
                                     'are specified, component mass priors must be constraint priors.')
        else:
            warnings.warn('Using default mass priors. Please check that the limits are appropriate.')
            if not mc_q_present:
                self['mass_ratio'] = Uniform(minimum=0.125, maximum=1, name='mass_ratio')
                self['chirp_mass'] = Uniform(minimum=25, maximum=100, name='chirp_mass')

            # For the default prior we add constraint component mass priors
            self['mass_1'] = Constraint(minimum=5, maximum=100, name='mass_1')
            self['mass_2'] = Constraint(minimum=5, maximum=100, name='mass_2')

    def _check_spin_parameters(self, key_set):
        """Check presence of spin parameters.

        These are not considered essential at the moment.
        """
        spin_params = {'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl'}
        if not spin_params < key_set:
            spin_keys_present = spin_params & key_set
            warnings.warn('Prior does not contain a complete prescription of generic spins!'
                          f'Only found spin parameters {spin_keys_present}')

    def _check_required_parameters(self, key_set):
        """
        Check for several required parameters and use a default prior if missing.
        """
        default_prior_dict = {
            'theta_jn': Sine(minimum=0, maximum=3.141592653589793, name='theta_jn'),
            'psi': Uniform(minimum=0, maximum=3.141592653589793, name='psi', boundary='periodic'),
            'phase': Uniform(minimum=0, maximum=6.283185307179586, name='phase', boundary='periodic'),
            'dec': Cosine(minimum=-1.5707963267948966, maximum=1.5707963267948966, name='dec'),
            'ra': Uniform(minimum=0, maximum=6.283185307179586, name='ra', boundary='periodic')
        }
        for k in ['theta_jn', 'phase', 'dec', 'ra', 'psi']:
            if not (k in key_set):
                warnings.warn(f'Missing prior for {k}. Adding default prior.')
                self[k] = default_prior_dict[k]

        if not ('luminosity_distance' in key_set):
            warnings.warn('Missing prior for luminosity_distance. Please check distance limits.')
            self['luminosity_distance'] = UniformSourceFrame(minimum=100.0, maximum=5000.0,
                cosmology=Planck15, name='luminosity_distance')


    def _check_prior_completeness(self):
        """Check whether the prior specification includes required parameters
        and add default priors for required parameters.
        """
        key_set = set(self.keys())
        self._check_mass_parameters(key_set)
        self._check_spin_parameters(key_set)
        self._check_required_parameters(key_set)
        # TODO: add support for (zenith, azimuth) for sky position instead of (ra, dec)

    @property
    def intrinsic_parameters(self) -> Set:
        """
        The set of intrinsic parameters.

        This is the intersection of the full set of intrinsic parameters
        with the set of user specified parameters. At the very least it includes
        two mass parameters, phase and time.
        """
        return self.keys() & self._intrinsic_parameters

    @property
    def extrinsic_parameters(self) -> Set:
        """
        The set of extrinsic parameters.
        """
        return self.keys() & self._extrinsic_parameters

    def sample_intrinsic(self, size: int = None) -> Dict[str, np.ndarray]:
        """
        Sample from the intrinsic prior distribution.

        Parameters
        ----------
        size : int
            The number of samples to draw.
        """
        samples = self.sample_subset(keys=list(self.intrinsic_parameters), size=size)
        samples = self.default_conversion_function(samples)
        # Note:
        #   * total mass and symmetric mass-ratio are added automatically
        #   * mass-ratio and chirp-mass are also added if not being sampled in
        # Keep only samples in our intrinsic parameter list:
        return {k: samples[k] for k in self.intrinsic_parameters}

    def sample_extrinsic(self, size=None):
        """
        Sample from the extrinsic prior distribution.

        Parameters
        ----------
        size : int
            The number of samples to draw.
        """
        return self.sample_subset(keys=list(self.extrinsic_parameters), size=size)

