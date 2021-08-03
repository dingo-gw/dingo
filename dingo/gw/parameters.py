# Import all of the prior classes needed to define the default BBH prior by hand.
# We're not using all of them. We could make the default prior more explicit
# by defining it ourselves in the constructor.
from bilby.gw.prior import BBHPriorDict, UniformSourceFrame
from bilby.core.prior import Uniform, Constraint, Sine, Cosine
from astropy.cosmology import Planck15

import numpy as np
from typing import Dict, Set, Any
import warnings


# Silence INFO and WARNING messages from bilby
import logging
logging.getLogger('bilby').setLevel("ERROR")


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


    This class also stores reference values for two parameters which are
    used for waveform generation and later updated when sampling from
    extrinsic parameters: luminosity_distance, geocent_time.
    """

    def __init__(self,
                 parameter_prior_dict: Dict = None,
                 geocent_time_ref: float = 1126259642.413,  # s
                 luminosity_distance_ref: float = 500.0,  # Mpc
                 reference_frequency: float = 20.0):  # Hz
        """
        Parameters
        ----------
        parameter_prior_dict : Dict
            A dictionary of parameter names and 1-dimensional prior distribution
            objects. If None, we use a default binary black hole prior.
        geocent_time_ref : float
            The geocentric GPS time reference value in seconds.
            This is also used to determine the time prior.
        luminosity_distance_ref : float
            The luminosity distance reference value in Mpc.
        reference_frequency : float
            The frequency in Hz at which the binary's spins are defined.
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

        # Add time prior if missing
        if not ('geocent_time' in self):
            self['geocent_time'] = Uniform(
                minimum=geocent_time_ref - 0.1, maximum=geocent_time_ref + 0.1,
                name='geocent_time', latex_label='$t_c$', unit='$s$')

        # Store reference values
        self._geocent_time_ref = geocent_time_ref
        self._luminosity_distance_ref = luminosity_distance_ref
        self._reference_frequency = reference_frequency

        # TODO: Please check whether this is what we want
        self._intrinsic_parameters = ['mass_1', 'mass_2', 'mass_ratio',
                                      'chirp_mass', 'phase', 'theta_jn',
                                      'a_1', 'a_2', 'tilt_1', 'tilt_2',
                                      'phi_12', 'phi_jl']
        self._extrinsic_parameters = ['luminosity_distance', 'dec', 'ra', 'psi',
                                      'geocent_time']


    @property
    def reference_geocentric_time(self) -> float:
        """The value of the geocentric reference (GPS) time in seconds."""
        return self._geocent_time_ref

    @property
    def reference_luminosity_distance(self) -> float:
        """The value of the reference luminosity distance."""
        return self._luminosity_distance_ref

    @property
    def reference_frequency(self) -> float:
        """The frequency in Hz at which the binary's spins are defined."""
        return self._reference_frequency


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
        # Use bilby.gw.utils.zenith_azimuth_to_ra_dec()
        # TestSkyFrameConversion()
        # Note that this requires the list of detectors!

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

    def sample_intrinsic(self, size: int = None,
                         add_reference_values=True) -> Dict[str, np.ndarray]:
        """
        Sample from the intrinsic prior distribution.

        Parameters
        ----------
        size : int
            The number of samples to draw.
        add_reference_values : bool
            If True, add reference frequency, distance and time to the output dict.
            These are fixed values needed, not r.v.'s, but are added for each sample.
            Reference frequency and distance are needed for waveform generation, and
            reference time is used when projecting onto the detectors.

        Note that:
          * total mass and symmetric mass-ratio are added automatically
          * mass-ratio and chirp-mass are added if not being sampled in
        """
        samples = self.sample_subset(keys=list(self.intrinsic_parameters), size=size)
        samples = self.default_conversion_function(samples)
        # Keep only samples in our intrinsic parameter list:
        sample_dict = {k: samples[k] for k in self.intrinsic_parameters}

        if add_reference_values:
            ref_array = np.ones_like(sample_dict[self._intrinsic_parameters[0]])
            sample_dict.update({'f_ref': self.reference_frequency * ref_array,
                                'luminosity_distance': self.reference_luminosity_distance * ref_array,
                                'geocent_time': self.reference_geocentric_time * ref_array
                                })
        return sample_dict


    def sample_extrinsic(self, size: int = None) -> Dict[str, np.ndarray]:
        """
        Sample from the extrinsic prior distribution.

        Parameters
        ----------
        size : int
            The number of samples to draw.
        """
        return self.sample_subset(keys=list(self.extrinsic_parameters), size=size)


def generate_parameter_prior_dictionary(parameter_prior_dict_kwargs: Dict[str, Dict[str, Any]]) -> Dict[str, object]:
    """
    Generate dictionary of prior class instances from nested
    dictionary of prior parameters.

    Parameters
    ----------
    parameter_prior_dict_kwargs: Dict[str, Dict[str, Any]]
        Nested dictionary. Each key corresponds to a physical
        parameter, for which a dictionary of prior parameters
        needs to be specified. These inner parameters must
        include 'class_name': the name of the bilby prior class,
        and key value pairs for all parameters that are needed
        by this prior class.
    """
    pp_dict = parameter_prior_dict_kwargs.copy()
    if not all(['class_name' in item.keys() for item in pp_dict.values()]):
        raise ValueError('Parameter dictionaries must contain a key "class_name"'
                         'specifying the name of the prior class for each parameter.')
    # Assume that all required prior classes have been imported into this module
    return {k: globals()[v.pop('class_name')](**v) for k, v in pp_dict.items()}


def generate_default_prior_dictionary() -> Dict[str, object]:
    """
    Generate default binary black hole 15 dimensional prior dictionary
    in terms of bilby prior classes.
    """
    parameter_prior_dict = {
        'mass_1': {'class_name': 'Constraint', 'minimum': 5, 'maximum': 100},
        'mass_2': {'class_name': 'Constraint', 'minimum': 5, 'maximum': 100},
        'mass_ratio': {'class_name': 'Uniform', 'minimum': 0.125, 'maximum': 1},
        'chirp_mass': {'class_name': 'Uniform', 'minimum': 25, 'maximum': 100},
        'luminosity_distance': {'class_name': 'UniformSourceFrame', 'minimum': 100.0, 'maximum': 5000.0, 'cosmology': Planck15},
        'dec': {'class_name': 'Cosine', 'minimum': -1.5707963267948966, 'maximum': 1.5707963267948966},
        'ra': {'class_name': 'Uniform', 'minimum': 0, 'maximum': 6.283185307179586, 'boundary': 'periodic'},
        'theta_jn': {'class_name': 'Sine', 'minimum': 0, 'maximum': 3.141592653589793},
        'psi': {'class_name': 'Uniform', 'minimum': 0, 'maximum': 3.141592653589793, 'boundary': 'periodic'},
        'phase': {'class_name': 'Uniform', 'minimum': 0, 'maximum': 6.283185307179586, 'boundary': 'periodic'},
        'a_1': {'class_name': 'Uniform', 'minimum': 0, 'maximum': 0.99},
        'a_2': {'class_name': 'Uniform', 'minimum': 0, 'maximum': 0.99},
        'tilt_1': {'class_name': 'Sine', 'minimum': 0, 'maximum': 3.141592653589793},
        'tilt_2': {'class_name': 'Sine', 'minimum': 0, 'maximum': 3.141592653589793},
        'phi_12': {'class_name': 'Uniform', 'minimum': 0, 'maximum': 6.283185307179586, 'boundary': 'periodic'},
        'phi_jl': {'class_name': 'Uniform', 'minimum': 0, 'maximum': 6.283185307179586, 'boundary': 'periodic'},
        'geocent_time': {'class_name': 'Uniform', 'minimum': 1126259642.3130002, 'maximum': 1126259642.513}
    }
    for k, v in parameter_prior_dict.items():
        v['name'] = k
    return generate_parameter_prior_dictionary(parameter_prior_dict)
