# Import all of the prior classes needed to define the default BBH prior by hand.
# We're not using all of them. We could make the default prior more explicit
# by defining it ourselves in the constructor.
from bilby.gw.prior import BBHPriorDict, UniformSourceFrame
from bilby.core.prior import Uniform, Constraint, Sine

from typing import Dict

# Silence INFO and WARNING messages from bilby
import logging
logging.getLogger('bilby').setLevel("ERROR")


# This is subclassing the Bilby BBHPriorDict.
#
# TODO:
#  * [x] It would be useful to incorporate the ability to separately sample intrinsic
#  and extrinsic parameters. From the list of parameters, it should therefore be able
#  to determine which ones fall into each type.
#  * For some extrinsic parameters (e.g., distance, time of coalescence) we need to also
#  set reference values to be able to generate waveforms and position the detectors.
#  I am wondering whether this class should also take care of that.
#    - We could set delta function priors for some parameters, but this would mean
#      that we can't sample from their usual distributions. Perhaps it is better to
#      do this elsewhere?

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

        if (parameter_prior_dict is not None) and \
                not isinstance(parameter_prior_dict, dict):
            raise ValueError('Expected dictionary of parameters and priors, '
                             'but got:', parameter_prior_dict)

        # Build this prior dict
        super().__init__(dictionary=parameter_prior_dict)

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

    @property
    def intrinsic_parameters(self):
        return self._intrinsic_parameters

    @property
    def extrinsic_parameters(self):
        return self._extrinsic_parameters

    def sample_intrinsic(self, size=None):
        samples = self.sample_subset(keys=self._intrinsic_parameters, size=size)
        samples = self.default_conversion_function(samples)
        # Drop total mass and symmetric mass-ratio which are added automatically
        # We could also keep them if they are useful
        for k in ['total_mass', 'symmetric_mass_ratio']:
            samples.pop(k)
        return samples

    def sample_extrinsic(self, size=None):
        return self.sample_subset(keys=self._extrinsic_parameters, size=size)


    # Anything below is probably not needed

    # import copy
    # def copy(self):
    #     return copy.deepcopy(self)
    #
    # def define_intrinsic_extrinsic_priors(self):
    #     # FIXME: If needed we could define separate intrinsic and extrinsic prior dicts
    #     #  This could be done like so, but I'm not sure it is needed.
    #     priors_intrinsic = self.copy()
    #     priors_extrinsic = self.copy()
    #     for k in self._extrinsic_parameters:
    #         priors_intrinsic.pop(k)
    #     for k in self._intrinsic_parameters:
    #         priors_extrinsic.pop(k)
    #     self.priors_intrinsic = priors_intrinsic
    #     self.priors_extrinsic = priors_extrinsic
    #
    #     # With these new prior objects defined, we can sample from them like so:
    #     print(self.priors_intrinsic.sample())
    #     #self.priors_extrinsic.sample()  # Error because masses etc are missing
    #     print(self.priors_extrinsic.sample_subset(keys=self._extrinsic_parameters))
    #
    #
    # # TODO: Do we need to return partial prior objects? What would they be used for?
    # #  We can sample from the intrinsic and extrinsic parameters with the methods below.
    # def intrinsic_priors(self):
    #     pass
    #
    # def extrinsic_priors(self):
    #     pass
