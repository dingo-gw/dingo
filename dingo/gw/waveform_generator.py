from domains import Domain
from parameters import PriorDict
from typing import Dict, List, Tuple
from enum import Enum, auto

class SpinType(Enum):
    NON_SPINNING = auto()
    ALIGNED_SPIN = auto()
    PRECESSING = auto()

# TODO:
#  * GPNPE needs to be put in: need to carry around detector coalescence times
#  * Need to transform into input data for NN. This will require an extra class, not in this module
#  * whitening will be done later as a transform: ASD
#    - Rather treat the noise as an additional transform: add noise and whiten
#    similar for noise transients / realistic noise


class WaveformGenerator:
    """Generate GW polarizations, either in the time or frequency domain.

    TODO:
        - should the output only be the whitened waveform? No, whitening will be done somewhere else
        - should the output be TD or FD or either?
            That is prescribed by the Domain that is passed in
            Should the domains know about how to convert or should this be done here? could just be functions
        - Think about storage for wfs: pd.Dataframe or dict(array)
        - Technical note for later:
            - if there are multiple ways of generating the approximant wf, choose the domain
            which is the output domain, i.e. if FD: SEOBNRv4PHM and domain = "uFD" choose LS.SimInspiralFD
            - There is a complication: for EOB spins are always defined at the starting frequency, and
             there is no way to set a reference frequency != f_start
             In principle, could update to "correct" spins from EOB dynamics.
        - call LAL functions, not pycbc
    """

    def __init__(self, approximant: str, domain: Domain):

        self.approximant = approximant
        self.domain = domain
        # What type is "domain"? Just a string or a subclass of Domain?
        # The domain knows about frequencies and times and thus would be useful to refer to

    def convert_parameters(self, p: Dict, spins: SpinType):
        """Convert from source frame to Cartesian parameters"""
        # TODO: distinguish cases:
        #  - aligned vs precessing spins vs non-spinning
        #  - inclination given or not: easier to demand that inclination is always given
        #
        # TODO: look at how this is implemented in bilby

    def generate_FD_waveform(self):
        pass

    def generate_TD_waveform(self):
        pass

    def generate_hplus_hcross(self, parameters: Dict) -> Tuple: # (hp, hc) or Dict?
        # FIXME: which parameters are we expecting in "parameters"
        #  - intrinsic & extrinsic?
        #  - which parameters have to be present and which are optional?
        #  - for which parameters do we allow default values and which values should these be?

        parameters_lal = self.convert_parameters(parameters)
        # TODO: what format should parameters_lal have: Dict?

        # see WaveformDataset._generate_whitened_waveform()
        # This is way too long and needs to be split
        # One function to generate just the polarizations for FD, TD (separate functions)
        # One function to do whitening or should this happen outside of this class?
        # -- Needs PSD info which we may not want to couple with this class


# TODO: look at pytorch transforms
# What are the requirements: __call__ or inheritance?
class StandardizeParameters:
    # standardize_parameters
    # Need to save this info -- separate transform?


class RandomProjectToDetectors(object):
    """Given a sample waveform (in terms of its polarizations, and intrinsic parameters),
    draw a sample from the extrinsic parameter prior distribution and project on the
    given detector network. Return the strain (FD?)


    (This is an example of a pytorch-like transform.)
    """

    def __init__(self, domain: Domain, extrinsic_prior: PriorDict):

        self.domain = domain
        self.extrinsic_prior = extrinsic_prior

    def __call__(self, sample: Dict):

        extrinsic_parameters = self.extrinsic_prior.sample()
        return self.project_to_detectors(sample['hplus'], sample['hcross'],
                                         sample['parameters'], extrinsic_parameters)

    def project_to_detectors(self, hplus, hcross, old_parameters, new_extrinsic_parameters):
        # FIXME: need info about detector network: list or dict of Detector objects

        pass
        # see WaveformDataset.get_detector_waveforms()
        # Given ra, dec, psi, self.ref_time and a list of detector objects
        # loop over detectors, and compute h+ * F+ + hx + Fx and timeshift at detector
        # code is long and needs to be split up - do we need all cases?

        # Detector objects:
        # see WaveformDataset.init_detectors()
        # So far using pycbc.detector.Detector -- get rid of this dependency? Look at structure of this class and what is used in existing code

    # Which other methods should this class have?