from bilby.gw.prior import BBHPriorDict

# This is subclassing the Bilby BBHPriorDict. It would be useful to incorporate
# the ability to separately sample intrinsic and extrinsic parameters. From the list
# of parameters, it should therefore be able to determine which ones fall into each
# type.

# For some extrinsic parameters (e.g., distance, time of coalescence) we need to also
# set reference values to be able to generate waveforms and position the detectors.
# I am wondering whether this class should also take care of that.


class ParameterDict(BBHPriorDict):

    @property
    def intrinsic_parameters(self):
        pass

    @property
    def extrinsic_parameters(self):
        pass

    def sample_extrinsic(self, num):
        pass

    def sample_intrinsic(self, num):
        pass