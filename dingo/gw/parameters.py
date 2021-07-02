from bilby.gw.prior import BBHPriorDict

# This is subclassing the Bilby BBHPriorDict.
#
# TODO:
#  * It would be useful to incorporate the ability to separately sample intrinsic
#  and extrinsic parameters. From the list of parameters, it should therefore be able
#  to determine which ones fall into each type.
#  * For some extrinsic parameters (e.g., distance, time of coalescence) we need to also
#  set reference values to be able to generate waveforms and position the detectors.
#  I am wondering whether this class should also take care of that.


# Note: We keep the detectors somewhat fixed at some reference time -- we asssume
# that Detectors don't move between events
# in post-processing fix that by looking at how much the earth has rotated
# They are not totally fixed (plug in reference time + dt)


class PriorDict(BBHPriorDict):
    """Collect the prior distributions of parameters the data generating
    model depends on, distinguish intrinsic and extrinsic parameters (see below),
    and provide methods to sample from subsets of parameters.

    Intrinsic parameters are parameters that the data model depends on in a
    complicated way and, therefore, there parameters need to be sampled when
    generating the dataset. In contrast, extrinsic parameters are parameters
    on which the model depends in a simple way. Therefore, intrinsic samples
    from the data model can be augmented during training to include samples
    of extrinsic parameters by using simple transformations.
    """
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