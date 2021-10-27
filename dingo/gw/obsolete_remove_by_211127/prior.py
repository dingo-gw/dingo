import numpy as np
import bilby.gw.prior
import bilby.core.prior

# Any additional priors we wish to support should be wrapped here

class Uniform(bilby.core.prior.Uniform):
    """
    Add analytical mean and standard deviation methods.
    """
    def mean(self):
        return (self.maximum + self.minimum) / 2.0

    def std(self):
        var = (self.maximum - self.minimum)**2 / 12.0
        return np.sqrt(var)


class Sine(bilby.core.prior.Sine):
    """
    Add analytical mean and standard deviation methods.
    """
    def mean(self):
        return np.pi/2

    def std(self):
        var = 0.25*np.pi**2 - 2
        return np.sqrt(var)


class Cosine(bilby.core.prior.Cosine):
    """
    Add analytical mean and standard deviation methods.
    """

    def mean(self):
        return 0.0

    def std(self):
        var = 0.25*np.pi**2 - 2
        return np.sqrt(var)


class PriorManualMeanStdev:
    """
    Add mean and standard deviation to a prior class.
    Initialize with NaN.
    """
    def __init__(self):
        """
        Set mean and stdev to placeholder values.
        """
        self._fake_values = True
        self._mean = np.nan
        self._std = np.nan

    def set_mean_std_from_samples(self, samples: np.ndarray):
        """
        Set mean and standard deviation given samples.
        """
        self._mean = np.mean(samples)
        self._std = np.std(samples)
        self._fake_values = False

    def mean(self):
        return self._mean

    def std(self):
        return self._std

class Constraint(bilby.core.prior.Constraint, PriorManualMeanStdev):
    """
    Add mean and standard deviation methods for constraint prior
    which cannot be sampled directly.
    """
    def __init__(self, minimum, maximum, name=None, latex_label=None, unit=None):
        super().__init__(minimum=minimum, maximum=maximum, name=name,
                         latex_label=latex_label, unit=unit)
        PriorManualMeanStdev.__init__(self)


class UniformSourceFrame(bilby.gw.prior.UniformSourceFrame, PriorManualMeanStdev):
    """
    Add mean and standard deviation methods for distance prior.
    """
    def __init__(self, minimum, maximum, cosmology=None, name=None,
                 latex_label=None, unit=None, boundary=None):
        super().__init__(minimum=minimum, maximum=maximum, cosmology=cosmology,
                         name=name, latex_label=latex_label, unit=unit,
                         boundary=boundary)
        PriorManualMeanStdev.__init__(self)
