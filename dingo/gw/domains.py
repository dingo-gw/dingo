import functools
import numpy as np


class Domain(object):

    @property
    def noise_std(self):
        pass


class FrequencyDomain(Domain):

    def __init__(self, f_min, f_max, delta_f, window_factor):

        self.f_min = f_min
        self.f_max = f_max
        self.delta_f = delta_f
        self.window_factor = window_factor

    @property
    def Nf(self):
        return int(self.f_max / self.delta_f) + 1

    @property
    @functools.lru_cache()
    def sample_frequencies(self):
        return np.linspace(0.0, self.f_max, num=self.Nf, endpoint=True, dtype=np.float32)

    @property
    @functools.lru_cache()
    def frequency_mask(self):
        return (self.sample_frequencies >= self.f_min)

    @property
    def noise_std(self):
        return np.sqrt(self.window_factor) / np.sqrt(4.0 * self.delta_f)


class TimeDomain(Domain):

    pass


class PCADomain(Domain):

    pass


class NonuniformFrequencyDomain(Domain):

    pass
