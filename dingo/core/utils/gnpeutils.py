from scipy.stats import kstest, gaussian_kde
import numpy as np

class ConvergenceTracker:
    def __init__(self, x=None):
        self.x = x
        self.ks_result = None

    def update(self, y):
        if self.x is not None:
            statistic, pvalue = [], []
            for k in y.keys():
                ks_result = kstest(self.x[k], y[k])
                statistic.append(ks_result[0])
                pvalue.append(ks_result[1])
            self.ks_result = {"statistics": statistic, "pvalue": pvalue}
        self.x = y

    @property
    def pvalue_min(self):
        if self.ks_result is None:
            return -np.inf
        else:
            return min(self.ks_result["pvalue"])

    # def remove_outliers(self, x):
    #     xc = np.concatenate([v[None, :] for v in self.x.values()], axis=0)
    #     yc = np.concatenate([v[None, :] for v in y.values()], axis=0)
    #     gaussian_kde(yc).logpdf(yc)