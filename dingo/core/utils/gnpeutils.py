from scipy.stats import kstest, gaussian_kde
import numpy as np


class IterationTracker:
    def __init__(self, data=None, store_data=False):
        self.data = data
        self.ks_result = None
        self.store_data = store_data

    def update(self, new_data):
        """
        Append new_data to self.data.

        Parameters
        ----------
        new_data: dict
            dict with numpy arrays to append to data

        Returns
        -------

        """
        if self.data is None:
            self.data = {k: v.copy()[None, :] for k, v in new_data.items()}

        else:
            x = {k: v[-1, :] for k, v in self.data.items()}
            y = new_data
            # get ks
            statistic, pvalue = [], []
            for k in y.keys():
                N = len(x[k]) // 2
                ks_result = kstest(x[k][:N], y[k][N:])
                statistic.append(ks_result[0])
                pvalue.append(ks_result[1])
            self.ks_result = {"statistics": statistic, "pvalue": pvalue}

            if not self.store_data:
                self.data = {k: v.copy()[None, :] for k, v in y.items()}
            else:
                self.data = {
                k: np.concatenate((v, y[k][None, :]), axis=0)
                for k, v in self.data.items()
            }

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
