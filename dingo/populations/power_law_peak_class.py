from . import pdfs

import bilby.core.prior

try:
    from cupyx.scipy import special
    import cupy as xp
except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp
    import scipy.special as special

class PowerLawPeak(bilby.core.prior.Prior):

    def __init__(self, alpha, minimum, maximum, delta, lam, mu, sigma, beta, mlow=2., mhigh=100., qlow=None, qhigh=1.,name=None, latex_label=None,unit=None, boundary=None):
        """Power law with bounds and alpha, spectral index

        Parameters
        ==========
        alpha: float
            Power law exponent parameter for m1
        minimum: float
            maximum mass of smoothed power-law
        maximum: float
            maximum mass of power-law
        delta: float
            smoothing scale
        lam: float
            weight of gaussian component
        mu: float
            mean of gaussian
        sigma: float
            width of gaussian
        xlow: float
            minimum range of the whole pdf, below xlow it is truncated
        xhigh: float
            maximum range of the whole pdf, above xhigh it is truncated
        beta: float
            Power law exponent parameter for q

        """
        super(PowerLawPeak, self).__init__(name=name, latex_label=latex_label,
                                       minimum=minimum, maximum=maximum, unit=unit,
                                       boundary=boundary)

        self.params={}

        self.params['mu']=mu
        self.params['sigma']=sigma
        self.params['lam']=lam
        self.params['alpha']=alpha
        self.params['xmin']=minimum
        self.params['xmax']=maximum
        self.params['delta']=delta
        self.params['beta']=beta

        self.mlow=mlow
        self.mhigh=mhigh

        self.qlow=qlow
        self.qhigh=qhigh

    def sample(self,size=1):
        """
        Draw sample from pdf

        Returns
        =======
        sample from power law+peak pdf
        """

        if size is None:
            size=1

        # print(self.params)

        x1s=pdfs.draw_power_law_peak(self.params,size=size,xlow=self.mlow,xhigh=self.mhigh)

        x2s=pdfs.draw_power_law_smooth_conditioned(self.params,x1s,size=size,xlow=self.qlow,xhigh=self.qhigh)

        return x1s,x2s

    def prob(self, val):
        """Return the prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Prior probability of val
        """

        if val[0].ndim==0:
            return pdfs.pdf_power_law_peak(xp.array([val[0]]),self.params,xlow=self.xlow,xhigh=self.xhigh)*pdfs.pdf_power_law_smooth_conditioned(xp.array([val[1]]),xp.array([val[0]]),self.params,xlow=self.xlow,xhigh=self.xhigh)

        else:
            return pdfs.pdf_power_law_peak(val,self.params,xlow=self.xlow,xhigh=self.xhigh)*pdfs.pdf_power_law_smooth_conditioned(val[:,1],val[:,0],self.params,xlow=self.xlow,xhigh=self.xhigh)

    def ln_prob(self, val):
        """Return the logarithmic prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float:

        """
        if val[0].ndim==0:
            return xp.log(pdfs.pdf_power_law_peak(xp.array([val[0]]),self.params,xlow=self.xlow,xhigh=self.xhigh))+xp.log(pdfs.pdf_power_law_smooth_conditioned(xp.array([val[1]]),xp.array([val[0]]),self.params,xlow=self.xlow,xhigh=self.xhigh))

        else:
            return xp.log(pdfs.pdf_power_law_peak(val,self.params,xlow=self.xlow,xhigh=self.xhigh))+xp.log(pdfs.pdf_power_law_smooth_conditioned(val[:,1],val[:,0],self.params,xlow=self.xlow,xhigh=self.xhigh))
