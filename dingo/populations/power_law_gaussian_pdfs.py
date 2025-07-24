# code copied/adapted from https://github.com/simone-mastrogiovanni/icarogw#

import numpy as np
import scipy
import copy

def check_bounds_1D(x,minval,maxval):
    return (x<minval) | (x>maxval)

def check_bounds_2D(x1,x2,y):
    return (x1<x2) | np.isnan(y)

def get_gaussian_norm(ming,maxg,meang,sigmag):
    '''
    Returns the normalization of the gaussian distribution
    
    Parameters
    ----------
    ming, maxg, meang,sigmag: Minimum, maximum, mean and standard deviation of the gaussian distribution
    '''
    max_point = (maxg-meang)/(sigmag*np.sqrt(2.))
    min_point = (ming-meang)/(sigmag*np.sqrt(2.))
    return 0.5*scipy.special.erf(max_point)-0.5*scipy.special.erf(min_point)

def PL_normfact(minpl,maxpl,alpha):
    '''
    Returns the Powerlaw normalization factor
    
    Parameters
    ----------
    minpl, maxpl,alpha: Minimum, maximum and power law exponent of the distribution
    '''
    if alpha == -1:
        norm_fact=np.log(maxpl/minpl)
    else:
        norm_fact=(np.power(maxpl,alpha+1.)-np.power(minpl,alpha+1))/(alpha+1)
    return norm_fact

class basic_1dimpdf(object):
    
    def __init__(self,minval,maxval):
        '''
        Basic class for a 1-dimensional pdf
        
        Parameters
        ----------
        minval,maxval: float
            minimum and maximum values within which the pdf is defined
        '''
        self.minval=minval
        self.maxval=maxval 
        
    def _check_bound_pdf(self,x,y):
        '''
        Check if x is between the pdf boundaries and set y to -np.inf where x is outside
        
        Parameters
        ----------
        x,y: np.array
            Array where the log pdf is evaluated and values of the log pdf
        
        Returns
        -------
        log pdf values: np.array
            log pdf values updates to -np.inf outside the boundaries
            
        '''
        
        indx=check_bounds_1D(x,self.minval,self.maxval)
        y[indx]=-np.inf
        return y
    
    def _check_bound_cdf(self,x,y):
        '''
        Check if x is between the pdf boundaries nd set the cdf y to 0 and 1 outside the boundaries
        
        Parameters
        ----------
        x,y: np.array
            Array where the log cdf is evaluated and values of the log cdf
        
        Returns
        -------
        log cdf values: np.array
            log cdf values updates to 0 and 1 outside the boundaries
            
        '''
        
        y[x<self.minval],y[x>self.maxval]=-np.inf,0.
        return y
    
    def log_pdf(self,x):
        '''
        Evaluates the log_pdf
        
        Parameters
        ----------
        x: np.array
            where to evaluate the log_pdf
        
        Returns
        -------
        log_pdf: np.array
        '''
        y=self._log_pdf(x)
        y=self._check_bound_pdf(x,y)
        return y
    
    def log_cdf(self,x):
        '''
        Evaluates the log_cdf
        
        Parameters
        ----------
        x: np.array
            where to evaluate the log_cdf
        
        Returns
        -------
        log_cdf: np.array
        '''
        y=self._log_cdf(x)
        y=self._check_bound_cdf(x,y)
        return y
    
    def pdf(self,x):
        '''
        Evaluates the pdf
        
        Parameters
        ----------
        x: np.array
            where to evaluate the pdf
        
        Returns
        -------
        pdf: np.array
        '''
        
        return np.exp(self.log_pdf(x))
    
    def cdf(self,x):
        '''
        Evaluates the cdf
        
        Parameters
        ----------
        x: np.array
            where to evaluate the cdf
        
        Returns
        -------
        cdf: np.array
        '''
        
        return np.exp(self.log_cdf(x))
    
    def sample(self,N):
        '''
        Samples from the pdf
        
        Parameters
        ----------
        N: int
            Number of samples to generate
        
        Returns
        -------
        Samples: np.array
        '''
        sarray=np.linspace(self.minval,self.maxval,10000)
        cdfeval=self.cdf(sarray)
        randomcdf=np.random.rand(N)
        return np.interp(randomcdf,cdfeval,sarray)

class PowerLaw(basic_1dimpdf):
    
    def __init__(self,minpl,maxpl,alpha):
        '''
        Class for a  powerlaw probability
        
        Parameters
        ----------
        minpl,maxpl,alpha: float
            Minimum, Maximum and exponent of the powerlaw 
        '''
        super().__init__(minpl,maxpl)
        self.minpl,self.maxpl,self.alpha=minpl, maxpl, alpha
        self.norm_fact=PL_normfact(minpl,maxpl,alpha)
        
    def _log_pdf(self,x):
        '''
        Evaluates the log_pdf
        
        Parameters
        ----------
        x: np.array
            where to evaluate the log_pdf
        
        Returns
        -------
        log_pdf: np.array
        '''
        
        toret=self.alpha*np.log(x)-np.log(self.norm_fact)
        return toret
    
    def _log_cdf(self,x):
        '''
        Evaluates the log_cdf
        
        Parameters
        ----------
        x: np.array
            where to evaluate the log_cdf
        
        Returns
        -------
        log_cdf: np.array
        '''
        
        if self.alpha == -1.:
            toret = np.log(np.log(x/self.minval)/self.norm_fact)
        else:
            toret =np.log(((np.power(x,self.alpha+1)-np.power(self.minpl,self.alpha+1))/(self.alpha+1))/self.norm_fact)
        return toret
    
class TruncatedGaussian(basic_1dimpdf):
    
    def __init__(self,meang,sigmag,ming,maxg):
        '''
        Class for a Truncated gaussian probability
        
        Parameters
        ----------
        meang,sigmag,ming,maxg: float
            mean, sigma, min value and max value for the gaussian
        '''
        super().__init__(ming,maxg)
        self.meang,self.sigmag,self.ming,self.maxg=meang,sigmag,ming,maxg
        self.norm_fact= get_gaussian_norm(ming,maxg,meang,sigmag)
        
    def _log_pdf(self,x):
        '''
        Evaluates the log_pdf
        
        Parameters
        ----------
        x: np.array
            where to evaluate the log_pdf
        
        Returns
        -------
        log_pdf: np.array
        '''
        
        toret=-np.log(self.sigmag)-0.5*np.log(2*np.pi)-0.5*np.power((x-self.meang)/self.sigmag,2.)-np.log(self.norm_fact)
        return toret
    
    def _log_cdf(self,x):
        '''
        Evaluates the log_cdf
        
        Parameters
        ----------
        x: np.array
            where to evaluate the log_cdf
        
        Returns
        -------
        log_cdf: np.array
        '''
        
        max_point = (x-self.meang)/(self.sigmag*np.sqrt(2.))
        min_point = (self.ming-self.meang)/(self.sigmag*np.sqrt(2.))
        toret = np.log((0.5*scipy.special.erf(max_point)-0.5*scipy.special.erf(min_point))/self.norm_fact)
        return toret

class PowerLawGaussian(basic_1dimpdf):
    
    def __init__(self,minpl,maxpl,alpha,lambdag,meang,sigmag,ming,maxg):
        '''
        Class for a Power Law + Gaussian probability
        
        Parameters
        ----------
        minpl,maxpl,alpha,lambdag,meang,sigmag,ming,maxg: float
            In sequence, minimum, maximum, exponential of the powerlaw part. Fraction, mean, sigma, min value, max value of the gaussian
        '''
        super().__init__(min(minpl,ming),max(maxpl,maxg))
        self.minpl,self.maxpl,self.alpha,self.lambdag,self.meang,self.sigmag,self.ming,self.maxg=minpl,maxpl,alpha,lambdag,meang,sigmag,ming,maxg
        self.PL=PowerLaw(minpl,maxpl,alpha)
        self.TG=TruncatedGaussian(meang,sigmag,ming,maxg)
        
    def _log_pdf(self,x):
        '''
        Evaluates the log_pdf
        
        Parameters
        ----------
        x: np.array
            where to evaluate the log_pdf
        
        Returns
        -------
        log_pdf: np.array
        '''
        
        toret=np.logaddexp(np.log1p(-self.lambdag)+self.PL.log_pdf(x),np.log(self.lambdag)+self.TG.log_pdf(x))
        return toret
    
    def _log_cdf(self,x):
        '''
        Evaluates the log_cdf
        
        Parameters
        ----------
        x: np.array
            where to evaluate the log_cdf
        
        Returns
        -------
        log_cdf: np.array
        '''
        
        toret=np.log((1-self.lambdag)*self.PL.cdf(x)+self.lambdag*self.TG.cdf(x))
        return toret
    
def _highpass_filter(mass, mmin,delta_m):
    '''
    This function returns the value of the window function defined as Eqs B6 and B7 of https://arxiv.org/pdf/2010.14533.pdf

    Parameters
    ----------
    mass: np.array or float
        array of x or masses values
    mmin: float or np.array (in this case len(mmin) == len(mass))
        minimum value of window function
    delta_m: float or np.array (in this case len(delta_m) == len(mass))
        width of the window function

    Returns
    -------
    Values of the window function
    '''

    to_ret = np.ones_like(mass)
    if delta_m == 0:
        return to_ret

    mprime = mass-mmin

    # Defines the different regions of thw window function ad in Eq. B6 of  https://arxiv.org/pdf/2010.14533.pdf
    select_window = (mass>mmin) & (mass<(delta_m+mmin))
    select_one = mass>=(delta_m+mmin)
    select_zero = mass<=mmin

    effe_prime = np.ones_like(mass)

    # Defines the f function as in Eq. B7 of https://arxiv.org/pdf/2010.14533.pdf
    # This line might raise a warnig for exp orverflow, however this is not important as it enters at denominator
    effe_prime[select_window] = np.exp(np.nan_to_num((delta_m/mprime[select_window])+(delta_m/(mprime[select_window]-delta_m))))
    to_ret = 1./(effe_prime+1)
    to_ret[select_zero]=0.
    to_ret[select_one]=1.
    return to_ret
    
class conditional_2dimpdf(object):
    
    def __init__(self,pdf1,pdf2):
        '''
        Basic class for a 2-dimensional pdf, where x2<x1
        
        Parameters
        ----------
        pdf1, pdf2: basic_1dimpdf, basic_2dimpdf
            Two classes of pdf functions
        '''
        self.pdf1=pdf1
        self.pdf2=pdf2
        
    def _check_bound_pdf(self,x1,x2,y):
        '''
        Check if x1 and x2 are between the pdf boundaries and set y to -np.inf where x1<x2 is outside
        
        Parameters
        ----------
        x1,x2,y: np.array
            Array where the log pdf is evaluated and values of the log pdf
        
        Returns
        -------
        log pdf values: np.array
            log pdf values updated to -np.inf outside the boundaries
            
        '''
        
        indx=check_bounds_2D(x1,x2,y)
        y[indx]=-np.inf
        return y
    
    def log_pdf(self,x1,x2):
        '''
        Evaluates the log_pdf
        
        Parameters
        ----------
        x1,x2: np.array
            where to evaluate the log_pdf
        
        Returns
        -------
        log_pdf: np.array
        '''
        # This line might create some nan since p(m2|m1) = p(m2)/CDF_m2(m1) = 0/0 if m2 and m1 < mmin.
        # This nan is eliminated with the _check_bound_pdf
        y=self.pdf1.log_pdf(x1)+self.pdf2.log_pdf(x2)-self.pdf2.log_cdf(x1)
        y=self._check_bound_pdf(x1,x2,y)
        return y 
    
    def pdf(self,x1,x2):
        '''
        Evaluates the pdf
        
        Parameters
        ----------
        x1,x2: np.array
            where to evaluate the pdf
        
        Returns
        -------
        pdf: np.array
        '''
        
        return np.exp(self.log_pdf(x1,x2))
    
    def sample(self,N):
        '''
        Samples from the pdf
        
        Parameters
        ----------
        N: int
            Number of samples to generate
        
        Returns
        -------
        Samples: np.array
        '''
        sarray1=np.linspace(self.pdf1.minval,self.pdf1.maxval,10000)
        cdfeval1=self.pdf1.cdf(sarray1)
        randomcdf1=np.random.rand(N)

        sarray2=np.linspace(self.pdf2.minval,self.pdf2.maxval,10000)
        cdfeval2=self.pdf2.cdf(sarray2)
        randomcdf2=np.random.rand(N)
        x1samp=np.interp(randomcdf1,cdfeval1,sarray1)
        x2samp=np.interp(randomcdf2*self.pdf2.cdf(x1samp),cdfeval2,sarray2)
        return x1samp,x2samp
    
class LowpassSmoothedProb(basic_1dimpdf):
    def __init__(self,originprob,bottomsmooth):
        '''
        Class for a smoother probability
        
        Parameters
        ----------
        originprob: class
            Original probability class
        bottomsmooth: float
            float corresponding to the smooth of the prior
        '''
        self.origin_prob = copy.deepcopy(originprob)
        self.bottom_smooth = bottomsmooth
        self.bottom = originprob.minval
        super().__init__(originprob.minval,originprob.maxval)
        
        # Find the values of the integrals in the region of the window function before and after the smoothing
        int_array = np.linspace(originprob.minval,originprob.minval+bottomsmooth,1000)
        integral_before = np.trapz(self.origin_prob.pdf(int_array),int_array)
        integral_now = np.trapz(self.origin_prob.pdf(int_array)*_highpass_filter(int_array, self.bottom,self.bottom_smooth),int_array)

        self.integral_before = integral_before
        self.integral_now = integral_now
        # Renormalize the smoother function.
        self.norm = 1 - integral_before + integral_now

        self.x_eval_cpu = np.linspace(self.bottom,self.bottom+self.bottom_smooth,1000)
        self.cdf_numeric_cpu = np.cumsum(self.pdf((self.x_eval_cpu[:-1:]+self.x_eval_cpu[1::])*0.5))*(self.x_eval_cpu[1::]-self.x_eval_cpu[:-1:])
        
        self.x_eval_gpu = self.x_eval_cpu
        self.cdf_numeric_gpu = self.cdf_numeric_cpu
        
    def _log_pdf(self,x):
        '''
        Evaluates the log_pdf
        
        Parameters
        ----------
        x: np.array
            where to evaluate the log_pdf
        
        Returns
        -------
        log_pdf: np.array
        '''

        # Return the window function
        window = _highpass_filter(x, self.bottom,self.bottom_smooth)
        # The line below might raise warnings for log(0), however python is able to handle it.
        prob_ret =self.origin_prob.log_pdf(x)+np.log(window)-np.log(self.norm)
        return prob_ret

    def _log_cdf(self,x):
        '''
        Evaluates the log_cdf
        
        Parameters
        ----------
        x: np.array
            where to evaluate the log_cdf
        
        Returns
        -------
        log_cdf: np.array
        '''
        
        cdf_numeric = self.cdf_numeric_cpu
        x_eval = self.x_eval_cpu
        
        origin=x.shape
        ravelled=np.ravel(x)
        
        toret = np.ones_like(ravelled)
        toret[ravelled<self.bottom] = 0.        
        toret[(ravelled>=self.bottom) & (ravelled<=(self.bottom+self.bottom_smooth))] = np.interp(ravelled[(ravelled>=self.bottom) & (ravelled<=(self.bottom+self.bottom_smooth))]
                           ,(x_eval[:-1:]+x_eval[1::])*0.5,cdf_numeric)
        # The line below might contain some log 0, which is automatically accounted for in python
        toret[ravelled>=(self.bottom+self.bottom_smooth)]=(self.integral_now+self.origin_prob.cdf(
        ravelled[ravelled>=(self.bottom+self.bottom_smooth)])-self.origin_prob.cdf(np.array([self.bottom+self.bottom_smooth])))/self.norm
        
        return np.log(toret).reshape(origin)
    
class pm_prob(object):
    def pdf(self,mass_1_source):
        return self.prior.pdf(mass_1_source)
    def log_pdf(self,mass_1_source):
        return self.prior.log_pdf(mass_1_source)
    
class pm1m2_prob(object):
    def pdf(self,mass_1_source,mass_2_source):
        return self.prior.pdf(mass_1_source,mass_2_source)
    def log_pdf(self,mass_1_source,mass_2_source):
        return self.prior.log_pdf(mass_1_source,mass_2_source)

class massprior_PowerLawPeak(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha','mmin','mmax','mu_g','sigma_g','lambda_peak']
    def update(self,**kwargs):
        self.prior=PowerLawGaussian(
            kwargs['mmin'],kwargs['mmax'],-kwargs['alpha'],kwargs['lambda_peak'],kwargs['mu_g'],
            kwargs['sigma_g'],kwargs['mmin'],kwargs['mu_g']+5*kwargs['sigma_g']
        )

class m1m2_conditioned_lowpass(pm1m2_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters+['beta','delta_m']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
        p1 = LowpassSmoothedProb(self.wrapper_m.prior,kwargs['delta_m'])
        p2 = LowpassSmoothedProb(PowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta']),kwargs['delta_m'])
        self.prior=conditional_2dimpdf(p1,p2)

def build_massprior_PowerLawPeak(params_in):

    params = copy.deepcopy(params_in)

    # rename params
    params['mmin'] = params['minimum_mass']
    params['mmax'] = params['maximum_mass']

    list_params = ['alpha', 'beta', 'mmin', 'mmax', 'delta_m', 'mu_g', 'sigma_g', 'lambda_peak']

    p = {key: params[key] for key in list_params if key in params}

    mw = massprior_PowerLawPeak()
    mw = m1m2_conditioned_lowpass(mw)
    mw.update(**p)

    return mw