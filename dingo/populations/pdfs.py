import numpy as np
from scipy.interpolate import RectBivariateSpline as spline2d
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.stats as stats
from scipy.interpolate import RegularGridInterpolator as splinend
import scipy




try:
    from cupyx.scipy import special
    import cupy as xp
except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp
    import scipy.special as special

# from hyp2f1_cupy import hyp2f1


def trapz(y, x=None, dx=1.0, axis=-1):
    """
    Lifted from `numpy <https://github.com/numpy/numpy/blob/v1.15.1/numpy/lib/function_base.py#L3804-L3891>`_.

    Integrate along the given axis using the composite trapezoidal rule.
    Integrate `y` (`x`) along given axis.

    Parameters
    ==========
    y : array_like
        Input array to integrate.
    x : array_like, optional
        The sample points corresponding to the `y` values. If `x` is None,
        the sample points are assumed to be evenly spaced `dx` apart. The
        default is None.
    dx : scalar, optional
        The spacing between sample points when `x` is None. The default is 1.
    axis : int, optional
        The axis along which to integrate.

    Returns
    =======
    trapz : float
        Definite integral as approximated by trapezoidal rule.


    References
    ==========
    .. [1] Wikipedia page: http://en.wikipedia.org/wiki/Trapezoidal_rule

    Examples
    ========
    >>> trapz([1,2,3])
    4.0
    >>> trapz([1,2,3], x=[4,6,8])
    8.0
    >>> trapz([1,2,3], dx=2)
    8.0
    >>> a = xp.arange(6).reshape(2, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> trapz(a, axis=0)
    array([ 1.5,  2.5,  3.5])
    >>> trapz(a, axis=1)
    array([ 2.,  8.])
    """
    y = xp.asanyarray(y)
    if x is None:
        d = dx
    else:
        x = xp.asanyarray(x)
        if x.ndim == 1:
            d = xp.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = xp.diff(x, axis=axis)
    ndim = y.ndim
    slice1 = [slice(None)] * ndim
    slice2 = [slice(None)] * ndim
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    product = d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0
    try:
        ret = product.sum(axis)
    except ValueError:
        ret = xp.add.reduce(product, axis)
    return ret

def power_law(x,ampl,alpha,xmin,xmax):


    if alpha==-1:

        norm=xp.log(xmax/xmin)

    else:

        norm=(1./(1.+alpha))*(xmax**(1.+alpha)-xmin**(1.+alpha))


    ans=x**(alpha)*(ampl/norm)
    ans[x<xmin]=0
    ans[x>xmax]=0



    return ans


def gaussian(x, ampl, mean, std):


    return (ampl/(xp.sqrt(2.*xp.pi)*std))*xp.exp(-((x - mean) ** 2) / (2*std ** 2))



def gaussian_truncated(x, ampl, mean, std, xlow, xhigh):

    gauss=gaussian(x, ampl, mean, std)

    fact=0.5*(special.erf((xhigh-mean)/(xp.sqrt(2)*std))+special.erf((mean-xlow)/(xp.sqrt(2)*std)))

    ans=gauss/fact

    ans[x<xlow]=0

    ans[x>xhigh]=0

    ans[gauss==0]=0


    return ans


def smooth_exp(x,smooth_scale):

    return xp.exp((smooth_scale/x)+(smooth_scale/(x-smooth_scale)))


def smooth_function(x,low_cut,smooth_scale):


    ans=(1.+smooth_exp(x-low_cut,smooth_scale))**(-1)

    ans[x<low_cut]=0.
    ans[x>(low_cut+smooth_scale)]=1.

    return ans


def pdf_power_law_peak(x,params,xlow=2.,xhigh=100.):

    #this function implements the power law+peak function as in the LVK

    ans0=pdf_not_norm_power_law_peak(x,params,xlow=xlow,xhigh=xhigh)

    x_grid=xp.linspace(xlow,xhigh,1000)
    pdf_grid=pdf_not_norm_power_law_peak(x_grid,params)
    norm=trapz(pdf_grid,x=x_grid)

    ans=ans0/norm
    ans[ans0==0]=0

    return ans


def pdf_not_norm_power_law_peak(x,params,xlow=2.,xhigh=100.):

    # print(params)
    gauss=gaussian_truncated(x,params['lam'],params['mu'],params['sigma'],xlow=xlow,xhigh=xhigh)
    pl=power_law(x,1.-params['lam'],params['alpha'],params['xmin'],params['xmax'])

    # breakpoint()

    ans=gauss+pl

    ans*=smooth_function(x,params['xmin'],params['delta'])

    return ans


def draw_power_law_peak(params,size=1,xlow=2.,xhigh=100.):

    #this function draws samples from the power law+peak pdf

    samples=xp.zeros(size)

    xs=np.linspace(xlow,xhigh,1000)
    pdfs=pdf_not_norm_power_law_peak(xs,params,xlow=xlow,xhigh=xhigh)
    pdf_max=xp.amax(pdfs)

    for k in range(size):

        draw=True

        while draw:
            xtry=np.random.uniform(xlow,xhigh)
            pdf_try=pdf_not_norm_power_law_peak(np.array([xtry]),params,xlow=xlow,xhigh=xhigh)
            pkeep=np.random.rand()
            if pkeep*pdf_max<pdf_try:
                draw=False

        samples[k]=xtry

    if size==1:
        return samples[0]
    else:
        return samples

def pdf_power_law_smooth_conditioned(x,x0,params,xlow=None,xhigh=1.):

    #this function implements the mass ratio pdf used in the LVK, it is power law smoothed at low q to go to 0 at mmin/m1. It it a conditional pdf on m1, here represented by x0.

    ans0=pdf_not_norm_power_law_smooth_conditioned(x,x0,params,xlow=xlow,xhigh=xhigh)

    if x0.ndim==0:
        if xlow is None:
            x_grid=xp.linspace(params['xmin']/x0,xhigh,1000)
        else:
            x_grid=xp.linspace(xlow,xhigh,1000)
        pdf_grid=pdf_not_norm_power_law_smooth_conditioned(x_grid,x0,params,xlow=xlow,xhigh=xhigh)
        norm=trapz(pdf_grid,x=x_grid)
    else:
        nx0=len(x0)
        norm=np.zeros(nx0)
        for k in range(nx0):
            if xlow is None:
                x_grid=xp.linspace(params['xmin']/x0[k],xhigh,1000)
            else:
                x_grid=xp.linspace(xlow,xhigh,1000)
            pdf_grid=pdf_not_norm_power_law_smooth_conditioned(x_grid,x0[k],params,xlow=xlow,xhigh=xhigh)
            norm[k]=trapz(pdf_grid,x=x_grid)

    ans=ans0/norm
    ans[ans0==0]=0

    return ans

def pdf_not_norm_power_law_smooth_conditioned(x,x0,params,xlow=None,xhigh=1.):

    ans=x**(params['beta'])*smooth_function(x*x0,params['xmin'],params['delta'])

    ans[x>xhigh]=0.

    if xlow is not None:
        ans[x<xlow]=0.

    return ans


def draw_power_law_smooth_conditioned(params,x0,size=1,xlow=None,xhigh=1.):

    #this function draws samples for the mass ratio conditioned on m1, here represented by x0.

    samples=xp.zeros(size)

    if x0.ndim==0:

        if xlow is None:
            xlow=params['xmin']/x0

        xs=np.linspace(xlow,xhigh,1000)
        pdfs=pdf_not_norm_power_law_smooth_conditioned(xs,x0,params,xlow=xlow,xhigh=xhigh)
        pdf_max=xp.amax(pdfs)

        for k in range(size):

            draw=True

            while draw:
                xtry=np.random.uniform(xlow,xhigh)
                pdf_try=pdf_not_norm_power_law_smooth_conditioned(np.array([xtry]),x0,params,xlow=xlow,xhigh=xhigh)
                pkeep=np.random.rand()
                if pkeep*pdf_max<pdf_try:
                    draw=False

            samples[k]=xtry

    elif x0.ndim==1:

        for k in range(size):

            if xlow is None:
                xlow=params['xmin']/x0[k]

            xs=np.linspace(xlow,xhigh,1000)
            pdfs=pdf_not_norm_power_law_smooth_conditioned(xs,x0[k],params,xlow=xlow,xhigh=xhigh)
            pdf_max=xp.amax(pdfs)

            draw=True

            while draw:
                xtry=np.random.uniform(xlow,xhigh)
                pdf_try=pdf_not_norm_power_law_smooth_conditioned(np.array([xtry]),x0[k],params,xlow=xlow,xhigh=xhigh)
                pkeep=np.random.rand()
                if pkeep*pdf_max<pdf_try:
                    draw=False

            samples[k]=xtry

    if size==1:
        return samples[0]
    else:
        return samples
