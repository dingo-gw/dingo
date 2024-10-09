import numpy as np

from bilby.core.prior import Uniform, PriorDict

def build_bilby_prior_dict(prior_dict):
    """
    Build a Bilby PriorDict from a dictionary of priors.

    Parameters
    ----------
    prior_dict: dict
        Dictionary of priors.

    Returns
    -------
    PriorDict
        Bilby PriorDict.
    """
    prior_dict_bilby = PriorDict()

    for k in prior_dict:
        type_ = prior_dict[k]["type"]

        if(type_ == "bilby.core.prior.Uniform"):
            prior_dict_bilby[k] = Uniform(prior_dict[k]["minimum"], prior_dict[k]["maximum"])

    return prior_dict_bilby


def calculate_mean_and_std(distribution):

    """
    If the distribution is Uniform, we can analytically calculate 
    the mean and standard deviation.

    Parameters
    ----------

    distribution: bilby.core.prior.Prior

    Returns
    -------
    mean: float
        The mean of the distribution.
    std: float
        The standard deviation of the distribution.

    """
    
    # Check if the entry is an instance of the Uniform class
    if isinstance(distribution, Uniform):
        # Calculate the mean
        mean = (distribution.minimum + distribution.maximum) / 2
        # Calculate the standard deviation
        std = (distribution.maximum - distribution.minimum) / np.sqrt(12)
        return mean, std
    else:
        raise ValueError("The provided distribution is not a Uniform distribution.")