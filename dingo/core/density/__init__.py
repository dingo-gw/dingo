"""
This submodule contains tools for density estimation from samples.
This is required for instance to recover the posterior density from GNPE samples,
since the density is intractable with GNPE.
"""
from .uncogit nditional_density_estimation import train_unconditional_density_estimator