"""
This submodule contains tools for density estimation from samples.
This is required for instance to recover the posterior density from GNPE samples,
since the density is intractable with GNPE.
"""
from .unconditional_density_estimation import train_unconditional_density_estimator
from .interpolation import (
    interpolated_sample_and_log_prob_multi,
    interpolated_log_prob_multi,
)
from .nde_settings import get_default_nde_settings_3d
