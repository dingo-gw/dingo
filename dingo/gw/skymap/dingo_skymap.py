"""Generation of ligo skymaps from dingo results."""
from ligo.skymap import kde
from typing import Optional, Dict, Any
from bilby.gw.prior import UniformComovingVolume
from bilby.core.prior.analytical import PowerLaw
import numpy as np

from dingo.gw.result import Result


def generate_skymap_from_dingo_result(
    dingo_result: Result,
    num_samples: int = 5_000,
    num_trials: int = 1,
    num_jobs: int = 1,
    prior_distance_power: Optional[float] = 2,
    cosmology: bool = False,
    weight_clipping_kwargs: Optional[Dict[str, Any]] = None,
    return_aux: bool = False,
    allow_duplicates: bool = True,
):
    """Generate a skymap based on the estimated sky position of the dingo result.

    Parameters
    ----------
    dingo_result: Result
        dingo result file
    num_samples: int
        number of samples for skymap kde
    num_trials: int
        number of trials for skymap kde
    num_jobs: int
        number of parallel job for skymap kde
    prior_distance_power: int, optional
        The power of distance that appears in the prior
        default: 2, uniform in volume
        if None, use prior from dingo result (i.e., no reweighting)
    cosmology: bool, optional
        Set to enable a uniform in comoving volume prior (default: false).
    weight_clipping_kwargs: dict
        if set, clip the weights before rejection sampling based on these kwargs
    return_aux: bool
        if True, return dict with aux information
    allow_duplicates: bool
        if False, the KDE samples are sampled without replacement

    Returns
    -------
    skymap: skymap for estimated dingo position
    aux: dict with aux information
    """
    samples = dingo_result.samples

    # extract weights, also veto invalid [ra, dec] values
    if "weights" in samples:
        weights = np.array(samples["weights"])
    else:
        weights = np.ones(len(samples))
    weights *= np.array(samples["ra"]) >= 0
    weights *= np.array(samples["ra"]) <= 2 * np.pi
    weights *= np.array(samples["dec"]) >= -np.pi / 2
    weights *= np.array(samples["dec"]) <= np.pi / 2

    # apply distance reweighting
    distance = np.array(samples["luminosity_distance"])
    prior_result = dingo_result.prior["luminosity_distance"]
    if cosmology:
        if prior_distance_power is not None:
            raise ValueError(
                "Only one of prior_distance_power and cosmology can be used."
            )
        prior_updated = UniformComovingVolume(
            prior_result.minimum, prior_result.maximum, name="luminosity_distance"
        )
        weights = weights * np.exp(
            prior_updated.ln_prob(distance) - prior_result.ln_prob(distance)
        )
        weights[np.where(np.isinf(prior_result.ln_prob(distance)))[0]] = 0
        weights /= weights.mean()
    elif prior_distance_power is not None:
        prior_updated = PowerLaw(
            prior_distance_power, prior_result.minimum, prior_result.maximum
        )
        weights = weights * np.exp(
            prior_updated.ln_prob(distance) - prior_result.ln_prob(distance)
        )
        weights[np.where(np.isinf(prior_result.ln_prob(distance)))[0]] = 0
        weights /= weights.mean()

    if weight_clipping_kwargs is not None:
        if not allow_duplicates:
            raise ValueError(
                "Weight clipping should not be set when dropping duplicates."
            )
        weights = clip_weights(weights, **weight_clipping_kwargs)

    samples = samples.sample(num_samples, weights=weights, replace=allow_duplicates)
    ra_dec_dL = np.array(samples[["ra", "dec", "luminosity_distance"]])
    skypost = kde.Clustered2DSkyKDE(ra_dec_dL, trials=num_trials, jobs=num_jobs)
    skymap = skypost.as_healpix()

    if not return_aux:
        return skymap
    else:
        aux = {
            "n_eff": int(np.sum(weights) ** 2 / np.sum(weights ** 2)),
            "n_rejection": int(np.sum(weights / np.max(weights))),
        }
        return skymap, aux


def clip_weights(weights, num_clip, mode="mean", print_stats=False):
    """Clip the num_clip highest weights and return the result.

    See e.g. https://ieeexplore.ieee.org/document/8450722.
    Clipping is leads to asymptotically correct results if num_clip <= sqrt(num_samples).

    Parameters
    ----------
    weights: array with weights
    num_clip: number of samples to clip
    mode: whether to set the clipped weight to mean or min weight of the clipped samples
    print_stats: if True, print the ess and ns before and after clipping

    Returns
    -------
    weights: updated (clipped) weights
    """
    weights = np.copy(weights)
    ess_before = np.sum(weights) ** 2 / np.sum(weights ** 2)
    ns_before = np.sum(weights / np.max(weights))

    # get indices of clipped samples
    # clipped_indices = np.argsort(-weights)[:num_clip]
    clipped_indices = np.argpartition(-weights, num_clip)[:num_clip]  # more efficient

    # get new weight for clipped samples
    if mode == "mean":
        weight_new = np.mean(weights[clipped_indices])
    elif mode == "min":
        weight_new = np.min(weights[clipped_indices])
    else:
        raise ValueError(f"Mode should be mean or min, got {mode}.")

    # set new weight
    weights[clipped_indices] = weight_new
    ess_after = np.sum(weights) ** 2 / np.sum(weights ** 2)
    ns_after = np.sum(weights / np.max(weights))

    if print_stats:
        print(f"Statistics before / after clipping (N = {len(weights)}):")
        print(f"ESS:\t{ess_before:.0f} / {ess_after:.0f}")
        print(f"NS: \t{ns_before:.0f} / {ns_after:.0f}")

    weights = weights / np.mean(weights)

    return weights
