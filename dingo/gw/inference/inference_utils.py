from typing import Optional

import numpy as np

from dingo.gw.inference.gw_samplers import GWSampler


def prepare_log_prob(
    sampler,
    num_samples: int,
    nde_settings: dict,
    batch_size: Optional[int] = None,
    threshold_std: Optional[float] = np.inf,
    remove_init_outliers: Optional[float] = 0.0,
    low_latency_label: str = None,
    outdir: str = None,
):
    """
    Prepare gnpe sampling with log_prob. This is required, since in its vanilla
    form gnpe does not provide the density for its samples.

    Specifically, we train an unconditional neural density estimator (nde) for the
    gnpe proxies. This requires running the gnpe sampler till convergence, and
    extracting the gnpe proxies after the final gnpe iteration. The nde is trained
    to match the distribution over gnpe proxies, which provides a way of rapidly
    sampling (converged!) gnpe proxies *and* evaluating the log_prob.

    After this preparation step, self.run_sampler can leverage
    self.gnpe_proxy_sampler (which is based on the aforementioned trained nde) to
    sample gnpe proxies, such that one gnpe iteration is sufficient. The
    log_prob of
    the samples in the *joint* space (inference parameters + gnpe proxies) is then
    simply given by the sum of the corresponding log_probs (from self.model and
    self.gnpe_proxy_sampler.model).

    Parameters
    ----------
    num_samples: int
        number of samples for training of nde
    batch_size: int = None
        batch size for sampler
    threshold_std: float = np.inf
        gnpe proxies deviating by more then threshold_std standard deviations from
        the proxy mean (along any axis) are discarded.
    low_latency_label: str = None
        File label for low latency samples (= samples used for training nde).
        If None, these samples are not saved.
    outdir: str = None
        Directory in which low latency samples are saved. Needs to be set if
        low_latency_label is not None.
    """
    sampler.remove_init_outliers = remove_init_outliers
    sampler.run_sampler(num_samples, batch_size)
    if low_latency_label is not None:
        sampler.to_hdf5(label=low_latency_label, outdir=outdir)
    result = sampler.to_result()
    nde_settings["training"]["device"] = str(sampler.model.device)
    unconditional_model = result.train_unconditional_flow(
        sampler.gnpe_proxy_parameters,
        nde_settings,
        threshold_std=threshold_std,
    )

    # Prepare sampler with unconditional model as initialization. This should only use
    # one iteration and also not remove any outliers.
    sampler.init_sampler = GWSampler(model=unconditional_model)
    sampler.num_iterations = 1
    sampler.remove_init_outliers = 0.0  # Turn off for final sampler.
