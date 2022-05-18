from dingo.core.dataset import DingoDataset


class SamplesDataset(DingoDataset):
    """
    A dataset class to hold a collection of samples, implementing I/O.

    Attributes
    ----------
    samples : pd.Dataframe
        Contains parameter samples, as well as (possibly) log_prob, log_likelihood,
        weights, log_prior.
    context : dict
        Context data on which the samples were produced (e.g., strain data, ASDs).
    log_evidence : float
    effective_sample_size : float
    """
    def __init__(self, file_name=None, dictionary=None):
        super().__init__(
            file_name=file_name,
            dictionary=dictionary,
            data_keys=[
                "samples",
                "context",
                "log_evidence",
                "effective_sample_size",
            ],
        )
