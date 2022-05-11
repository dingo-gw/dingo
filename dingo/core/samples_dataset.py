from dingo.core.dataset import DingoDataset


class SamplesDataset(DingoDataset):
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
