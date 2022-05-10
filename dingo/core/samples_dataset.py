from dingo.core.dataset import DingoDataset


class SamplesDataset(DingoDataset):
    def __init__(self, file_name, dictionary):
        super().__init__(
            file_name=file_name,
            dictionary=dictionary,
            data_keys=[
                "samples",
                "injection_parameters",
                "context",
                "log_evidence",
                "event_metadata",
            ],
        )
