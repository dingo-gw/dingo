from dingo.core.dataset import DingoDataset


class EventDataset(DingoDataset):
    """Dataset class for storing single event."""
    def __init__(self, file_name=None, dictionary=None):
        self.data = None
        self.event_metadata = None
        self.settings = None
        super().__init__(
            file_name=file_name,
            dictionary=dictionary,
            data_keys=["data", "event_metadata"],
        )
