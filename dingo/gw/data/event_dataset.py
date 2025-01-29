from dingo.core.dataset import DingoDataset


class EventDataset(DingoDataset):
    """Dataset class for storing single event."""

    dataset_type = "event_dataset"

    def __init__(self, file_name=None, dictionary=None):
        self.data = None
        self.settings = None
        super().__init__(
            file_name=file_name,
            dictionary=dictionary,
            data_keys=["data"],
        )

class EventDatasetList(DingoDataset):
    
    def __init__(self, file_name=None, dictionary=None):
        super().__init__(
            file_name=file_name,
            dictionary=dictionary,
            data_keys=["event_dataset_dict_list"],
        )