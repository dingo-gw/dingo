class UnpackDict(object):
    """
    Unpacks the dictionary to prepare it for final output of the dataloader.
    Only returns elements specified in selected_keys.
    """
    def __init__(self, selected_keys):
        self.selected_keys = selected_keys

    def __call__(self, input_sample):
        return [input_sample[k] for k in self.selected_keys]


class SelectKeys(object):
    """
    Restricts the sample dictionary to selected_keys, to prepare it for final output
    of the dataloader. In contrast to UnpackDict, the sample remains a dictionary, so
    that batches are keyed by name rather than by position.
    """

    def __init__(self, selected_keys):
        self.selected_keys = selected_keys

    def __call__(self, input_sample):
        missing = [k for k in self.selected_keys if k not in input_sample]
        if missing:
            raise KeyError(
                f"Sample is missing keys {missing}: expected {self.selected_keys}, "
                f"got {sorted(input_sample)}."
            )
        return {k: input_sample[k] for k in self.selected_keys}