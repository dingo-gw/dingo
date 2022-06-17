class GetItem:

    def __init__(self, key):
        self.key = key

    def __call__(self, sample):
        return sample[self.key]

class RenameKey:

    def __init__(self, old, new):
        self.old = old
        self.new = new

    def __call__(self, input_sample : dict):
        sample = input_sample.copy()
        sample[self.new] = sample.pop(self.old)
        return sample
