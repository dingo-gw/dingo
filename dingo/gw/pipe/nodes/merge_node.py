from bilby_pipe.job_creation.nodes import MergeNode as BilbyMergeNode


class MergeNode(BilbyMergeNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, detectors=[])

    @property
    def executable(self):
        return self._get_executable_path("dingo_result")

    @property
    def result_file(self):
        label = self.label.removesuffix("_merge")
        return f"{self.inputs.result_directory}/{label}.hdf5"
