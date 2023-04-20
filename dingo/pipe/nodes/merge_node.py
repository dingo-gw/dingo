from bilby_pipe.job_creation.nodes import MergeNode as BilbyMergeNode

from dingo.pipe.utils import _strip_unwanted_submission_keys


class MergeNode(BilbyMergeNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, detectors=[])
        if self.inputs.simple_submission:
            _strip_unwanted_submission_keys(self.job)

    @property
    def executable(self):
        return self._get_executable_path("dingo_result")

    @property
    def result_file(self):
        label = self.label.replace("_merge", "")
        return f"{self.inputs.result_directory}/{label}.hdf5"
