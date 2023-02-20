from bilby_pipe.job_creation.nodes import MergeNode as BilbyMergeNode


class MergeNode(BilbyMergeNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, detectors=[])

    @property
    def executable(self):
        return self._get_executable_path("dingo_result")
