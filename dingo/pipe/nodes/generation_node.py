import os

from bilby_pipe.job_creation.nodes import GenerationNode as BilbyGenerationNode

from dingo.pipe.utils import _strip_unwanted_submission_keys


class GenerationNode(BilbyGenerationNode):

    def __init__(self, inputs, importance_sampling=False, **kwargs):
        self.importance_sampling = importance_sampling
        super().__init__(inputs, **kwargs)

        if self.inputs.simple_submission:
            _strip_unwanted_submission_keys(self.job)

    @property
    def executable(self):
        return self._get_executable_path("dingo_pipe_generation")

    def setup_arguments(self, **kwargs):
        super().setup_arguments(**kwargs)
        if self.importance_sampling:
            self.arguments.add_flag("importance-sampling-generation")

    @property
    def job_name(self):
        flag = "_IS" if self.importance_sampling else ""
        return super().job_name + flag

    @property
    def event_data_file(self):
        return os.path.join(
            self.inputs.data_directory, "_".join([self.label, "event_data.hdf5"])
        )
