import os

from bilby_pipe.job_creation.nodes import GenerationNode as BilbyGenerationNode


class GenerationNode(BilbyGenerationNode):
    @property
    def executable(self):
        return self._get_executable_path("dingo_pipe_generation")

    @property
    def event_data_file(self):
        return os.path.join(
            self.inputs.data_directory, "_".join([self.label, "event_data.hdf5"])
        )
