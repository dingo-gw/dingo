from bilby_pipe.job_creation.nodes import GenerationNode as BilbyGenerationNode


class GenerationNode(BilbyGenerationNode):

    @property
    def executable(self):
        return self._get_executable_path("dingo_pipe_generation")
