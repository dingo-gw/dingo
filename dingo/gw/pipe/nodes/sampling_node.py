import os

from bilby_pipe.job_creation.nodes import AnalysisNode


class SamplingNode(AnalysisNode):
    def __init__(self, inputs, generation_node, dag):
        super(AnalysisNode, self).__init__(inputs)
        self.dag = dag
        self.generation_node = generation_node
        self.request_cpus = inputs.request_cpus

        data_label = generation_node.job_name
        base_name = data_label.replace("generation", "sampling")
        self.job_name = base_name
        self.label = self.job_name

        self.setup_arguments()

        # if self.inputs.transfer_files or self.inputs.osg:
        #     data_dump_file = generation_node.data_dump_file
        #     input_files_to_transfer = [
        #         str(data_dump_file),
        #         str(self.inputs.complete_ini_file),
        #     ]
        #     self.extra_lines.extend(
        #         self._condor_file_transfer_lines(
        #             input_files_to_transfer,
        #             [self._relative_topdir(self.inputs.outdir, self.inputs.initialdir)],
        #         )
        #     )
        #     self.arguments.add("outdir", os.path.relpath(self.inputs.outdir))

        # Add extra arguments for dingo
        self.arguments.add("label", self.label)
        self.arguments.add("event-data-file", generation_node.event_data_file)

        # TODO: Set whether to recover the log probability. (Config should be in input
        #  file.)  Other settings needed?

        self.extra_lines.extend(self._checkpoint_submit_lines())
        if inputs.known_args.device == "cuda":
            self.extra_lines.extend(["request_gpus = 1"])
            # TODO: set memory requirement in .ini file?
            self.requirements.extend(["TARGET.CUDAGlobalMemoryMb > 40000"])
        # if self.request_cpus > 1:
        #     self.extra_lines.extend(['environment = "OMP_NUM_THREADS=1"'])

        self.process_node()
        self.job.add_parent(generation_node.job)

    @property
    def executable(self):
        return self._get_executable_path("dingo_pipe_sampling")

    @property
    def samples_file(self):
        return os.path.join(
            self.inputs.result_directory, "_".join([self.label, "result.hdf5"])
        )
