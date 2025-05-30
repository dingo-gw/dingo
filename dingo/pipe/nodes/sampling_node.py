import os

from bilby_pipe.job_creation.nodes import AnalysisNode

from dingo.pipe.utils import _strip_unwanted_submission_keys


class SamplingNode(AnalysisNode):
    def __init__(self, inputs, generation_node, dag):
        super(AnalysisNode, self).__init__(inputs)
        self.dag = dag
        self.generation_node = generation_node
        self.request_cpus = inputs.request_cpus
        self.device = inputs.device

        data_label = generation_node.job_name
        base_name = data_label.replace("generation", "sampling")
        self.job_name = base_name
        self.label = self.job_name

        self.setup_arguments()


        # Add extra arguments for dingo
        self.arguments.add("label", self.label)
        self.arguments.add("event-data-file", generation_node.event_data_file)

        self.extra_lines.extend(self._checkpoint_submit_lines())
        env_vars = []
        # if self.request_cpus > 1:
        #     env_vars.append("OMP_NUM_THREADS=1")
        if getattr(self, "disable_hdf5_locking", None):
            env_vars.append("USE_HDF5_FILE_LOCKING=FALSE")
        if env_vars:
            self.extra_lines.append(f"environment = \"{' '.join(env_vars)}\"")

        for req in inputs.sampling_requirements:
            self.requirements.append(req)

        if self.device == "cuda":
            self.extra_lines.append("request_gpus = 1")


        if self.inputs.transfer_files or self.inputs.osg:
            input_files_to_transfer = [
                str(generation_node.event_data_file),
                str(self.inputs.complete_ini_file),
            ]
            if self.inputs.osg:
                sites = self.inputs.gpu_desired_sites
                if sites is not None:
                    self.extra_lines.append(f'MY.DESIRED_Sites = "{sites}"')
                self.requirements.append("IS_GLIDEIN=?=True")

                # only supporting OSDF transfers for now
                # stripping osdf prefix as it is not needed
                network_files = [s.replace("/osdf", "") for s in [self.inputs.model, self.inputs.model_init] if s is not None]
                network_files = [f"igwn+osdf://{s}" for s in network_files]
                input_files_to_transfer.extend(network_files)
                # This is needed to access the networks which are in OSDF
                self.extra_lines.extend(self.scitoken_lines)

            self.extra_lines.extend(
                self._condor_file_transfer_lines(
                    input_files_to_transfer,
                    [self._relative_topdir(self.inputs.outdir, self.inputs.initialdir)],
                )
            )

            self.arguments.add("outdir", os.path.relpath(self.inputs.outdir))

        self.process_node()
        self.job.add_parent(generation_node.job)

        if self.inputs.simple_submission:
            _strip_unwanted_submission_keys(self.job)

    @property
    def executable(self):
        return self._get_executable_path("dingo_pipe_sampling")

    @property
    def samples_file(self):
        # TODO: Maybe remove -- not needed.
        return os.path.join(
            self.inputs.result_directory, self.label + ".hdf5"
        )

    @property
    def result_file(self):
        return self.samples_file
