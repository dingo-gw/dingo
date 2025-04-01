import os

from bilby_pipe.job_creation.nodes import AnalysisNode

from dingo.pipe.utils import _strip_unwanted_submission_keys


class ImportanceSamplingNode(AnalysisNode):
    def __init__(self, inputs, sampling_node, generation_node, parallel_idx, dag):
        super(AnalysisNode, self).__init__(inputs)
        self.dag = dag
        self.sampling_node = sampling_node
        self.generation_node = generation_node
        self.parallel_idx = parallel_idx
        self.request_cpus = inputs.request_cpus_importance_sampling

        data_label = sampling_node.job_name
        base_name = data_label.replace("sampling", "importance_sampling")
        self.base_job_name = base_name
        if parallel_idx != "":
            self.job_name = f"{base_name}_{parallel_idx}"
        else:
            self.job_name = base_name
        self.label = self.job_name

        proposal_samples_file = os.path.join(
            self.inputs.result_directory,
            self.label.replace("importance_sampling", "sampling") + ".hdf5",
        )

        self.setup_arguments()

        if self.inputs.transfer_files or self.inputs.osg:
           
            input_files_to_transfer = [
                proposal_samples_file,
                str(generation_node.event_data_file),
                str(self.inputs.complete_ini_file),
                *(self.inputs.spline_calibration_envelope_dict.values() if self.inputs.spline_calibration_envelope_dict else [])
            ]

            # if running on the OSG we need to specify the sites
            if self.inputs.osg:
                sites = self.inputs.desired_sites
                if sites is not None:
                    self.extra_lines.append(f'MY.DESIRED_Sites = "{sites}"')
                self.requirements.append("IS_GLIDEIN=?=True")
                
            self.extra_lines.extend(
                self._condor_file_transfer_lines(
                    input_files_to_transfer,
                    [self._relative_topdir(self.inputs.outdir, self.inputs.initialdir)],
                )
            )
            self.arguments.add("outdir", os.path.relpath(self.inputs.outdir))

            
        # Add extra arguments for dingo
        self.arguments.add("label", self.label)
        self.arguments.add("proposal-samples-file", proposal_samples_file)
        self.arguments.add("event-data-file", generation_node.event_data_file)

        env_vars = []
        # if self.request_cpus > 1:
        #     env_vars.append("OMP_NUM_THREADS=1")
        if getattr(self, "disable_hdf5_locking", None):
            env_vars.append("USE_HDF5_FILE_LOCKING=FALSE")
        if env_vars:
            self.extra_lines.append(f"environment = \"{' '.join(env_vars)}\"")

        self.process_node()

        # We need both of these as parents because importance sampling can in principle
        # use different data than sampling. In that case, the generation node will not
        # be a parent of the sampling node.
        self.job.add_parent(sampling_node.job)
        self.job.add_parent(generation_node.job)

        if self.inputs.simple_submission:
            _strip_unwanted_submission_keys(self.job)

    @property
    def executable(self):
        return self._get_executable_path("dingo_pipe_importance_sampling")

    @property
    def result_file(self):
        return f"{self.inputs.result_directory}/{self.job_name}.hdf5"
