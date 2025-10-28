import os

from bilby_pipe.job_creation.nodes import GenerationNode as BilbyGenerationNode
from bilby_pipe.utils import BilbyPipeError

from dingo.pipe.utils import _strip_unwanted_submission_keys


class GenerationNode(BilbyGenerationNode):

    def __init__(self, inputs, trigger_time, idx, dag, parent=None, importance_sampling=False):
        """
        Node for data generation jobs with DINGO-specific customizations
        """
        self.importance_sampling = importance_sampling
        
        # Call Node.__init__ directly, skipping BilbyGenerationNode.__init__
        super(BilbyGenerationNode, self).__init__(inputs, retry=3)
        
        # Copy all the BilbyGenerationNode logic but add DINGO customizations
        if not inputs.osg and inputs.generation_pool == "igwn-pool":
            raise BilbyPipeError(
                "Generation job requested to use the igwn-pool "
                "(OSG, --generation-pool=igwn-pool), but --osg=False"
            )
        else:
            self.run_node_on_osg = inputs.generation_pool == "igwn-pool"
        self.inputs = inputs
        self.trigger_time = trigger_time
        self.inputs.trigger_time = trigger_time
        self.idx = idx
        self.dag = dag
        self.request_cpus = 1

        self.setup_arguments()
        self.arguments.add("label", self.label)
        self.arguments.add("idx", self.idx)
        self.arguments.add("trigger-time", self.trigger_time)
        if self.inputs.injection_file is not None:
            self.arguments.add("injection-file", self.inputs.injection_file)
        if self.inputs.timeslide_file is not None:
            self.arguments.add("timeslide-file", self.inputs.timeslide_file)

        frame_files, success = self.resolve_frame_files
        need_scitokens = not success

        if self.inputs.transfer_files or self.inputs.osg:
            input_files_to_transfer = list()
            for attr in [
                "complete_ini_file",
                "prior_file",
                "injection_file",
                "gps_file",
                "timeslide_file",
            ]:
                if (value := getattr(self.inputs, attr)) is not None:
                    input_files_to_transfer.append(os.path.abspath(str(value)))

            # modifying the paths for OSDF networks
            network_files = []
            for s in [self.inputs.model, self.inputs.model_init]:
                if s is None:
                    continue
                if "osdf" in s:
                    # stripping osdf prefix as it is not needed
                    network_files.append(f"igwn+osdf://{s.replace('/osdf', '')}")
                else:
                    network_files.append(s)

            input_files_to_transfer.extend(network_files)

            if self.transfer_container:
                input_files_to_transfer.append(self.inputs.container)
            for value in [
                self.inputs.psd_dict,
                self.inputs.spline_calibration_envelope_dict,
                frame_files,
            ]:
                input_files_to_transfer.extend(self.extract_paths_from_dict(value))
            input_files_to_transfer.extend(self.inputs.additional_transfer_paths)

            input_files_to_transfer, need_auth = self.job_needs_authentication(
                input_files_to_transfer
            )

            # Credentials are needed to access any OSDF files
            if any(["osdf" in s for s in input_files_to_transfer]):
                need_auth = True
            need_scitokens = need_scitokens or need_auth

            self.extra_lines.extend(
                self._condor_file_transfer_lines(
                    input_files_to_transfer,
                    [self._relative_topdir(self.inputs.outdir, self.inputs.initialdir)],
                )
            )
            self.arguments.add("outdir", os.path.relpath(self.inputs.outdir))

        elif new_frames := [
            fname
            for fname in self.extract_paths_from_dict(frame_files)
            if fname.startswith(self.inputs.data_find_urltype)
        ]:
            from bilby_pipe.utils import logger
            logger.warning(
                "The following frame files were identified by gwdatafind for this analysis. "
                "These frames may not be found by the data generation stage as file "
                "transfer is not being used. You should either set transfer-files=True or "
                "pass these frame files to the data-dict option. You may need to "
                f"remove a prefix, e.g., file://localhost.\n\t{new_frames}"
            )
        if need_scitokens:
            self.extra_lines.extend(self.scitoken_lines)

        # DINGO-specific customization: Add site selection
        if self.inputs.generation_pool == "igwn-pool":
            sites = getattr(self.inputs, 'generation_desired_sites', None)
            if sites is not None:
                self.extra_lines.append(f'MY.DESIRED_Sites = "{sites}"')

        self.process_node()
        if parent:
            self.job.add_parent(parent.job)

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
