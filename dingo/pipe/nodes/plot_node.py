import os

from bilby_pipe.job_creation.nodes import PlotNode as BilbyPlotNode

from dingo.pipe.utils import _strip_unwanted_submission_keys


class PlotNode(BilbyPlotNode):

    run_node_on_osg = False
    
    def __init__(self, inputs, merged_node, dag):
        # Call Node.__init__ directly, skipping BilbyPlotNode.__init__ to customize
        super(BilbyPlotNode, self).__init__(inputs)
        self.dag = dag
        self.job_name = merged_node.job_name.replace("_merge", "") + "_plot"
        self.label = merged_node.job_name.replace("_merge", "") + "_plot"
        self.request_cpus = 1

        # Add file transfer logic (copied from BilbyPlotNode)
        if self.inputs.transfer_files or self.inputs.osg:
            input_files_to_transfer = [
                self._relative_topdir(merged_node.result_file, self.inputs.initialdir),
            ] + inputs.additional_transfer_paths
            if self.transfer_container:
                input_files_to_transfer.append(self.inputs.container)

            input_files_to_transfer, need_scitokens = self.job_needs_authentication(
                input_files_to_transfer
            )
            if need_scitokens:
                self.extra_lines.extend(self.scitoken_lines)

            self.extra_lines.extend(
                self._condor_file_transfer_lines(
                    input_files_to_transfer,
                    [self._relative_topdir(self.inputs.outdir, self.inputs.initialdir)],
                )
            )

        self.setup_arguments(
            add_ini=False, add_unknown_args=False, add_command_line_args=False
        )
        self.arguments.add("label", self.label)
        if self.inputs.transfer_files or self.inputs.osg:
            self.arguments.add("result", os.path.relpath(merged_node.result_file))
            self.arguments.add("outdir", os.path.relpath(self.inputs.result_directory))
        else:
            self.arguments.add("result", merged_node.result_file)
            self.arguments.add("outdir", self.inputs.result_directory)
        for plot_type in ["corner", "weights", "log_probs"]:
            if getattr(inputs, f"plot_{plot_type}", False):
                self.arguments.add_flag(plot_type)
        # self.arguments.add("format", inputs.plot_format)

        if getattr(self, "disable_hdf5_locking", None):
            self.extra_lines.append('environment = "HDF5_USE_FILE_LOCKING=FALSE"')

        self.process_node()
        self.job.add_parent(merged_node.job)

        if self.inputs.simple_submission:
            _strip_unwanted_submission_keys(self.job)

    @property
    def executable(self):
        return self._get_executable_path("dingo_pipe_plot")


class PlotPPNode(BilbyPlotNode):
    def __init__(self, inputs, merged_node_list, dag):
        super(BilbyPlotNode, self).__init__(inputs)
        self.dag = dag
        self.request_cpus = 1
        self.job_name = f"{self.inputs.label}_plot_pp"
        self.setup_arguments(
            add_ini=False, add_unknown_args=False, add_command_line_args=False
        )
        self.arguments.add_positional_argument(self.inputs.result_directory)

        if getattr(self, "disable_hdf5_locking", None):
            self.extra_lines.append('environment = "HDF5_USE_FILE_LOCKING=FALSE"')

        self.process_node()
        for node in merged_node_list:
            self.job.add_parent(node.job)

        if self.inputs.simple_submission:
            _strip_unwanted_submission_keys(self.job)

    @property
    def executable(self):
        return self._get_executable_path("dingo_pipe_pp_test")

    @property
    def request_memory(self):
        return "16 GB"