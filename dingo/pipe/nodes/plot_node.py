from bilby_pipe.job_creation.nodes import PlotNode as BilbyPlotNode

from dingo.pipe.utils import _strip_unwanted_submission_keys


class PlotNode(BilbyPlotNode):
    def __init__(self, inputs, merged_node, dag):
        super(BilbyPlotNode, self).__init__(inputs)
        self.dag = dag
        self.job_name = merged_node.job_name.replace("_merge", "") + "_plot"
        self.label = merged_node.job_name.replace("_merge", "") + "_plot"
        self.request_cpus = 1
        self.setup_arguments(
            add_ini=False, add_unknown_args=False, add_command_line_args=False
        )
        self.arguments.add("label", self.label)
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
