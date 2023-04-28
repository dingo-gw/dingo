from bilby_pipe.job_creation.nodes import PESummaryNode as BilbyPESummaryNode
from bilby_pipe.utils import BilbyPipeError, logger

logger.name = "dingo_pipe"


class PESummaryNode(BilbyPESummaryNode):
    def __init__(self, inputs, merged_node_list, generation_node_list, dag):
        super(BilbyPESummaryNode, self).__init__(inputs)
        self.dag = dag
        self.job_name = f"{inputs.label}_pesummary"
        self.request_cpus = 1

        n_results = len(merged_node_list)
        result_files = [merged_node.result_file for merged_node in merged_node_list]
        labels = [merged_node.label for merged_node in merged_node_list]

        self.setup_arguments(
            add_ini=False, add_unknown_args=False, add_command_line_args=False
        )
        self.arguments.add("webdir", self.inputs.webdir)
        if self.inputs.email is not None:
            self.arguments.add("email", self.inputs.email)
        self.arguments.add(
            "config", " ".join([self.inputs.complete_ini_file] * n_results)
        )
        self.arguments.add("samples", f"{' '.join(result_files)}")

        # Using append here as summary pages doesn't take a full name for approximant
        self.arguments.append("-a")
        self.arguments.append(" ".join([self.inputs.waveform_approximant] * n_results))

        # if len(generation_node_list) == 1:
        #     self.arguments.add("gwdata", generation_node_list[0].data_dump_file)
        # elif len(generation_node_list) > 1:
        #     logger.info(
        #         "Not adding --gwdata to PESummary job as there are multiple files"
        #     )
        existing_dir = self.inputs.existing_dir
        if existing_dir is not None:
            self.arguments.add("existing_webdir", existing_dir)

        if isinstance(self.inputs.summarypages_arguments, dict):
            if "labels" not in self.inputs.summarypages_arguments.keys():
                self.arguments.append(f"--labels {' '.join(labels)}")
            else:
                if len(labels) != len(result_files):
                    raise BilbyPipeError(
                        "Please provide the same number of labels for postprocessing "
                        "as result files"
                    )
            not_recognised_arguments = {}
            for key, val in self.inputs.summarypages_arguments.items():
                if key == "nsamples_for_skymap":
                    self.arguments.add("nsamples_for_skymap", val)
                elif key == "gw":
                    self.arguments.add_flag("gw")
                elif key == "no_ligo_skymap":
                    self.arguments.add_flag("no_ligo_skymap")
                elif key == "burnin":
                    self.arguments.add("burnin", val)
                elif key == "kde_plot":
                    self.arguments.add_flag("kde_plot")
                elif key == "gracedb":
                    self.arguments.add("gracedb", val)
                elif key == "palette":
                    self.arguments.add("palette", val)
                elif key == "include_prior":
                    self.arguments.add_flag("include_prior")
                elif key == "notes":
                    self.arguments.add("notes", val)
                elif key == "publication":
                    self.arguments.add_flag("publication")
                elif key == "labels":
                    self.arguments.add("labels", f"{' '.join(val)}")
                else:
                    not_recognised_arguments[key] = val
            if not_recognised_arguments != {}:
                logger.warn(
                    "Did not recognise the summarypages_arguments {}. To find "
                    "the full list of available arguments, please run "
                    "summarypages --help".format(not_recognised_arguments)
                )

        self.process_node()
        for merged_node in merged_node_list:
            self.job.add_parent(merged_node.job)
