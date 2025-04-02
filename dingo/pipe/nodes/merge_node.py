from bilby_pipe.job_creation.nodes import MergeNode as BilbyMergeNode

from dingo.pipe.utils import _strip_unwanted_submission_keys


class MergeNode(BilbyMergeNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, detectors=[])
        if self.inputs.simple_submission:
            _strip_unwanted_submission_keys(self.job)

        if self.inputs.transfer_files or self.inputs.osg:
            # extracting hdf5 files
            input_files_to_transfer = []
            args = self.arguments.argument_list
            if '--result' in args:
                start_index = args.index('--result') + 1
                for i in range(start_index, len(args)):
                    if args[i].startswith('--'):  # Stop when another flag is found
                        break
                    if args[i].endswith('.hdf5'):
                        input_files_to_transfer.append(args[i])

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

    @property
    def executable(self):
        return self._get_executable_path("dingo_result")

    @property
    def result_file(self):
        label = self.label.replace("_merge", "")
        return f"{self.inputs.result_directory}/{label}.hdf5"
