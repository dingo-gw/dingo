from bilby_pipe.job_creation.nodes import MergeNode as BilbyMergeNode

from dingo.pipe.utils import _strip_unwanted_submission_keys


class MergeNode(BilbyMergeNode):

    run_node_on_osg = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs, detectors=[])
        
        # Add site selection for merge jobs
        if self.inputs.osg:
            # Check for merge-specific desired sites, fall back to cpu_desired_sites or nogrid
            sites = getattr(self.inputs, 'cpu_desired_sites', 'nogrid')
            if sites == 'nogrid' or sites is None:
                self.extra_lines.append("MY.flock_local = True")
                self.extra_lines.append('MY.DESIRED_Sites = "nogrid"')
            else:
                self.extra_lines.append(f'MY.DESIRED_Sites = "{sites}"')
        
        if self.inputs.simple_submission:
            _strip_unwanted_submission_keys(self.job)

    @property
    def executable(self):
        return self._get_executable_path("dingo_result")

    @property
    def result_file(self):
        label = self.label.replace("_merge", "")
        return f"{self.inputs.result_directory}/{label}.hdf5"
