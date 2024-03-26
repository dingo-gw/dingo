# adapted from the Asimov Bilby Pipeline interface
import configparser
import glob
import importlib
import os
import re
import subprocess
import time


from asimov import config, logger

from asimov.pipeline import (
    Pipeline,
    PipelineException,
    PipelineLogger,
    PESummaryPipeline,
)


class Dingo(Pipeline):
    """
    The Dingo Pipeline.

    Parameters
    ----------
    production : :class:`asimov.Production`
       The production object.
    category : str, optional
        The category of the job.
        Defaults to "C01_offline".
    """

    with importlib.resources.path("dingo.asimov", "dingo.ini") as template_file:
        config_template = template_file

    name = "dingo"
    STATUS = {"wait", "stuck", "stopped", "running", "finished"}

    def __init__(self, production, category=None):
        super(Dingo, self).__init__(production, category)
        self.logger = logger
        if not production.pipeline.lower() == self.name:
            raise PipelineException

    def detect_completion(self):
        """
        Check for the production of the posterior file to signal that the job has completed.
        """
        self.logger.info("Checking if the dingo job has completed")

        results_dir = glob.glob(f"{self.production.rundir}/result")
        if len(results_dir) > 0:
            if (
                len(
                    glob.glob(
                        os.path.join(results_dir[0], f"*importance_sampling.hdf5")
                    )
                )
                > 0
            ):
                self.logger.info("Results files found, the job is finished.")
                return True
            else:
                self.logger.info("No results files found.")
                return False
        else:
            self.logger.info("No results directory found")
            return False

    def before_submit(self):
        """Pre-submit hook."""
        self.logger.info("Running the before_submit hook")
        pass

    def build_dag(self, psds=None, user=None, clobber_psd=False, dryrun=False):
        """
        Construct a DAG file in order to submit a production to the
        condor scheduler using DingoPipe.

        Parameters
        ----------
        production : str
           The production name.
        psds : dict, optional
           The PSDs which should be used for this DAG. If no PSDs are
           provided the PSD files specified in the ini file will be used
           instead.
        user : str
           The user accounting tag which should be used to run the job.
        dryrun: bool
           If set to true the commands will not be run, but will be printed to standard output. Defaults to False.



        Raises
        ------
        PipelineException
           Raised if the construction of the DAG fails.
        """

        cwd = os.getcwd()
        self.logger.info(f"Working in {cwd}")

        if self.production.event.repository:
            ini = self.production.event.repository.find_prods(
                self.production.name, self.category
            )[0]
            ini = os.path.join(cwd, ini)
        else:
            ini = f"{self.production.name}.ini"

        rundir = self.production.rundir

        if "job label" in self.production.meta:
            job_label = self.production.meta["job label"]
        else:
            job_label = self.production.name

        command = [
            os.path.join(config.get("pipelines", "environment"), "bin", "dingo_pipe"),
            ini,
        ]

        if dryrun:
            print(" ".join(command))
        else:
            self.logger.info(" ".join(command))
            pipe = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            out, err = pipe.communicate()
            self.logger.info(out)

            if err or "DAG generation complete, to submit jobs" not in str(out):
                self.logger.error(err)
                raise PipelineException(
                    f"DAG file could not be created.\n{command}\n{out}\n\n{err}",
                    production=self.production.name,
                )

            else:
                time.sleep(10)
                return PipelineLogger(message=out, production=self.production.name)

    def samples(self):
        """
        Collect the combined samples files for PESummary.
        """

        results_dir = os.path.join(self.production.rundir, "result")
        results_filenames = glob.glob(
            os.path.join(results_dir, f"*importance_sampling.hdf5")
        )
        return os.path.join(results_dir, results_filenames[0])

    def upload_assets(self):
        """
        Upload the samples from this job.
        """

        asset = self.collect_assets()["samples"]
        self.production.event.repository.add_file(
            asset,
            "posterior_samples.hdf5",
            commit_message=f"Added posterior_samples.",
        )

    def collect_assets(self):
        """
        Gather all of the results assets for this job.
        """
        return {"samples": self.samples()}

    def collect_logs(self):
        """
        Collect all of the log files which have been produced by this production and
        return their contents as a dictionary.
        """
        logs = glob.glob(f"{self.production.rundir}/submit/*.err") + glob.glob(
            f"{self.production.rundir}/log*/*.err"
        )
        logs += glob.glob(f"{self.production.rundir}/*/*.out")
        messages = {}
        for log in logs:
            try:
                with open(log, "r") as log_f:
                    message = log_f.read()
                    message = message.split("\n")
                    messages[log.split("/")[-1]] = "\n".join(message[-100:])
            except FileNotFoundError:
                messages[log.split("/")[-1]] = (
                    "There was a problem opening this log file."
                )
        return messages

    def submit_dag(self, dryrun=False):
        """
        Submit a DAG file to the condor cluster.

        Parameters
        ----------
        dryrun : bool
           If set to true the DAG will not be submitted,
           but all commands will be printed to standard
           output instead. Defaults to False.

        Returns
        -------
        int
           The cluster ID assigned to the running DAG file.
        PipelineLogger
           The pipeline logger message.

        Raises
        ------
        PipelineException
           This will be raised if the pipeline fails to submit the job.
        """

        cwd = os.getcwd()
        self.logger.info(f"Working in {cwd}")

        self.before_submit()

        try:
            if "job label" in self.production.meta:
                job_label = self.production.meta["job label"]
            else:
                job_label = self.production.name
            dag_filename = f"dag_{job_label}.submit"
            command = [
                "condor_submit_dag",
                "-batch-name",
                f"dingo/{self.production.event.name}/{self.production.name}",
                os.path.join(self.production.rundir, "submit", dag_filename),
            ]

            if dryrun:
                print(" ".join(command))
            else:
                dagman = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                )

                self.logger.info(" ".join(command))

                stdout, stderr = dagman.communicate()

                if "submitted to cluster" in str(stdout):
                    cluster = re.search(
                        r"submitted to cluster ([\d]+)", str(stdout)
                    ).groups()[0]
                    self.logger.info(
                        f"Submitted successfully. Running with job id {int(cluster)}"
                    )
                    self.production.status = "running"
                    self.production.job_id = int(cluster)
                    return cluster, PipelineLogger(stdout)
                else:
                    self.logger.error("Could not submit the job to the cluster")
                    self.logger.info(stdout)
                    self.logger.error(stderr)

                    raise PipelineException(
                        "The DAG file could not be submitted.",
                    )

        except FileNotFoundError as error:
            self.logger.exception(error)
            raise PipelineException(
                "It looks like condor isn't installed on this system.\n"
                f"""I wanted to run {" ".join(command)}."""
            ) from error

    def after_completion(self):
        post_pipeline = PESummaryPipeline(production=self.production)
        self.logger.info("Job has completed. Running PE Summary.")
        cluster = post_pipeline.submit_dag()
        self.production.meta["job id"] = int(cluster)
        self.production.status = "processing"
        self.production.event.update_data()

    def detect_completion_processing(self):
        # no post processing currently performed
        return True

    def resurrect(self):
        """
        Attempt to ressurrect a failed job.
        """
        try:
            count = self.production.meta["resurrections"]
        except:
            count = 0
        if (count < 5) and (
            len(glob.glob(os.path.join(self.production.rundir, "submit", "*.rescue*")))
            > 0
        ):
            count += 1
            self.submit_dag()

    @classmethod
    def read_ini(cls, filepath):
        """
        Read and parse a dingo configuration file.

        Parameters
        ----------
        filepath: str
           The path to the ini file.
        """

        with open(filepath, "r") as f:
            file_content = f.read()

        config_parser = configparser.RawConfigParser()
        config_parser.read_string(file_content)

        return config_parser
