import configparser
import glob
import os
import re
import subprocess
import time


from asimov import config, logger
from asimov.utils import set_directory

from asimov.pipeline import Pipeline, PipelineException, PipelineLogger



class Dingo(Pipeline):
    """
    The LALInference Pipeline.

    Parameters
    ----------
    production : :class:`asimov.Production`
       The production object.
    category : str, optional
        The category of the job.
        Defaults to "C01_offline".
    """

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
        # TODO: correct results directory
        results_dir = glob.glob(f"{self.production.rundir}/posterior_samples")
        if len(results_dir) > 0:
            if len(glob.glob(os.path.join(results_dir[0], f"posterior_*.hdf5"))) > 0:
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


        gps_file = self.production.get_timefile()

        if self.production.rundir:
            rundir = self.production.rundir
        else:
            rundir = os.path.join(
                os.path.expanduser("~"),
                self.production.event.name,
                self.production.name,
            )
            self.production.rundir = rundir

        if "job label" in self.production.meta:
            job_label = self.production.meta["job label"]
        else:
            job_label = self.production.name


        command = [
            os.path.join(config.get("pipelines", "environment"), "bin", "dingo_pipe"),
            ini,
            "--label",
            job_label,
            "--outdir",
            f"{cwd}/{self.production.rundir}",
            # "--accounting",
            # f"{self.production.meta['scheduler']['accounting group']}",
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
        return glob.glob(os.path.join(self.production.rundir, "posterior_samples", "posterior*.hdf5"))


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
                messages[
                    log.split("/")[-1]
                ] = "There was a problem opening this log file."
        return messages


    def submit_dag(self, dryrun=False):
        """
        Submit a DAG file to the condor cluster.

        Parameters
        ----------
        category : str, optional
           The category of the job.
           Defaults to "C01_offline".
        production : str
           The production name.

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
        cluster = self.run_pesummary()
        self.production.meta['job id'] = int(cluster)
        self.production.status = "processing"

    def resurrect(self):
        """
        Attempt to ressurrect a failed job.
        """
        try:
            count = self.production.meta['resurrections']
        except:
            count = 0
        if (count < 5) and (len(glob.glob(os.path.join(self.production.rundir, "submit", "*.rescue*"))) > 0):
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

        config_parser = ConfigParser.RawConfigParser()
        config_parser.read_string(file_content)

        return config_parser
