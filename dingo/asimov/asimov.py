# adapted from the Asimov Bilby Pipeline interface
import configparser
import glob
import importlib.resources
import os
import re
import subprocess
import warnings
from importlib.metadata import version

from asimov.pipeline import Pipeline, PipelineException, PipelineLogger
from packaging.version import Version

# Check if using asimov < 0.7.0 (legacy version with PESummaryPipeline)
ASIMOV_LEGACY = Version(version("asimov")) < Version("0.7.0")

if ASIMOV_LEGACY:
    warnings.warn(
        "Using asimov < 0.7.0 is deprecated and support will be removed. "
        "Please upgrade to asimov >= 0.7.0.",
        DeprecationWarning,
    )


class DingoPipeline(Pipeline):
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
        from asimov import logger

        super(DingoPipeline, self).__init__(production, category)
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
           If set to true the commands will not be run, but will be printed to standard output.
           Defaults to False.

        Raises
        ------
        PipelineException
           Raised if the construction of the DAG fails.
        """
        import time

        from asimov import config

        cwd = os.getcwd()
        self.logger.info(f"Working in {cwd}")

        if self.production.event.repository:
            ini = self.production.event.repository.find_prods(
                self.production.name, self.category
            )[0]
            ini = os.path.join(cwd, ini)
        else:
            ini = f"{self.production.name}.ini"

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

    def samples(self, absolute=False):
        """
        Collect available combined samples files for PESummary.
        """
        if absolute:
            rundir = os.path.abspath(self.production.rundir)
        else:
            rundir = self.production.rundir
        self.logger.info(f"Rundir for samples: {rundir}")
        result_files = glob.glob(
            os.path.join(rundir, "result", f"*importance_sampling.hdf5")
        )
        if len(result_files) == 0:
            # Asimov also probes assets before a dependency has run.
            return []

        from dingo.gw.result import Result

        # pesummary can't presently read a result file containing MultibandedFrequencyDomain
        # This is a hotfix to create a copy of the result file with UniformFrequencyDomain
        result = Result(file_name=result_files[0])
        settings = result.settings["dataset_settings"]["domain"]
        if settings["type"] == "MultibandedFrequencyDomain":
            result.settings["dataset_settings"]["domain"] = {
                "type": "UniformFrequencyDomain",
                "delta_f": settings["delta_f_initial"],
                "f_min": settings["base_domain"]["f_min"],
                "f_max": settings["base_domain"]["f_max"],
            }
            filename, ext = os.path.splitext(result_files[0])
            result_files[0] = f"{filename}_pesummary{ext}"
            result.to_file(result_files[0])

        return result_files

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

    def _fmin_max_are_compatible(self, prod_meta, net_meta, reasons):
        """
        Check if the network min/max frequencies are compatible with the data.

        Take possible domain updates and random frequency masking into account.
        """
        f_min = prod_meta["likelihood"]["minimum frequency"].values()
        f_max = prod_meta["likelihood"]["maximum frequency"].values()

        # Get network values
        domain = net_meta["dataset_settings"]["domain"]["base_domain"]
        domain_update = net_meta["train_settings"]["data"].get("domain_update", {})

        # Network f_min and f_max (with possible updates)
        net_f_min = domain_update.get("f_min", domain["f_min"])
        net_f_max = domain_update.get("f_max", domain["f_max"])

        # Random strain cropping bounds (if they exist)
        net_f_min_upper = (
            net_meta["train_settings"]["data"]["random_strain_cropping"].get(
                "f_min_upper", None
            )
            if "random_strain_cropping" in net_meta["train_settings"]["data"]
            else None
        )
        net_f_max_lower = (
            net_meta["train_settings"]["data"]["random_strain_cropping"].get(
                "f_max_lower", None
            )
            if "random_strain_cropping" in net_meta["train_settings"]["data"]
            else None
        )

        # Check f_min
        if net_f_min_upper is None:
            f_min_match = min(f_min) == max(f_min) == net_f_min
        else:
            f_min_match = (min(f_min) >= net_f_min) and (max(f_min) <= net_f_min_upper)

        # Check f_max
        if net_f_max_lower is None:
            f_max_match = min(f_max) == max(f_max) == net_f_max
        else:
            f_max_match = (min(f_max) >= net_f_max_lower) and (max(f_max) <= net_f_max)

        if not f_min_match:
            upper = net_f_min if net_f_min_upper is None else net_f_min_upper
            reasons.append(
                f"minimum frequencies {prod_meta['likelihood']['minimum frequency']} "
                f"outside allowed range [{net_f_min}, {upper}] Hz"
            )
        if not f_max_match:
            lower = net_f_max if net_f_max_lower is None else net_f_max_lower
            reasons.append(
                f"maximum frequencies {prod_meta['likelihood']['maximum frequency']} "
                f"outside allowed range [{lower}, {net_f_max}] Hz"
            )
        return f_min_match and f_max_match

    @staticmethod
    def _is_transformer(metadata):
        model = metadata["train_settings"].get("model", {})
        return (model.get("embedding_type") or "").lower() == "transformer"

    @staticmethod
    def _net_max_luminosity_distance(metadata):
        prior = metadata["train_settings"]["data"]["extrinsic_prior"][
            "luminosity_distance"
        ]
        match = re.search(r"\bmaximum\s*=\s*([^,)]+)", prior)
        if match is None:
            raise ValueError(f"Missing luminosity-distance prior maximum: {prior!r}")
        return float(match.group(1))

    def network_is_compatible(self, prod_meta, net_meta, reasons=None):
        """Check compatibility, optionally appending rejection reasons to a list."""
        reasons = [] if reasons is None else reasons
        duration = prod_meta["data"]["segment length"]
        ifos = prod_meta["interferometers"]

        if self._is_transformer(net_meta):
            from copy import deepcopy

            from dingo.core.utils.backward_compatibility import \
                update_data_config
            from dingo.gw.domains import build_domain_from_model_metadata
            from dingo.gw.frequency_updates import (check_detector_update,
                                                    check_frequency_updates)

            net_meta = deepcopy(net_meta)
            update_data_config(net_meta)
            domain = build_domain_from_model_metadata(net_meta, base=True)
            if round(domain.duration) != duration:
                reasons.append(
                    f"duration {domain.duration} s does not match requested {duration} s"
                )

            bounds = []
            for key in ("minimum frequency", "maximum frequency"):
                bound = prod_meta.get("likelihood", {}).get(key)
                if isinstance(bound, dict):
                    bound = {ifo: bound[ifo] for ifo in ifos if ifo in bound}
                bounds.append(bound)
            try:
                check_detector_update(net_meta, ifos)
                check_frequency_updates(net_meta, *bounds, detectors=ifos)
            except ValueError as err:
                reasons.append(str(err))
            return not reasons

        net_duration = round(
            1 / net_meta["dataset_settings"]["domain"]["base_domain"]["delta_f"]
        )
        net_ifos = net_meta["train_settings"]["data"]["detectors"]

        if net_duration != duration:
            reasons.append(
                f"duration {net_duration} s does not match requested {duration} s"
            )
        if sorted(net_ifos) != sorted(ifos):
            reasons.append(f"detectors {net_ifos} do not match requested {ifos}")
        self._fmin_max_are_compatible(prod_meta, net_meta, reasons)
        return not reasons

    def _filter_compatible_networks(self, prod_meta):
        """Load available networks and return those compatible with the production.

        Returns a list of (network_config, net_meta) tuples.
        """
        import torch

        compatible_networks = []
        rejections = []
        for networks in prod_meta["available networks"]:
            try:
                f = torch.load(
                    networks["model"], map_location="meta", weights_only=False
                )
                net_meta = f["metadata"]
            except FileNotFoundError as err:
                msg = f"Could not find network: '{networks['model']}'.."
                raise PipelineException(msg, production=self.production.name) from err
            except KeyError as err:
                msg = "Could not load metadata from network: '{networks['model']}'.."
                raise PipelineException(msg, production=self.production.name) from err

            reasons = []
            if self.network_is_compatible(prod_meta, net_meta, reasons):
                if self._is_transformer(net_meta):
                    networks = {**networks, "model init": None}
                compatible_networks.append((networks, net_meta))
            else:
                rejection = f"  {networks['model']}: " + "; ".join(reasons)
                rejections.append(rejection)
                self.logger.info(rejection)

        if not compatible_networks:
            raise PipelineException(
                "No compatible DINGO network found for this production.\n"
                + "\n".join(rejections),
                production=self.production.name,
            )

        return compatible_networks

    def _select_preferred_network(self, prod_meta, compatible_networks):
        """Select the best network from compatible candidates by luminosity distance.

        Prefers the network with the tightest max luminosity distance that still
        covers the production prior. Falls back to the largest if none fully cover it.

        Returns the selected network config.
        """
        if len({self._is_transformer(meta) for _, meta in compatible_networks}) > 1:
            raise PipelineException(
                "Both transformer and GNPE/legacy networks match this production. "
                "Set 'networks' explicitly or restrict 'available networks' "
                "to one architecture.",
                production=self.production.name,
            )
        if len(compatible_networks) == 1:
            return compatible_networks[0][0]

        distances = [
            self._net_max_luminosity_distance(x[1]) for x in compatible_networks
        ]
        if len(distances) > len(set(distances)):
            raise PipelineException(
                "Multiple DINGO networks match this production..",
                production=self.production.name,
            )

        prod_max_luminosity_distance = prod_meta["priors"]["luminosity distance"][
            "maximum"
        ]
        compatible_networks = sorted(
            compatible_networks, key=lambda x: self._net_max_luminosity_distance(x[1])
        )
        for networks, net_meta in compatible_networks:
            net_max_luminosity_distance = self._net_max_luminosity_distance(net_meta)
            if net_max_luminosity_distance >= prod_max_luminosity_distance:
                return networks
        return compatible_networks[-1][0]

    def before_config(self, dryrun=False):
        """Parse available networks before building the ini."""
        super().before_config(dryrun=dryrun)
        meta = self.production.meta
        if "networks" in meta:  # Set networks as override
            return
        if "available networks" not in meta:
            raise ValueError(
                "Neither 'networks' nor 'available networks' found in production "
                "metadata. Set one of these in the analysis or configuration blueprint."
            )

        compatible_networks = self._filter_compatible_networks(meta)
        meta["networks"] = self._select_preferred_network(
            meta, compatible_networks
        ).copy()

        # Update the ledger
        if not dryrun:
            self.production.meta.update(meta)

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
        super().after_completion()

        # Only run PESummary if it was requested
        if not "pesummary" in self.production.meta.get("postprocessing", {}):
            self.logger.info("No pesummary requested, skipping..")
            return

        try:
            from asimov_pesummary import PESummary
        except ImportError:
            self.logger.warning(
                "asimov-pesummary not available, skipping post-processing"
            )
            return

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


class DingoLegacy(DingoPipeline):
    """
    The Dingo Pipeline for asimov < 0.7.0 with legacy PESummaryPipeline support.
    """

    def after_completion(self):
        """
        Legacy implementation using PESummaryPipeline for post-processing.
        """
        from asimov.pipeline import PESummaryPipeline

        post_pipeline = PESummaryPipeline(production=self.production)
        self.logger.info("Job has completed. Running PE Summary.")
        cluster = post_pipeline.submit_dag()
        self.production.meta["job id"] = int(cluster)
        self.production.status = "processing"
        self.production.event.update_data()

    def detect_completion_processing(self):
        # no post processing currently performed
        return True

    def before_submit(self):
        """Pre-submit hook."""
        self.logger.info("Running the before_submit hook")
        pass


class Dingo:
    """
    Factory class that returns the appropriate Dingo pipeline implementation.

    Uses __new__ to transparently return either DingoPipeline or DingoLegacy
    instances based on the asimov version.

    Parameters
    ----------
    production : :class:`asimov.Production`
       The production object.
    category : str, optional
        The category of the job.
        Defaults to "C01_offline".
    """

    def __new__(cls, *args, **kwargs):
        """
        Return an instance of the appropriate pipeline class.

        If asimov < 0.7.0, returns a DingoLegacy instance.
        Otherwise, returns a DingoPipeline instance.
        """
        if ASIMOV_LEGACY:
            dingo_cls = DingoLegacy
        else:
            dingo_cls = DingoPipeline
        return dingo_cls(*args, **kwargs)
