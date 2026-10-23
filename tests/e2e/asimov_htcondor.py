"""Run the Asimov -> Dingo -> PESummary smoke test on a local HTCondor pool.

Usage: python tests/e2e/asimov_htcondor.py /path/to/empty/project

All test assertions are in assert_outputs(). Checkpoint construction lives
in model_fixtures.py. This script is not collected by pytest.
"""

import argparse
import getpass
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml
from model_fixtures import create_model_checkpoints, write_noise_psd

from dingo.gw.domains import MultibandedFrequencyDomain
from dingo.gw.result import Result

EXAMPLES = Path(__file__).resolve().parents[2] / "examples/asimov"
MODEL_SETTINGS = Path(__file__).with_name("model.yaml")
ANALYSES = ("transformer", "gnpe")
MINIMUM_FREQUENCY = {"H1": 24.0, "L1": 28.0}
MAXIMUM_FREQUENCY = {"H1": 96.0, "L1": 112.0}
NUM_SAMPLES = 32


def assert_outputs(event_name: str) -> None:
    """Check successful jobs, frequency propagation, and usable summary pages."""
    # Import after the final monitor call to read the finished project's ledger.
    from asimov import current_ledger

    productions = {
        production.name: production
        for production in current_ledger.get_event(event_name)[0].productions
    }
    for name in ANALYSES:
        production = productions[name]
        assert production.status == "finished", name
        run_directory = Path(production.rundir)
        dag_jobs = read_job_history(
            find_single_file(run_directory, "submit/*.dagman.log")
        )
        assert len(dag_jobs) == 1, dag_jobs
        assert dag_jobs[0].get("ExitCode") == 0, dag_jobs

        result_path = find_single_file(
            run_directory, "result/*_importance_sampling.hdf5"
        )
        result = Result(file_name=str(result_path))
        assert isinstance(result.domain, MultibandedFrequencyDomain), name
        assert len(result.samples) == NUM_SAMPLES, name
        assert np.isfinite(result.log_evidence), name
        assert result.minimum_frequency == MINIMUM_FREQUENCY, name
        assert result.maximum_frequency == MAXIMUM_FREQUENCY, name

        # Check that the requested bounds affect the likelihood, too.
        result._build_likelihood()
        frequencies = result.likelihood.data_domain()
        for detector in MINIMUM_FREQUENCY:
            inside = (frequencies >= MINIMUM_FREQUENCY[detector]) & (
                frequencies <= MAXIMUM_FREQUENCY[detector]
            )
            asd = result.likelihood.asd[detector]
            assert np.all(asd[~inside] == 1.0), (name, detector)
            assert np.all(asd[inside] < 1e-15), (name, detector)

        summary_name = f"{name}-summary"
        assert productions[summary_name].status == "finished", summary_name
        web_directory = Path("pages") / event_name / summary_name / "pesummary"
        assert (web_directory / "home.html").stat().st_size > 0, summary_name
        with h5py.File(web_directory / "samples/posterior_samples.h5", "r") as summary:
            samples = summary[summary_name]["posterior_samples"][()]
            assert len(samples) > 0, summary_name

    # Both summary jobs share the event's job log. Read only this run's jobs.
    summary_jobs = read_job_history(Path("working") / event_name / "pesummary.log")
    assert len(summary_jobs) == len(ANALYSES), summary_jobs
    assert all(job.get("ExitCode") == 0 for job in summary_jobs), summary_jobs


def run_command(*arguments, timeout: int = 180) -> None:
    """Run a command with live terminal output and fail on a nonzero exit."""
    command = [str(argument) for argument in arguments]
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, check=True, timeout=timeout)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def find_single_file(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {directory / pattern}; found {matches}")
    return matches[0]


def read_job_history(log_path: Path) -> list[dict]:
    """Read exit information from an existing HTCondor job log."""
    output = subprocess.check_output(
        ["condor_history", "-userlog", str(log_path), "-json"], text=True, timeout=30
    )
    return json.loads(output)


def wait_for_jobs(log_path: Path, timeout: int = 600) -> None:
    """Let HTCondor wait for all jobs in this log; assert_outputs checks success."""
    run_command("condor_wait", "-wait", timeout, log_path, timeout=timeout + 30)


def initialize_project(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError(f"Use an empty output directory: {directory}")
    os.chdir(directory)

    # Keep Asimov's Git identity inside the disposable project.
    git_config = directory / ".gitconfig"
    git_config.write_text(
        "[user]\n\tname = Dingo CI\n\temail = dingo-ci@example.invalid\n"
        "[init]\n\tdefaultBranch = master\n"
    )
    os.environ["GIT_CONFIG_GLOBAL"] = str(git_config)
    run_command("asimov", "init", "Dingo end-to-end test")
    for key, value in (
        ("condor/user", getpass.getuser()),
        ("condor/cache_time", "0"),
        ("pipelines/environment", sys.prefix),
        ("general/webroot", "pages"),
    ):
        run_command("asimov", "configuration", "update", key, value)


def build_project_blueprint(recovery_settings: dict) -> dict:
    """Reduce the example's resource and sampling settings for a CPU smoke test."""
    project = load_yaml(EXAMPLES / "project_dingo.yaml")
    pipeline = project["pipelines"]["dingo"]
    pipeline["scheduler"].pop("CUDA Global Memory Mb", None)
    pipeline["scheduler"].update(
        {
            "accounting group": "dingo.e2e",
            "generation pool": "local-pool",
            "request cpus sampling": 1,
            "request cpus importance sampling": 1,
            "request memory sampling": 2,
            "request memory generation": 2,
            "request memory importance sampling": 2,
            "request disk": 1,
            "n parallel": 1,
            "osg": False,
            "transfer files": True,
            "environment variables": {
                "OMP_NUM_THREADS": 1,
                "OPENBLAS_NUM_THREADS": 1,
                "MKL_NUM_THREADS": 1,
                "HDF5_USE_FILE_LOCKING": "FALSE",
            },
        }
    )
    pipeline["sampler"].update(
        {
            "device": "cpu",
            "num samples": NUM_SAMPLES,
            "batch size": 16,
            "num gnpe iterations": 2,
            "sampling seed": 42,
            "density recovery settings": recovery_settings,
        }
    )
    pipeline["plotting"] = False
    project["postprocessing"]["pesummary"] = {
        "accounting group": "dingo.e2e",
        "multiprocess": 1,
        "skymap samples": 10,
    }
    return project


def configure_event(event: dict, psd_path: str) -> None:
    """Use short Gaussian data with detector-specific lower frequency bounds."""
    event["data"].update(
        {
            "segment length": 4,
            "psd length": 4,
            "gaussian noise": True,
            "n simulation": 1,
            "generation seed": 42,
        }
    )
    event["psds"] = {detector: psd_path for detector in event["interferometers"]}
    event["likelihood"].update(
        {
            "minimum frequency": MINIMUM_FREQUENCY,
            "sample rate": 256,
        }
    )


def apply_blueprint(name: str, blueprint: dict, event_name: str | None = None) -> None:
    path = Path("blueprints") / f"{name}.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(blueprint, sort_keys=False))
    event_arguments = ["-e", event_name] if event_name else []
    run_command("asimov", "apply", "-f", path, *event_arguments)


def apply_analyses(event_name: str, checkpoints: dict[str, str]) -> None:
    """Pair each example Dingo analysis with a dependent PESummary analysis."""
    for name in ANALYSES:
        analysis = load_yaml(EXAMPLES / "analysis_dingo.yaml")
        analysis["name"] = name
        analysis["networks"].update(
            {
                "model": checkpoints[name],
                "model init": checkpoints["init"] if name == "gnpe" else None,
            }
        )
        # Analysis upper bounds complement the event's lower bounds.
        analysis["likelihood"] = {"maximum frequency": MAXIMUM_FREQUENCY}
        apply_blueprint(name, analysis, event_name)

        summary = load_yaml(EXAMPLES / "analysis_pesummary.yaml")
        summary["name"] = f"{name}-summary"
        summary["needs"] = [name]
        apply_blueprint(summary["name"], summary, event_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_directory", type=Path, help="Empty Asimov project directory"
    )
    arguments = parser.parse_args()
    initialize_project(arguments.output_directory.expanduser().resolve())

    event = load_yaml(EXAMPLES / "event_dingo.yaml")
    settings = load_yaml(MODEL_SETTINGS)
    fixture_directory = Path("fixtures").resolve()
    fixture_directory.mkdir()
    checkpoints = create_model_checkpoints(event, settings, fixture_directory)
    psd_path = write_noise_psd(settings, fixture_directory / "psd.txt")
    configure_event(event, psd_path)
    apply_blueprint("project", build_project_blueprint(settings["recovery"]))
    apply_blueprint("event", event)
    apply_analyses(event["name"], checkpoints)

    run_command("asimov", "manage", "build", "submit")
    event_directory = Path("working") / event["name"]
    for name in ANALYSES:
        wait_for_jobs(find_single_file(event_directory / name, "submit/*.dagman.log"))
    run_command("asimov", "monitor")

    # Monitoring satisfies the dependencies; build/submit starts PESummary.
    run_command("asimov", "manage", "build", "submit")
    wait_for_jobs(event_directory / "pesummary.log", timeout=900)
    run_command("asimov", "monitor")

    assert_outputs(event["name"])
    print("Passed: transformer, GNPE, frequency bounds, and PESummary.", flush=True)


if __name__ == "__main__":
    main()
