import os
import argparse
from typing import Dict
from pycondor import Job, Dagman
import yaml
import copy

# Fixed file names
svd_fn = "svd.hdf5"
settings_part_fn = "settings_part.yaml"
settings_svd_part_fn = "settings_svd_part.yaml"
svd_dataset_part_prefix = "svd_dataset_part_"
svd_dataset_fn = "svd_dataset.hdf5"
dataset_part_prefix = "dataset_part_"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--settings_file",
        type=str,
        required=True,
        help="YAML file containing database settings",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="waveform_dataset.hdf5",
        help="Name of file for storing dataset.",
    )
    parser.add_argument(
        "--num_jobs",
        type=int,
        required=True,
        help="Number of condor jobs over which to split work.",
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default="temp",
        help="Directory for storing temporary files.",
    )
    parser.add_argument(
        "--env_path",
        type=str,
        required=True,
        help="Absolute path to the dingo Python environment. "
        'We will execute scripts in "env_path/bin/".',
    )

    # condor arguments
    parser.add_argument("--request_cpus", type=int, default=None, help="CPUs per job.")
    parser.add_argument(
        "--request_memory", type=int, default=None, help="Memory per job."
    )
    parser.add_argument(
        "--request_memory_high", type=int, default=None, help="Memory per job, "
                                                              "used only for "
                                                              "aggregating data and "
                                                              "building SVD."
    )
    parser.add_argument("--error", type=str, default="condor/error")
    parser.add_argument("--output", type=str, default="condor/output")
    parser.add_argument("--log", type=str, default="condor/log")
    parser.add_argument("--submit", type=str, default="condor/submit")

    return parser.parse_args()


def modulus_check(a: int, b: int, a_label: str, b_label: str):
    """Raise error if a % b != 0."""
    if a % b != 0:
        raise ValueError(
            f"Expected {a_label} mod {b_label} to be zero. "
            f"But got {a} mod {b} = {a % b}."
        )


def create_args_string(args_dict: Dict):
    """Generate argument string from dictionary of argument names and arguments."""
    return "".join([f"--{k} {v} " for k, v in args_dict.items()])


def configure_runs(settings, num_jobs, temp_dir):
    """Prepare and save settings .yaml files for generating subsets of the dataset.
    Generally this will produce two .yaml files, one for generating the main dataset,
    one for the SVD training.

    Parameters
    ----------
    settings : dict
        Settings for full dataset configuration.
    num_jobs : int
        Number of jobs over which to split the run.
    temp_dir : str
        Name of (temporary) directory in which to place temporary output files.
    """

    settings_part = copy.deepcopy(settings)
    num_samples = settings["num_samples"]
    modulus_check(num_samples, num_jobs, "num_samples", "num_jobs")
    settings_part["num_samples"] = num_samples // num_jobs

    if "compression" in settings:
        if "svd" in settings["compression"]:

            # Configure a set of runs to generate waveforms from which to construct the
            # SVD. These should be uncompressed waveforms.
            settings_svd_part = copy.deepcopy(settings)
            num_samples = settings["compression"]["svd"]["num_training_samples"]
            num_samples += settings["compression"]["svd"].get(
                "num_validation_samples", 0
            )
            modulus_check(
                num_samples,
                num_jobs,
                "(number of SVD training + validation samples)",
                "num_jobs",
            )
            settings_svd_part["num_samples"] = num_samples // num_jobs
            del settings_svd_part["compression"]["svd"]
            with open(os.path.join(temp_dir, settings_svd_part_fn), "w") as f:
                yaml.dump(
                    settings_svd_part, f, default_flow_style=False, sort_keys=False
                )

            # Set the runs for the main dataset to use the saved SVD basis (assume it was
            # produced already, based on the above).
            settings_part["compression"]["svd"] = {
                "file": os.path.join(temp_dir, svd_fn)
            }

    with open(os.path.join(temp_dir, settings_part_fn), "w") as f:
        yaml.dump(settings_part, f, default_flow_style=False, sort_keys=False)


def create_dag(args, settings):
    """
    Create a Condor DAG from command line arguments to carry out the five steps in the
    workflow.

    """
    kwargs = {
        "request_cpus": args.request_cpus,
        "request_memory": args.request_memory,
        "submit": args.submit,
        "error": args.error,
        "output": args.output,
        "log": args.log,
        "getenv": True,
    }
    kwargs_high_memory = kwargs.copy()
    if args.request_memory_high is not None:
        kwargs_high_memory["request_memory"] = args.request_memory_high

    # scripts are installed in the env's bin directory
    path = os.path.join(args.env_path, "bin")
    temp_dir = args.temp_dir

    # DAG ---------------------------------------------------------------------
    dagman = Dagman(name="dingo_generate_dataset_dagman", submit=args.submit)

    # 1. Prepare SVD basis ----------------------------------------------------
    # This is only needed if we are using SVD compression
    if "compression" in settings:
        if "svd" in settings["compression"]:

            # --- (a) Generate dataset for SVD training. Split this over multiple jobs.
            executable = os.path.join(path, "dingo_generate_dataset")
            args_dict = {
                "settings_file": os.path.join(temp_dir, settings_svd_part_fn),
                "num_processes": args.request_cpus,
                "out_file": os.path.join(
                    temp_dir, svd_dataset_part_prefix + "$(Process).hdf5"
                ),
            }
            args_str = create_args_string(args_dict)
            generate_svd_dataset_part = Job(
                name="generate_svd_dataset_part",
                executable=executable,
                queue=args.num_jobs,
                dag=dagman,
                arguments=args_str,
                **kwargs,
            )

            # --- (b) Consolidate dataset.
            executable = os.path.join(path, "dingo_merge_datasets")
            args_dict = {
                "prefix": os.path.join(temp_dir, svd_dataset_part_prefix),
                "num_parts": args.num_jobs,
                "out_file": os.path.join(temp_dir, svd_dataset_fn),
            }
            args_str = create_args_string(args_dict)
            consolidate_svd_dataset = Job(
                name="consolidate_svd_dataset",
                executable=executable,
                dag=dagman,
                arguments=args_str,
                **kwargs_high_memory,
            )
            consolidate_svd_dataset.add_parent(generate_svd_dataset_part)

            # --- (c) Build SVD basis
            executable = os.path.join(path, "dingo_build_svd")
            args_dict = {
                "dataset_file": os.path.join(temp_dir, svd_dataset_fn),
                "size": settings["compression"]["svd"]["size"],
                "out_file": os.path.join(temp_dir, svd_fn),
                "num_train": settings["compression"]["svd"]["num_training_samples"],
            }
            args_str = create_args_string(args_dict)
            build_svd = Job(
                name="build_svd",
                executable=executable,
                dag=dagman,
                arguments=args_str,
                **kwargs_high_memory,
            )
            build_svd.add_parent(consolidate_svd_dataset)

    # 2. Prepare main dataset -------------------------------------------------

    # --- (a) Generate dataset. Split over multiple jobs.
    executable = os.path.join(path, "dingo_generate_dataset")
    args_dict = {
        "settings_file": os.path.join(temp_dir, settings_part_fn),
        "num_processes": args.request_cpus,
        "out_file": os.path.join(temp_dir, dataset_part_prefix + "$(Process).hdf5"),
    }
    args_str = create_args_string(args_dict)
    generate_dataset_part = Job(
        name="generate_dataset_part",
        executable=executable,
        queue=args.num_jobs,
        dag=dagman,
        arguments=args_str,
        **kwargs,
    )
    if "compression" in settings:
        if "svd" in settings["compression"]:
            generate_dataset_part.add_parent(build_svd)

    # --- (b) Consolidate dataset
    executable = os.path.join(path, "dingo_merge_datasets")
    args_dict = {
        "prefix": os.path.join(temp_dir, dataset_part_prefix),
        "num_parts": args.num_jobs,
        "out_file": args.out_file,
        "settings_file": args.settings_file,
    }
    args_str = create_args_string(args_dict)
    consolidate_dataset = Job(
        name="consolidate_dataset",
        executable=executable,
        dag=dagman,
        arguments=args_str,
        **kwargs_high_memory,
    )
    consolidate_dataset.add_parent(generate_dataset_part)

    return dagman


def main():
    args = parse_args()

    # Load settings
    with open(args.settings_file, "r") as f:
        settings = yaml.safe_load(f)

    # create temporary directory
    if not os.path.exists(args.temp_dir):
        os.mkdir(args.temp_dir)

    # Set up component .yaml files
    configure_runs(settings, args.num_jobs, args.temp_dir)

    dagman = create_dag(args, settings)

    try:
        dagman.visualize("waveform_dataset_generation_workflow.png")
    except:
        pass

    dagman.build()
    print(f"DAG submission file written to {args.submit}.")


if __name__ == "__main__":
    main()
