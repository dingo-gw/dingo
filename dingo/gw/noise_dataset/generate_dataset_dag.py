import os
import pickle
from os.path import join
from typing import Dict

import numpy as np
import yaml
from pycondor import Job, Dagman

from dingo.gw.noise_dataset.estimation import get_time_segments


def create_args_string(args_dict: Dict):
    """Generate argument string from dictionary of argument names and arguments."""
    return "".join([f"--{k} {v} " for k, v in args_dict.items()])


def split_time_segments(time_segments, condor_dir, num_jobs):

    time_segments_path_list = []
    segment_path = join(condor_dir, "time_segments")
    os.makedirs(segment_path, exist_ok=True)

    for det in time_segments.keys():

        segments = np.array_split(time_segments[det], num_jobs)

        for i, segs in enumerate(segments):

            save_dict = {det: segs}
            filename = join(segment_path, f"seg_{det}_{i:05d}.pkl")
            with open(filename, "wb") as f:
                pickle.dump(save_dict, f)
            time_segments_path_list.append(filename)

    return time_segments_path_list


def create_dag(data_dir, settings_file, time_segments):
    """
    Create a Condor DAG from command line arguments to carry out the five steps in the
    workflow.

    """

    with open(settings_file, "r") as f:
        settings = yaml.safe_load(f)

    condor_dir = join(data_dir, "tmp", "condor")
    kwargs = {
        "request_cpus": settings["local"]["condor"]["num_cpus"],
        "request_memory": settings["local"]["condor"]["memory_cpus"],
        "submit": os.path.join(condor_dir, "submit"),
        "error": os.path.join(condor_dir, "error"),
        "output": os.path.join(condor_dir, "output"),
        "log": os.path.join(condor_dir, "logging"),
        "getenv": True,
    }
    # scripts are installed in the env's bin directory
    env_path = os.path.join(settings["local"]["env_path"], "bin")

    # DAG ---------------------------------------------------------------------
    dagman = Dagman(name="dingo_generate_ASD_dataset_dagman", submit=join(condor_dir, "dag"))

    executable = os.path.join(env_path, "dingo_estimate_psds")
    num_jobs = settings["local"]["condor"]["num_jobs"]

    time_segments = get_time_segments(data_dir, settings["dataset_settings"])
    time_segment_path_list = split_time_segments(time_segments, condor_dir, num_jobs)

    # --- (a) Download and estimate PSDs corresponding to each time segment
    job_list = []
    for i, seg in enumerate(time_segment_path_list):

        args_dict = {
            "data_dir": data_dir,
            "settings_file": settings_file,
            "time_segments_file": seg
        }
        args_str = create_args_string(args_dict)

        psd_job = Job(
            name=f"psd_estimation_{i}",
            executable=executable,
            dag=dagman,
            arguments=args_str,
            **kwargs,
        )

        job_list.append(psd_job)

    # --- (b) Consolidate dataset
    executable = os.path.join(env_path, "dingo_merge_ASD_datasets")
    args_dict = {
        "data_dir": data_dir,
        "settings_file": settings_file
    }
    args_str = create_args_string(args_dict)
    consolidate_dataset = Job(
        name="consolidate_dataset",
        executable=executable,
        dag=dagman,
        arguments=args_str,
        **kwargs,
    )
    for job in job_list:
        consolidate_dataset.add_parent(job)

    return dagman