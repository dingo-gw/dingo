import os
import pickle
from os.path import join
from typing import Dict

import numpy as np
import yaml
from pycondor import Job, Dagman


def create_args_string(args_dict: Dict):
    """Generate argument string from dictionary of argument names and arguments."""
    return "".join([f"--{k} {v} " for k, v in args_dict.items()])


def split_time_segments(time_segments, condor_dir, num_jobs):
    """
    Split up all time segments used for estimating PSDs into num_jobs-many
    segments and save them into a condor directory

    Parameters
    ----------
    time_segments : dict
        contains all time segments used for estimating PSDs
    condor_dir : str
        path to a directory where condr-related files are stored
    num_jobs : int
        number of jobs that should be used per detector to parallelize the PSD estimation
    Returns
    -------
    List of paths where the files including the subsets of all time segments are stored
    """
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


def create_dag(data_dir, settings_file, time_segments, out_name):
    """
    Create a Condor DAG to (a) download, estimate,
    individual PSDs and (b) merge them into one dataset

    Parameters
    ----------
    data_dir : str
        Path to the directory where the PSD dataset will be stored
    settings_file : str
        Settings : Path to settings file relevant for PSD generation
    time_segments : dict
        contains all time segments used for estimating PSDs
    out_name : str
        path where the resulting ASD dataset should be stored

    Returns
    -------
        Condor DAG
    """

    with open(settings_file, "r") as f:
        settings = yaml.safe_load(f)

    condor_dir = join(data_dir, "tmp", "condor")
    kwargs = {
        "request_cpus": settings["condor"]["num_cpus"],
        "request_memory": settings["condor"]["memory_cpus"],
        "submit": os.path.join(condor_dir, "submit"),
        "error": os.path.join(condor_dir, "error"),
        "output": os.path.join(condor_dir, "output"),
        "log": os.path.join(condor_dir, "logging"),
        "getenv": True,
    }
    # scripts are installed in the env's bin directory
    env_path = os.path.join(settings["condor"]["env_path"], "bin")
    executable = os.path.join(env_path, "dingo_estimate_psds")
    num_jobs = settings["condor"]["num_jobs"]

    time_segment_path_list = split_time_segments(time_segments, condor_dir, num_jobs)

    # DAG ---------------------------------------------------------------------
    dagman = Dagman(
        name="dingo_generate_ASD_dataset_dagman", submit=join(condor_dir, "dag")
    )

    # --- (a) Download and estimate PSDs corresponding to each time segment
    job_list = []
    for seg_path in time_segment_path_list:

        args_dict = {
            "data_dir": data_dir,
            "settings_file": settings_file,
            "time_segments_file": seg_path,
        }
        args_str = create_args_string(args_dict)

        psd_job = Job(
            name="psd_estimation",
            executable=executable,
            dag=dagman,
            arguments=args_str,
            **kwargs,
        )

        job_list.append(psd_job)

    # --- (b) Consolidate dataset
    executable = os.path.join(env_path, "dingo_merge_asd_datasets")
    args_dict = {
        "data_dir": data_dir,
        "settings_file": settings_file,
        "time_segments_file": join(
            data_dir, "tmp", settings["dataset_settings"]["observing_run"], "psd_time_segments.pkl"
        )
    }
    if out_name is not None:
        args_dict["out_name"] = out_name

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