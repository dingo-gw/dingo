"""Run the heavy end-to-end DINGO smoke pipeline and print the sample efficiency.

Stages: waveform dataset -> ASD dataset (fixed GPS) -> train -> dingo_pipe local
injection with importance sampling. Reads the IS sample efficiency from the
result file and prints it for the pytest wrapper to parse.
"""
import argparse
import glob
import os
import shutil
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, cwd):
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def timed_stage(name, fn):
    """Run fn(), return elapsed seconds, and return the result."""
    t0 = time.time()
    result = fn()
    elapsed = time.time() - t0
    return elapsed, result


def find_is_efficiency(outdir):
    """Locate the importance-sampled result and return sample_efficiency (0-1)."""
    from dingo.gw.result import Result

    candidates = sorted(
        glob.glob(os.path.join(outdir, "**", "*.hdf5"), recursive=True)
    )
    for path in candidates:
        try:
            result = Result(file_name=path)
        except Exception:
            continue
        eff = getattr(result, "sample_efficiency", None)
        if eff is not None:
            return eff, path
    raise RuntimeError(
        f"No importance-sampled result with sample_efficiency found under {outdir}. "
        f"Scanned: {candidates}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=os.path.join(HERE, "config"))
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args()

    # Default to /tmp so the script works inside a read-only SIF container.
    # /tmp is always writable (apptainer mounts a tmpfs there).
    # Always start from a clean workdir: apptainer bind-mounts the host /tmp,
    # so stale files from a previous (possibly failed) run may be present.
    workdir = args.workdir or os.path.join("/tmp", "dingo_run")
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir, exist_ok=True)
    for name in (
        "waveform_dataset_settings.yaml",
        "asd_dataset_settings.yaml",
        "train_settings.yaml",
        "injection.ini",
    ):
        shutil.copy(os.path.join(args.config_dir, name), os.path.join(workdir, name))

    stage_times = {}
    pipeline_start = time.time()

    # 1. Waveform dataset
    def _stage_waveform():
        run(
            ["dingo_generate_dataset", "--settings_file", "waveform_dataset_settings.yaml",
             "--num_processes", "8", "--out_file", "waveform_dataset.hdf5"],
            cwd=workdir,
        )

    stage_times["waveform_dataset"], _ = timed_stage("waveform_dataset", _stage_waveform)

    # 2. ASD dataset at a fixed GPS time (GWOSC, no auth)
    def _stage_asd():
        ts_path = os.path.join(workdir, "time_segments.pkl")
        run(["python", os.path.join(HERE, "make_time_segments.py"), "--out", ts_path], cwd=workdir)
        run(
            ["dingo_generate_asd_dataset", "--settings_file", "asd_dataset_settings.yaml",
             "--data_dir", workdir, "--time_segments_file", ts_path,
             "--out_name", "asds_O1.hdf5"],
            cwd=workdir,
        )

    stage_times["asd_dataset"], _ = timed_stage("asd_dataset", _stage_asd)

    # 3. Train
    def _stage_train():
        run(["dingo_train", "--settings_file", "train_settings.yaml", "--train_dir", workdir],
            cwd=workdir)

    stage_times["train"], _ = timed_stage("train", _stage_train)

    # 4. Inference + importance sampling via dingo_pipe (local mode).
    # dingo_pipe generates submit/bash_<label>.sh; execute it to run all nodes locally.
    def _stage_inference():
        run(["dingo_pipe", "injection.ini"], cwd=workdir)
        bash_script = os.path.join(workdir, "inference_out", "submit", "bash_heavy_ci.sh")
        run(["bash", bash_script], cwd=workdir)

    stage_times["inference"], _ = timed_stage("inference", _stage_inference)

    total_seconds = time.time() - pipeline_start

    eff, path = find_is_efficiency(os.path.join(workdir, "inference_out"))
    print(f"Sample efficiency = {100 * eff:.2f}%", flush=True)
    print(f"RESULT_FILE = {path}", flush=True)
    for stage_name, secs in stage_times.items():
        print(f"STAGE_SECONDS {stage_name} = {secs:.1f}", flush=True)
    print(f"TOTAL_SECONDS = {total_seconds:.1f}", flush=True)


if __name__ == "__main__":
    main()
