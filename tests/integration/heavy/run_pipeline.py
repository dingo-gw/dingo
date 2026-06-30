"""Run the heavy end-to-end DINGO smoke pipeline and print the sample efficiency.

Stages: waveform dataset -> ASD dataset (fixed GPS) -> train -> dingo_pipe local
injection with importance sampling. Reads the IS sample efficiency from the
result file and prints it for the pytest wrapper / architecture search to parse.
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

    workdir = args.workdir or os.path.join(HERE, "_run")
    os.makedirs(workdir, exist_ok=True)
    for name in (
        "waveform_dataset_settings.yaml",
        "asd_dataset_settings.yaml",
        "train_settings.yaml",
        "injection.ini",
    ):
        shutil.copy(os.path.join(args.config_dir, name), os.path.join(workdir, name))

    # 1. Waveform dataset (skip if already present — large file, expensive to regenerate)
    wf_path = os.path.join(workdir, "waveform_dataset.hdf5")
    if os.path.exists(wf_path):
        print(f"Reusing existing waveform_dataset.hdf5 in {workdir}", flush=True)
    else:
        run(
            ["dingo_generate_dataset", "--settings_file", "waveform_dataset_settings.yaml",
             "--num_processes", "8", "--out_file", "waveform_dataset.hdf5"],
            cwd=workdir,
        )

    # 2. ASD dataset at a fixed GPS time (GWOSC, no auth; skip if already present)
    asd_path = os.path.join(workdir, "asds_O1.hdf5")
    ts_path = os.path.join(workdir, "time_segments.pkl")
    if os.path.exists(asd_path):
        print(f"Reusing existing asds_O1.hdf5 in {workdir}", flush=True)
    else:
        run(["python", os.path.join(HERE, "make_time_segments.py"), "--out", ts_path], cwd=workdir)
        run(
            ["dingo_generate_asd_dataset", "--settings_file", "asd_dataset_settings.yaml",
             "--data_dir", workdir, "--time_segments_file", ts_path,
             "--out_name", "asds_O1.hdf5"],
            cwd=workdir,
        )

    # 3. Train (timed)
    t0 = time.time()
    run(["dingo_train", "--settings_file", "train_settings.yaml", "--train_dir", workdir],
        cwd=workdir)
    train_seconds = time.time() - t0

    # 4. Inference + importance sampling via dingo_pipe (local mode).
    # dingo_pipe generates submit/bash_<label>.sh; execute it to run all nodes locally.
    run(["dingo_pipe", "injection.ini"], cwd=workdir)
    bash_script = os.path.join(workdir, "inference_out", "submit", "bash_heavy_ci.sh")
    run(["bash", bash_script], cwd=workdir)

    eff, path = find_is_efficiency(os.path.join(workdir, "inference_out"))
    print(f"Sample efficiency = {100 * eff:.2f}%", flush=True)
    print(f"TRAIN_SECONDS = {train_seconds:.1f}", flush=True)
    print(f"RESULT_FILE = {path}", flush=True)


if __name__ == "__main__":
    main()
