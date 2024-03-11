import numpy as np
import os

base_ini = "/Users/maxdax/Documents/Projects/GW-Inference/01_bns/scripts/dingo_pipe_sweep/GW170817.ini"
sweep_file = "/Users/maxdax/Documents/Projects/GW-Inference/01_bns/scripts/dingo_pipe_sweep/t-f.txt"
parent_dir = "/Users/maxdax/Desktop/testdir"

sweep = np.loadtxt(sweep_file)

with open(base_ini, "r") as f:
    lines = f.readlines()
    line_idx_is = [
        i for i, l in enumerate(lines) if l.startswith("importance-sampling-updates")
    ][0]
    line_idx_masking = [
        i for i, l in enumerate(lines) if l.startswith("frequency-masking")
    ][0]
    line_idx_time = [
        i for i, l in enumerate(lines) if l.startswith("post-trigger-duration")
    ][0]
    line_idx_outdir = [i for i, l in enumerate(lines) if l.startswith("outdir")][0]


for idx, (t, f_max) in enumerate(sweep):
    # make directory
    outdir = os.path.join(parent_dir, str(idx))
    os.makedirs(outdir, exist_ok=True)
    # modify dingo_pipe file
    lines[line_idx_is] = (
        f"importance-sampling-updates = {{maximum_frequency:" f" {f_max:.2f}}}\n"
    )
    lines[line_idx_masking] = f"frequency-masking = {{f_max: {f_max:.2f}}}\n"
    lines[line_idx_time] = f"post-trigger-duration = {t:.2f}\n"
    lines[line_idx_outdir] = f"outdir = {outdir}\n"
    # save dingo_pipe file
    with open(os.path.join(outdir, "settings_dingo_pipe.ini"), "w") as f:
        f.writelines(lines)
    # if idx > 3:
    #     break
