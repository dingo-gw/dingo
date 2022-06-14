import os
from os.path import dirname, basename, join
import argparse
import pandas as pd
import numpy as np
import pdb

parser = argparse.ArgumentParser(description="Merge dingo smaple files")
parser.add_argument("--prefix", type=str, required=True,
                    help="Prefix for sample files. Includes dirname + local prefix.")
parser.add_argument("--log_probs_target_min", type=float, default=None,
                    help="Save only samples with log_probs_target >= log_probs_target_min.")
args = parser.parse_args()

directory = dirname(args.prefix)
prefix = basename(args.prefix)
outname = f"merged_{prefix}.pkl"

# valid filenames, starting with prefix
files = sorted([f for f in os.listdir(directory) if f.startswith(prefix)])

# load frames and concatenate them
print(f"Merging {len(files)} frames with dingo samples.")
frames = []
for f in files:
    frames.append(pd.read_pickle(join(directory, f)))
new_frame = pd.concat(frames)

# add metadata to new frame
new_frame.attrs = frames[0].attrs

# optionally filter samples
if args.log_probs_target_min is not None:
    log_probs_target_offset = np.max(new_frame.log_probs_target)
    n0 = len(new_frame)
    new_frame = new_frame[new_frame.log_probs_target - log_probs_target_offset > args.log_probs_target_min]
    n1 = len(new_frame)
    new_frame.attrs["filter"] = {
        "log_probs_target_min": args.log_probs_target_min,
        "log_probs_target_offset": log_probs_target_offset,
        "num_initial_samples": n0,
        "num_filtered_samples": n1,
    }
    print(
        f"Filtered samples for log_prob > {args.log_probs_target_min}.",
        f"Kept {n1} out of {n0} samples ({n1/n0 * 100:.1f}%)."
    )
    # drop log probs
    new_frame.drop(columns="log_probs_target", inplace=True)

# save new frame
new_frame.to_pickle(join(directory, outname))
print(f"Done. New frame contains {len(new_frame)} samples.")
print(f"Output save to {join(directory, outname)}.")
