from os.path import join, getmtime, commonpath, relpath, getsize
import time
from datetime import timedelta
from tabulate import tabulate
import subprocess
import argparse

parser = argparse.ArgumentParser(description="Monitor cluster runs for dingo")
parser.add_argument("--condor_name", type=str, default=None,
                    help="If set, job IDs are fetched via condor_q args.condor_name.")
args = parser.parse_args()


def is_condor_process(row):
    try:
        float(row[0])
        return True
    except:
        return False


def get_condor_info(train_dir, condor_processes=None):
    if condor_processes is None:
        return ""
    else:
        condor_process = [pr for pr in condor_processes if train_dir == pr[10]]
        if len(condor_process) == 0:
            return ""
        elif len(condor_process) > 1:
            IDs = ", ".join([pr[0] for pr in condor_process])
            print(f"Warning: multiple processes for run {train_dir}: IDs {IDs}.")
        condor_process = condor_process[0]
        return " ".join((condor_process[4], condor_process[0], condor_process[5]))


def get_info(train_dir, common_path="", condor_processes=None):
    name = relpath(train_dir, common_path)
    # get condor info
    condor_info = get_condor_info(train_dir, condor_processes)
    # parse history.txt for epoch, train_loss, lr and time_since_last epoch
    try:
        with open(join(train_dir, "history.txt"), "r") as f:
            last_line = f.readlines()[-1].strip("\n").split("\t")
        epoch = int(last_line[0])
        train_loss = float(last_line[1])
        lr = float(last_line[-1])
        time_since_last_epoch = time.time() - getmtime(join(train_dir, "history.txt"))
    except FileNotFoundError:
        epoch = 0
        train_loss = ""
        lr = ""
        time_since_last_epoch = ""
    # parse info.err to check for errors
    try:
        if getsize(join(train_dir, "info.err")) > 0:
            error = "X"
        else:
            error = ""
    except FileNotFoundError:
        error = ""
    # parse info.out to get train time of last epoch
    try:
        time_epoch = ""
        with open(join(train_dir, "info.out"), "r") as f:
            lines = f.readlines()
        for idx, line in enumerate(reversed(lines)):
            if line.startswith("Start testing epoch"):
                mins, secs = lines[-idx-2].strip("Done. This took min.\n").split(":")
                time_epoch = timedelta(minutes=int(mins), seconds=int(secs))
                break
    except:
        time_epoch = ""

    return [
        name,
        condor_info,
        timedelta(seconds=int(time_since_last_epoch)),
        time_epoch,
        epoch,
        f"{train_loss:.2f}",
        f"{lr:.2e}",
        error,
    ]


with open("cluster_runs.txt", "r") as f:
    runs = sorted(f.read().splitlines())
common_path = commonpath(runs)
print(f"Absolute path: {common_path}\n")

if args.condor_name is not None:
    condor_processes = subprocess.run(
        ["condor_q", args.condor_name], stdout=subprocess.PIPE
    ).stdout.decode("utf-8").split("\n")
    condor_processes = [pr.split() for pr in condor_processes]
    condor_processes = [
        pr for pr in condor_processes if is_condor_process(pr)
    ]
else:
    condor_processes = None


table_data = []
for train_dir in runs:
    table_data.append(get_info(train_dir, common_path, condor_processes))

headers = [
    "train dir",
    "condor_info",
    "time since last epoch",
    "time last epoch",
    "epoch",
    "train loss",
    "learning rate",
    "error",
]
print("\n")
print(tabulate(table_data, headers=headers, stralign="right", colalign=("left",)))

# def commonprefix(m):
#     "Given list of pathnames, find longest common prefix"
#     if not m:
#         return ""
#     s1 = min(m)
#     s2 = max(m)
#     for i, c in enumerate(s1):
#         if c != s2[i]:
#             return s1[:i]
#         return s1
