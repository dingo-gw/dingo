from os.path import join, getmtime, commonpath, relpath, getsize
import time
from datetime import timedelta
from tabulate import tabulate


def get_info(train_dir, common_path=""):
    name = relpath(train_dir, common_path)
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
        timedelta(seconds=int(time_since_last_epoch)),
        time_epoch,
        epoch,
        f"{train_loss:.2f}",
        f"{lr:.2e}",
        error,
    ]


with open("cluster_runs.txt", "r") as f:
    runs = f.read().splitlines()
common_path = commonpath(runs)

table_data = []
for train_dir in runs:
    table_data.append(get_info(train_dir, common_path))

print(f"Absolute path: {common_path}\n")
headers = [
    "train dir",
    "time since last epoch",
    "time last epoch",
    "epoch",
    "train loss",
    "learning rate",
    "error",
]
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
