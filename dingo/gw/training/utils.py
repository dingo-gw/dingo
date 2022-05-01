import numpy as np
import torch
import argparse
import yaml


def append_stage():

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--stage_settings_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--replace", type=int)
    args = parser.parse_args()

    # Typically training is done on the GPU, so the model could be saved on a GPU
    # device. Since this routine may be run on a CPU machine, allow for a remap of the
    # torch tensors.
    if torch.cuda.is_available():
        d = torch.load(args.checkpoint)
    else:
        d = torch.load(args.checkpoint, map_location=torch.device("cpu"))

    stages = [
        v
        for k, v in d["metadata"]["train_settings"]["training"].items()
        if k.startswith("stage_")
    ]
    num_stages = len(stages)
    print(f"Checkpoint training plan consists of {num_stages} stages.")

    with open(args.stage_settings_file, "r") as f:
        new_stage = yaml.safe_load(f)

    if args.replace is not None:
        if args.replace < 0 or args.replace >= num_stages:
            raise ValueError(f"Invalid argument replace={args.replace}. Valid values "
                             f"are {list(range(num_stages))}.")
        current_epoch = d["epoch"]
        stage_epoch = np.sum([s["epochs"] for s in stages[: args.replace]])
        if current_epoch > stage_epoch:
            print(
                f"WARNING: Modification to training plan changes a training stage "
                f"that has already started. Current model epoch is {current_epoch}. "
                f"Proceed at your own risk!"
            )
        print(f"Replacing planned stage {args.replace} with new stage.")
        new_stage_number = args.replace
    else:
        print(f"Appending new stage to training plan.")
        new_stage_number = num_stages

    d["metadata"]["train_settings"]["training"][f"stage_{new_stage_number}"] = new_stage
    print("Summary of new training plan:")
    print(
        yaml.dump(
            d["metadata"]["train_settings"]["training"],
            default_flow_style=False,
            sort_keys=False,
        )
    )

    torch.save(d, args.out_file)
