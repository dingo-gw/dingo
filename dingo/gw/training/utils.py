import torch
import argparse
import yaml


def append_stage():

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--stage_settings_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    args = parser.parse_args()

    d = torch.load(args.checkpoint)
    num_stages = len(
        [
            k
            for k in d["metadata"]["train_settings"]["training"]
            if k.startswith("stage_")
        ]
    )
    print(f"Checkpoint training plan consists of {num_stages} stages.")

    with open(args.stage_settings_file, "r") as f:
        new_stage = yaml.safe_load(f)

    d["metadata"]["train_settings"]["training"][f"stage_{num_stages}"] = new_stage
    print("Summary of new training plan:")
    print(
        yaml.dump(
            d["metadata"]["train_settings"]["training"],
            default_flow_style=False,
            sort_keys=False,
        )
    )

    torch.save(d, args.out_file)
