import torch
import argparse
import yaml


def append_stage():

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--stage_settings_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    args = parser.parse_args()

    # Typically training is done on the GPU, so the model could be saved on a GPU
    # device. Since this routine may be run on a CPU machine, allow for a remap of the
    # torch tensors.
    if torch.cuda.is_available():
        d = torch.load(args.checkpoint)
    else:
        d = torch.load(args.checkpoint, map_location=torch.device('cpu'))

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
