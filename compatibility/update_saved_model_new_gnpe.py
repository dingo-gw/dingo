import torch
import argparse

import yaml


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    # Typically training is done on the GPU, so the model could be saved on a GPU
    # device. Since this routine may be run on a CPU machine, allow for a remap of the
    # torch tensors.
    if torch.cuda.is_available():
        d = torch.load(args.checkpoint)
    else:
        d = torch.load(args.checkpoint, map_location=torch.device("cpu"))

    data_settings = d["metadata"]["train_settings"]["data"]

    if "gnpe_time_shifts" in data_settings:

        data_settings["gnpe_time_shifts"][
            "kernel"
        ] = "bilby.core.prior.Uniform(minimum=-0.001, maximum=0.001)"
        del data_settings["gnpe_time_shifts"]["kernel_kwargs"]

        data_settings["context_parameters"] = [
            f"{ifo}_time_proxy" for ifo in data_settings["detectors"][1:]
        ]

        for p in data_settings["context_parameters"]:
            data_settings["standardization"]["mean"][p] = 0.0
            data_settings["standardization"]["std"][p] = data_settings[
                "standardization"
            ]["std"]["geocent_time"]

    data_settings["inference_parameters"] = data_settings["selected_parameters"]
    del data_settings["selected_parameters"]

    print("New data_settings:")
    print(yaml.dump(data_settings, default_flow_style=False, sort_keys=False))

    torch.save(d, args.out_file)


if __name__ == "__main__":
    main()
