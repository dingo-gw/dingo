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
        if data_settings["gnpe_time_shifts"]["exact_equiv"]:

            # Rename context variables "L1_time_proxy" to "L1_time_proxy_relative", etc.
            for i, p in enumerate(data_settings["context_parameters"]):
                if p.endswith('_time_proxy'):
                    data_settings["context_parameters"][i] = p + '_relative'
                    data_settings["standardization"]["mean"][p + '_relative'] = \
                        data_settings["standardization"]["mean"].pop(p)
                    data_settings["standardization"]["std"][p + '_relative'] = \
                        data_settings["standardization"]["std"].pop(p)

    print("New data_settings:")
    print(yaml.dump(data_settings, default_flow_style=False, sort_keys=False))

    torch.save(d, args.out_file)


if __name__ == "__main__":
    main()