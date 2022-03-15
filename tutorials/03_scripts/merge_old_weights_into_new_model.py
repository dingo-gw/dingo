import torch
import h5py
import argparse

DEFAULT_PARAMS = [
    "mass_1",
    "mass_2",
    "phase",
    "geocent_time",
    "luminosity_distance",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "theta_jn",
    "psi",
    "ra",
    "dec",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load weights of PRL model and settings of dingo model, "
                    "and save these to a new model.",
    )
    parser.add_argument(
        "--model_PRL",
        type=str,
        required=True,
        help="PRL model with target weights.",
    )
    parser.add_argument(
        "--model_dingo",
        type=str,
        required=True,
        help="Dingo model with target settings.",
    )
    parser.add_argument(
        "--model_new",
        type=str,
        required=True,
        help="Name of new model.",
    )
    parser.add_argument(
        "--PRL_wf_supp",
        type=str,
        required=True,
        help="PRL waveforms supplementary with standardizations.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    model_PRL = torch.load(args.model_PRL)
    model_dingo = torch.load(args.model_dingo)
    keys_PRL = model_dingo["model_state_dict"].keys()
    keys_dingo = model_PRL["model_state_dict"].keys()
    assert len(keys_PRL) == len(keys_dingo)
    # dingo keys start with transforms and then enet, for PRL its vice versa.
    # Other than that, we can rely on the order to identify parameters.
    keys_enet = [k for k in keys_dingo if k.startswith("_embedding_net")]
    keys_flow = [k for k in keys_dingo if k.startswith("_transform")]
    keys_dingo_in_order = keys_enet + keys_flow
    assert len(keys_dingo) == len(keys_dingo_in_order)
    for k1, k2 in zip(keys_PRL, keys_dingo_in_order):
        assert (
            model_dingo["model_state_dict"][k1].shape
            == model_PRL["model_state_dict"][k2].shape
        )
        model_dingo["model_state_dict"][k1] = model_PRL["model_state_dict"][k2]
    # delete optimizer state dict;
    # if this is required, we need to add another conversion here
    # model_dingo.pop('optimizer_state_dict')
    # set epoch to -1 to indicate that this is not a proper dingo run
    # model_dingo["epoch"] = 1
    # epoch determines training stage and thus asd dataset
    model_dingo["epoch"] = 449
    # do only validation
    # model_dingo["metadata"]["train_settings"]["data"]["train_fraction"] = 0.00
    # set correct inference parameters and standardizations
    model_dingo["metadata"]["train_settings"]["data"][
        "inference_parameters"
    ] = DEFAULT_PARAMS
    with h5py.File(args.PRL_wf_supp, "r") as fp:
        mean = fp["parameters_mean"][:]
        std = fp["parameters_std"][:]
    params = model_dingo["metadata"]["train_settings"]["data"]["inference_parameters"]
    mean = {params[i]: mean[i] for i in range(len(params))}
    std = {params[i]: std[i] for i in range(len(params))}
    mean["L1_time_proxy"] = mean["geocent_time"]
    std["L1_time_proxy"] = std["geocent_time"]
    model_dingo["metadata"]["train_settings"]["data"]["standardization"] = {
        "mean": mean,
        "std": std,
    }
    # save new model
    torch.save(model_dingo, args.model_new)
