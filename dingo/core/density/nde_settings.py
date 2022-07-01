"""Default settings for unconditional density estimation"""


def get_default_nde_settings_3d(
    device="cpu",
    num_workers=0,
    inference_parameters=None,
):
    settings = {
        "data": {
            "inference_parameters": inference_parameters,
        },
        "model": {
            "type": "nsf",
            "num_flow_steps": 5,
            "base_transform_kwargs": {
                "hidden_dim": 128,
                "num_transform_blocks": 2,
                "activation": "elu",
                "dropout_probability": 0.1,
                "batch_norm": True,
                "num_bins": 8,
                "base_transform_type": "rq-coupling",
            },
            "input_dim": 3,
            "context_dim": None,
        },
        "training": {
            "device": device,
            "num_workers": num_workers,
            "train_fraction": 0.9,
            "batch_size": 4096,
            "epochs": 10,
            "optimizer": {"type": "adam", "lr": 0.005},
            "scheduler": {"type": "cosine", "T_max": 10},
        },
    }
    return settings


DEFAULT_NDE_SETTINGS_2D = {
    "data": {
        "inference_parameters": ["GNPE:H1_time_proxy", "GNPE:L1_time_proxy"],
        "parameter_samples": None,
    },
    "model": {
        "type": "nsf",
        "num_flow_steps": 5,
        "base_transform_kwargs": {
            "hidden_dim": 64,
            "num_transform_blocks": 2,
            "activation": "elu",
            "dropout_probability": 0.1,
            "batch_norm": "true",
            "num_bins": 8,
            "base_transform_type": "rq-coupling",
        },
    },
    # nde training
    "training": {
        "device": "cpu",
        "num_workers": 0,
        "train_fraction": 0.9,
        "batch_size": 4096,
        "epochs": 10,
        "optimizer": {"type": "adam"},
        "lr": 0.001,
        "scheduler": {
            "type": "cosine",
            "T_max": 10,
        },
    },
}
