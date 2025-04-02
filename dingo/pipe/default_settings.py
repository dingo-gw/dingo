DENSITY_RECOVERY_SETTINGS = {
    "ProxyRecoveryDefault": {
        "num_samples": 400_000,
        "threshold_std": 5,
        "nde_settings": {
            "model": {
                "posterior_model_type": "normalizing_flow",
                "posterior_kwargs": {
                    "num_flow_steps": 5,
                    "base_transform_kwargs": {
                        "hidden_dim": 256,
                        "num_transform_blocks": 4,
                        "activation": "elu",
                        "dropout_probability": 0.1,
                        "batch_norm": True,
                        "num_bins": 8,
                        "base_transform_type": "rq-coupling",
                    },
                },
            },
            "training": {
                "num_workers": 0,
                "train_fraction": 0.9,
                "batch_size": 4096,
                "epochs": 20,
                "optimizer": {
                    "type": "adam",
                    "lr": 0.002,
                },
                "scheduler": {
                    "type": "cosine",
                    "T_max": 20,
                },
            },
        },
    },
}

IMPORTANCE_SAMPLING_SETTINGS = {
    "PhaseRecoveryDefault": {
        "synthetic_phase": {
            "approximation_22_mode": False,
            "n_grid": 5001,
            "uniform_weight": 0.01,
        },
    },
    "MultibandingDefault": {
        "use_base_domain": True,
    },
}
