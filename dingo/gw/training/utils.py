import logging

import hydra
import numpy as np
import torch
import yaml
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from dingo.core.utils.backward_compatibility import torch_load_with_fallback

logger = logging.getLogger(__name__)
logging.captureWarnings(True)


@hydra.main(
    version_base="1.3",
    config_path="../../../configs",
    config_name="append_training_stage",
)
def append_stage(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # trying to load on CUDA, MPS, HIP or CPU
    d, _ = torch_load_with_fallback(to_absolute_path(cfg["checkpoint"]))

    stages = [
        v
        for k, v in d["metadata"]["train_settings"]["training"].items()
        if k.startswith("stage_")
    ]
    num_stages = len(stages)
    logger.info(f"Checkpoint training plan consists of {num_stages} stages.")

    new_stage = cfg["stage"]

    if cfg["replace"] is not None:
        if cfg["replace"] < 0 or cfg["replace"] >= num_stages:
            raise ValueError(
                f"Invalid argument replace={cfg['replace']}. Valid values "
                f"are {list(range(num_stages))}."
            )
        current_epoch = d["epoch"]
        stage_epoch = np.sum([s["epochs"] for s in stages[: cfg["replace"]]])
        if current_epoch > stage_epoch:
            logger.info(
                f"WARNING: Modification to training plan changes a training stage "
                f"that has already started. Current model epoch is {current_epoch}. "
                f"Proceed at your own risk!"
            )
        logger.info(f"Replacing planned stage {cfg['replace']} with new stage.")
        new_stage_number = cfg["replace"]
    else:
        logger.info("Appending new stage to training plan.")
        new_stage_number = num_stages

    d["metadata"]["train_settings"]["training"][f"stage_{new_stage_number}"] = new_stage
    logger.info("Summary of new training plan:")
    logger.info(
        yaml.dump(
            d["metadata"]["train_settings"]["training"],
            default_flow_style=False,
            sort_keys=False,
        )
    )

    torch.save(d, cfg["out_file"])
