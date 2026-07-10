import logging
from os.path import join

import hydra
import os
import pickle
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from dingo.gw.noise.asd_estimation import (
    download_and_estimate_psds,
)
from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.noise.utils import merge_datasets, get_time_segments

logger = logging.getLogger(__name__)
logging.captureWarnings(True)


@hydra.main(
    version_base="1.3",
    config_path="../../../configs",
    config_name="generate_asd_dataset",
)
def generate_dataset(cfg: DictConfig):
    """
    Creates and saves an ASD dataset
    """
    data_dir = to_absolute_path(cfg.data_dir)
    settings = OmegaConf.to_container(cfg, resolve=True)
    time_segments_file = settings.pop("time_segments_file")
    out_name = settings.pop("out_name")
    verbose = settings.pop("verbose")
    settings.pop("data_dir")

    if time_segments_file:
        with open(to_absolute_path(time_segments_file), "rb") as f:
            time_segments = pickle.load(f)
    else:
        time_segments = get_time_segments(settings["dataset_settings"])
        time_segments_path = join(
            data_dir, "tmp", settings["dataset_settings"]["observing_run"]
        )
        os.makedirs(time_segments_path, exist_ok=True)
        with open(join(time_segments_path, "psd_time_segments.pkl"), "wb") as f:
            pickle.dump(time_segments, f)

    if "condor" in settings:

        raise NotImplementedError(
            "Hydra Stage 1 only supports local ASD dataset generation. "
            "Condor/DAG ASD generation is deferred to Stage 3."
        )
        # Legacy Condor/DAG implementation to translate in Stage 3:
        #
        # dagman = create_dag(data_dir, settings_file, time_segments, out_name)
        #
        # try:
        #     dagman.visualize(
        #         join(data_dir, "tmp", "condor", "ASD_dataset_generation_workflow.png")
        #     )
        # except:
        #     pass
        #
        # dagman.build()
        # logger.info("DAG submission file written.")

    else:

        logger.info("Downloading strain data and estimating PSDs...")
        asd_filename_list = download_and_estimate_psds(
            data_dir, settings, time_segments, verbose=verbose
        )
        asd_dataset_list = {
            det: [ASDDataset(asd_file) for asd_file in asd_file_list]
            for det, asd_file_list in asd_filename_list.items()
        }
        logger.info("Merging single dataset files into one...")
        dataset = merge_datasets(asd_dataset_list)
        filename = out_name
        if filename is None:
            run = settings["dataset_settings"]["observing_run"]
            filename = join(data_dir, f"asds_{run}.hdf5")
        dataset.to_file(filename)
