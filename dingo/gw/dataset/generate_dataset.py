import copy
import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Tuple
from functools import partial

import hydra
import numpy as np
import pandas as pd
from bilby.gw.prior import BBHPriorDict
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from threadpoolctl import threadpool_limits
from torchvision.transforms import Compose

from dingo.core.utils.hydra_utils import instantiate_with_runtime_dependencies
from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.SVD import ApplySVD
from dingo.gw.waveform_generator import (
    WaveformGenerator,
    generate_waveforms_parallel,
)
from dingo.core.utils.misc import call_func_strict_output_dim

logger = logging.getLogger(__name__)
logging.captureWarnings(True)


def generate_parameters_and_polarizations(
    waveform_generator: WaveformGenerator,
    prior: BBHPriorDict,
    num_samples: int,
    num_processes: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Generate a dataset of waveforms based on parameters drawn from the prior.

    Parameters
    ----------
    waveform_generator : WaveformGenerator
    prior : Prior
    num_samples : int
    num_processes : int

    Returns
    -------
    pandas DataFrame of parameters
    dictionary of numpy arrays corresponding to waveform polarizations
    """
    logger.info("Generating dataset of size " + str(num_samples))
    parameters = pd.DataFrame(prior.sample(num_samples))

    if num_processes > 1:
        with threadpool_limits(limits=1, user_api="blas"):
            with Pool(processes=num_processes) as pool:
                polarizations = generate_waveforms_parallel(
                    waveform_generator, parameters, pool
                )
    else:
        polarizations = generate_waveforms_parallel(waveform_generator, parameters)

    # Find cases where waveform generation failed and only return data for successful ones
    wf_failed = np.any(np.isnan(polarizations["h_plus"]), axis=1)
    if wf_failed.any():
        idx_failed = np.where(wf_failed)[0]
        idx_ok = np.where(~wf_failed)[0]
        polarizations_ok = {k: v[idx_ok] for k, v in polarizations.items()}
        parameters_ok = parameters.iloc[idx_ok]
        failed_percent = 100 * len(idx_failed) / len(parameters)
        logger.warning(
            f"{len(idx_failed)} out of {len(parameters)} configuration ({failed_percent:.1f}%) failed to generate."
        )
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            logger.warning(parameters.iloc[idx_failed].to_string())
        logger.warning(
            f"Only returning the {len(idx_ok)} successfully generated configurations."
        )
        return parameters_ok, polarizations_ok

    return parameters, polarizations


def _instantiate_compression_transform(transform_config: Dict, runtime_dependencies: Dict):
    transform_config = copy.deepcopy(transform_config)
    if "svd_basis" not in transform_config:
        return instantiate_with_runtime_dependencies(
            transform_config,
            runtime_dependencies,
        )

    svd_basis_config = transform_config.pop("svd_basis")
    svd_basis = instantiate_with_runtime_dependencies(
        svd_basis_config,
        runtime_dependencies,
    )
    return instantiate(transform_config, svd_basis=svd_basis)


def _settings_from_config(cfg: DictConfig) -> Dict:
    settings = OmegaConf.to_container(cfg, resolve=True)
    settings.pop("out_file", None)
    settings.pop("num_processes", None)
    return settings


def generate_dataset(settings: Dict, num_processes: int) -> WaveformDataset:
    """
    Generate a waveform dataset.

    Parameters
    ----------
    settings : dict
        Dictionary of settings to configure the dataset
    num_processes : int

    Returns
    -------
    A WaveformDataset based on the settings.
    """

    prior = instantiate(settings["intrinsic_prior"])
    domain = instantiate(settings["domain"])
    waveform_generator = instantiate(settings["waveform_generator"], domain=domain)

    dataset_dict = {"settings": settings}

    if settings.get("compression", None) is not None:
        compression_transforms = []
        compression_settings = []
        runtime_dependencies = {
            "domain": domain,
            "waveform_generator": waveform_generator,
            "prior": prior,
            "num_processes": num_processes,
            "settings": settings,
            "compression_transforms": compression_transforms,
            "compression_settings": compression_settings,
        }

        for transform_config in settings["compression"]:
            compression_transform = _instantiate_compression_transform(
                transform_config,
                runtime_dependencies,
            )
            compression_transforms.append(compression_transform)
            compression_settings.append(copy.deepcopy(transform_config))

            if isinstance(compression_transform, ApplySVD):
                dataset_dict["svd"] = compression_transform.svd_basis.to_dictionary()

        waveform_generator.transform = Compose(compression_transforms)

    func = partial(
        generate_parameters_and_polarizations,
        waveform_generator,
        prior,
        num_processes=num_processes,
    )
    parameters, polarizations = call_func_strict_output_dim(
        func, settings["num_samples"]
    )
    dataset_dict["parameters"] = parameters
    dataset_dict["polarizations"] = polarizations

    dataset_dict[settings["num_samples"]] = len(parameters)
    dataset = WaveformDataset(dictionary=dataset_dict)
    return dataset


@hydra.main(
    version_base="1.3",
    config_path="../../../configs",
    config_name="generate_dataset",
)
def main(cfg: DictConfig) -> None:
    out_file = cfg.out_file
    if not Path(out_file).parent.is_dir():
        raise FileNotFoundError(
            f"dataset generation: can not create {out_file}: "
            f"{Path(out_file).parent} does not exist"
        )
    settings = _settings_from_config(cfg)
    dataset = generate_dataset(settings, cfg.num_processes)
    dataset.to_file(str(out_file))


if __name__ == "__main__":
    main()
