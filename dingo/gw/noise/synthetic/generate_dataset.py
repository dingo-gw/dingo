import copy
import logging
import numpy as np
from typing import Dict

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.noise.synthetic.asd_parameterization import parameterize_asd_dataset
from dingo.gw.noise.synthetic.asd_sampling import KDE, get_rescaling_params
from dingo.gw.noise.synthetic.utils import reconstruct_psds_from_parameters

logger = logging.getLogger(__name__)
logging.captureWarnings(True)


def generate_dataset(
    real_dataset, settings: Dict, num_samples, num_processes: int, verbose: bool
):
    """
    Generate a synthetic ASD dataset from an existing dataset of real ASDs.

    Parameters
    ----------
    real_dataset : ASDDataset
        Existing dataset of real ASDs.
    settings : dict
        Dictionary containing the settings for the parameterization and sampling.
    num_processes : int
        Number of processes to use in pool for parallel parameterization.
    verbose : bool
        Whether to print progress information.

    """
    parameters_dict = parameterize_asd_dataset(
        real_dataset,
        settings["parameterization_settings"],
        num_processes,
        verbose=verbose,
    )

    synthetic_dataset_dict = copy.deepcopy(real_dataset.to_dictionary())
    synthetic_dataset_dict["settings"]["parameterization_settings"] = settings[
        "parameterization_settings"
    ]

    sampling_settings = settings.get("sampling_settings")
    if sampling_settings:
        kde = KDE(parameters_dict, sampling_settings)
        kde.fit()
        rescaling_params = None
        if "rescaling_asd_paths" in sampling_settings:
            rescaling_params = get_rescaling_params(
                sampling_settings["rescaling_asd_paths"],
                settings["parameterization_settings"],
            )
        parameters_dict = kde.sample(num_samples, rescaling_params)
        synthetic_dataset_dict["settings"]["sampling_settings"] = settings[
            "sampling_settings"
        ]

    asds_dict = {}
    for det, params in parameters_dict.items():
        psds = reconstruct_psds_from_parameters(
            params,
            real_dataset.domain,
            settings["parameterization_settings"],
        )

        asds_dict[det] = np.sqrt(
            psds[:, real_dataset.domain.min_idx : real_dataset.domain.max_idx + 1]
        )

    synthetic_dataset_dict["asd_parameterizations"] = parameters_dict
    synthetic_dataset_dict["asds"] = asds_dict

    return ASDDataset(dictionary=synthetic_dataset_dict)


@hydra.main(
    version_base="1.3",
    config_path="../../../../configs",
    config_name="generate_synthetic_asd_dataset",
)
def main(cfg: DictConfig):
    settings = OmegaConf.to_container(cfg, resolve=True)
    asd_dataset = to_absolute_path(settings.pop("asd_dataset"))
    num_samples = settings.pop("num_samples")
    num_processes = settings.pop("num_processes")
    out_file = settings.pop("out_file")
    verbose = settings.pop("verbose")

    real_dataset = ASDDataset(file_name=asd_dataset)
    synthetic_dataset = generate_dataset(
        real_dataset, settings, num_samples, num_processes, verbose
    )

    synthetic_dataset.to_file(out_file)


if __name__ == "__main__":
    main()
