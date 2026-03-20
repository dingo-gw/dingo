"""Core functions for generating waveform datasets (new-style API).

This module provides dataset generation using the new-style WaveformGenerator
hierarchy and typed dataclasses. The existing generate_dataset.py is preserved
for backward compatibility with the legacy dict-based API.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from bilby.gw.prior import BBHPriorDict

from dingo.gw.compression.svd import SVDBasis
from dingo.gw.compression.transforms import (
    ApplySVD,
    ComposeTransforms,
    Transform,
    WhitenAndUnwhiten,
)
from dingo.gw.domains import Domain, DomainParameters, build_domain
from dingo.gw.prior import new_build_prior_with_defaults
from dingo.gw.waveform_generator.new_api import (
    NewWaveformGenerator,
    build_waveform_generator,
)
from dingo.gw.waveform_generator.polarizations import BatchPolarizations
from dingo.gw.waveform_generator.waveform_parameters import BBHWaveformParameters
from .compression_settings import CompressionSettings
from .dataset_settings import DatasetSettings
from .generation_types import WaveformGeneratorConfig, WaveformResult
from .new_waveform_dataset import NewWaveformDataset

_logger = logging.getLogger(__name__)

# Worker process state (standard ProcessPoolExecutor initializer pattern)
_worker_generator: Optional[NewWaveformGenerator] = None
_worker_domain: Optional[Domain] = None


def _init_worker(
    wfg_config: WaveformGeneratorConfig, domain_params: DomainParameters
) -> None:
    global _worker_generator, _worker_domain
    _worker_domain = build_domain(domain_params)
    _worker_generator = build_waveform_generator(
        wfg_config.to_dict(), _worker_domain
    )


def _generate_single_waveform_optimized(
    parameters_dict: dict,
) -> WaveformResult:
    global _worker_generator
    try:
        wf_params = BBHWaveformParameters(**parameters_dict)
        polarization = _worker_generator.generate_hplus_hcross(wf_params)
        return WaveformResult.success_result(
            polarization.h_plus, polarization.h_cross
        )
    except Exception as e:
        _logger.warning(
            f"Failed to generate waveform for parameters {parameters_dict}: {e}"
        )
        return WaveformResult.failure_result(str(e))


def _generate_waveform_batch(
    params_batch: List[dict],
) -> List[WaveformResult]:
    global _worker_generator
    results = []
    for parameters_dict in params_batch:
        try:
            wf_params = BBHWaveformParameters(**parameters_dict)
            polarization = _worker_generator.generate_hplus_hcross(wf_params)
            results.append(
                WaveformResult.success_result(
                    polarization.h_plus, polarization.h_cross
                )
            )
        except Exception as e:
            _logger.warning(
                f"Failed to generate waveform for parameters {parameters_dict}: {e}"
            )
            results.append(WaveformResult.failure_result(str(e)))
    return results


def _generate_single_waveform(
    parameters_dict: dict,
    wfg_config: WaveformGeneratorConfig,
    domain_params: DomainParameters,
) -> WaveformResult:
    try:
        domain = build_domain(domain_params)
        wfg = build_waveform_generator(wfg_config.to_dict(), domain)
        wf_params = BBHWaveformParameters(**parameters_dict)
        polarization = wfg.generate_hplus_hcross(wf_params)
        return WaveformResult.success_result(
            polarization.h_plus, polarization.h_cross
        )
    except Exception as e:
        _logger.warning(
            f"Failed to generate waveform for parameters {parameters_dict}: {e}"
        )
        return WaveformResult.failure_result(str(e))


def generate_waveforms_sequential(
    waveform_generator: NewWaveformGenerator,
    parameters: pd.DataFrame,
) -> BatchPolarizations:
    """Generate waveforms sequentially (single process)."""
    h_plus_list = []
    h_cross_list = []

    _logger.info(f"Generating {len(parameters)} waveforms sequentially...")

    for idx, row in parameters.iterrows():
        try:
            wf_params = BBHWaveformParameters(**row.to_dict())
            polarization = waveform_generator.generate_hplus_hcross(wf_params)
            h_plus_list.append(polarization.h_plus)
            h_cross_list.append(polarization.h_cross)
        except Exception as e:
            _logger.warning(f"Failed to generate waveform {idx}: {e}")
            domain_length = len(
                waveform_generator._waveform_gen_params.domain
            )
            h_plus_list.append(
                np.full(domain_length, np.nan, dtype=complex)
            )
            h_cross_list.append(
                np.full(domain_length, np.nan, dtype=complex)
            )

    polarizations = BatchPolarizations(
        h_plus=np.array(h_plus_list),
        h_cross=np.array(h_cross_list),
    )

    if waveform_generator.transform is not None:
        polarizations = apply_transforms_to_polarizations(
            polarizations, waveform_generator.transform
        )

    return polarizations


def generate_waveforms_parallel(
    waveform_generator: NewWaveformGenerator,
    parameters: pd.DataFrame,
    num_processes: int = 4,
) -> BatchPolarizations:
    """Generate waveforms in parallel using ProcessPoolExecutor."""
    if num_processes == 1:
        return generate_waveforms_sequential(waveform_generator, parameters)

    _logger.info(
        f"Generating {len(parameters)} waveforms with {num_processes} processes..."
    )

    wfg_params = waveform_generator._waveform_gen_params
    domain_params = wfg_params.domain.get_parameters()
    wfg_config = WaveformGeneratorConfig(
        approximant=str(wfg_params.approximant),
        f_ref=wfg_params.f_ref,
        spin_conversion_phase=wfg_params.spin_conversion_phase,
    )

    results = {}
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {
            executor.submit(
                _generate_single_waveform,
                row.to_dict(),
                wfg_config,
                domain_params,
            ): idx
            for idx, row in parameters.iterrows()
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                _logger.error(f"Worker failed for waveform {idx}: {e}")
                results[idx] = WaveformResult.failure_result(str(e))

    h_plus_list = []
    h_cross_list = []
    for idx in sorted(results.keys()):
        result = results[idx]
        h_plus_list.append(result.h_plus)
        h_cross_list.append(result.h_cross)

    polarizations = BatchPolarizations(
        h_plus=np.array(h_plus_list),
        h_cross=np.array(h_cross_list),
    )

    if waveform_generator.transform is not None:
        polarizations = apply_transforms_to_polarizations(
            polarizations, waveform_generator.transform
        )

    return polarizations


def generate_waveforms_parallel_optimized(
    waveform_generator: NewWaveformGenerator,
    parameters: pd.DataFrame,
    num_processes: int = 4,
    batch_size: Optional[int] = None,
) -> BatchPolarizations:
    """Generate waveforms in parallel with optimized worker initialization."""
    if num_processes == 1:
        return generate_waveforms_sequential(waveform_generator, parameters)

    _logger.info(
        f"Generating {len(parameters)} waveforms with {num_processes} processes "
        f"(optimized with worker initialization and batching)..."
    )

    wfg_params = waveform_generator._waveform_gen_params
    domain_params = wfg_params.domain.get_parameters()
    wfg_config = WaveformGeneratorConfig(
        approximant=str(wfg_params.approximant),
        f_ref=wfg_params.f_ref,
        spin_conversion_phase=wfg_params.spin_conversion_phase,
    )

    if batch_size is None:
        num_waveforms = len(parameters)
        ideal_num_tasks = num_processes * 6
        batch_size = max(1, num_waveforms // ideal_num_tasks)
        batch_size = min(batch_size, 100)
        _logger.debug(f"Auto-computed batch_size: {batch_size}")

    param_dicts = [row.to_dict() for idx, row in parameters.iterrows()]

    if batch_size == 1:
        batches = [[p] for p in param_dicts]
    else:
        batches = [
            param_dicts[i : i + batch_size]
            for i in range(0, len(param_dicts), batch_size)
        ]

    _logger.debug(
        f"Split {len(parameters)} waveforms into {len(batches)} batches"
    )

    all_results = []
    with ProcessPoolExecutor(
        max_workers=num_processes,
        initializer=_init_worker,
        initargs=(wfg_config, domain_params),
    ) as executor:
        futures = [
            executor.submit(_generate_waveform_batch, batch)
            for batch in batches
        ]

        for future in as_completed(futures):
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
            except Exception as e:
                _logger.error(f"Batch processing failed: {e}")
                batch_size_actual = len(batches[0])
                all_results.extend(
                    [WaveformResult.failure_result(str(e))] * batch_size_actual
                )

    h_plus_list = [r.h_plus for r in all_results]
    h_cross_list = [r.h_cross for r in all_results]

    polarizations = BatchPolarizations(
        h_plus=np.array(h_plus_list),
        h_cross=np.array(h_cross_list),
    )

    if waveform_generator.transform is not None:
        polarizations = apply_transforms_to_polarizations(
            polarizations, waveform_generator.transform
        )

    return polarizations


def new_generate_parameters_and_polarizations(
    waveform_generator: NewWaveformGenerator,
    prior: BBHPriorDict,
    num_samples: int,
    num_processes: int = 1,
) -> Tuple[pd.DataFrame, BatchPolarizations]:
    """Generate dataset of waveforms based on parameters drawn from prior."""
    _logger.info(f"Generating dataset of size {num_samples}")

    parameters = pd.DataFrame(prior.sample(num_samples))

    if num_processes > 1:
        polarizations = generate_waveforms_parallel(
            waveform_generator, parameters, num_processes
        )
    else:
        polarizations = generate_waveforms_sequential(
            waveform_generator, parameters
        )

    wf_failed = np.any(np.isnan(polarizations.h_plus), axis=1)
    if wf_failed.any():
        idx_failed = np.where(wf_failed)[0]
        idx_ok = np.where(~wf_failed)[0]
        polarizations_ok = BatchPolarizations(
            h_plus=polarizations.h_plus[idx_ok],
            h_cross=polarizations.h_cross[idx_ok],
        )
        parameters_ok = parameters.iloc[idx_ok].reset_index(drop=True)
        failed_percent = 100 * len(idx_failed) / len(parameters)
        _logger.warning(
            f"{len(idx_failed)} out of {len(parameters)} configurations "
            f"({failed_percent:.1f}%) failed to generate."
        )
        _logger.info(
            f"Returning {len(idx_ok)} successfully generated configurations."
        )
        return parameters_ok, polarizations_ok

    return parameters, polarizations


def train_svd_basis(
    polarizations: BatchPolarizations,
    parameters: pd.DataFrame,
    size: int,
    n_train: int,
) -> Tuple[SVDBasis, int, int]:
    """Train and validate an SVD basis from waveform data."""
    n_total = len(polarizations)
    n_train = min(n_train, n_total)
    n_validation = n_total - n_train

    _logger.info(
        f"Training SVD basis: {n_train} train, {n_validation} validation samples"
    )

    train_data = np.concatenate(
        [polarizations.h_plus[:n_train], polarizations.h_cross[:n_train]],
        axis=0,
    )

    basis = SVDBasis()
    basis.generate_basis(train_data, n_components=size, method="scipy")

    if n_validation > 0:
        _logger.info("Computing validation mismatches...")
        val_data = np.concatenate(
            [polarizations.h_plus[n_train:], polarizations.h_cross[n_train:]],
            axis=0,
        )
        val_params = pd.concat(
            [parameters[n_train:], parameters[n_train:]], ignore_index=True
        )
        basis.compute_mismatches(val_data, val_params)

    return basis, n_train, n_validation


def apply_transforms_to_polarizations(
    polarizations: BatchPolarizations,
    transforms: Optional[ComposeTransforms],
) -> BatchPolarizations:
    """Apply transform pipeline to polarizations."""
    if transforms is None:
        return polarizations

    pol_dict = {
        "h_plus": polarizations.h_plus,
        "h_cross": polarizations.h_cross,
    }
    transformed = transforms(pol_dict)
    return BatchPolarizations(
        h_plus=transformed["h_plus"],
        h_cross=transformed["h_cross"],
    )


def build_compression_transforms(
    compression_settings: CompressionSettings,
    domain: Domain,
    prior: BBHPriorDict,
    waveform_generator: NewWaveformGenerator,
    num_processes: int,
) -> Tuple[Optional[ComposeTransforms], Optional[SVDBasis]]:
    """Build compression transform pipeline from settings."""
    transforms: List[Transform] = []
    svd_basis: Optional[SVDBasis] = None

    if compression_settings.whitening is not None:
        _logger.info(
            f"Adding whitening transform with ASD from {compression_settings.whitening}"
        )
        transforms.append(
            WhitenAndUnwhiten(
                domain, compression_settings.whitening, inverse=False
            )
        )

    if compression_settings.svd is not None:
        svd_settings = compression_settings.svd

        if svd_settings.file is not None:
            _logger.info(f"Loading SVD basis from {svd_settings.file}")
            svd_basis = SVDBasis.load(svd_settings.file)
        else:
            _logger.info("Generating SVD basis from training waveforms...")

            if transforms:
                waveform_generator.transform = ComposeTransforms(transforms)

            n_total = (
                svd_settings.num_training_samples
                + svd_settings.num_validation_samples
            )
            train_parameters, train_polarizations = (
                new_generate_parameters_and_polarizations(
                    waveform_generator, prior, n_total, num_processes
                )
            )

            svd_basis, n_train, n_val = train_svd_basis(
                train_polarizations,
                train_parameters,
                svd_settings.size,
                svd_settings.num_training_samples,
            )

            waveform_generator.transform = None

        transforms.append(ApplySVD(svd_basis, inverse=False))
        _logger.info(
            f"Added SVD compression with {svd_basis.n_components} components"
        )

    if transforms:
        return ComposeTransforms(transforms), svd_basis
    else:
        return None, None


def new_generate_waveform_dataset(
    settings: DatasetSettings, num_processes: int = 1
) -> NewWaveformDataset:
    """Generate a waveform dataset based on settings."""
    settings.validate()

    _logger.info("Building domain, prior, and waveform generator...")
    domain = build_domain(settings.domain)
    prior = new_build_prior_with_defaults(settings.intrinsic_prior)
    wfg_dict = settings.waveform_generator.to_dict()
    waveform_generator = build_waveform_generator(wfg_dict, domain)

    compression_transforms = None
    svd_basis = None
    if settings.compression is not None:
        _logger.info("Building compression pipeline...")
        compression_transforms, svd_basis = build_compression_transforms(
            settings.compression,
            domain,
            prior,
            waveform_generator,
            num_processes,
        )

        if compression_transforms is not None:
            waveform_generator.transform = compression_transforms
            _logger.info(f"Compression pipeline: {compression_transforms}")

    parameters, polarizations = new_generate_parameters_and_polarizations(
        waveform_generator, prior, settings.num_samples, num_processes
    )

    dataset = NewWaveformDataset(
        parameters=parameters,
        polarizations=polarizations,
        settings=settings,
        svd_basis=svd_basis,
    )

    _logger.info(
        f"Dataset generated successfully with {len(parameters)} samples."
    )
    if svd_basis is not None:
        _logger.info(
            f"Dataset includes SVD compression with {svd_basis.n_components} components"
        )

    return dataset
