"""
Test consistency between parser.py arguments and dingo.ini template.

When adding new arguments to parser.py, they should either be:
1. Added to dingo.ini template (if they need to be configurable via Asimov)
2. Added to EXCLUDED_FROM_INI set (if they are internal or not needed in Asimov)

This test ensures that no arguments that are added to parser.py are forgotten
in dingo.ini by explicitly adding them to the EXCLUDED_FROM_INI set if needed.
"""

import re

import pytest

from importlib.resources import files
from dingo.pipe.parser import create_parser


# Arguments that do NOT need to be in dingo.ini template.
# When adding a new parser argument that shouldn't be in dingo.ini, add it here.
EXCLUDED_FROM_INI = {
    "additional_transfer_paths",
    "allow_tape",
    "analysis_executable",
    "analysis_executable_parser",
    "asd_dataset",
    "catch_waveform_errors",
    "condor_job_priority",
    "data_dump_file",
    "data_find_url",
    "data_find_urltype",
    "data_format",
    "default_prior",
    "deltaT",
    "density_recovery_settings",
    "dingo_injection",
    "disable_hdf5_locking",
    "duration",
    "email",
    "event_data_file",
    "existing_dir",
    "extra_lines",
    "fetch_open_data_kwargs",
    "final_result",
    "final_result_nsamples",
    "gaussian_noise",
    "generation_desired_sites",
    "generation_seed",
    "getenv",
    "gps_file",
    "gps_tuple",
    "idx",
    "ignore_gwpy_data_quality_check",
    "importance_sample",
    "importance_sampling_generation",
    "importance_sampling_settings",
    "importance_sampling_updates",
    "ini",
    "injection",
    "injection_file",
    "injection_frequency_domain_source_model",
    "injection_numbers",
    "injection_waveform_approximant",
    "injection_waveform_arguments",
    "injection_waveform_generator_constructor_dict",
    "local_generation",
    "local_plot",
    "log_directory",
    "model_reference_time",
    "n_simulation",
    "notification",
    "numerical_relativity_file",
    "overwrite_outdir",
    "plot_data",
    "plot_injection",
    "plot_pp",
    "plot_spectrogram",
    "post_trigger_duration",
    "prior_dict",
    "prior_file",
    "proposal_samples_file",
    "psd_fractional_overlap",
    "psd_maximum_duration",
    "psd_method",
    "psd_start_time",
    "queue",
    "reference_frequency",
    "resampling_method",
    "result_format",
    "save_bilby_data_dump",
    "scheduler",
    "scheduler_analysis_time",
    "scheduler_args",
    "scheduler_env",
    "scheduler_module",
    "submit",
    "summarypages_arguments",
    "time_reference",
    "timeslide_dict",
    "timeslide_file",
    "Toffset",
    "verbose",
    "waveform_approximant",
    "zero_noise",
}


def get_parser_arguments():
    """Extract all argument names from the parser."""
    # Get arguments from both top_level=True and top_level=False
    parser_top = create_parser(top_level=True)
    parser_internal = create_parser(top_level=False)

    args = set()
    for parser in [parser_top, parser_internal]:
        for action in parser._actions:
            if action.dest and action.dest != "help":
                args.add(action.dest)
    return args


def get_ini_arguments():
    """Extract argument names referenced in dingo.ini template."""
    ini_content = files("dingo.asimov").joinpath("dingo.ini").read_text()

    # Match lines like "argument-name=" or "argument-name ="
    # This regex captures the argument name before the = sign
    pattern = r"^([a-z][a-z0-9-]*)\s*="
    matches = re.findall(pattern, ini_content, re.MULTILINE)

    # Convert kebab-case to snake_case to match parser dest names
    args = {match.replace("-", "_") for match in matches}
    return args


@pytest.mark.asimov
def test_parser_arguments_in_ini_or_excluded():
    """
    Test that all parser arguments are either in dingo.ini or explicitly excluded.

    If this test fails, you added a new argument to parser.py but forgot to either:
    1. Add it to dingo.ini template (if it should be configurable via Asimov)
    2. Add it to EXCLUDED_FROM_INI set (if it's internal/not needed in Asimov)
    """
    parser_args = get_parser_arguments()
    ini_args = get_ini_arguments()

    # Arguments that should be in ini but aren't
    missing_from_ini = parser_args - ini_args - EXCLUDED_FROM_INI

    if missing_from_ini:
        missing_list = "\n  - ".join(sorted(missing_from_ini))
        pytest.fail(
            f"The following parser arguments are not in dingo.ini and not in "
            f"EXCLUDED_FROM_INI:\n  - {missing_list}\n\n"
            f"Please either:\n"
            f"  1. Add them to dingo/asimov/dingo.ini template, OR\n"
            f"  2. Add them to EXCLUDED_FROM_INI in this test file"
        )


@pytest.mark.asimov
def test_excluded_args_exist_in_parser():
    """
    Test that all excluded arguments actually exist in the parser.

    This catches typos in EXCLUDED_FROM_INI and removes stale entries.
    """
    parser_args = get_parser_arguments()

    # Excluded args that don't exist in parser (typos or removed args)
    invalid_excluded = EXCLUDED_FROM_INI - parser_args

    if invalid_excluded:
        invalid_list = "\n  - ".join(sorted(invalid_excluded))
        pytest.fail(
            f"The following entries in EXCLUDED_FROM_INI don't exist in parser.py:\n"
            f"  - {invalid_list}\n\n"
            f"Please remove them from EXCLUDED_FROM_INI (they may be typos or "
            f"removed arguments)."
        )


@pytest.mark.asimov
def test_ini_args_exist_in_parser():
    """
    Test that all arguments in dingo.ini exist in the parser.

    This catches typos in dingo.ini or arguments that were removed from parser.
    """
    parser_args = get_parser_arguments()
    ini_args = get_ini_arguments()

    # INI args that don't exist in parser
    invalid_ini_args = ini_args - parser_args

    if invalid_ini_args:
        invalid_list = "\n  - ".join(sorted(invalid_ini_args))
        pytest.fail(
            f"The following arguments in dingo.ini don't exist in parser.py:\n"
            f"  - {invalid_list}\n\n"
            f"Please fix typos or remove stale arguments from dingo.ini."
        )
