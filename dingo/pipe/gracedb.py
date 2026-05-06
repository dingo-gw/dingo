"""
dingo_pipe_gracedb: Generate a dingo_pipe config file from a GraceDB event.

Works analogously to bilby_pipe_gracedb but generates configuration for
dingo_pipe rather than bilby_pipe. Reuses bilby_pipe's GraceDB interface
and data utilities.
"""

import argparse
import json
import os
import subprocess

from bilby_pipe.gracedb import (
    CHANNEL_DICTS,
    _read_cbc_candidate,
    calibration_dict_lookup,
    extract_psds_from_xml,
    read_from_gracedb,
    read_from_json,
)
from bilby_pipe.utils import (
    BilbyPipeError,
    check_directory_exists_and_if_not_mkdir,
    logger,
)

GRACEDB_URL = "https://gracedb.ligo.org/api/"


def _get_analysis_duration(chirp_mass):
    """Return analysis duration in seconds based on chirp mass.

    Uses the same boundaries as bilby_pipe_gracedb so that the data
    segment length is consistent with standard online PE practice.
    """
    if chirp_mass > 13.53:
        return 4
    elif chirp_mass > 8.73:
        return 8
    elif chirp_mass > 5.66:
        return 16
    elif chirp_mass > 3.68:
        return 32
    elif chirp_mass > 2.39:
        return 64
    return 128


def _extract_prior_from_model(model_path):
    """Extract prior specifications from a dingo model checkpoint.

    Reads the model metadata and merges the intrinsic and extrinsic prior
    specifications stored there. Entries in the intrinsic prior that are
    fixed scalar values (parameters not sampled during training) are
    excluded. Extrinsic prior entries override intrinsic ones for shared
    parameters (e.g. luminosity_distance, geocent_time).

    Parameters
    ----------
    model_path : str
        Path to the trained dingo model checkpoint (.pt file).

    Returns
    -------
    dict
        Mapping parameter_name -> prior_string (or "default").
    """
    import torch

    d = torch.load(model_path, map_location="cpu", weights_only=False)
    metadata = d["metadata"]

    intrinsic_prior = metadata["dataset_settings"]["intrinsic_prior"]
    extrinsic_prior = metadata["train_settings"]["data"]["extrinsic_prior"]

    # Keep only string entries from the intrinsic prior; fixed scalar values
    # (e.g. luminosity_distance = 100.0) are not priors and must be excluded.
    prior = {k: v for k, v in intrinsic_prior.items() if isinstance(v, str)}

    # Extrinsic prior entries take precedence over intrinsic ones for shared
    # parameters (e.g. luminosity_distance has a proper range in extrinsic_prior).
    prior.update(extrinsic_prior)

    return prior


def _write_config_file(config_dict, filename, comment=None):
    """Write a dingo_pipe INI config file from a plain dictionary."""
    with open(filename, "w") as f:
        if comment:
            print(f"# {comment}\n", file=f)
        for key, value in config_dict.items():
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value)
            key = key.replace("_", "-")
            print(f"{key}={value}", file=f)
        print("", file=f)


def prepare_dingo_config(
    candidate,
    gracedb,
    outdir,
    model,
    device="cuda",
    num_samples=50000,
    batch_size=None,
    importance_sample=True,
    channel_dict=None,
    psd_cut=0.95,
    settings=None,
    webdir=None,
):
    """Generate a dingo_pipe INI config for a GraceDB CBC event.

    Parameters
    ----------
    candidate : dict
        GraceDB event JSON dictionary (as returned by read_from_json or
        read_from_gracedb).
    gracedb : str
        GraceDB event or superevent ID used for labelling.
    outdir : str
        Directory where the INI file and all outputs are written.
    model : str
        Path to the trained dingo model (.pt file). The prior is extracted
        directly from this checkpoint, so no separate prior file is needed.
    device : str
        Device for the neural network forward pass ('cuda' or 'cpu').
    num_samples : int
        Number of posterior samples to draw from the model.
    batch_size : int, optional
        Batch size for sampling. Defaults to num_samples if None.
    importance_sample : bool
        Whether to run the importance sampling second stage.
    channel_dict : dict, optional
        Mapping from detector name to channel name. If None, the config
        omits a channel specification (provide via --settings or data_dict).
    psd_cut : float
        Maximum frequency is capped at this fraction of the pipeline PSD's
        maximum frequency to avoid likelihood overflow from the low-pass roll-off.
    settings : dict, optional
        Additional key-value pairs that override any computed defaults.
        Applied last, so they take precedence over all other settings.
    webdir : str, optional
        Directory for HTML summary pages. Defaults to outdir/results_page.

    Returns
    -------
    str
        Path to the written dingo_config.ini.
    """
    if settings is None:
        settings = {}

    # Parse CBC candidate metadata
    (
        trigger_values,
        superevent,
        trigger_time,
        ifos,
        reference_frame,
        time_reference,
    ) = _read_cbc_candidate(candidate)

    chirp_mass = trigger_values["chirp_mass"]
    duration = _get_analysis_duration(chirp_mass)
    minimum_frequency = 20.0
    maximum_frequency = 1024.0

    # Extract PSDs from coinc.xml when available
    psd_dict = {}
    if candidate.get("coinc_file"):
        psd_dict, psd_max_freq = extract_psds_from_xml(
            candidate["coinc_file"], ifos, outdir
        )
        if psd_max_freq is not None:
            psd_max_freq *= min(psd_cut, 1)
            if maximum_frequency > psd_max_freq:
                maximum_frequency = psd_max_freq
                logger.info(
                    f"maximum_frequency reduced to {psd_max_freq:.1f} Hz "
                    "due to pipeline PSD bandwidth"
                )

    # Calibration lookup (gracefully returns (None, None) off-cluster)
    calibration_model, calib_dict = calibration_dict_lookup(trigger_time, ifos)

    # Extract the prior from the model checkpoint so that inference uses
    # exactly the prior the model was trained on.
    prior_dict = _extract_prior_from_model(model)

    if webdir is None:
        webdir = os.path.join(outdir, "results_page")

    config = {
        "label": gracedb,
        "outdir": outdir,
        "accounting": "ligo.dev.o4.cbc.pe.dingo",
        # Data settings
        "trigger_time": trigger_time,
        "detectors": ifos,
        "duration": duration,
        "sampling_frequency": 4096,
        "minimum_frequency": minimum_frequency,
        "maximum_frequency": maximum_frequency,
        "reference_frequency": 20.0,
        "deltaT": 0.2,
        "reference_frame": reference_frame,
        "time_reference": time_reference,
        # Prior extracted from model
        "prior_dict": prior_dict,
        # Dingo model
        "model": model,
        "device": device,
        "num_samples": num_samples,
        "recover_log_prob": True,
        # Importance sampling
        "importance_sample": importance_sample,
        # Output
        "overwrite_outdir": True,
        "result_format": "hdf5",
        "webdir": webdir,
    }

    if batch_size is not None:
        config["batch_size"] = batch_size

    if psd_dict:
        config["psd_dict"] = psd_dict

    if channel_dict:
        config["channel_dict"] = channel_dict

    if calibration_model is not None:
        config.update(
            {
                "calibration_model": calibration_model,
                "spline_calibration_envelope_dict": calib_dict,
                "spline_calibration_nodes": 10,
            }
        )

    # User overrides are applied last
    config.update(settings)

    filename = os.path.join(outdir, "dingo_config.ini")
    _write_config_file(
        config,
        filename,
        comment=(
            f"Configuration generated by dingo_pipe_gracedb "
            f"for event {gracedb} (superevent {superevent})"
        ),
    )
    logger.info(f"Wrote dingo config to {filename}")
    return filename


def create_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a dingo_pipe configuration file from a GraceDB event. "
            "Works analogously to bilby_pipe_gracedb."
        )
    )

    event = parser.add_mutually_exclusive_group(required=True)
    event.add_argument(
        "--gracedb", type=str,
        help="GraceDB superevent or event ID (e.g. S230914ax)",
    )
    event.add_argument(
        "--json", type=str,
        help="Path to a local GraceDB JSON file (no network required)",
    )

    parser.add_argument(
        "--psd-file", type=str, default=None,
        help="Path to ligolw-xml file containing PSDs (overrides coinc.xml download)",
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory (default: outdir_{gracedb_id})",
    )
    parser.add_argument(
        "--output", type=str, default="full",
        choices=["ini", "full", "full-local", "full-submit"],
        help=(
            "Execution mode: 'ini' writes the config only; "
            "'full' creates the HTCondor DAG; "
            "'full-local' runs locally; "
            "'full-submit' creates and submits the DAG (default: full)"
        ),
    )
    parser.add_argument(
        "--gracedb-url", type=str, default=GRACEDB_URL,
        help=(
            f"GraceDB service URL (default: {GRACEDB_URL}). "
            "Use https://gracedb-playground.ligo.org/api/ for testing."
        ),
    )
    parser.add_argument(
        "--channel-dict", type=str, default="online",
        choices=list(CHANNEL_DICTS.keys()),
        help="Channel preset (default: online)",
    )
    parser.add_argument("--psd-cut", type=float, default=0.95)
    parser.add_argument(
        "--settings", type=str, default=None,
        help="Path to JSON file with additional settings to override defaults",
    )
    parser.add_argument("--webdir", type=str, default=None)

    # Required dingo argument
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained dingo model (.pt file)",
    )

    # Optional dingo arguments
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help="Device for neural network inference (default: cuda)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=50000,
        help="Number of posterior samples to draw (default: 50000)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size for sampling (default: num-samples)",
    )
    parser.add_argument(
        "--no-importance-sampling", dest="importance_sample",
        action="store_false",
        help="Skip the importance sampling stage",
    )
    parser.set_defaults(importance_sample=True)

    return parser


def main(args=None):
    if args is None:
        args = create_parser().parse_args()

    gracedb_url = args.gracedb_url
    outdir = args.outdir

    if args.json:
        candidate = read_from_json(args.json)
        gracedb = candidate["graceid"]
    else:
        gracedb = args.gracedb
        if outdir is None:
            outdir = f"outdir_{gracedb}"
        check_directory_exists_and_if_not_mkdir(outdir)
        candidate = read_from_gracedb(gracedb, gracedb_url, outdir)

    if outdir is None:
        outdir = f"outdir_{gracedb}"
    check_directory_exists_and_if_not_mkdir(outdir)

    if args.psd_file is not None and os.path.isfile(args.psd_file):
        candidate["coinc_file"] = args.psd_file

    extra_settings = {}
    if args.settings is not None:
        with open(args.settings) as f:
            extra_settings = json.load(f)

    channel_dict = CHANNEL_DICTS[args.channel_dict.lower()]

    filename = prepare_dingo_config(
        candidate=candidate,
        gracedb=gracedb,
        outdir=outdir,
        model=args.model,
        device=args.device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        importance_sample=args.importance_sample,
        channel_dict=channel_dict,
        psd_cut=args.psd_cut,
        settings=extra_settings,
        webdir=args.webdir,
    )

    if args.output == "ini":
        logger.info(f"Config generated. Run with:\n  dingo_pipe {filename}")
    else:
        cmd = ["dingo_pipe", filename]
        if args.output == "full-local":
            cmd.append("--local")
        elif args.output == "full-submit":
            cmd.append("--submit")
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
