"""
dingo_pipe_gracedb: Generate a dingo_pipe config file from a GraceDB event.

Works analogously to bilby_pipe_gracedb but generates configuration for
dingo_pipe rather than bilby_pipe. Reuses bilby_pipe's GraceDB interface
and data utilities.
"""

import argparse
import json
import os
import re
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


def _load_model_metadata(model_path):
    """Load and return the metadata dict from a dingo model checkpoint."""
    import torch

    d = torch.load(model_path, map_location="cpu", weights_only=False)
    return d["metadata"]


def _extract_prior_from_metadata(metadata):
    """Extract prior specifications from model metadata.

    Merges the intrinsic and extrinsic prior specifications stored in the
    metadata. Entries in the intrinsic prior that are fixed scalar values
    (parameters not sampled during training) are excluded. Extrinsic prior
    entries override intrinsic ones for shared parameters (e.g.
    luminosity_distance, geocent_time).

    Parameters
    ----------
    metadata : dict
        Model metadata as stored in the checkpoint under 'metadata'.

    Returns
    -------
    dict
        Mapping parameter_name -> prior_string (or "default").
    """
    intrinsic_prior = metadata["dataset_settings"]["intrinsic_prior"]
    extrinsic_prior = metadata["train_settings"]["data"]["extrinsic_prior"]

    # Keep only string entries from the intrinsic prior; fixed scalar values
    # (e.g. luminosity_distance = 100.0) are not priors and must be excluded.
    prior = {k: v for k, v in intrinsic_prior.items() if isinstance(v, str)}

    # Extrinsic prior entries take precedence over intrinsic ones for shared
    # parameters (e.g. luminosity_distance has a proper range in extrinsic_prior).
    prior.update(extrinsic_prior)

    return prior


def _extract_prior_from_model(model_path):
    """Extract prior specifications from a dingo model checkpoint.

    Convenience wrapper around _load_model_metadata + _extract_prior_from_metadata.
    """
    return _extract_prior_from_metadata(_load_model_metadata(model_path))


def _check_model_compatibility(trigger_chirp_mass, metadata):
    """Raise ValueError if the trigger requires longer segments than the model.

    Dingo models are trained on a fixed segment length determined by their
    frequency domain. If the trigger's chirp mass implies a longer required
    segment (e.g. BNS or NSBH), the model cannot produce valid posteriors and
    the run should not start.

    Parameters
    ----------
    trigger_chirp_mass : float
        Chirp mass estimated from the GraceDB trigger (solar masses).
    metadata : dict
        Model metadata as stored in the checkpoint under 'metadata'.

    Raises
    ------
    ValueError
        If the required analysis duration exceeds the model's training duration.
    """
    # Use the domain builder so this works for both UniformFrequencyDomain and
    # MultibandedFrequencyDomain models. base=True returns the underlying
    # uniform domain, whose duration is the model's training segment length.
    from ..gw.domains.build_domain import build_domain_from_model_metadata

    domain = build_domain_from_model_metadata(metadata, base=True)
    model_duration = round(domain.duration)
    required_duration = _get_analysis_duration(trigger_chirp_mass)

    if required_duration > model_duration:
        # Parse the model's chirp mass prior range for the error message.
        chirp_mass_prior_str = metadata["dataset_settings"]["intrinsic_prior"].get(
            "chirp_mass", ""
        )
        mc_min_match = re.search(r"minimum=([0-9.eE+\-]+)", chirp_mass_prior_str)
        mc_min = float(mc_min_match.group(1)) if mc_min_match else None

        detail = (
            f" The model's chirp mass prior minimum is {mc_min} Msun."
            if mc_min is not None
            else ""
        )
        raise ValueError(
            f"Trigger chirp mass {trigger_chirp_mass:.2f} Msun requires "
            f"{required_duration}s segments, but the model was trained on "
            f"{model_duration}s segments.{detail} "
            f"This is likely a BNS or NSBH event incompatible with this model. "
            f"Use a model trained for longer segments or do not run dingo_pipe_gracedb."
        )


def _is_gnpe_model(metadata):
    """Whether the model uses GNPE and therefore requires an init model.

    Same criterion as GWSamplerGNPE (gw/inference/gw_samplers.py): any gnpe_*
    block in train_settings.data.
    """
    data_settings = metadata.get("train_settings", {}).get("data", {})
    return any(
        data_settings.get(key)
        for key in ("gnpe_time_shifts", "gnpe_chirp", "gnpe_phase")
    )


def _find_init_model(model_path):
    """Look for the conventional init-model file next to a GNPE main model.

    Two naming conventions are in use (e.g. for the O4c production networks):
    ``<name>.pt`` with ``<name>_init.pt``, and ``<name>_main.pt`` with
    ``<name>_init.pt``. For ``_main`` models the documented convention
    (``<name>_init.pt``) is tried first. Returns the first existing
    candidate, or None.
    """
    base, ext = os.path.splitext(model_path)
    candidates = []
    if base.endswith("_main"):
        candidates.append(f"{base[: -len('_main')]}_init{ext}")
    candidates.append(f"{base}_init{ext}")
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


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
    model_init=None,
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
        directly from this checkpoint; the trigger is also checked for
        compatibility with the model's training duration.
    model_init : str, optional
        Path to the GNPE init model (.pt). GNPE models require one; if None
        and the model's metadata shows it is GNPE, a conventional
        ``*_init.pt`` sibling of ``model`` is used when present, and a
        ValueError is raised otherwise.
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

    Raises
    ------
    ValueError
        If the trigger's required analysis duration exceeds the model's
        training duration (e.g. a BNS event queried against a BBH model).
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
    minimum_frequency = 20.0
    maximum_frequency = 1024.0

    # Load model metadata once — used for both compatibility check and prior.
    model_metadata = _load_model_metadata(model)

    # Raise early if the trigger requires a longer analysis than the model supports.
    _check_model_compatibility(chirp_mass, model_metadata)

    # GNPE models require an init model. Validate/resolve here: sampling keys
    # on model_init alone and would otherwise fail only on the GPU node.
    is_gnpe = _is_gnpe_model(model_metadata)
    if model_init is not None:
        if not os.path.isfile(model_init):
            raise ValueError(f"model_init file does not exist: {model_init}")
        if not is_gnpe:
            raise ValueError(
                f"--model-init was given, but model {model} is not a GNPE "
                "model (no gnpe settings in its metadata). dingo_pipe_sampling "
                "would wrongly use the GNPE sampler; drop --model-init."
            )
    elif is_gnpe:
        model_init = _find_init_model(model)
        if model_init is None:
            raise ValueError(
                f"Model {model} is a GNPE model (gnpe settings present in its "
                "metadata) and requires an init model, but --model-init was "
                "not given and no '*_init.pt' file was found next to the model."
            )
        logger.info(f"GNPE model detected; using init model {model_init}")

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

    # The calibration archive (/home/cal) exists only on CIT; fall back to no
    # calibration elsewhere. bilby_pipe catches only its own BilbyPipeError:
    # OSError (missing archive) and KeyError (O3-era non-H1/L1/V1 detector)
    # leak out and must be handled here.
    try:
        calibration_model, calib_dict = calibration_dict_lookup(trigger_time, ifos)
    except (OSError, KeyError) as e:
        logger.warning(f"Calibration lookup failed ({e!r}); proceeding without it.")
        calibration_model, calib_dict = None, None

    if webdir is None:
        webdir = os.path.join(outdir, "results_page")

    config = {
        "label": gracedb,
        "outdir": outdir,
        "accounting": "ligo.dev.o4.cbc.pe.dingo",
        # Data settings. duration, reference_frequency and deltaT are
        # deliberately NOT written: dingo_pipe fills them from the model
        # (the source of truth). Writing mismatching values would push them
        # into importance_sampling_updates, and rebuilding a
        # MultibandedFrequencyDomain for changed settings is not implemented
        # (dingo/gw/result.py:_rebuild_domain) — the IS stage would crash.
        # The chirp-mass duration ladder is still used for the early
        # model-compatibility check above.
        "trigger_time": trigger_time,
        "detectors": ifos,
        "sampling_frequency": 4096,
        "minimum_frequency": minimum_frequency,
        "maximum_frequency": maximum_frequency,
        "time_reference": time_reference,
        # Prior comes from the model; override via prior-dict-updates.
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

    if model_init is not None:
        config["model_init"] = model_init

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
    parser.add_argument(
        "--model-init", type=str, default=None,
        help=(
            "Path to the GNPE init model (.pt file). GNPE models require one; "
            "if omitted, a '*_init.pt' file next to --model is used when present"
        ),
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

    if args.psd_file is not None:
        if not os.path.isfile(args.psd_file):
            raise ValueError(
                f"--psd-file {args.psd_file} does not exist. Refusing to "
                "proceed with a different PSD source than requested."
            )
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
        model_init=args.model_init,
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
