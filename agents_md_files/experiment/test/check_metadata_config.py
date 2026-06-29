"""Objective test for dingo issue #327.

Issue #327 asks that the model *metadata* (not the full ~5 GB network) be carried into the
data-generation / plotting steps, by incorporating the model metadata into the completed
config INI. On unfixed code the completed config contains only derived scalars
(duration, detectors, frequencies, prior-dict) but NOT the nested model metadata
(``dataset_settings`` / ``train_settings``), so generation must still load the network.

This script generates the completed config from ``input.ini`` + the reference model using the
dingo package on ``PYTHONPATH`` (so a worktree's modified code is exercised), then measures how
much of the model's metadata key structure is present in the completed config. Baseline HEAD
scores ~0; a correct fix embeds the metadata and scores ~1.

Usage:
    PYTHONPATH=<worktree> uv run python check_metadata_config.py --repo <worktree> --out result.json
"""
import argparse
import json
import os
import re
import sys
import tempfile
import traceback

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PASS_THRESHOLD = 0.90  # fraction of metadata key-tokens that must appear in the completed config


def metadata_key_tokens(meta):
    """Return the set of unique key tokens appearing anywhere in the metadata dict."""
    tokens = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                tokens.add(str(k))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(meta)
    return tokens


def generate_complete_config(repo, outdir):
    """Run the minimal dingo_pipe path that writes the completed config; return its path.

    Mirrors dingo.pipe.main.main() up to write_complete_config_file (skips generate_dag).
    Tolerates downstream errors as long as the completed config file was written.
    """
    from bilby_pipe.utils import parse_args
    from dingo.pipe.parser import create_parser
    from dingo.pipe.main import (
        MainInput,
        fill_in_arguments_from_model,
        write_complete_config_file,
    )

    # Resolve the input INI into the temp outdir.
    with open(os.path.join(TEST_DIR, "input.ini")) as f:
        ini_text = f.read().replace("__OUTDIR__", outdir)
    resolved = os.path.join(outdir, "input.ini")
    with open(resolved, "w") as f:
        f.write(ini_text)

    parser = create_parser(top_level=True)
    args, unknown_args = parse_args([resolved], parser)
    importance_sampling_updates, _ = fill_in_arguments_from_model(args)
    inputs = MainInput(args, unknown_args, importance_sampling_updates)
    write_complete_config_file(parser, args, inputs)
    return inputs.complete_ini_file


def bundle_text(complete_ini, outdir):
    """Text the completed config carries: the INI itself plus any metadata artifact written
    alongside it (json/ini/yaml/txt read as text; .pkl/.pt unpickled and stringified).

    The issue asks that metadata be carried by the completed config; a fix may embed it inline
    or write a small sidecar referenced from the config. Both make the metadata recoverable
    from the completed-config bundle, so both are credited. The reference model .pt is never
    counted (only artifacts inside this run's outdir).
    """
    import base64
    import re as _re

    parts = []
    ini_text = ""
    if complete_ini and os.path.isfile(complete_ini):
        with open(complete_ini, encoding="utf-8", errors="replace") as f:
            ini_text = f.read()
        parts.append(ini_text)
    skip = {os.path.basename(complete_ini or ""), "input.ini"}
    for root, _dirs, files in os.walk(outdir):
        for fn in files:
            if fn in skip:
                continue
            p = os.path.join(root, fn)
            try:
                if fn.endswith(".pt"):
                    import torch
                    parts.append(str(torch.load(p, map_location="cpu", weights_only=False)))
                elif fn.endswith((".pkl", ".pickle")):
                    import pickle
                    with open(p, "rb") as fh:
                        parts.append(str(pickle.load(fh)))
                elif fn.endswith((".json", ".ini", ".yaml", ".yml", ".txt", ".cfg")):
                    with open(p, encoding="utf-8", errors="replace") as fh:
                        parts.append(fh.read())
                else:  # try pickle then torch then text for unknown sidecars
                    try:
                        import pickle
                        with open(p, "rb") as fh:
                            parts.append(str(pickle.load(fh)))
                    except Exception:
                        with open(p, encoding="utf-8", errors="replace") as fh:
                            parts.append(fh.read())
            except Exception:
                pass  # unreadable artifact -> just skip it
    # Decode any long base64 blobs embedded in the config (some fixes base64 the metadata
    # so that '#'/quote chars survive the INI round-trip). Recoverable metadata still counts.
    for blob in _re.findall(r"[A-Za-z0-9+/]{120,}={0,2}", ini_text):
        try:
            parts.append(base64.b64decode(blob).decode("utf-8", "replace"))
        except Exception:
            pass
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="worktree path being tested (for the record)")
    ap.add_argument("--out", required=True, help="where to write the JSON result")
    args = ap.parse_args()

    # Guarantee the worktree's code is imported, not an editable-installed copy.
    sys.path.insert(0, os.path.abspath(args.repo))

    reference = json.load(open(os.path.join(TEST_DIR, "model_metadata.json")))
    tokens = metadata_key_tokens(reference)
    top_level = list(reference.keys())  # ['dataset_settings', 'train_settings']

    result = {
        "repo": args.repo,
        "passed": False,
        "coverage": 0.0,
        "num_tokens": len(tokens),
        "num_present": 0,
        "present_top_level": [],
        "missing_top_level": list(top_level),
        "complete_ini_written": False,
        "dingo_main_file": None,
        "error": None,
    }

    outdir = tempfile.mkdtemp(prefix="metatest_")
    try:
        complete_ini = generate_complete_config(args.repo, outdir)
        import dingo.pipe.main as _m
        result["dingo_main_file"] = getattr(_m, "__file__", None)
        result["complete_ini_written"] = bool(complete_ini and os.path.isfile(complete_ini))
        # The completed-config bundle: the INI plus any metadata sidecar it carries.
        text = bundle_text(complete_ini, outdir)
        # Token coverage: a metadata key token counts as present if it appears as a
        # whole word anywhere in the completed-config bundle.
        present = {t for t in tokens if re.search(r"\b" + re.escape(t) + r"\b", text)}
        result["num_present"] = len(present)
        result["coverage"] = round(len(present) / max(1, len(tokens)), 4)
        result["present_top_level"] = [t for t in top_level if t in present]
        result["missing_top_level"] = [t for t in top_level if t not in present]
        # PASS: both nested metadata roots embedded AND high overall key coverage.
        result["passed"] = (
            all(t in present for t in top_level)
            and result["coverage"] >= PASS_THRESHOLD
        )
    except Exception:
        result["error"] = traceback.format_exc()

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "error"}, indent=2))
    if result["error"]:
        print("ERROR (non-fatal if complete_ini_written=true):\n", result["error"], file=sys.stderr)
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
