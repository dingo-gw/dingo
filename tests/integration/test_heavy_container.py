"""Heavy end-to-end test: build an Apptainer image from this checkout and run the
scaled-down train -> sample -> importance-sample pipeline inside it.

Build strategy:
  We build the image entirely in Python rather than from an Apptainer .def file.
  A .def build runs its %post section under ``--fakeroot``, which needs the kernel
  to set up a user-namespace uid map and therefore an /etc/subuid range for the
  caller -- unavailable on shared HPC clusters.  The three-step sandbox workflow
  below never runs %post, so it works with any non-setuid Apptainer, with or
  without subuid:

    1. ``apptainer build --sandbox sandbox docker://python:3.11``
       Pull the base image into an unpacked directory (no %post, no user-ns needed).
    2. Stage source + install via ``apptainer exec --writable``.
       Pip writes files owned by the calling user, which is fine because the sandbox
       directory is already owned by that user.
    3. ``apptainer build dingo-heavy.sif sandbox``
       Pack the sandbox into a read-only SIF.  No user-ns required.
"""
import math
import os
import re
import subprocess
import tempfile

import pytest

from tests.integration.conftest import APPTAINER_CMD, HAS_APPTAINER, HAS_GPU

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HEAVY_DIR = os.path.join(os.path.dirname(__file__), "heavy")
EFFICIENCY_FLOOR_PCT = 3.0  # asserted floor sits below the 5% search target

# Version fed to setuptools_scm so pip install succeeds without a .git directory.
# ``git archive`` omits .git, so SETUPTOOLS_SCM_PRETEND_VERSION must be set.
_DINGO_VERSION = "0.9.9"

# Per-stage ceiling guards (seconds).  These are loose smoke guards tied to the
# untuned scaled-down config (Task 6).  Each is ~3x the observed container
# baseline so a stuck/hung stage will fail rather than quietly eat the 60-min
# budget.  Revisit these ceilings once Task 7 config tuning lands.
#
# Observed inside container (A100, apptainer --nv, untuned config):
#   waveform_dataset ~82 s   -> ceiling 250 s
#   asd_dataset      ~30 s   -> ceiling 125 s  (generous; download-bound)
#   train            ~324 s  -> ceiling 1000 s
#   inference        ~548 s  -> ceiling 1650 s (includes synthetic-phase grid)
#   gw150914         ~736 s  -> ceiling 2200 s (~3x observed; includes GWOSC download + IS)
_STAGE_CEILINGS = {
    "waveform_dataset": 250,
    "asd_dataset": 125,
    "train": 1000,
    "inference": 1650,
    "gw150914": 2200,
}

# GW150914 reference values (GWTC-1 medians).
_GW150914_CHIRP_MASS_TRUTH = 31.2   # Msun, detector-frame
_GW150914_MASS_RATIO_TRUTH = 0.864

# Loose recovery window for chirp_mass: ±8 Msun of the truth.
# This is a smoke guard only — the scaled-down network is not expected to give
# publication-quality posteriors.  A failure here indicates a gross problem
# (wrong data, wrong model, unit error) rather than just sub-optimal training.
_GW150914_CHIRP_MASS_WINDOW = 8.0


def _parse_stage_times(output: str) -> dict:
    """Return {stage_name: seconds} from STAGE_SECONDS lines in output."""
    times = {}
    for m in re.finditer(r"STAGE_SECONDS (\w+) = ([0-9.]+)", output):
        times[m.group(1)] = float(m.group(2))
    return times


def _parse_total_seconds(output: str):
    """Return total pipeline seconds from TOTAL_SECONDS line, or None."""
    m = re.search(r"TOTAL_SECONDS = ([0-9.]+)", output)
    return float(m.group(1)) if m else None


def _print_timing_table(stage_times: dict, total: float | None) -> None:
    """Print a human-readable per-stage timing table to stdout."""
    header = f"{'Stage':<20} {'Seconds':>10}  {'Ceiling':>10}"
    print("\n--- pipeline stage timings ---")
    print(header)
    print("-" * len(header))
    for name, secs in stage_times.items():
        ceiling = _STAGE_CEILINGS.get(name, "?")
        ceiling_str = f"{ceiling}" if isinstance(ceiling, int) else ceiling
        print(f"{name:<20} {secs:>10.1f}  {ceiling_str:>10}")
    if total is not None:
        print("-" * len(header))
        print(f"{'TOTAL':<20} {total:>10.1f}")
    print("------------------------------\n")


pytestmark = [
    pytest.mark.heavy,
    pytest.mark.skipif(not HAS_APPTAINER, reason="apptainer/singularity not available"),
    pytest.mark.skipif(not HAS_GPU, reason="no GPU available"),
]


def test_heavy_container_e2e():
    with tempfile.TemporaryDirectory() as build_dir:
        sandbox = os.path.join(build_dir, "sandbox")

        # ------------------------------------------------------------------
        # Step 1: pull base image into a writable sandbox directory.
        # No %post section is executed, so no user-namespace mapping is needed.
        # ------------------------------------------------------------------
        subprocess.run(
            [APPTAINER_CMD, "build", "--sandbox", sandbox,
             "docker://python:3.11"],
            check=True,
        )

        # ------------------------------------------------------------------
        # Step 2a: stage DINGO source (tracked files only) into the sandbox.
        # ------------------------------------------------------------------
        dingo_src_tar = os.path.join(build_dir, "dingo-src.tar")
        subprocess.run(
            ["git", "archive", "--format=tar", "-o", dingo_src_tar, "HEAD"],
            cwd=REPO_ROOT, check=True,
        )
        dingo_dst = os.path.join(sandbox, "opt", "dingo")
        os.makedirs(dingo_dst, exist_ok=True)
        subprocess.run(
            ["tar", "-xf", dingo_src_tar, "-C", dingo_dst],
            check=True,
        )

        # Step 2b: stage the heavy pipeline scripts.
        subprocess.run(
            ["cp", "-r", HEAVY_DIR, os.path.join(sandbox, "opt", "heavy")],
            check=True,
        )

        # Step 2c: set the runscript (replace the docker CMD default).
        runscript_path = os.path.join(sandbox, ".singularity.d", "runscript")
        with open(runscript_path, "w") as fh:
            fh.write("#!/bin/sh\nexec python3 /opt/heavy/run_pipeline.py \"$@\"\n")
        os.chmod(runscript_path, 0o755)

        # ------------------------------------------------------------------
        # Step 2d: install PyTorch for the host's CUDA driver (12.8 / 570.x).
        # The default PyPI torch wheel targets CUDA 13.0 which requires a newer
        # driver; pinning to cu128 keeps the wheel within the installed driver.
        # ------------------------------------------------------------------
        subprocess.run(
            [APPTAINER_CMD, "exec", "--writable", sandbox,
             "pip3", "install", "--no-cache-dir",
             "torch", "torchvision",
             "--index-url", "https://download.pytorch.org/whl/cu128"],
            check=True,
        )

        # ------------------------------------------------------------------
        # Step 2e: install DINGO and all remaining deps inside the writable
        # sandbox.  apptainer exec --writable runs as the calling user; because
        # the sandbox directory is owned by that user pip can write to
        # site-packages.  SETUPTOOLS_SCM_PRETEND_VERSION avoids the "no git"
        # version error.
        # ------------------------------------------------------------------
        build_env = dict(
            os.environ,
            SETUPTOOLS_SCM_PRETEND_VERSION=_DINGO_VERSION,
        )
        subprocess.run(
            [APPTAINER_CMD, "exec", "--writable", sandbox,
             "pip3", "install", "--no-cache-dir", "/opt/dingo"],
            env=build_env, check=True,
        )

        # ------------------------------------------------------------------
        # Step 3: pack the sandbox into a read-only SIF.
        # ------------------------------------------------------------------
        sif = os.path.join(build_dir, "dingo-heavy.sif")
        subprocess.run(
            [APPTAINER_CMD, "build", sif, sandbox],
            check=True,
        )

        # ------------------------------------------------------------------
        # Run the pipeline.
        # ------------------------------------------------------------------
        run_env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        proc = subprocess.run(
            [APPTAINER_CMD, "run", "--nv", sif],
            cwd=build_dir, env=run_env, capture_output=True, text=True,
            # Timeout must exceed the sum of all per-stage ceilings so that a
            # pathological slow run is caught by the per-stage ceiling assertions
            # (which report which stage hung) rather than by a bare TimeoutExpired.
            # _STAGE_CEILINGS sum ~5225 s (~87 min); +30 min for image build overhead.
            timeout=sum(_STAGE_CEILINGS.values()) + 30 * 60,
        )

    output = proc.stdout + "\n" + proc.stderr
    assert proc.returncode == 0, f"pipeline failed:\n{output}"

    match = re.search(r"Sample efficiency = ([0-9.]+)%", output)
    assert match, f"no 'Sample efficiency' line in output:\n{output}"
    efficiency = float(match.group(1))
    assert efficiency >= EFFICIENCY_FLOOR_PCT, (
        f"sample efficiency {efficiency}% below floor {EFFICIENCY_FLOOR_PCT}%"
    )

    # Parse and display per-stage timings.
    stage_times = _parse_stage_times(output)
    total_secs = _parse_total_seconds(output)
    _print_timing_table(stage_times, total_secs)

    # Guard: each stage must be under its ceiling (loose smoke check).
    for stage, ceiling in _STAGE_CEILINGS.items():
        if stage in stage_times:
            assert stage_times[stage] <= ceiling, (
                f"stage '{stage}' took {stage_times[stage]:.1f}s, "
                f"exceeding ceiling {ceiling}s — check for hangs or config regressions"
            )

    # --- GW150914 real-event assertions ---

    # Hard assert: GW150914 stage must have completed and produced finite results.
    gw_eff_match = re.search(r"GW150914_EFFICIENCY = ([0-9.]+)%", output)
    assert gw_eff_match, f"no GW150914_EFFICIENCY line in output:\n{output[-3000:]}"
    gw_eff = float(gw_eff_match.group(1))
    print(f"GW150914 sample efficiency: {gw_eff:.2f}%")
    assert math.isfinite(gw_eff) and gw_eff > 0.0, (
        f"GW150914 efficiency {gw_eff}% is not positive-finite"
    )

    gw_cm_match = re.search(r"GW150914_CHIRP_MASS = ([0-9.]+)", output)
    assert gw_cm_match, f"no GW150914_CHIRP_MASS line in output:\n{output[-3000:]}"
    gw_cm = float(gw_cm_match.group(1))
    print(f"GW150914 recovered chirp_mass median: {gw_cm:.3f} Msun  (truth ~{_GW150914_CHIRP_MASS_TRUTH})")
    assert math.isfinite(gw_cm), f"GW150914 chirp_mass median is not finite: {gw_cm}"
    # Loose smoke guard: must lie within the training prior [20, 40] Msun.
    assert 20.0 <= gw_cm <= 40.0, (
        f"GW150914 chirp_mass median {gw_cm:.3f} Msun outside training prior [20, 40]"
    )
    # Loose smoke guard: within ±{_GW150914_CHIRP_MASS_WINDOW} Msun of GWTC-1 median.
    # This is a very broad window — purely a gross-error check, not a precision test.
    assert abs(gw_cm - _GW150914_CHIRP_MASS_TRUTH) <= _GW150914_CHIRP_MASS_WINDOW, (
        f"GW150914 chirp_mass median {gw_cm:.3f} Msun deviates more than "
        f"±{_GW150914_CHIRP_MASS_WINDOW} Msun from truth {_GW150914_CHIRP_MASS_TRUTH}"
    )

    gw_mr_match = re.search(r"GW150914_MASS_RATIO = ([0-9.]+)", output)
    assert gw_mr_match, f"no GW150914_MASS_RATIO line in output:\n{output[-3000:]}"
    gw_mr = float(gw_mr_match.group(1))
    print(f"GW150914 recovered mass_ratio median: {gw_mr:.4f}  (truth ~{_GW150914_MASS_RATIO_TRUTH})")
    assert math.isfinite(gw_mr), f"GW150914 mass_ratio median is not finite: {gw_mr}"
    # SANITY BOUND ONLY — NOT a recovery guard.
    # mass_ratio is intrinsically poorly constrained with this scaled-down network;
    # This assertion only guards against values outside the training prior
    # [0.5, 1.0] (a tautological bound for an NPE).  The meaningful recovery
    # guard is chirp_mass (±8 Msun window above) — that is what detects gross
    # problems (wrong data, wrong model, unit error).
    assert 0.5 <= gw_mr <= 1.0, (
        f"GW150914 mass_ratio median {gw_mr:.4f} outside training prior [0.5, 1.0]"
    )

    gw_rf_match = re.search(r"GW150914_RESULT_FILE = (.+)", output)
    if gw_rf_match:
        print(f"GW150914 result file: {gw_rf_match.group(1).strip()}")
