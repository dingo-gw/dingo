"""Heavy end-to-end test: build an Apptainer image from this checkout and run the
scaled-down train -> sample -> importance-sample pipeline inside it.

Build strategy (no-subuid fakeroot):
  On hosts without /etc/subuid support the standard
  ``apptainer build --fakeroot def_file`` path fails because the kernel blocks the
  user-namespace uid-map setup that fakeroot needs for %post.  Instead we use a
  three-step sandbox workflow that works with any non-setuid Apptainer:

    1. ``apptainer build --sandbox sandbox docker://python:3.11``
       Pull the base image into an unpacked directory (no %post, no user-ns needed).
    2. Stage source + install via ``apptainer exec --writable``.
       Pip writes files owned by the calling user, which is fine because the sandbox
       directory is already owned by that user.
    3. ``apptainer build dingo-heavy.sif sandbox``
       Pack the sandbox into a read-only SIF.  No user-ns required.
"""
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
            # generous timeout: train + importance-sample; tighten once config is tuned
            timeout=60 * 60,
        )

    output = proc.stdout + "\n" + proc.stderr
    assert proc.returncode == 0, f"pipeline failed:\n{output}"

    match = re.search(r"Sample efficiency = ([0-9.]+)%", output)
    assert match, f"no 'Sample efficiency' line in output:\n{output}"
    efficiency = float(match.group(1))
    assert efficiency >= EFFICIENCY_FLOOR_PCT, (
        f"sample efficiency {efficiency}% below floor {EFFICIENCY_FLOOR_PCT}%"
    )
