"""Heavy end-to-end test: build an Apptainer image from this checkout and run the
scaled-down train -> sample -> importance-sample pipeline inside it."""
import os
import re
import subprocess
import tempfile

import pytest

from tests.integration.conftest import APPTAINER_CMD, HAS_APPTAINER, HAS_GPU

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HEAVY_DIR = os.path.join(os.path.dirname(__file__), "heavy")
EFFICIENCY_FLOOR_PCT = 3.0  # asserted floor sits below the 5% search target

pytestmark = [
    pytest.mark.heavy,
    pytest.mark.skipif(not HAS_APPTAINER, reason="apptainer/singularity not available"),
    pytest.mark.skipif(not HAS_GPU, reason="no GPU available"),
]


def test_heavy_container_e2e():
    with tempfile.TemporaryDirectory() as build_dir:
        # Stage the PR's code (tracked files only) + the heavy tree as the build context.
        subprocess.run(
            ["git", "archive", "--format=tar", "-o",
             os.path.join(build_dir, "dingo-src.tar"), "HEAD"],
            cwd=REPO_ROOT, check=True,
        )
        subprocess.run(
            ["cp", "-r", HEAVY_DIR, os.path.join(build_dir, "heavy")], check=True,
        )

        sif = os.path.join(build_dir, "dingo-heavy.sif")
        subprocess.run(
            [APPTAINER_CMD, "build", "--fakeroot", sif,
             os.path.join("heavy", "dingo-heavy.def")],
            cwd=build_dir, check=True,
        )

        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        proc = subprocess.run(
            [APPTAINER_CMD, "run", "--nv", sif],
            cwd=build_dir, env=env, capture_output=True, text=True,
            # generous timeout: config not yet size-tuned (search task deferred); tighten toward ~20min once tuned
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
