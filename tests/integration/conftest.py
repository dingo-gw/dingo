"""Shared skip conditions for heavy integration tests."""
import shutil
import subprocess


def _detect_apptainer():
    for cmd in ("apptainer", "singularity"):
        if shutil.which(cmd):
            return cmd
    return None


def _detect_gpu():
    if not shutil.which("nvidia-smi"):
        return False
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30
        )
        return out.returncode == 0 and "GPU 0" in out.stdout
    except (subprocess.SubprocessError, OSError):
        return False


APPTAINER_CMD = _detect_apptainer()
HAS_APPTAINER = APPTAINER_CMD is not None
HAS_GPU = _detect_gpu()
