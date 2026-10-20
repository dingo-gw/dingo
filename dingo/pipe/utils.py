import bilby_pipe
from packaging.version import Version


# Since 1.9.0, Bilby uses container_image, which HTCondor transfers automatically.
# Only older versions need the image explicitly added to transfer_input_files.
BILBY_PIPE_USES_CONTAINER_UNIVERSE = Version(bilby_pipe.__version__) >= Version("1.9.0")


def _strip_unwanted_submission_keys(job):
    job.getenv = None
    job.universe = None
    extra_lines = []
    for line in job.extra_lines:
        if not line.startswith("priority") and not line.startswith("accounting_group") and not line.startswith("ENV GET HTGETTOKENOPTS"):
            extra_lines.append(line)
    job.extra_lines = extra_lines

def dict_to_string(d):
    """Convert a dictionary to a string representation."""
    return "{"  + ", ".join(f"{key}:{value}" for key, value in d.items()) + "}"
