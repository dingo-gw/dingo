import bilby_pipe
from packaging.version import Version


# Since 1.9.0, Bilby uses container_image, which HTCondor transfers automatically.
# Only older versions need the image explicitly added to transfer_input_files.
BILBY_PIPE_USES_CONTAINER_UNIVERSE = Version(bilby_pipe.__version__) >= Version("1.9.0")


def configure_file_transfer(
    node,
    input_files,
    output_directory=None,
    *,
    network_files=(),
    include_container=True,
    authenticate=True,
    require_osdf_auth=False,
):
    """Add transfer and credential directives before ``node.process_node()``.

    Call inside the node's transfer-files/OSG branch. The caller selects input
    paths (including any additional-transfer-paths); this copies that list.
    Output defaults to outdir, with webdir passed explicitly for summary jobs.
    ``network_files`` adds model paths, skipping None and converting mounted
    /osdf/ paths to transfer URLs. Existing URLs are left intact.

    ``include_container`` permits adding legacy images only; native container
    images and their credentials remain Bilby's/configuration's responsibility.
    ``authenticate`` uses Bilby's input URL/credential handling. Analysis nodes
    disable it to retain their existing URLs. ``require_osdf_auth`` additionally
    requests credentials for any OSDF input, including already-prefixed URLs.
    """
    # Careful, Bilby's authentication helper can rewrite paths in place.
    files = list(input_files)
    for path in network_files:
        if path is None:
            continue
        if path.startswith("/osdf/"):
            path = f"igwn+osdf://{path.removeprefix('/osdf')}"
        files.append(path)

    if (
        include_container
        and not BILBY_PIPE_USES_CONTAINER_UNIVERSE
        and node.transfer_container
    ):
        files.append(node.inputs.container)

    need_scitokens = False
    if authenticate:
        files, need_scitokens = node.job_needs_authentication(files)
    if require_osdf_auth and any("osdf" in path for path in files):
        # Already-prefixed OSDF URLs may not trigger Bilby's authentication check.
        need_scitokens = True
    if (
        node.transfer_container
        and (BILBY_PIPE_USES_CONTAINER_UNIVERSE or include_container)
        and node.inputs.container.startswith(("osdf://", "igwn+osdf://"))
    ):
        # Native container_image transfers are not in transfer_input_files, so
        # Bilby's input authentication check above cannot see their token needs.
        need_scitokens = True
    if need_scitokens:
        # Generation may already have requested tokens for frame discovery.
        node.extra_lines.extend(
            line for line in node.scitoken_lines if line not in node.extra_lines
        )

    if output_directory is None:
        output_directory = node.inputs.outdir
    node.extra_lines.extend(
        node._condor_file_transfer_lines(
            files,
            [node._relative_topdir(output_directory, node.inputs.initialdir)],
        )
    )


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
