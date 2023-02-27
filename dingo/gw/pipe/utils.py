def _strip_unwanted_submission_keys(job):
    job.getenv = None
    job.universe = None
    extra_lines = []
    for line in job.extra_lines:
        if not line.startswith("priority") and not line.startswith("accounting_group"):
            extra_lines.append(line)
    job.extra_lines = extra_lines
