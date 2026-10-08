"""Fatal metric-backend errors that must not be rewritten as N/A."""


class MetricLoadError(RuntimeError):
    """Checkpoint or backend failed to load.

    Callers must fail the shard/job. Do not convert this into
    ``eligibility_status=not_applicable`` or a silent per-sample skip.
    """
