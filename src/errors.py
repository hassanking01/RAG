class Patherror(Exception):
    """Report an invalid or inaccessible project path."""
    ...


class EmptyData(Exception):
    """Report missing or empty input data."""
    ...


class MaxChunkSizeError(Exception):
    """Report a chunk size outside the supported range."""
    ...


class EvaluationError(Exception):
    """Report an invalid evaluation result or dataset relationship."""
    ...


class ModelError(Exception):
    """Report invalid data for a project model."""
    pass
