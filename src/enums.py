from enum import Enum


class Filestatus(Enum):
    """Describe how a source file changed since the last index."""
    DELETED = 1
    ADDED = 2
    MODIFIED = 3


class FileType(Enum):
    """Identify the supported source-file formats."""
    PYTHON = "py"
    MARKDOWN = "md"
    TEXT = "txt"
