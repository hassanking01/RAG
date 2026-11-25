from enum import Enum


class Filestatus(Enum):
    DELETED = 1
    ADDED = 2
    MODIFIED = 3


class FileType(Enum):
    PYTHON = "py"
    MARKDOWN = "md"
    TEXT = "txt"
