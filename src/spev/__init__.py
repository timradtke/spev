from importlib.metadata import version

from importlib.util import find_spec

_HAS_SHUFFLER = False
_HAS_CHRONOS = False

if find_spec("shuffler") is not None:
    _HAS_SHUFFLER = True

if find_spec("chronos") is not None:
    _HAS_CHRONOS = True

from .task import (
    SamplePathTask,
)

__version__ = version("spev")

__all__ = [
    "SamplePathTask",
]