try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._reader import napari_get_reader
from ._widget import (
    tesellate_area,
    fit_focus_plane,
)
from ._writer import write_single_shape

__all__ = (
    "napari_get_reader",
    "write_single_shape",
    "tesellate_area",
    "fit_focus_plane",
    "__version__",
)
