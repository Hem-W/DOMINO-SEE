"""Climate networks computation package based on Xarray."""
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

__author__ = """Hui-Min Wang"""
__email__ = "wanghuimin@u.nus.edu"
try:
    __version__ = _pkg_version("dominosee")
except PackageNotFoundError:  # package not installed (e.g. source checkout)
    __version__ = "0+unknown"

# Import submodules to make them available when importing the package
from . import conventions
from . import eca
from . import engine
from . import es
from . import eventorize
from . import grid
from . import network
from . import utils

# Naming conventions and data-model helpers form the top-level API
from .conventions import (
    EVENT,
    LAYER,
    LAYER_I,
    LAYER_J,
    NODE,
    NODE_I,
    NODE_J,
    TIME,
    as_pair,
    from_node_format,
    from_pair,
    is_intralayer,
    to_node_format,
    transpose_network,
)
