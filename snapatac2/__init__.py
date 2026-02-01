from sys import stderr
from ._io import read_10x_mtx
from importlib.metadata import version
from . import preprocessing as pp
from . import tools as tl
from . import metrics
from . import plotting as pl
from . import export as ex

from snapatac2._snapatac2 import (
    set_write_options, get_write_options,
    AnnData, AnnDataSet, PyDNAMotif, PyDNAMotifScanner, PyDNAMotifTest, concat,
    read, read_mtx, read_dataset, read_motifs,
)

__version__ = version("snapatac2")

__all__ = [
    "pp", "tl", "pl", "ex", "metrics",
    "set_write_options", "get_write_options",
    "AnnData", "AnnDataSet", "concat", "read", "read_mtx", "read_dataset", "read_10x_mtx", 
    "PyDNAMotif", "PyDNAMotifScanner", "PyDNAMotifTest", "read_motifs",
]

import logging
logging.basicConfig(
    stream=stderr,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO, 
)