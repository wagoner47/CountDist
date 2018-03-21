from ._version import __version__, __version_info__
import os, glob

version = __version__

cd_dir = os.path.dirname(__file__)

#from .calc_distances import run_calc
from .file_io import read_files, resave
from .utils import MyConfigObj
from . import calc_distances
from .calc_distances import run_calc
