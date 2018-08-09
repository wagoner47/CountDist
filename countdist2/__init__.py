from ._version import __version__, __version_info__
import os, glob

version = __version__

cd_dir = os.path.dirname(__file__)

#from .calc_distances import run_calc
from .file_io import make_query, read_db, read_db_multiple, read_metadata
from .fit_probabilities import SingleFitter, AnalyticSingleFitter, \
    ProbFitter, TooFewGroupsError
from .utils import MyConfigObj
from . import calc_distances
from .calc_distances import run_calc
