from ._version import __version__, __version_info__
import os, glob

version = __version__

cd_dir = os.path.dirname(__file__)

#from .calc_distances import run_calc
import calculate_distances as _calculate_distances
from calculate_distances import (
    BinSpecifier, NNCounts1D, NNCounts2D, NNCounts3D,
    ExpectedNNCounts1D, ExpectedNNCounts2D, ExpectedNNCounts3D)
from .options import set_logging_level
from .fit_probabilities import (
    SingleFitter, AnalyticSingleFitter, ProbFitter, TooFewGroupsError,
    add_extra_columns, get_delta_stats, add_corr_column, get_corr_stats,
    stats_table_to_stats_df, stats_df_to_stats_table)
from . import fit_probabilities
from .utils import MyConfigObj, ndigits
from . import calc_distances
from .calc_distances import (
    calculate_separations, calculate_separations_from_params,
    get_3d_pair_counts, get_3d_pair_counts_from_params,
    make_single_realization, convolve_pair_counts)
from .string_manip_helpers import (
    delatexify, strip_dollar_signs, numeric, stringify, double_curly_braces,
    strip_dollars_and_double_braces, round_to_significant_figures,
    pretty_print_number)
