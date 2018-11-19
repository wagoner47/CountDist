from __future__ import print_function
from .utils import ndigits, init_logger, _initialize_cosmology
from . import string_manip_helpers as smh
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from chainconsumer import ChainConsumer
import CatalogUtils
from scipy.stats import kstat, kstatvar, describe
from scipy.optimize import minimize, OptimizeResult
from scipy.linalg import block_diag
import emcee
import numpy as np
import math
import pandas as pd
from astropy.table import Table
import pickle
import os, itertools, warnings
import copy, logging
import seaborn as sns

plt.rcParams["figure.facecolor"] = "white"
plt.style.use("seaborn-colorblind")
sns.set_palette("colorblind")

glogger = init_logger(os.path.splitext(os.path.basename(__file__))[0])


# Custom ValueError for too few groups (for catching during debugging)
class TooFewGroupsError(ValueError): pass


# Empty MultiIndex creation
def empty_multi_index(names):
    levels = [[] for i in range(len(names))]
    labels = [[] for i in range(len(names))]
    return pd.MultiIndex(levels=levels, labels=labels, names=names)

def empty_multi_index_nonames(nlevels):
    levels = [[] for i in range(nlevels)]
    labels = [[] for i in range(nlevels)]
    return pd.MultiIndex(levels=levels, labels=labels)


def corr_coeff(x, y, x_mean, y_mean, x_var, y_var):
    """
    Calculate the correlation coefficient between x and y. All inputs must be
    scalar or 1D array-like with the same shape

    :param x: The first value to correlate
    :type x: scalar or 1D array-like `float`
    :param y: The second value to correlate
    :type y: scalar or 1D array-like `float`
    :param x_mean: The mean of :param:`x`
    :type x_mean: scalar or 1D array-like `float`
    :param y_mean: The mean of :param:`y`
    :type y_mean: scalar or 1D array-like `float`
    :param x_var: The variance of :param:`x`
    :type x_var: scalar or 1D array-like `float`
    :param y_var: The variance of :param:`y`
    :type y_var: scalar or 1D array-like `float`
    :return r: The correlation between x and y
    :rtype r: scalar or 1D :class:`pandas.Series` `float`
    """
    if not hasattr(x_var, "__len__"):
        r = ((x - x_mean) * (y - y_mean)) / math.sqrt(x_var * y_var)
    else:
        r = ((x - x_mean) * (y - y_mean)) / (np.sqrt(x_var * y_var))
    return r



class MeanY(object):
    params = [r"$a$", r"$\alpha$", r"$\beta$", r"$b$", r"$c$", r"$d$"]
    __name__ = "mean_y"
    def __call__(self, rpo, rlo, a, alpha, beta, b, c, d, *, index=None,
                 rpo_scale=1.0, rlo_scale=1.0, **kwargs):
        """
        The mean of :math:`\frac{\Delta R_\parallel}{R_\parallel^O}`. This
        function looks like :math:`a (y - \alpha) (y - \beta) \Theta\!\left[(y -
        \alpha) (\beta - y)\right] - b \exp\left[-\frac{x^2}{2 c^2} -
        \frac{y^2}{2 d^2}\right]`, where x is the perpendicular separation and y
        is the parallel separation. If :param:`rpo` or :param:`rlo` has a
        length, they must be broadcastable.
        
        :param rpo: The observed perpendicular separation
        :type rpo: scalar or array-like `float`
        :param rlo: The observed parallel separation
        :type rlo: scalar or array-like `float`
        :param a: The constant for large and small separations
        :type a: scalar `float`
        :param alpha: The small scale boundary
        :type alpha: scalar `float`
        :param beta: The large scale boundary
        :type beta: scalar `float`
        :param b: The amplitude of the exponential term
        :type b: scalar `float`
        :param c: The width of the exponential in terms of the perpendicular
        separation
        :type c: scalar `float`
        :param d: The width of the exponential in terms of the parallel
        separation
        :type d: scalar `float`
        :key index: Optional index to use for returning a Series rather than a
        scalar or an array. Default `None`
        :type index: :class:`pandas.Index` or :class:`pandas.MultiIndex`
        :key rpo_scale: Optional scaling to apply to the observed perpendicular
        separation. The passed perpendicular separations are divided by this
        value. The default is 1.0 (in the same units as :param:`rpo`)
        :type rpo_scale: `float`
        :key rlo_scale: Optional scaling to apply to the observed parallel
        separation. The passed parallel separations are divided by this value.
        The default is 1.0 (in the same units as :param:`rlo`)
        :type rlo_scale: `float`
        :return f: The function evaluated at the separation(s)
        :rtype f: scalar, :class:`numpy.ndarray`, or :class:`pandas.Series`
        `float`
        """
        scaled_rpo = copy.deepcopy(rpo) / rpo_scale
        scaled_rlo = copy.deepcopy(rlo) / rlo_scale
        # Because the Heaviside function is only defined in numpy, it doesn't
        # make sense to use math.exp even for scalars
        if hasattr(rpo, "__len__"):
            scaled_rpo = np.asarray(scaled_rpo)
        if hasattr(rlo, "__len__"):
            scaled_rlo = np.asarray(scaled_rlo)
        if (hasattr(rpo, "__len__") or hasattr(rlo, "__len__")) and not all(
            (m == n) or (m == 1) or (n == 1) for m, n in zip(
                np.atleast_1d(scaled_rpo).shape[::-1],
                np.atleast_1d(scaled_rlo).shape[::-1])):
            raise ValueError("Array-like separations must be broadcastable")
        f = ((a * (scaled_rlo - alpha) * (scaled_rlo - beta) *
              np.heaviside((scaled_rlo - alpha) * (beta - scaled_rlo), 0)) -
             (b * np.exp(-0.5 * ((scaled_rpo / c)**2 + (scaled_rlo / d)**2))))
        if index is not None:
            f = pd.Series(f, index=index)
        return f
mean_y = MeanY()

# def mean_y(rpo, rlo, a, alpha, beta, s, **kwargs):
#     """
#     The mean of :math:`\frac{\Delta R_\parallel}{R_\parallel^O}`. This function
#     looks like :math:`a x^\alpha y^\beta \exp[-y^2 / 2 s^2]`, where x is the
#     perpendicular separation and y is the parallel separation. If :param:`rpo`
#     or :param:`rlo` has a length, it will be assumed that they are the indices
#     from a Series/DataFrame, and a Series will be returned.
# 
#     :param rpo: The observed perpendicular separation
#     :type rpo: scalar or array-like `float`
#     :param rlo: The observed parallel separation
#     :type rlo: scalar or array-like `float`
#     :param a: The amplitude of the function
#     :type a: scalar `float`
#     :param alpha: The power on the observed perpendicular separation
#     :type alpha: scalar `float`
#     :param beta: The power law on the observed parallel separation
#     :type beta: scalar `float`
#     :param s: The scale of the exponential term
#     :type s: scalar `float`
#     :key index: Optional index to use for returning a Series rather than a
#     scalar or an array. Default `None`
#     :type index: :class:`pandas.Index` or :class:`pandas.MultiIndex`
#     :key rpo_scale: Optional scaling to apply to the observed perpendicular
#     separation. The default is 1.0 (in the same units as :param:`rpo`)
#     :type rpo_scale: `float`
#     :key rlo_scale: Optional scaling to apply to the observed parallel
#     separation. The default is 1.0 (in the same units as :param:`rlo`)
#     :type rlo_scale: `float`
#     :return f: The function evaluated at the separation(s)
#     :rtype f: scalar, :class:`numpy.ndarray`, or :class:`pandas.Series` `float`
#     """
#     mean_y.params = [r"$a$", r"$\alpha$", r"$\beta$", r"$s$"]
#     index = kwargs.pop("index", None)
#     scaled_rpo = copy.deepcopy(rpo) / kwargs.pop("rpo_scale", 1.0)
#     scaled_rlo = copy.deepcopy(rlo) / kwargs.pop("rlo_scale", 1.0)
#     if not hasattr(rpo, "__len__") and not hasattr(rlo, "__len__"):
#         fexp = math.exp
#     elif hasattr(rpo, "__len__") and hasattr(rlo, "__len__"):
#         scaled_rpo = np.atleast_1d(scaled_rpo)
#         scaled_rlo = np.atleast_1d(scaled_rlo)
#         if not all((m == n) or (m == 1) or (n == 1) for m, n in
#                    zip(scaled_rpo.shape[::-1], scaled_rlo.shape[::-1])):
#             raise ValueError("Array-like separations must be broadcastable: "\
#                                  "shape(rpo) = {}, shape(rlo) = "\
#                                  "{}".format(scaled_rpo.shape,
#                                              scaled_rlo.shape))
#         fexp = np.exp
#     else:
#         raise ValueError("Separations must either be both scalar or both "\
#                              "array-like")
#     f = (a * (scaled_rpo**alpha) * (scaled_rlo**beta) *
#          fexp(-0.5 * scaled_rlo**2 / s**2))
#     if index is not None:
#         f = pd.Series(f, index=index)
#     return f


class VarY(object):
    params = [r"$a$", r"$b$", r"$s_1$", r"$s_2$", r"$\rho$"]
    __name__ = "var_y"
    def __call__(self, rpo, rlo, a, b, s_1, s_2, rho, *, index=None,
                 rpo_scale=1.0, rlo_scale=1.0, **kwargs):
        """
        The variance of :math:`\frac{\Delta R_\parallel}{\sqrt{2} \chi'(\bar{z})
        \sigma_z(\bar{z})}`. This function looks like :math:`a - b \exp[-0.5
        \vec{ r}^T C^{-1} \vec{r}]`, where :math:`\vec{r}` is a vector of the
        observed perpendicular and parallel separations, and C looks like a
        covariance matrix if :param:`s1` and :param:`s2` are variances and
        :param:`rho` is the correlation coefficient. If :param:`rpo` or
        :param:`rlo` has a length, both will assumed to be indices from a
        Series/DataFrame, and a Series will be returned.

        :param rpo: The observed perpendicular separations
        :type rpo: scalar or array-like `float`
        :param rlo: The observed parallel separations
        :type rlo: scalar or array-like `float`
        :param a: The constant approached at large :param:`rpo` and :param:`rlo`
        :type a: scalar `float`
        :param b: The amplitude on the exponential term
        :type b: scalar `float`
        :param s_1: The width of the exponential associated with the observed
        perpendicular separation
        :type s_1: scalar `float`
        :param s_2: The width of the exponential associated with the observed
        parallel separation
        :type s_2: scalar `float`
        :param rho: The mixing of the perpendicular and parallel contriubtions
        to the exponential
        :type rho: scalar `float`
        :key index: Optional index to use for returning a Series rather than an
        array for array-like :param:`rpo` and/or :param:`rlo`. Default `None`
        :type index: :class:`pandas.Index` or :class:`pandas.MultiIndex`,
        optional
        :key rpo_scale: Optional scaling to apply to the observed perpendicular
        separation. The passed perpendicular separations are divided by this
        value. The default is 1.0 (in the same units as :param:`rpo`)
        :type rpo_scale: `float`
        :key rlo_scale: Optional scaling to apply to the observed parallel
        separation. The passed parallel separations are divided by this value.
        The default is 1.0 (in the same units as :param:`rlo`)
        :type rlo_scale: `float`
        :return f: The function evaluated at the separation(s)
        :rtype f: scalar, :class:`numpy.ndarray`, or :class:`pandas.Series`
        `float`
        """
        scaled_rpo = copy.deepcopy(rpo) / rpo_scale
        scaled_rlo = copy.deepcopy(rlo) / rlo_scale
        inv_weight = 1. / (s_1**2 * s_2**2 * (1 - rho**2))
        cinv = [x * inv_weight for x in [s_2**2, s_1**2, -2 * rho * s_1 * s_2]]
        if hasattr(rpo, "__len__"):
            scaled_rpo = np.asarray(scaled_rpo)
        if hasattr(rlo, "__len__"):
            scaled_rlo = np.asarray(scaled_rlo)
        if (hasattr(rpo, "__len__") or hasattr(rlo, "__len__")) and not all(
            (m == n) or (m == 1) or (n == 1) for m, n in zip(
                np.atleast_1d(scaled_rpo).shape[::-1],
                np.atleast_1d(scaled_rlo).shape[::-1])):
            raise ValueError("Array-like separations must be broadcastable")
        f = a - b * np.exp(-0.5 * (scaled_rpo**2 * cinv[0] +
                                   scaled_rlo**2 * cinv[1] +
                                   scaled_rpo * scaled_rlo * cinv[2]))
        if index is not None:
            f = pd.Series(f, index=index)
        return f
var_y = VarY()


# def var_y(rpo, rlo, a, b, s1, s2, rho, **kwargs):
#     """
#     The variance of :math:`\frac{\Delta R_\parallel}{\sqrt{2} \chi'(\bar{z})
#     \sigma_z(\bar{z})}`. This function looks like :math:`a - b \exp[-0.5 \vec{
#     r}^T C^{-1} \vec{r}]`, where :math:`\vec{r}` is a vector of the observed
#     perpendicular and parallel separations, and C looks like a covariance matrix
#     if :param:`s1` and :param:`s2` are variances and :param:`rho` is the
#     correlation coefficient. If :param:`rpo` or :param:`rlo` has a length, both
#     will assumed to be indices from a Series/DataFrame, and a Series will be
#     returned.
# 
#     :param rpo: The observed perpendicular separations
#     :type rpo: scalar or array-like `float`
#     :param rlo: The observed parallel separations
#     :type rlo: scalar or array-like `float`
#     :param a: The constant approached at large :param:`rpo` and :param:`rlo`
#     :type a: scalar `float`
#     :param b: The amplitude on the exponential term
#     :type b: scalar `float`
#     :param s1: The width of the exponential associated with the observed
#     perpendicular separation
#     :type s1: scalar `float`
#     :param s2: The width of the exponential associated with the observed
#     parallel separation
#     :type s2: scalar `float`
#     :param rho: The mixing of the perpendicular and parallel contriubtions to
#     the exponential
#     :type rho: scalar `float`
#     :key index: Optional index to use for returning a Series rather than an
#     array for array-like :param:`rpo` and/or :param:`rlo`. Default `None`
#     :type index: :class:`pandas.Index` or :class:`pandas.MultiIndex`, optional
#     :key rpo_scale: Optional scaling to apply to the observed perpendicular
#     separation. The default is 1.0 (in the same units as :param:`rpo`)
#     :type rpo_scale: `float`
#     :key rlo_scale: Optional scaling to apply to the observed parallel
#     separation. The default is 1.0 (in the same units as :param:`rlo`)
#     :type rlo_scale: `float`
#     :return f: The function evaluated at the separation(s)
#     :rtype f: scalar, :class:`numpy.ndarray`, or :class:`pandas.Series` `float`
#     """
#     var_y.params = [r"$a$", r"$b$", r"$s_1$", r"$s_2$", r"$\rho$"]
#     index = kwargs.pop("index", None)
#     scaled_rpo = copy.deepcopy(rpo) / kwargs.pop("rpo_scale", 1.0)
#     scaled_rlo = copy.deepcopy(rlo) / kwargs.pop("rlo_scale", 1.0)
#     inv_weight = 1. / (s1**2 * s2**2 * (1 - rho**2))
#     cinv = [x * inv_weight for x in [s2**2, s1**2, -2 * rho * s1 * s2]]
#     if not hasattr(rpo, "__len__") and not hasattr(rlo, "__len__"):
#         fexp = math.exp
#     elif hasattr(rpo, "__len__") and hasattr(rlo, "__len__"):
#         scaled_rpo = np.atleast_1d(scaled_rpo)
#         scaled_rlo = np.atleast_1d(scaled_rlo)
#         if not all((m == n) or (m == 1) or (n == 1) for m, n in
#                    zip(scaled_rpo.shape[::-1], scaled_rlo.shape[::-1])):
#             raise ValueError("Array-like separations must be broadcastable: "\
#                                  "shape(rpo) = {}, shape(rlo) = "\
#                                  "{}".format(scaled_rpo.shape,
#                                              scaled_rlo.shape))
#         fexp = np.exp
#     else:
#         raise ValueError("Separations must either be both scalar or both "\
#                              "array-like")
#     f = a - b * fexp(-0.5 * (scaled_rpo**2 * cinv[0] +
#                              scaled_rlo**2 * cinv[1] +
#                              scaled_rpo * scaled_rlo * cinv[2]))
#     if index is not None:
#         f = pd.Series(f, index=index)
#     return f

def stats_table_to_stats_df(table):
    df = table.to_pandas()
    df.set_index(["RPO_BIN", "RLO_BIN"], inplace=True)
    df.columns = pd.MultiIndex.from_tuples([("_".join(col.split("_")[:2]),
                                             col.split("_")[2]) for col in
                                            df.columns])
    return df

def stats_df_to_stats_table(df):
    df_renamed = df.reset_index()
    df_renamed.columns = ["_".join(col).rstrip("_") for col in
                          df_renamed.columns]
    return Table.from_pandas(df_renamed)

def _check_pickleable(attrs):
    pickles = []
    for attr in attrs:
        try:
            pickle.dumps(attr)
            pickles.append(False)
        except (pickle.PicklingError, TypeError):
            pickles.append(True)
    return (not any(pickles), pickles)

def _perp_mean_scale(rpo, rlo, zbar, sigma_z):
    return rpo

def _perp_var_scale(rpo, rlo, zbar, sigma_z):
    return ((np.sqrt(0.5) * rpo * sigma_z * (1 + zbar) *
             CatalogUtils.dr_dz(zbar)) / CatalogUtils.dist(zbar))

def _par_mean_scale(rpo, rlo, zbar, sigma_z):
    return rlo

def _par_var_scale(rpo, rlo, zbar, sigma_z):
    return (np.sqrt(2.0) * sigma_z * (1 + zbar) *
            CatalogUtils.dr_dz(zbar))

def _add_bin_column(seps, orig_col_name, bin_col_name, bin_size):
    seps[bin_col_name] = np.floor(seps.loc[:,orig_col_name] / bin_size).astype(
        int)

def _add_zbar(seps):
    seps["ZBAR"] = CatalogUtils.z_at_chi(seps["AVE_D_OBS"])

def _add_delta_column(seps, direction, scale_func, dcol_name, sigma_z):
    tcol_name = "R_{}_T".format(direction)
    ocol_name = "R_{}_O".format(direction)
    scale = 1. / scale_func(seps.loc[:,"R_PERP_O"], seps.loc[:,"R_PAR_O"],
                            seps.loc[:,"ZBAR"], sigma_z)
    seps[dcol_name] = seps[tcol_name].sub(seps[ocol_name]).mul(scale)

def add_extra_columns(seps, perp_bin_size, par_bin_size, sigma_z):
    """This function adds some of the extra data columns to the input DataFrame
    that are needed for grouping and generating the statistics. It does not add
    the column for the correlation, that is handled by a separate function.

    :param seps: The DataFrame of the separations which should already contain,
    at a minimum, columns 'R_PERP_O', 'R_PAR_O', 'R_PERP_T', 'R_PAR_T', and
    'AVE_D_OBS'. Additional columns are ignored
    :type seps: :class:`pandas.DataFrame`
    :param perp_bin_size: The bin size to use for binning 'R_PERP_O', in the
    same units as 'R_PERP_O'
    :type perp_bin_size: `float`
    :param par_bin_size: The bin size to use for binning 'R_PAR_O', in the same
    units as 'R_PAR_O'
    :type par_bin_size: `float`
    :param sigma_z: The redshift error assumed for the separations
    :type sigma_z: `float`
    """
    logger = glogger.getChild(__name__)
    logger.debug("Add column RPO_BIN")
    _add_bin_column(seps, "R_PERP_O", "RPO_BIN", perp_bin_size)
    logger.debug("Add column RLO_BIN")
    _add_bin_column(seps, "R_PAR_O", "RLO_BIN", par_bin_size)
    logger.debug("Add column ZBAR")
    _add_zbar(seps)
    logger.debug("Add column DELTA_R_PERP")
    _add_delta_column(seps, "PERP", _perp_mean_scale, "DELTA_R_PERP", sigma_z)
    logger.debug("Add column x")
    _add_delta_column(seps, "PERP", _perp_var_scale, "x", sigma_z)
    logger.debug("Add column DELTA_R_PAR")
    _add_delta_column(seps, "PAR", _par_mean_scale, "DELTA_R_PAR", sigma_z)
    logger.debug("Add column y")
    _add_delta_column(seps, "PAR", _par_var_scale, "y", sigma_z)

def get_delta_stats(seps, min_counts=200, min_bins=2):
    """Calculate the sample means/variances and the variances on them for each
    of the four 'delta' columns: 'DELTA_R_PERP', 'x', 'DELTA_R_PAR', and 'y'.
    The return will have MultiIndex rows labeled by their perpendicular and
    parallel separation bins ('RPO_BIN' and 'RLO_BIN'), and MultiIndex columns
    where the 0th level is which statistic it is ('mean_x' for 'DELTA_R_PERP',
    'var_x' for 'x', 'mean_y' for 'DELTA_R_PAR', and 'var_y' for 'y') and the
    1st level labels which item it is ('mean' is always the statistic and
    'variance' is always the variance on it, although 'mean' is actually the
    sample variance in the case of 'var_x' and 'var_y').

    If the extra columns needed for the calculations have not been added yet via
    :function:`add_extra_columns`, an AttributeError will be raised. Please make
    sure to call that function first to have all needed columns

    :param seps: The DataFrame of the separations, which must have columns
    'RPO_BIN', 'RLO_BIN', 'DELTA_R_PERP', 'x', 'DELTA_R_PAR', and 'y'. All of
    these columns may be added with a single call to
    :function:`add_extra_columns`
    :type seps: :class:`pandas.DataFrame`
    :param min_counts: The minimum number of pairs in each bin in order to
    keep the bin. If there are fewer than :param:`min_bins` of these, an error
    will be thrown and the statistics will not be calculated. Default 200
    :type min_counts: `int`, optional
    :param min_bins: The minimum number of bins needed after filtering based
    on number of pairs. If there aren't at least this many bins, an error
    will be thrown, and the statistics will not be calculated. Default 2
    :type min_bins: `int`, optional
    :returns stats: A DataFrame containing the statistics functions calculated.
    See above for the index names and column descriptions
    :rtype stats: :class:`pandas.DataFrame`
    :raises AttributeError: if any of the needed columns are missing
    :riases TooFewGroupsError: if there are fewer than :param:`min_bins`
    bins with at least :param:`min_counts` pairs
    """
    logger = glogger.getChild(__name__)
    logger.debug("Drop NAN columns")
    seps_filt = seps.dropna(axis=1)
    required_cols = ["RPO_BIN", "RLO_BIN",
                     "DELTA_R_PERP", "x",
                     "DELTA_R_PAR", "y"]
    logger.debug("Check for missing required columns")
    missing_cols = [col not in seps_filt for col in required_cols]
    if np.any(missing_cols):
        raise AttributeError("Missing required column(s): {}".format(
                list(itertools.compress(required_cols, missing_cols))))
    logger.debug("Select only needed columns for grouping")
    sub_seps = seps_filt.loc[:,required_cols].copy()
    logger.debug("Group separations and filter based on number of pairs")
    grouped = sub_seps.groupby(["RPO_BIN", "RLO_BIN"]).filter(
        lambda x: len(x) >= min_counts).groupby(["RPO_BIN", "RLO_BIN"])
    logger.debug("Check for enough bins")
    if len(grouped) <= min_bins:
        raise TooFewGroupsError(
            "Only {} bins with at least {} pairs. Consider decreasing "\
                "the minimum number of pairs, increasing the bin size, "\
                "or using a larger catalog".format(len(grouped), min_counts))
    logger.debug("Get sample mean of (R_PERP_T - R_PERP_O) / R_PERP_O")
    stats_mx = pd.concat([grouped["DELTA_R_PERP"].agg(kstat, 1),
                          grouped["DELTA_R_PERP"].agg(kstatvar, 1)],
                         keys=["mean", "variance"], axis=1)
    logger.debug("Get sample variance of x")
    stats_vx = pd.concat([grouped["x"].agg(kstat, 2),
                          grouped["x"].agg(kstatvar, 2)],
                         keys=["mean", "variance"], axis=1)
    logger.debug("Get sample mean of (R_PAR_T - R_PAR_O) / R_PAR_O")
    stats_my = pd.concat([grouped["DELTA_R_PAR"].agg(kstat, 1),
                          grouped["DELTA_R_PAR"].agg(kstatvar, 1)],
                         keys=["mean", "variance"], axis=1)
    logger.debug("Get sample variance of y")
    stats_vy = pd.concat([grouped["y"].agg(kstat, 2),
                          grouped["y"].agg(kstatvar, 2)],
                         keys=["mean", "variance"], axis=1)
    logger.debug("Concatenate statistics")
    stats = pd.concat([stats_mx, stats_vx, stats_my, stats_vy],
                      keys=["mean_x", "var_x", "mean_y", "var_y"],
                      axis=1)
    return stats

def add_corr_column(seps, x_mean, y_mean, x_var, y_var):
    """Add a column for the correlation for each pair to the input DataFrame.
    the means and variances may be calculated from the data or fit. The added
    column has name 'r'

    :param seps: The DataFrame of the separations, which must have a minimum
    columns 'R_PERP_T' and 'R_PAR_T'
    :type seps: :class:`pandas.DataFrame`
    :param x_mean: The mean of 'R_PERP_T' for each pair, based on their
    observed positions and separations
    :type x_mean: 1D array-like `float`
    :param y_mean: The mean of 'R_PAR_T' for each pair, based on their
    observed positions and separations
    :type y_mean: 1D array-like `float`
    :param x_var: The variance of 'R_PERP_T' for each pair, based on their
    observed positions and separations
    :type x_var: 1D array-like `float`
    :param y_var: The variance of 'R_PERP_T' for each pair, based on their
    observed positions and separations
    :type y_var: 1D array-like `float`
    """
    seps["r"] = corr_coeff(seps.loc[:,"R_PERP_T"], seps.loc[:,"R_PAR_T"],
                           x_mean, y_mean, x_var, y_var)

def get_corr_stats(seps, min_counts=200, min_bins=2):
    """Get the sample mean and variance on sample mean of the correlation.  The
    returned DataFrame will have columns 'mean' and 'variance', and indices
    labeled 'RPO_BIN' and 'RLO_BIN'.

    :param seps: The DataFrame of the separations, containing at least columns
    'RPO_BIN', 'RLO_BIN', and 'r'
    :type seps: :class:`pandas.DataFrame`
    :param min_counts: The minimum number of pairs needed in each bin. Bins
    with fewer pairs will be discarded. Default 200
    :type min_counts: `int`, optional
    :param min_bins: The minimum number of bins that need to be kept. Default 2
    :type min_bins: `int`, optional
    :returns stats: A DataFrame containing a column 'mean' for the sample mean
    of 'r' and 'variance' for the variance on the sample mean
    :rtype stats: :class:`pandas.DataFrame`
    :raises AttributeError: if any of the required columns are missing
    :raises TooFewGroupsError: if fewer than :param:`min_bins` bins have at
    least :param:`min_counts` pairs
    """
    logger = glogger.getChild(__name__)
    logger.debug("Drop NAN columns")
    seps_filt = seps.dropna(axis=1)
    required_cols = ["RPO_BIN", "RLO_BIN", "r"]
    logger.debug("Check for missing required columns")
    missing_cols = [col not in seps_filt for col in required_cols]
    if np.any(missing_cols):
        raise AttributeError("Missing required column(s): {}".format(
            list(itertools.compress(required_cols, missing_cols))))
    logger.debug("Select only needed columns for grouping")
    sub_seps = seps_filt.loc[:,required_cols].copy()
    logger.debug("Groupd separations and filter based on number of pairs")
    grouped = sub_seps.groupby(["RPO_BIN", "RLO_BIN"]).filter(
        lambda x: len(x) >= min_counts).groupby(["RPO_BIN", "RLO_BIN"])
    logger.debug("Check for enough bins")
    if len(grouped) <= min_bins:
        raise TooFewGroupsError(
            "Only {} bins with at least {} pairs. Consider decreasing "\
                "the minimum number of pairs, increasing the bin size, "\
                "or using a larger catalog".format(len(grouped), min_counts))
    logger.debug("Get sample mean of Corr(R_PERP_T, R_PAR_T)")
    stats = pd.concat([grouped["r"].agg(kstat, 2),
                       grouped["r"].agg(kstatvar, 2)],
                      keys=[("mean_r", "mean"), ("mean_r", "variance")],
                      axis=1)
    return stats


def __make_clean_dict__(orig, param_names=None):
    """
    Make a single `dict` or a 1D array-like into a `dict` with keys that have
    possibly been stripped of latex formatting.
    """
    if isinstance(orig, dict):
        return dict((smh.delatexify(key), value) for key, value in orig.items())
    elif not hasattr(orig[0], "__len__"):
        if param_names is None:
            raise ValueError("Must give parameter names if creating dict from "\
                                 "array-like")
        return dict(zip(smh.delatexify(param_names), orig))
    else:
        return np.array(list(map(lambda o: __make_clean_dict__(
                    o, param_names), orig)))


# This class was borrowed/stolen (with slight modification) from emcee...
class _FitFunctionWrapper(object):
    """
    A hack to make dynamically set functions with `args` and/or `kwargs`
    pickleable. This also handles vectorization (with respect to the
    parameters theta), in which case a map object is returned
    """
    def __init__(self, f, params, args=None, kwargs=None):
        self.f = f
        self.params = params
        self.func_name = f.__name__
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __call__(self, x, y, theta, **kwargs):
        theta = __make_clean_dict__(theta, self.params)
        if isinstance(theta, dict):
            try:
                return self.f(x, y, *self.args, **theta, **self.kwargs,
                              **kwargs)
            except TypeError:
                print("TypeError when calling ", self.func_name, flush=True)
                print("\ttheta = ", theta, flush=True)
                print("\targs = ", self.args, flush=True)
                print("\tfixed kwargs = ", self.kwargs, flush=True)
                print("\tadditional kwargs = ", kwargs, flush=True)
                raise
        else:
            return map(lambda t: self.__call__(x, y, t, **kwargs), theta)

def flat_prior(theta, extents):
    """A generic flat prior with a set of extents that also must be given.
    This will be wrapped within :class:`_FlatPriorFunctionWrapper`
    
    :param theta: The parameter values to check, as a dictionary
    :type theta: `dict`
    :param extents: The allowed ranges of the parameters, as a dictionary
    :type extents: `dict`
    :return: The value of the prior, 0 in allowed parameter space or
    negative infinity otherwise
    :rtype: `float`
    """
    theta = __make_clean_dict__(theta)
    extents = __make_clean_dict__(extents)
    if all([pmin < theta[p] < pmax for p, (pmin, pmax) in extents.items()]):
        return 0.0
    return -math.inf

class _FlatPriorFunctionWrapper(object):
    """A wrapper for a flat prior function with extents that can be given or
    updated

    :param extents: The range of allowed values for each parameter, as a
    dictionary
    :type extents: `dict`
    """
    def __init__(self, extents):
        self.extents = extents

    def __getstate__(self):
        d = self.__dict__.copy()
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __call__(self, theta):
        """The call method to make this pickleable

        :param theta: The parameter values to be checked, as either a dict or
        an array-like of dicts
        :type theta: `dict` or 1D array-like of `dict`
        """
        return flat_prior(theta, self.extents)


class SingleFitter(object):
    """
    Fit a single function to a single set of data.
    """

    def __init__(self, data, index_names, col_names, func=None, prior=None,
                 param_names=None, fitter_name=None, *, rpo_size=1.0,
                 rlo_size=1.0, func_args=None, func_kwargs=None, **kwargs):
        """Initialize the SingleFitter

        :param data: A DataFrame containing the observed parallel and
        perpendicular separations as indices, as well as columns for the data
        and the variance. Note that this does force the variance to be diagonal.
        :type data: :class:`pandas.DataFrame`
        :param index_names: A list containing the names of the index columns,
        with the first element corresponding to the perpendicular separation and
        the second to the parallel separation.
        :type index_names: `list` `str`
        :param col_names: A list containing the names of the columns from the
        DataFrame to be used, with the first element corresponding to the data
        column and the second corresponding the the variance.
        :type col_names: `list` `str`
        :param func: The function to fit to the data. Must be set before
        fitting, and parameter names must be given simultaneously. Use
        :function:`SingleFitter.set_fit_func` to set or change this later.
        Default `None`
        :type func: `function` or `None`, optional
        :param prior: The prior to use for fitting the data. Must be set before
        doing a fit with the prior likelihood. Use
        :function:`SingleFitter.set_prior_func` to set or change this
        later. This could also be a dictionary of the parameter extents for a
        flat prior, which will be generated automatically. Default `None`
        :type prior: `function` or `None`, optional
        :param param_names: The names of the parameters for the fitting
        function. Must be set with function. Default `None`
        :type param_names: 1D array-like `str` or `None`, optional
        :param fitter_name: A name for the SingleFitter instance,
        for representation. If `None`, the name will be set to
        'SingleFitter'. Default `None`
        :type fitter_name: `str` or `None`, optional
        :key rpo_size: The size of the observed perpendicular separation bins
        to assume is used in the data. Default 1.0
        :type rpo_size: `float`
        :key rlo_size: The size of the observed parallel separation bins to
        assume is used in the data. Default 1.0
        :type rlo_size: `float`
        :key func_kwargs: The keyword arguments to use for the fitting
        function, if it has been given, as a dictionary. Default `None` for
        no keyword arguments if the function is given.
        :type func_kwargs: `dict`
        :key func_args: The positional arguments (other than the separations
        and the parameters) to use for the fitting function, if it has been
        given, as an array-like. Default `None` for no additional positional
        arguments if the function is given.
        :type func_args: array-like
        """
        self._get_name(fitter_name)
        self.logger = init_logger(self.name)
        self.logger.debug("Set up data and variance")
        self.logger.debug("Data columns: \n{}".format(data.columns))
        self.data = data
        self.index_names = index_names
        self.col_names = col_names
        self.rpo_size = rpo_size
        self.rlo_size = rlo_size
        self.rpo = ((self.data.index.get_level_values(self.index_names[0]) +
                     0.5) * self.rpo_size)
        self.rlo = ((self.data.index.get_level_values(self.index_names[1]) +
                     0.5) * self.rlo_size)
        self.logger.debug("Set fitting function and prior")
        self.set_fit_func(func, param_names, kwargs.pop("func_args", None),
                          kwargs.pop("func_kwargs", None))
        self.set_prior_func(prior)
        self.logger.debug("Initialize sampler and best fit parameters as None")
        self._best_fit_params = None
        self._samples = None
        self.logger.debug("__init__ complete")

    def __repr__(self):
        return "%s(ndim=%r, best_fit=%r)" % (self.name, self.ndim,
                                             self._best_fit_params)

    def __getstate__(self):
        d = self.__dict__.copy()
        if "logger" in d:
            d["logger"] = (d["logger"].name, d["logger"].getEffectiveLevel())
        if "c" in d:
            d["c"] = (d["c"].chains, d["c"].config, d["c"].config_truth)
        if "c_walkers" in d:
            d["c_walkers"] = (d["c_walkers"].chains, d["c_walkers"].config,
                              d["c_walkers"].config_truth)
        return d

    def __setstate__(self, d):
        if "logger" in d:
            level = d["logger"][1]
            d["logger"] = logging.getLogger(d["logger"][0])
            d["logger"].setLevel(level)
        if "c" in d:
            c = ChainConsumer()
            [c.add_chain(chain.chain, parameters=chain.parameters,
                         name=chain.name, posterior=chain.posterior,
                         color=chain.color, walkers=chain.walkers) for
             chain in d["c"][0]]
            for i, chain in enumerate(d["c"][0]):
                c.chains[i].config = chain.config
            c.config = d["c"][1]
            c.config_truth = d["c"][2]
            d["c"] = c
        if "c_walkers" in d:
            c = ChainConsumer()
            [c.add_chain(chain.chain, parameters=chain.parameters,
                         name=chain.name, posterior=chain.posterior,
                         color=chain.color, walkers=chain.walkers) for
             chain in d["c_walkers"][0]]
            for i, chain in enumerate(d["c_walkers"][0]):
                c.chains[i].config = chain.config
            c.config = d["c_walkers"][1]
            c.config_truth = d["c_walkers"][2]
            d["c_walkers"] = c
        self.__dict__.update(d)

    @property
    def nburnin(self):
        return self._nburnin

    @nburnin.setter
    def nburnin(self, value):
        self._nburnin = value

    @property
    def best_fit(self):
        return self._best_fit_params

    @property
    def samples(self):
        return self._samples

    @property
    def data_vs_rlo(self):
        if self.index_names[0] != self.data.index.names[0]:
            return self.data.swaplevel(0, 1, axis=0).sort_index()
        return self.data.copy()

    @property
    def data_vs_rpo(self):
        if self.index_names[0] == self.data.index.names[0]:
            return self.data.swaplevel(0, 1, axis=0).sort_index()
        return self.data.copy()

    def _get_name(self, fitter_name):
        self.name = ("{}.{}".format(self.__class__.__name__, fitter_name) if
                     fitter_name is not None else self.__class__.__name__)

    def set_fit_func(self, func, param_names, args=None, kwargs=None):
        """Set a (new) fitting function with parameter names

        :param func: The fitting function to use
        :type func: `function` or `None`
        :param param_names: The names of the parameters for the fitting function
        :type param_names: 1D array-like `str` or `None`
        :param args: The additional positional arguments to pass to the fitting
        function, as a list. Default None
        :type args: `list`, optional
        :param kwargs: Any keyword arguments to pass to the fitting function,
        as a dict. Default None
        :type kwargs: `dict`, optional
        """
        if func is None:
            self.logger.debug("Setting fitting function to None")
            self.params = None
            self.ndim = None
            self.f = None
        else:
            if param_names is None:
                if hasattr(func, "params"):
                    param_names = func.params
                else:
                    raise ValueError("Parameter names must be given when "
                                     "fitting function is specified")
            self.logger.debug("Setting fitting function and parameters")
            self.params = param_names
            self.ndim = len(param_names)
            self.f = _FitFunctionWrapper(func, param_names, args, kwargs)
        self.logger.debug("done")

    def set_prior_func(self, prior):
        """Set a (new) prior likelihood function

        :param prior: The prior probability function
        :type prior: `function` or `None`
        """
        if prior is None:
            self.logger.debug("Setting prior to None")
            self.pr = prior
        else:
            self.logger.debug("Setting up prior function")
            if isinstance(prior, dict):
                if not hasattr(self, "pr") or self.pr is None:
                    self.pr = _FlatPriorFunctionWrapper(prior)
                else:
                    self.pr.extents = prior
            else:
                self.pr = prior
        self.logger.debug("done")

    def model(self, rpo, rlo, index=None):
        """Get the best fit model at the given separations

        :param rpo: The observed perpendicular separation
        :type rpo: scalar or array-like `float`
        :param rlo: The observed parallel separation
        :type rlo: scalar or array-like `float`
        :param index: The index to add to the Series. Ignored for scalar
        returns. If `None`, the index will be created from :param:`rpo` and
        :param:`rlo`, with names ['RPO_BIN', 'RLO_BIN']. Default `None`
        :type index: :class:`pandas.Index`, :class:`pandas.MultiIndex`, or
        `None`, optional
        :return m: The value of the best fit model at the points. This will
        be 1D if :param:`rpo` and :param:`rlo` are both array-like, with both
        separations flattened
        :rtype m: scalar or :class:`pandas.Series` `float`
        """
        if self._best_fit_params is None:
            raise AttributeError("Cannot get best fit model without "\
                                     "best fit parameters")
        m = self.f(rpo, rlo, self._best_fit_params, index=index)
        return m

    def model_with_errors(self, rpo, rlo, index=None):
        """Get the median and 68 percent contours of the model at the
        separations

        :param rpo: The observed perpendicular separation
        :type rpo: scalar or array-like `float`
        :param rlo: The observed parallel separation
        :type rlo: scalar or array-like `float`
        :param index: The index to add to the Series. Ignored for single
        separation. If `None`, the index will be created from :param:`rpo` and
        :param:`rlo`, with names ['RPO_BIN', 'RLO_BIN']. Default `None`
        :type index: :class:`pandas.Index`, :class:`pandas.MultiIndex`, or
        `None`, optional
        :return m: The 16th, 50th, and 84th percentiles of the model at the
        points. The 0th axis is the percentiles, and the first axis is the
        flattened points, if not both scalar
        :rtype m: :class:`pandas.Series` or :class:`pandas.DataFrame` `float`
        """
        if self._samples is None:
            raise AttributeError("Cannot get model with errors if samples "\
                                     "not available")
        samples = self._samples[:,self._nburnin:,:].reshape((-1, self.ndim))
        meval = self.f(rpo, rlo, samples, index=index)
        if not hasattr(rpo, "__len__") and not hasattr(rlo, "__len__"):
            meval = pd.Series(list(meval))
            m = meval.quantile([0.16, 0.5, 0.84])
        else:
            meval = pd.DataFrame(dict(zip(range(len(samples)), meval)),
                                 index=index)
            m = meval.quantile(q=[0.16, 0.5, 0.84], axis="columns").T
        return m

    def plot(self, rpo_label, rlo_label, ylabel, bins, perp_bin_scale, 
             par_bin_scale, exp, is_rpo=False, logx=False, logy=False, 
             filename=None, figsize=None, display=False, text_size=22, 
             with_fit=False, point_alpha=1.0):
        """Plot the data (and optionally the best fit to the data) at a number
        of individual perpendicular or parallel separations.

        :param rpo_label: The label to use for the x-axis (if :param:`is_rpo`
        is `False`) or the axes (if :param:`is_rpo` is `True`)
        :type rpo_label: `str`
        :param rlo_label: The label to use for the x-axis (if :param:`is_rpo`
        is `True`) or the axes (if :param:`is_rpo` is `False`)
        :type rlo_label: `str`
        :param ylabel: The label for the y-axis
        :type ylabel: `str`
        :param bins: The values of the fixed separation to use
        :type bins: scalar or array-like `float`
        :param perp_bin_scale: The scale of the bins in the perpendicular
        direction, in the same units as the perpendicular separations
        :type perp_bin_scale: `float`
        :param par_bin_scale: The scale of the bins in the parallel direction,
        in the same units as the parallel separations
        :type par_bin_scale: `float`
        :param exp: The expected constant for the limiting behavior. This is
        likely to be 0 for means and 1 for variances
        :type exp: scalar `float`
        :param is_rpo: If `True`, assume the bins are fixed perpendicular
        separation. Otherwise, assume fixed parallel separation. Default `False`
        :type is_rpo: `bool`, optional
        :param logx: If `True`, use log scaling on the x-axis. Default `False`
        :type logx: `bool`, optional
        :param logy: If `True`, use log scaling on the y-axis. Default `False`
        :type logy: `bool`, optional
        :param filename: If given, specifies path at which to save plot.
        If `None`, plot is not saved. Default `None`
        :type filename: `str` or `None`, optional
        :param figsize: Figure size to use. If `None`, use default figure
        size from rcParams. Default `None`
        :type figsize: `tuple` or `None`, optional
        :param display: If `True`, show plot when finished. Default `False`
        :type display: `bool`, optional
        :param text_size: The size of text to use in the figure. Default 22
        :type text_size: scalar `float`
        :param with_fit: If `True`, include the best fit on the plots (
        requires that fitting has been done). Default `False`
        :type with_fit: `bool`, optional
        :param point_alpha: The transparency to use on the points. This is
        useful when plotting the combined statistics where a large number of
        points may be in on region of the plot. Default 1.0
        :type point_alpha: `float`
        :return fig: The figure that has been created
        :rtype fig: :class:`matplotlib.figure.Figure`
        """
        if figsize is None:
            figsize = plt.rcParams["figure.figsize"]
        plt.rcParams["font.size"] = text_size

        if is_rpo:
            # x-axis will be RLO_BIN, bins are drawn from RPO_BIN
            r_bin_size = self.rpo_size / perp_bin_scale
            x_bin_size = self.rlo_size / par_bin_scale
            rlabel = rpo_label
            xlabel = rlo_label
            data = self.data_vs_rlo.loc[bins]
        else:
            # x-axis will be RPO_BIN, bins are drawn from RLO_BIN
            r_bin_size = self.rlo_size / par_bin_scale
            x_bin_size = self.rpo_size / perp_bin_scale
            rlabel = rlo_label
            xlabel = rpo_label
            data = self.data_vs_rpo.loc[bins]

        axis_label = r"${} = {{}} \pm {}$".format(
            smh.strip_dollars_and_double_braces(rlabel),
            smh.strip_dollar_signs(
                smh.pretty_print_number(0.5 * r_bin_size, 2)))
        
        if with_fit:
            if self._best_fit_params is None:
                warnings.warn("Ignoring with_fit option when no fit is done")
                with_fit = False
            else:
                if isinstance(data.index, pd.MultiIndex):
                    all_r = data.index.get_level_values(0).unique(
                        ).sort_values()
                    all_x = data.index.get_level_values(1).unique(
                        ).sort_values()
                    mod_index = pd.MultiIndex.from_product(
                        [all_r, all_x], names=data.index.names)
                    all_r = mod_index.get_level_values(0)
                    all_x = mod_index.get_level_values(1)
                else:
                    all_x = data.index
                    all_r = np.repeat(bins, all_x.size)
                    mod_index = all_x
                if is_rpo:
                    model_args = ((all_r + 0.5) * self.rpo_size,
                                  (all_x + 0.5) * self.rlo_size)
                else:
                    model_args = ((all_x + 0.5) * self.rpo_size,
                                  (all_r + 0.5) * self.rlo_size)
                with_fill = True
                try:
                    mod = self.model_with_errors(*model_args, index=mod_index)
                except AttributeError:
                    with_fill = False
                    mod = self.model(*model_args, index=mod_index)
        
        fig = plt.figure(figsize=figsize)
        if not hasattr(bins, "__len__"):
            # Case: single bin, don't need any subplots
            r_val = (bins + 0.5) * r_bin_size
            plt.xlabel(xlabel)
            plt.ylabel(ylabel, labelpad=(2 * plt.rcParams["font.size"]))
            if logx:
                plt.xscale("log")
            if logy:
                plt.yscale("log")
            plt.axhline(exp, c="k")
            line = plt.errorbar((data.index + 0.5) * x_bin_size,
                                data.loc[:,self.col_names[0]],
                                yerr=np.sqrt(data.loc[:,self.col_names[1]]),
                                fmt="C0o", alpha=point_alpha)[0]
            if with_fit:
                if with_fill:
                    fit_fill = plt.fill_between((mod.index + 0.5) * x_bin_size,
                                                mod.loc[:,0.16],
                                                mod.loc[:,0.84], color="C1",
                                                alpha=0.4)
                fit_line, = plt.plot((mod.index + 0.5) * x_bin_size,
                                     mod.loc[:,0.5], "C1-")
            plt.legend([line], [axis_label.format(
                    smh.strip_dollar_signs(smh.pretty_print_number(r_val, 2)))],
                       loc=0, markerscale=0, frameon=False)
            plt.tight_layout()
        else:
            bins = np.reshape(bins, -1)
            grid = gridspec.GridSpec(bins.size, 1, hspace=0)
            full_ax = fig.add_subplot(grid[:])
            for loc in ["top", "bottom", "left", "right"]:
                full_ax.spines[loc].set_color("none")
            full_ax.tick_params(labelcolor="w", top="off", bottom="off", 
                    left="off", right="off", which="both")
            full_ax.set_ylabel(ylabel, labelpad=(2 * plt.rcParams["font.size"]))
            full_ax.set_xlabel(xlabel)
            for i, (r, r_val) in enumerate(zip(bins,
                                               (bins + 0.5) * r_bin_size)):
                ax = fig.add_subplot(grid[i], sharex=full_ax)
                if logx:
                    ax.set_xscale("log")
                if logy:
                    ax.set_yscale("log")
                ax.axhline(exp, c="k")
                line = ax.errorbar((data.loc[r].index + 0.5) * x_bin_size,
                                   data.loc[r,self.col_names[0]],
                                   yerr=data.loc[r,self.col_names[1]].apply(
                        math.sqrt), fmt="C0o", alpha=point_alpha)[0]
                if with_fit:
                    if with_fill:
                        fit_fill = ax.fill_between(
                            (mod.loc[r].index + 0.5) * x_bin_size,
                            mod.loc[r,0.16], mod.loc[r,0.84], color="C1",
                            alpha=0.4)
                    fit_line, = ax.plot((mod.loc[r].index + 0.5) * x_bin_size,
                                        mod.loc[r,0.5], "C1-")
                ax.legend([line], [axis_label.format(
                        smh.strip_dollar_signs(
                                smh.pretty_print_number(r_val, 2)))],
                          loc=0, markerscale=0, frameon=False)
                if ax.is_last_row():
                    ax.tick_params(axis="x", which="both", direction="inout",
                                   top=True, bottom=True)
                else:
                    ax.tick_params(axis="x", which="both", direction="inout",
                                   top=True, bottom=False, labelbottom=False)
            grid.tight_layout(fig, h_pad=0.0)
            grid.update(hspace=0)
        if filename is not None:
            fig.savefig(filename, bbox_inches="tight")
        if display:
            plt.show()
        return fig

    def lnlike(self, theta):
        if self.f is None:
            raise AttributeError("No fitting function set for likelihood")
        if not isinstance(theta, dict):
            theta = __make_clean_dict__(theta, self.params)
        fev = self.f(self.rpo, self.rlo, theta, index=self.data.index)
        diff2 = self.data.loc[:,self.col_names[0]].sub(fev).pow(2)
        diffdiv = diff2.div(self.data.loc[:,self.col_names[1]])
        return -0.5 * diffdiv.sum()
    
    def lnprob(self, theta):
        if self.pr is None:
            raise AttributeError("No prior function set for lnprob")
        if not isinstance(theta, dict):
            theta = __make_clean_dict__(theta, self.params)
        lp = self.pr(theta)
        if not math.isfinite(lp):
            return -math.inf
        return lp + self.lnlike(theta)
    
    def fit_minimize(self, init_guess):
        """Fit the data using :function:`scipy.optimize.minimize`.
        
        :param init_guess: The guess for the starting point of the fit
        :type init_guess: 1D array-like `float`
        :return res: The result of the minimizer, with the fit parameters and
        convergence information
        :rtype res: :class:`scipy.optimize.OptimizeResult`
        """
        self.logger.debug("Defining negative lnprob")
        nll = lambda theta: -self.lnprob(theta)
        self.logger.debug("Running minimization")
        res = minimize(nll, init_guess)
        self._best_fit_params = res.x
        return res


    def fit_mcmc(self, nsteps, nburnin=0, init_guess=None, nwalkers=None,
                 pool=None, sampler=None):
        """Fit the data using an MCMC (as implemented via :module:`emcee`). A
        sampler that has already been initialized may be passed, which will
        ignore the parameters :param:`nwalkers` and :param:`nthreads` unless the
        sampler number of dimensions is incorrect (in which case a new sampler
        will be created to replace it). If the sampler has already been
        previously run, and the number of parameters associated with it are
        still correct, an additional :param:`nsteps` will be taken from the
        current position, so that :param:`init_guess` is also not needed. The
        :param:`init_guess` parameter may be given as a 1D array-like with shape
        (self.ndim,), an instance of :class:`scipy.optimize.OptimizeResult`, or
        a 2D array-like with shape (nwalkers, self.ndim). The first option will
        be used to set the mean of the walker initial positions, and they will
        be scattered around that mean by a Gaussian random with width determined
        by the order of magnitude of each parameter value. The second option
        will be used similarly, but with the best fit parameters from the
        OptimizeResult being used as the mean. The final option is useful when
        some other method of populating the walkers over some area in allowed
        parameter space needs to be used, and is used as the initial positions
        without change.
        
        At the end of the MCMC, the best fit parameters are added to self,
        and the sampler is returned for analyzing walker behavior and checking
        parameter distributions. The samples are also added to the self, as
        well as the burn-in length, for future reference.
        
        :param nsteps: The number of MCMC steps to take with each walker
        :type nsteps: `int`
        :param nburnin: The number of steps to omit from each walker for the
        burn-in phase. Note that while this parameter is optional, the default
        value is not necessarily safe and a warning will be raised if this is
        not changed and the value in self is not set. Default 0 or self.nburnin
        :type nburnin: `int`, optional
        :param init_guess: The initial guess for the walker starting
        positions. As described above, this can either be used as the mean for
        generating walker positions, or it can be the precomputed positions for
        all walkers. Ignored if :param:`sampler` is valid for self
        and has already been run previously, but is required for a new MCMC run
        if there is no best fit set in self. Default `None` or
        :attr:`self.best_fit`
        :type init_guess: 1D array-like, 2D array-like, or
        :class:`scipy.optimize.OptimizeResult`, optional
        :param nwalkers: The number of walkers to utilize for the MCMC. Required
        for new samplers. Default `None`
        :type nwalkers: `int`, optional
        :param pool: A python multiprocessing pool instance to use for parallel
        runs. This will be overriden if no parallelization can be done. Use
        `None` to force run in serial. This parameter is ignored if a valid
        :param:`sampler` is given. Note that the pool object cannot be closed
        before continuing a run. Default `None`
        :type pool: :class:`multiprocessing.pool.Pool`
        :param sampler: A sampler instance to use for the MCMC, or for
        continuing an MCMC. If it is not valid given self, a new
        sampler will be created. The updated or newly created sampler is
        returned at the end of the function. Default `None`
        :type sampler: :class:`emcee.EnsembleSampler`
        :return new_sampler: The sampler that was used for this run. This is
        either a new sampler that was created and run, or the given sampler run
        for :param:`nsteps`.
        :rtype new_sampler: :class:`emcee.EnsembleSampler`
        """
        self.logger.debug("Check the burn-in")
        if nburnin == 0 and self._nburnin is None:
            warnings.warn("Using burn-in of 0 is ill-advised and will likely "\
                              "result in improper best fit values being set")
        elif nburnin == 0:
            nburnin = self._nburnin
        self._nburnin = nburnin
        
        self.logger.debug("Check if passed sampler is valid")
        new_sampler = copy.deepcopy(sampler)
        if new_sampler is not None and sampler.chain.shape[-1] != self.ndim:
            warnings.warn("Invalid sampler given, creating new sampler instead")
            new_sampler = None
        if new_sampler is None:
            self.logger.debug("Check pickleable attributes for parallelization")
            attrs_needed = [self.data, self.index_names, self.col_names,
                            self.f, self.pr, self.ndim, self.lnlike,
                            self.lnprob, self]
            attrs_needed_names = np.array(["data", "index names list",
                                           "column names list",
                                           "fit function", "prior funcion",
                                           "number of dimensions",
                                           "lnlike", "lnprob", "self"])
            pickleable = _check_pickleable(attrs_needed)
            if not pickleable[0]:
                self.logger.info("Setting pool to None because not "\
                                     "pickleable")
                self.logger.debug("Non-pickleable attributes: {}".format(
                        attrs_needed_names[pickleable[1]]))
                nthreads = None
            self.logger.debug("Check number of walkers")
            if nwalkers is None:
                raise ValueError("Must give number of walkers for new sampler")
            self.logger.debug("Create a new sampler")
            new_sampler = emcee.EnsembleSampler(nwalkers, self.ndim,
                                                self.lnprob, pool=pool)
        
        self.logger.debug("Set initial position")
        if new_sampler.chain.shape[1] > 0:
            self.logger.debug("Continue chain from current position")
            init_guess = None
        else:
            if init_guess is None:
                raise ValueError("Must give init_guess for new MCMC")
            if isinstance(init_guess, OptimizeResult):
                if len(init_guess.x) != self.ndim:
                    raise ValueError("Wrong number of parameters for initial "\
                                         "guess from OptimizeResult")
                init_guess = init_guess.x
                if not hasattr(init_guess[0], "__len__"):
                    logger.debug("Set initial walker positions from mean")
                    if len(init_guess) != self.ndim:
                        raise ValueError("Wrong number of parameters for "\
                                             "initial guess mean values")
                    init_guess = [init_guess + (10**(-4 + ndigits(init_guess)) *
                                                np.random.randn(self.ndim))
                                  for i in range(nwalkers)]
                else:
                    if len(init_guess) != nwalkers:
                        raise ValueError("Wrong number of walker positions "\
                                             "for initial guess positions")
                    if len(init_guess[0]) != self.ndim:
                        raise ValueError("Wrong number of parameters for "\
                                             "initial guess positions")
                    pass
        self.logger.debug("Running MCMC")
        new_sampler.run_mcmc(init_guess, nsteps)
        self.logger.debug("Add chain to fitter")
        self._samples = new_sampler.chain
        self.logger.debug("Calculate best fit parameters and add to fitter")
        samples = new_sampler.chain[:,nburnin:,:].reshape((-1, self.ndim))
        self._best_fit_params = np.median(samples, axis=0)
        self.logger.debug("Set up Chain Consumer objects")
        c_walks = ChainConsumer()
        c_walks.add_chain(new_sampler.flatchain, parameters=self.params,
                          posterior=new_sampler.flatlnprobability,
                          walkers=new_sampler.k)
        self.c_walkers = c_walks.divide_chain()
        lnprobs = new_sampler.lnprobability[:, nburnin:].flatten()
        self.c = ChainConsumer()
        self.c.add_chain(samples, parameters=self.params,
                         posterior=lnprobs)
        self.logger.debug("Done")
        return new_sampler


class AnalyticSingleFitter(object):
    """
    Another single data model fitter, but this time for a model than can be
    fit analytically. This case is only valid (so far) for a constant
    function with uncorrelated errors (i.e. diagonal covariance matrix)
    """

    def __init__(self, data, index_names, col_names, fitter_name=None, *,
                 rpo_size=1.0, rlo_size=1.0, **kwargs):
        """
        Initialize the analytic fitter. Please note that all array-likes should
        have the same size
        :param data: A Pandas DataFrame with indices giving the observed
        perpendicular and parallel separations and columns for the data and
        variance.
        :type data: :class:`pandas.DataFrame`
        :param index_names: A list of the index names, with the first element
        corresponding to the name for the observed perpendicular separation and
        the second to the observed parallel separation.
        :type index_names: `list` `str`
        :param col_names: A list of the needed column names from the DataFrame, 
        with the first element corresponding to the data and the second to the 
        variance.
        :type col_names: `list` `str`
        :param fitter_name: The name for this instance of the fitter. If
        `None`, the name will be set to 'AnalyticSingleFitter'. Default `None`
        :type fitter_name: `str` or `None`, optional
        :key rpo_size: The size of the observed perpendicular separation bins
        to assume is used in the data. Default 1.0
        :type rpo_size: `float`
        :key rlo_size: The size of the observed parallel separation bins to
        assume is used in the data. Default 1.0
        :type rlo_size: `float`
        """
        self._get_name(fitter_name)
        self.logger = init_logger(self.name)
        self.logger.debug("Setting up data and variance")
        self.logger.debug("Data columns: \n{}".format(data.columns))
        self.data = data
        self.index_names = index_names
        self.col_names = col_names
        self.rpo_size = rpo_size
        self.rlo_size = rlo_size
        self._c = None
        self._c_err = None
        self.logger.debug("__init__ complete")

    def __repr__(self):
        return "%s(c=%r, c_err=%r)" % (self.name, self._c, self._c_err)

    def __getstate__(self):
        d = self.__dict__.copy()
        if "logger" in d:
            d["logger"] = (d["logger"].name, d["logger"].getEffectiveLevel())
        return d

    def __setstate__(self, d):
        if "logger" in d:
            level = d["logger"][1]
            d["logger"] = logging.getLogger(d["logger"][0])
            d["logger"].setLevel(level)
        self.__dict__.update(d)

    @property
    def best_fit(self):
        return np.array([self._c]) if self._c is not None else None

    @property
    def best_fit_err(self):
        return np.array([self._c_err]) if self._c_err is not None else None

    @property
    def data_vs_rlo(self):
        if self.index_names[0] != self.data.index.names[0]:
            return self.data.swaplevel(0, 1, axis=0).sort_index()
        return self.data.copy()

    @property
    def data_vs_rpo(self):
        if self.index_names[0] == self.data.index.names[0]:
            return self.data.swaplevel(0, 1, axis=0).sort_index()
        return self.data.copy()

    def _get_name(self, fitter_name):
        self.name = ("{}.{}".format(self.__class__.__name__, fitter_name) if
                     fitter_name is not None else self.__class__.__name__)

    def _get_const(self):
        self._c = (self.data.loc[:,self.col_names[0]].div(
            self.data.loc[:,self.col_names[1]]).sum() /
                   self.data.loc[:,self.col_names[1]].apply(
                lambda x: 1. / x).sum())

    def _get_err(self):
        self._c_err = math.sqrt(1. /
                                self.data.loc[:,self.col_names[1]].apply(
                lambda x: 1. / x).sum())

    def fit(self):
        self._get_const()
        self._get_err()

    def model(self, rpo, rlo, index=None):
        """Get the best fit model at the given separations

        :param rpo: The observed perpendicular separation
        :type rpo: scalar or array-like `float`
        :param rlo: The observed parallel separation
        :type rlo: scalar or array-like `float`
        :param index: The index to add to the Series. Ignored for scalar
        returns. If `None`, the index will be created from :param:`rpo` and
        :param:`rlo`, with names ['RPO_BIN', 'RLO_BIN']. Default `None`
        :type index: :class:`pandas.Index`, :class:`pandas.MultiIndex`, or
        `None`, optional
        :return m: The value of the best fit model at the points. This will
        be 1D if :param:`rpo` and :param:`rlo` are both array-like, with both
        separations flattened
        :rtype m: scalar or :class:`pandas.Series` `float`
        """
        if self._c is None:
            raise AttributeError("Cannot get best fit model if fit has "\
                                     "not been done")
        c = self._c
        if not hasattr(rpo, "__len__") and not hasattr(rlo, "__len__"):
            m = c
        else:
            if index is None:
                index = pd.MultiIndex.from_arrays([rpo, rlo], names=["RPO_BIN",
                    "RLO_BIN"])
            m = pd.Series(c, index=index)
        return m

    def model_with_errors(self, rpo, rlo, index=None):
        """Get the median and 1 sigma error region of the model at the
        separations

        :param rpo: The observed perpendicular separation
        :type rpo: scalar or array-like `float`
        :param rlo: The observed parallel separation
        :type rlo: scalar or array-like `float`
        :param index: The index to add to the Series. Ignored for single
        separation. If `None`, the index will be created from :param:`rpo` and
        :param:`rlo`, with names ['RPO_BIN', 'RLO_BIN']. Default `None`
        :type index: :class:`pandas.Index`, :class:`pandas.MultiIndex`, or
        `None`, optional
        :return m: The (median - 1 sigma), median, and (median + 1 sigma)
        values of the model at the points. The 0th axis is the percentiles,
        and the first axis is the flattened points, if not both scalar
        :rtype m: :class:`pandas.Series` or :class:`pandas.DataFrame` `float`
        """
        if self._c is None or self._c_err is None:
            raise AttributeError("Cannot get best fit model with errors "\
                                     "if fit has not been done")
        c = self._c
        c_err = self._c_err
        if not hasattr(rpo, "__len__") and not hasattr(rlo, "__len__"):
            m = pd.Series([c - c_err, c, c + c_err], index=[0.16, 0.5, 0.84])
        else:
            if index is None:
                if not hasattr(rpo, "__len__"):
                    index = pd.Index(rlo)
                elif not hasattr(rlo, "__len__"):
                    index = pd.Index(rpo)
                else:
                    index = pd.MultiIndex.from_arrays([rpo, rlo], 
                            names=["RPO_BIN", "RLO_BIN"])
            m = pd.DataFrame({0.16: c - c_err, 0.5: c, 0.84: c + c_err},
                             index=index)
        return m

    def plot(self, rpo_label, rlo_label, ylabel, bins, perp_bin_scale,
             par_bin_scale, exp, is_rpo=False, logx=False, logy=False, 
             filename=None, figsize=None, display=False, text_size=22, 
             with_fit=False, point_alpha=1.0):
        """Plot the data (and optionally the best fit to the data) at a number
        of individual perpendicular or parallel separations.

        :param rpo_label: The label to use for the x-axis (if :param:`is_rpo`
        is `False`) or the axes (if :param:`is_rpo` is `True`)
        :type rpo_label: `str`
        :param rlo_label: The label to use for the x-axis (if :param:`is_rpo`
        is `True`) or the axes (if :param:`is_rpo` is `False`)
        :type rlo_label: `str`
        :param ylabel: The label for the y-axis
        :type ylabel: `str`
        :param bins: The values of the fixed separation to use
        :type bins: scalar or array-like `float`
        :param perp_bin_scale: The scale of the bins in the perpendicular
        direction, in the same units as the perpendicular separations
        :type perp_bin_scale: `float`
        :param par_bin_scale: The scale of the bins in the parallel direction,
        in the same units as the parallel separations
        :type par_bin_scale: `float`
        :param exp: The expected constant for the limiting behavior. This is
        likely to be 0 for means and 1 for variances
        :type exp: scalar `float`
        :param is_rpo: If `True`, assume the bins are fixed perpendicular
        separation. Otherwise, assume fixed parallel separation. Default `False`
        :type is_rpo: `bool`, optional
        :param logx: If `True`, use log scaling on the x-axis. Default `False`
        :type logx: `bool`, optional
        :param logy: If `True`, use log scaling on the y-axis. Default `False`
        :type logy: `bool`, optional
        :param filename: If given, specifies path at which to save plot.
        If `None`, plot is not saved. Default `None`
        :type filename: `str` or `None`, optional
        :param figsize: Figure size to use. If `None`, use default figure
        size from rcParams. Default `None`
        :type figsize: `tuple` or `None`, optional
        :param display: If `True`, show plot when finished. Default `False`
        :type display: `bool`, optional
        :param text_size: The size of text to use in the figure. Default 22
        :type text_size: scalar `float`
        :param with_fit: If `True`, include the best fit on the plots (
        requires that fitting has been done). Default `False`
        :type with_fit: `bool`, optional
        :param point_alpha: The transparency to use on the points. This is
        useful when plotting the combined statistics where a large number of
        points may be in on region of the plot. Default 1.0
        :type point_alpha: `float`
        :return fig: The figure that has been created
        :rtype fig: :class:`matplotlib.figure.Figure`
        """
        if figsize is None:
            figsize = plt.rcParams["figure.figsize"]
        plt.rcParams["font.size"] = text_size

        if is_rpo:
            # x-axis will be RLO_BIN, bins are drawn from RPO_BIN
            r_bin_size = self.rpo_size / perp_bin_scale
            x_bin_size = self.rlo_size / par_bin_scale
            rlabel = rpo_label
            xlabel = rlo_label
            data = self.data_vs_rlo.loc[bins]
        else:
            # x-axis will be RPO_BIN, bins are drawn from RLO_BIN
            r_bin_size = self.rlo_size / par_bin_scale
            x_bin_size = self.rpo_size / perp_bin_scale
            rlabel = rlo_label
            xlabel = rpo_label
            data = self.data_vs_rpo.loc[bins]

        axis_label = r"${} = {{}} \pm {}$".format(
            smh.strip_dollars_and_double_braces(rlabel),
            smh.strip_dollar_signs(
                smh.pretty_print_number(0.5 * r_bin_size, 2)))

        if with_fit:
            if (self._c is None or self._c_err is None):
                warnings.warn("Ignoring with_fit because no fit has been done")
                with_fit = False
            else:
                if isinstance(data.index, pd.MultiIndex):
                    all_r = data.index.get_level_values(0).unique(
                        ).sort_values()
                    all_x = data.index.get_level_values(1).unique(
                        ).sort_values()
                    mod_index = pd.MultiIndex.from_product(
                        [all_r, all_x], names=data.index.names)
                    all_r = mod_index.get_level_values(0)
                    all_x = mod_index.get_level_values(1)
                else:
                    all_x = data.index
                    all_r = np.repeat(bins, all_x.size)
                    mod_index = all_x
                if is_rpo:
                    model_args = ((all_r + 0.5) * self.rpo_size,
                                  (all_x + 0.5) * self.rlo_size)
                else:
                    model_args = ((all_x + 0.5) * self.rpo_size,
                                  (all_r + 0.5) * self.rlo_size)
                mod = self.model_with_errors(*model_args, index=mod_index)

        fig = plt.figure(figsize=figsize)
        if not hasattr(bins, "__len__"):
            # In this case, we don't need any subplots
            r_val = (bins + 0.5) * r_bin_size
            plt.xlabel(xlabel)
            plt.ylabel(ylabel, labelpad=(2 * plt.rcParams["font.size"]))
            if logx:
                plt.xscale("log")
            if logy:
                plt.yscale("log")
            plt.axhline(exp, c="k")
            line = plt.errorbar((data.index + 0.5) * x_bin_size,
                                data.loc[:,self.col_names[0]],
                                yerr=data.loc[:,self.col_names[1]].apply(
                    math.sqrt), fmt="C0o", alpha=point_alpha)[0]
            if with_fit:
                fit_fill = plt.fill_between((mod.index + 0.5) * x_bin_size,
                                            mod.loc[:,0.16],
                                            mod.loc[:,0.84],
                                            color="C1",
                                            alpha=0.4)
                fit_line, = plt.plot((mod.index + 0.5) * x_bin_size,
                                     mod.loc[:,0.5], "C1-")
            plt.legend([line], [axis_label.format(
                    smh.strip_dollar_signs(smh.pretty_print_number(r_val, 2)))],
                       loc=0, markerscale=0, frameon=False)
            plt.tight_layout()
        else:
            bins = np.reshape(bins, -1)
            grid = gridspec.GridSpec(bins.size, 1, hspace=0)
            full_ax = fig.add_subplot(grid[:])
            for loc in ["top", "bottom", "left", "right"]:
                full_ax.spines[loc].set_color("none")
            full_ax.tick_params(labelcolor="w", top="off", bottom="off",
                    left="off", right="off", which="both")
            full_ax.set_ylabel(ylabel, labelpad=(2 * plt.rcParams["font.size"]))
            full_ax.set_xlabel(xlabel)
            for i, (r, r_val) in enumerate(zip(bins,
                                               (bins + 0.5) * r_bin_size)):
                ax = fig.add_subplot(grid[i], sharex=full_ax)
                if logx:
                    ax.set_xscale("log")
                if logy:
                    ax.set_yscale("log")
                ax.axhline(exp, c="k")
                line = ax.errorbar((data.loc[r].index + 0.5) * x_bin_size,
                                   data.loc[r,self.col_names[0]],
                                   yerr=data.loc[r,self.col_names[1]].apply(
                        math.sqrt), fmt="C0o", alpha=point_alpha)[0]
                if with_fit:
                    fit_fill = ax.fill_between(
                        (mod.loc[r].index + 0.5) * x_bin_size,
                        mod.loc[r,0.16], mod.loc[r,0.84], color="C1", alpha=0.4)
                    fit_line, = ax.plot((mod.loc[r].index + 0.5) * x_bin_size,
                                        mod.loc[r,0.5], "C1-")
                ax.legend([line], [axis_label.format(
                        smh.strip_dollar_signs(
                                smh.pretty_print_number(r_val, 2)))],
                          loc=0, markerscale=0, frameon=False)
                if ax.is_last_row():
                    ax.tick_params(axis="x", which="both", direction="inout",
                                   top=True, bottom=True)
                else:
                    ax.tick_params(axis="x", which="both", labelbottom=False,
                                   direction="inout", top=True, bottom=False)
            grid.tight_layout(fig, h_pad=0.0)
            # To make sure tight_layout didn't separate things again
            grid.update(hspace=0.0)
        if filename is not None:
            fig.savefig(filename, bbox_inches="tight")
        if display:
            plt.show()
        return fig


class ProbFitter(object):
    """
    Fit a bivariate Gaussian of 2 variables to the data. This is not a
    generalized fitter, it assumes the means and variances follow specific
    functions that are not set by the user but hardcoded within.
    """
    _fitter_types = ["mean_x", "var_x", "mean_y", "var_y", "mean_r"]
    def __init__(self, statistics=None, fitter_name=None, *, rpo_scale=1.0,
                 rlo_scale=1.0, rpo_size=1.0, rlo_size=1.0,
                 mean_y_const=False, var_y_const=False, mean_y_func=None,
                 var_y_func=None, mean_y_extents=None, var_y_extents=None,
                 mean_r_const=True, mean_r_func=None, mean_r_extents=None,
                 **kwargs):
        """
        Initialize the fitter with calculated statistics
        
        :param statistics: A DataFrame containing pre-computed sample means
        and variances for at least one of the fitters. The DataFrame columns
        must be multi-level, with the 0th level giving the quantity being
        calculated (of 'mean_x', 'var_x', 'mean_y', 'var_y', 'mean_r') and
        the 1st level being columns 'mean' (for sample mean/variance) and
        'variance' (for variance on the 'mean' column). If not passed, will
        need to be set later.
        :type statistics: :class:`pandas.DataFrame`, optional
        :param fitter_name: A name to use for this instance of the fitter. If
        `None`, the name will be set to 'ProbFitter'. Default `None`
        :type fitter_name: `str` or `None`, optional
        :key rpo_scale: Optional scaling to use on the observed perpendicular
        separations in the fitting functions. Default 1.0 (in units of the
        perpendicular separation)
        :type rpo_scale: `float`
        :key rlo_scale: Optional scaling to use on the observed parallel
        separations in the fitting functions. Default 1.0 (in units of the
        parallel separation)
        :type rlo_scale: `float`
        :key rpo_size: The size of the observed perpendicular separation bins
        to assume is used in the statistics. Default 1.0 (in units of the
        perpendicular separation)
        :type rpo_size: `float`
        :key rlo_size: The size of the observed parallel separation bins to
        assume is used in the statistics. Default 1.0 (in units of the
        parallel separation)
        :type rlo_size: `float`
        :kwarg mean_y_const: Optionally assume that 'mean_y' can be fit as a
        constant, rather than the functional form derived previously. You can
        also change the value of this property, and doing so will create a new
        fitter instance, replacing the previous one, if the value has actually
        changed (setting to the same value as the internal object does
        nothing). In that case, a new fit will need to be performed for the
        appropriate fitter. When this parameter is `False, whether originally or
        because it is changed, the fitting function should be specified using
        the method on the resulting :class:`SingleFitter` instance. For
        convenience, a function that seems to work for this is defined within
        this module as `mean_y`, and will be used by default. Default `False`
        :type mean_y_const: `bool`
        :kwarg var_y_const: Optionally assume that 'var_y' can be fit as a
        constant, rather than the functional form derived previously. You can
        also change the value of this property, and doing so will create a new
        fitter instance, replacing the previous one, if the value has actually
        changed (setting to the same value as the internal object does
        nothing). In that case, a new fit will need to be performed for the
        appropriate fitter. when this parameter is `False`, either originally or
        when changed, the fitting function should be specified using the method
        on the resulting :class:`SingleFitter` instance. For convenience, a
        function that seems to work for this is defined within this module as
        `var_y`, and will be used by default. Default `False`
        :type var_y_const: `bool`
        :kwarg mean_y_extents: Extents to use for a flat prior on 'mean_y'.  If
        `None`, the prior will not be set, and must be set by hand in the
        :class:`SingleFitter` instance for 'mean_y'. This is stored even if
        :kwarg:`mean_y_const` is `True` to be used if needed. However, it is not
        necessary for a constant. Default `None`
        :type mean_y_extents: `dict`
        :kwarg var_y_extents: Extents to use for a flat prior on 'var_y'.  If
        `None`, the prior will not be set, and must be set by hand in the
        :class:`SingleFitter` instance for 'var_y'. This is stored even if
        :kwarg:`var_y_const` is `True` to be used if needed later. However, it
        is not necessary for a constant. Default `None`
        :type var_y_extents: `dict`
        :kwarg mean_r_const: Optionally assume that 'mean_r' can be fit as a
        constant, rather than the functional form derived previously. You can
        also change the value of this property, and doing so will create a new
        fitter instance, replacing the previous one, if the value has actually
        changed (setting to the same value as the internal object does
        nothing). In that case, a new fit will need to be performed for the
        appropriate fitter. When this parameter is `False, whether originally or
        because it is changed, the fitting function should be specified using
        the method on the resulting :class:`SingleFitter` instance. Default
        `True`
        :type mean_r_const: `bool`
        :kwarg mean_r_extents: Extents to use for a flat prior on 'mean_r'.  If
        `None`, the prior will not be set, and must be set by hand in the
        :class:`SingleFitter` instance for 'mean_r'. This is stored even if
        :kwarg:`mean_r_const` is `True` to be used if needed. However, it is not
        necessary for a constant. Default `None`
        :type mean_r_extents: `dict`
        """
        self._init_switcher = dict(mean_x=self.initialize_mean_x,
                                   var_x=self.initialize_var_x,
                                   mean_y=self.initialize_mean_y,
                                   var_y=self.initialize_var_y,
                                   mean_r=self.initialize_mean_r)
        self._fitter_names = dict.fromkeys(self.__class__._fitter_types)
        self._get_name(fitter_name)
        self.logger = init_logger(self.name)
        self._fitters = dict.fromkeys(self._fitter_types, None)
        self.logger.debug("Set bin sizes and separation scaling")
        self.rpo_size = rpo_size
        self.rlo_size = rlo_size
        self.rpo_scale = rpo_scale
        self.rlo_scale = rlo_scale
        self._mean_y_const = mean_y_const
        self._var_y_const = var_y_const
        self._mean_r_const = mean_r_const
        self._mean_y_extents = mean_y_extents
        self._var_y_extents = var_y_extents
        self._mean_r_extents = mean_r_extents
        self.logger.debug("Add statistics")
        self.add_stats(statistics)
        self.logger.debug("__init__ complete")

    def __repr__(self):
        return "{name}(mean_x={f[0]}, var_x={f[1]}, mean_y={f[2]}, var_y={f[3]}, mean_r={f[4]})".format(name=self.name, f=list(self._fitters.values()))

    def __getstate__(self):
        d = self.__dict__.copy()
        if "logger" in d:
            d["logger"] = (d["logger"].name, d["logger"].getEffectiveLevel())
        return d

    def __setstate__(self, d):
        if "logger" in d:
            level = d["logger"][1]
            d["logger"] = logging.getLogger(d["logger"][0])
            d["logger"].setLevel(level)
        self.__dict__.update(d)

    @property
    def mean_x(self):
        return self._fitters["mean_x"]

    @property
    def var_x(self):
        return self._fitters["var_x"]

    @property
    def mean_y(self):
        return self._fitters["mean_y"]

    @property
    def var_y(self):
        return self._fitters["var_y"]

    @property
    def mean_r(self):
        return self._fitters["mean_r"]

    @property
    def mean_y_const(self):
        return self._mean_y_const

    @mean_y_const.setter
    def mean_y_const(self, constness):
        if constness is not self._mean_y_const:
            self._mean_y_const = constness
            if self._fitters["mean_y"] is not None:
                self.initialize_mean_y(
                    pd.concat([self._fitters["mean_y"].data], axis=1,
                              keys=["mean_y"]))

    @property
    def var_y_const(self):
        return self._var_y_const

    @var_y_const.setter
    def var_y_const(self, constness):
        if constness is not self._var_y_const:
            self._var_y_const = constness
            if self._fitters["var_y"] is not None:
                self.initialize_var_y(
                    pd.concat([self._fitters["var_y"].data], axis=1,
                              keys=["var_y"]))

    @property
    def mean_r_const(self):
        return self._mean_r_const

    @mean_r_const.setter
    def mean_r_const(self, constness):
        if constness is not self._mean_r_const:
            self._mean_r_const = constness
            if self._fitters["mean_r"] is not None:
                self.initialize_mean_r(
                    pd.concat([self._fitters["mean_r"].data], axis=1,
                              keys=["mean_r"]))

    @property
    def mean_y_extents(self):
        return self._mean_y_extents

    @mean_y_extents.setter
    def mean_y_extents(self, extents):
        self._mean_y_extents = extents.copy()
        if not self._mean_y_const:
            self._fitters["mean_y"].set_prior_func(extents)

    @property
    def var_y_extents(self):
        return self._var_y_extents

    @var_y_extents.setter
    def var_y_extents(self, extents):
        self._var_y_extents = extents.copy()
        if not self._var_y_const:
            self._fitters["var_y"].set_prior_func(extents)

    @property
    def mean_r_extents(self):
        return self._mean_r_extents

    @mean_r_extents.setter
    def mean_r_extents(self, extents):
        self._mean_r_extents = extents.copy()
        if not self._mean_r_const:
            self._fitters["mean_r"].set_prior_func(extents)

    @property
    def rpo_bin(self):
        fitters_init = [f for f in list(self._fitters.values()) if f is
                        not None]
        if len(fitters_init) == 0:
            return None
        return fitters_init[0].data.index.get_level_values("RPO_BIN")

    @property
    def rlo_bin(self):
        fitters_init = [f for f in list(self._fitters.values()) if f is
                        not None]
        if len(fitters_init) == 0:
            return None
        return fitters_init[0].data.index.get_level_values("RLO_BIN")

    @property
    def stats_vs_rpo_rlo(self):
        names = [name for name in self._fitters.keys() if
                 self._fitters[name] is not None]
        data = [self._fitters[name].data_vs_rlo for name in names]
        return pd.concat(data, axis=1, keys=names)

    @property
    def stats_vs_rlo_rpo(self):
        names = [name for name in self._fitters.keys() if
                 self._fitters[name] is not None]
        data = [self._fitters[name].data_vs_rpo for name in names]
        return pd.concat(data, axis=1, keys=names)

    @property
    def stats_table(self):
        concat_keys = [name for name, fitter in self._fitters.items() if
                       fitter is not None]
        stats = pd.concat([fitter.data for fitter in self._fitters.values() if
                           fitter is not None], axis=1, keys=concat_keys)
        return stats_df_to_stats_table(stats)

    def _get_name(self, fitter_name):
        self.name = ("{}.{}".format(self.__class__.__name__, fitter_name) if
                     fitter_name is not None else self.__class__.__name__)
        for name in self._fitter_names.keys():
            self._fitter_names[name] = ("{}_{}".format(fitter_name, name) if
                fitter_name is not None else name)

    def initialize_mean_x(self, stats):
        self.logger.debug("init_mean_x")
        self._fitters["mean_x"] = AnalyticSingleFitter(
            stats["mean_x"].copy(), ["RPO_BIN", "RLO_BIN"],
            ["mean", "variance"], self._fitter_names["mean_x"],
            rpo_size=self.rpo_size, rlo_size=self.rlo_size)
        self.logger.debug("init_mean_x: {}".format(self._fitters["mean_x"]))

    def initialize_var_x(self, stats):
        self.logger.debug("init_var_x")
        self._fitters["var_x"] = AnalyticSingleFitter(
            stats["var_x"].copy(), ["RPO_BIN", "RLO_BIN"],
            ["mean", "variance"], self._fitter_names["var_x"],
            rpo_size=self.rpo_size, rlo_size=self.rlo_size)
        self.logger.debug("init_var_x: {}".format(self._fitters["var_x"]))

    def initialize_mean_y(self, stats):
        self.logger.debug("init_mean_y")
        if self._mean_y_const:
            self._fitters["mean_y"] = AnalyticSingleFitter(
                stats["mean_y"].copy(), ["RPO_BIN", "RLO_BIN"],
                ["mean", "variance"], self._fitter_names["mean_y"],
                rpo_size=self.rpo_size, rlo_size=self.rlo_size)
        else:
            self._fitters["mean_y"] = SingleFitter(
                stats["mean_y"].copy(), ["RPO_BIN", "RLO_BIN"],
                ["mean", "variance"], mean_y, self._mean_y_extents,
                mean_y.params,
                self._fitter_names["mean_y"], rpo_size=self.rpo_size,
                rlo_size=self.rlo_size,
                func_kwargs=dict(rpo_scale=self.rpo_scale,
                                 rlo_scale=self.rlo_scale))
        self.logger.debug("init_mean_y: {}".format(self._fitters["mean_y"]))

    def initialize_var_y(self, stats):
        self.logger.debug("init_var_y")
        if self._var_y_const:
            self._fitters["var_y"] = AnalyticSingleFitter(
                stats["var_y"].copy(), ["RPO_BIN", "RLO_BIN"],
                ["mean", "variance"], self._fitter_names["var_y"],
                rpo_size=self.rpo_size, rlo_size=self.rlo_size)
        else:
            self._fitters["var_y"] = SingleFitter(
                stats["var_y"].copy(), ["RPO_BIN", "RLO_BIN"],
                ["mean", "variance"], var_y, self._var_y_extents,
                var_y.params,
                self._fitter_names["var_y"], rpo_size=self.rpo_size,
                rlo_size=self.rlo_size,
                func_kwargs=dict(rpo_scale=self.rpo_scale,
                                 rlo_scale=self.rlo_scale))
        self.logger.debug("init_var_y: {}".format(self._fitters["var_y"]))

    def initialize_mean_r(self, stats):
        self.logger.debug("init_mean_r")
        if self._mean_r_const:
            self._fitters["mean_r"] = AnalyticSingleFitter(
                stats["mean_r"].copy(), ["RPO_BIN", "RLO_BIN"],
                ["mean", "variance"], self._fitter_names["mean_r"], 
                rpo_size=self.rpo_size, rlo_size=self.rlo_size)
        else:
            self._fitters["mean_r"] = SingleFitter(
                stats["mean_r"].copy(), ["RPO_BIN", "RLO_BIN"],
                ["mean", "variance"], prior=self._mean_r_extents,
                fitter_name=self._fitter_names["mean_r"],
                rpo_size=self.rpo_size, rlo_size=self.rlo_size)
        self.logger.debug("init_mean_r: {}".format(self._fitters["mean_r"]))

    def add_stats(self, stats_in):
        """Add statistics for initializing the fitters, with the columns as
        described in the __init__. This does the initialization for the
        appropriate fitters, if they haven't already been initialized

        :param stats_in: The statistics to use for initializing fitters
        :type stats_in: :class:`pandas.DataFrame` or :class:`astropy.Table`
        """
        if isinstance(stats_in, Table):
            stats = stats_table_to_stats_df(stats_in)
        else:
            stats = stats_in.copy(deep=True)
        if stats is not None:
            self.logger.debug("Drop NAN columns")
            stats_filt = stats.dropna(axis=1)
            if not stats_filt.empty:
                self.logger.debug("Stats (first 5): \n{}".format(
                        stats_filt.head()))
                self.logger.debug("Stats columns: \n{}".format(
                        stats_filt.columns))
                ftypes_in_stats = [col in stats_filt for col in
                                   self.__class__._fitter_types]
                self.logger.debug(ftypes_in_stats)
                if np.any(ftypes_in_stats):
                    for ftype in list(itertools.compress(
                            self.__class__._fitter_types, ftypes_in_stats)):
                        if self._fitters[ftype] is None:
                            self.logger.debug("Initialize {}".format(ftype))
                            self._init_switcher[ftype](stats_filt)
                            self.logger.debug(self._fitters[ftype])
                        else:
                            pass
        self.logger.debug(self.__repr__())

    def mean_rpt(self, rpo, rlo, zbar, sigma_z, *, index=None):
        """
        Get the mean of the true perpendicular separation. All inputs must be
        scalar or 1D array-like with the same size, except :param:`sigma_z`,
        which can only be scalar

        :param rpo: The observed perpendicular separation at which to calculate
        :type rpo: scalar or 1D array-like `float`
        :param rlo: The observed parallel separation at which to calculate
        :type rlo: scalar or 1D array-like `float`
        :param zbar: The average observed redshift at which to calculate
        :type zbar: scalar or 1D array-like `float`
        :param sigma_z: The redshift uncertainty
        :type sigma_z: scalar `float`
        :kwarg index: Optionally provide an index for returning a 
        :class:`pandas.Series`. Default `None`
        :type index: :class:`pandas.Index` or :class:`pandas.MultiIndex`
        :return: The mean with the scaling undone
        :rtype: scalar or 1D :class:`numpy.ndarray` `float`
        """
        return (_perp_mean_scale(rpo, rlo, zbar, sigma_z) * self.mean_x.model(
                rpo, rlo, index=index) + rpo)

    def var_rpt(self, rpo, rlo, zbar, sigma_z, *, index=None):
        """
        Get the variance of the true perpendicular separation. All inputs must
        be scalar or 1D array-like with the same size, except :param:`sigma_z`,
        which can only be scalar

        :param rpo: The observed perpendicular separation at which to calculate
        :type rpo: scalar or 1D array-like `float`
        :param rlo: The observed parallel separation at which to calculate
        :type rlo: scalar or 1D array-like `float`
        :param zbar: The average observed redshift at which to calculate
        :type zbar: scalar or 1D array-like `float`
        :param sigma_z: The redshift uncertainty
        :type sigma_z: scalar `float`
        :kwarg index: Optionally provide an index for returning a 
        :class:`pandas.Series`. Default `None`
        :type index: :class:`pandas.Index` or :class:`pandas.MultiIndex`
        :return: The variance with the scaling undone
        :rtype: scalar or 1D :class:`numpy.ndarray` `float`
        """
        return (_perp_var_scale(rpo, rlo, zbar, sigma_z)**2 * self.var_x.model(
                rpo, rlo, index=index))

    def mean_rlt(self, rpo, rlo, zbar, sigma_z, *, index=None):
        """
        Get the mean of the true parallel separation. All inputs must be scalar
        or 1D array-like with the same size, except :param:`sigma_z`, which can
        only be scalar

        :param rpo: The observed perpendicular separation at which to calculate
        :type rpo: scalar or 1D array-like `float`
        :param rlo: The observed parallel separation at which to calculate
        :type rlo: scalar or 1D array-like `float`
        :param zbar: The average observed redshift at which to calculate
        :type zbar: scalar or 1D array-like `float`
        :param sigma_z: The redshift uncertainty
        :type sigma_z: scalar `float`
        :kwarg index: Optionally provide an index for returning a 
        :class:`pandas.Series`. Default `None`
        :type index: :class:`pandas.Index` or :class:`pandas.MultiIndex`
        :return: The mean with the scaling undone
        :rtype: scalar or 1D :class:`numpy.ndarray` `float`
        """
        return (_par_mean_scale(rpo, rlo, zbar, sigma_z) * self.mean_y.model(
                rpo, rlo, index=index) + rlo)

    def var_rlt(self, rpo, rlo, zbar, sigma_z, *, index=None):
        """
        Get the variance of the true parallel separation. All inputs must be
        scalar or 1D array-like with the same size, except :param:`sigma_z`,
        which can only be scalar

        :param rpo: The observed perpendicular separation at which to calculate
        :type rpo: scalar or 1D array-like `float`
        :param rlo: The observed parallel separation at which to calculate
        :type rlo: scalar or 1D array-like `float`
        :param zbar: The average observed redshift at which to calculate
        :type zbar: scalar or 1D array-like `float`
        :param sigma_z: The redshift uncertainty
        :type sigma_z: scalar `float`
        :kwarg index: Optionally provide an index for returning a 
        :class:`pandas.Series`. Default `None`
        :type index: :class:`pandas.Index` or :class:`pandas.MultiIndex`
        :return: The variance with the scaling undone
        :rtype: scalar or 1D :class:`numpy.ndarray` `float`
        """
        return (_par_var_scale(rpo, rlo, zbar, sigma_z)**2 * self.var_y.model(
                rpo, rlo, index=index))

    def cov_rpt_rlt(self, rpo, rlo, zbar, sigma_z, *, index=None):
        """
        Get the covariance of the true perpendicular and parallel separations.
        All inputs must be scalar or 1D array-like with the same size, except
        :param:`sigma_z`, which can only be scalar

        :param rpo: The observed perpendicular separation at which to calculate
        :type rpo: scalar or 1D array-like `float`
        :param rlo: The observed parallel separation at which to calculate
        :type rlo: scalar or 1D array-like `float`
        :param zbar: The average observed redshift at which to calculate
        :type zbar: scalar or 1D array-like `float`
        :param sigma_z: The redshift uncertainty
        :type sigma_z: scalar `float`
        :kwarg index: Optionally provide an index for returning a 
        :class:`pandas.Series`. Default `None`
        :type index: :class:`pandas.Index` or :class:`pandas.MultiIndex`
        :return: The covariance between the perpendicular and parallel
        directions
        :rtype: scalar or 1D :class:`numpy.ndarray` or :class:`pandas.Series`
        `float`
        """
        return (np.sqrt(self.var_rpt(rpo, rlo, zbar, sigma_z) *
                        self.var_rlt(rpo, rlo, zbar, sigma_z)) *
                self.mean_r.model(rpo, rlo, index=index))

    def det_cov_matrix(self, rpo, rlo, zbar, sigma_z, *, index=None):
        """
        Get the determinant of the covariance matrix of the true perpendicular
        and parallel separations. All inputs must be scalar or 1D array-like
        with the same size, except :param:`sigma_z`, which can only be scalar.

        :param rpo: The observed perpendicular separation at which to calculate
        :type rpo: scalar or 1D array-like `float`
        :param rlo: The observed parallel separation at which to calculate
        :type rlo: scalar or 1D array-like `float`
        :param zbar: The average observed redshift at which to calculate
        :type zbar: scalar or 1D array-like `float`
        :param sigma_z: The redshift uncertainty
        :type sigma_z: scalar `float`
        :kwarg index: Optionally provide an index for returning a 
        :class:`pandas.Series`. Default `None`
        :type index: :class:`pandas.Index` or :class:`pandas.MultiIndex`
        :return d: The determinant of the covariance matrix between the
        perpendicular and parallel directions
        :rtype d: scalar or 1D :class:`numpy.ndarray` or :class:`pandas.Series`
        `float`
        """
        d = (self.var_rpt(rpo, rlo, zbar, sigma_z) *
             self.var_rlt(rpo, rlo, zbar, sigma_z) *
             (1. - self.mean_r.model(rpo, rlo, index=index)**2))
        return d

    def inverse_cov_matrix(self, rpo, rlo, zbar, sigma_z, *, index=None):
        """
        Get the inverse covariance matrix of the true perpendicular and parallel
        separations. All inputs must be scalar or 1D array-like with the same
        size, except :param:`sigma_z`, which can only be scalar. The output will
        be a nx2x2 matrix where n is the length of the inputs, and is 1 for
        scalars.

        :param rpo: The observed perpendicular separation at which to calculate
        :type rpo: scalar or 1D array-like `float`
        :param rlo: The observed parallel separation at which to calculate
        :type rlo: scalar or 1D array-like `float`
        :param zbar: The average observed redshift at which to calculate
        :type zbar: scalar or 1D array-like `float`
        :param sigma_z: The redshift uncertainty
        :type sigma_z: scalar `float`
        :kwarg index: Optionally provide an index for returning a 
        :class:`pandas.Series`. Default `None`
        :type index: :class:`pandas.Index` or :class:`pandas.MultiIndex`
        :return icov: The inverse covariance matrix between the perpendicular
        and parallel directions. This is not actually returned as a matrix, but 
        rather an array for easily computing the dot product in the Gaussian
        :rtype icov: :class:`numpy.ndarray` `float` with shape nx3, for input
        of length n (n = 1 for scalars)
        """
        if not hasattr(rpo, "__len__"):
            icov = np.empty((1, 3))
        else:
            icov = np.empty((len(rpo), 3))
        inv_det = 1. / self.det_cov_matrix(rpo, rlo, zbar, sigma_z)
        icov[:,0] = self.var_rlt(rpo, rlo, zbar, sigma_z) * inv_det
        icov[:,1] = self.var_rpt(rpo, rlo, zbar, sigma_z) * inv_det
        icov[:,2] = -(2 * self.mean_r.model(rpo, rlo, index=index) * 
                      np.sqrt(self.var_rpt(rpo, rlo, zbar, sigma_z) *
                              self.var_rlt(rpo, rlo, zbar, sigma_z)) * 
                      inv_det)
        if index is not None:
            icov = pd.DataFrame(icov, index=index)
        return icov


    def data_vector(self, rpt, rlt, rpo, rlo, zbar, sigma_z, *, index=None):
        """
        Get the "data vector" :math:`\vec{x} - \vec{\mu_x}` for the 2D data of
        the true parallel and perpendicular separations. The inputs must all be
        scalar or 1D array-like with the same size, except :param:`sigma_z`,
        which can only be scalar. The output will be an array of shape nx2,
        where n is the length of the inputs, and is 1 for scalars.

        :param rpt: The true perpendicular separation at which to calculate
        :type rpt: scalar or 1D array-like `float`
        :param rlt: The true parallel separation at which to calculate
        :type rlt: scalar or 1D array-like `float`
        :param rpo: The observed perpendicular separation at which to calculate
        :type rpo: scalar or 1D array-like `float`
        :param rlo: The observed parallel separation at which to calculate
        :type rlo: scalar or 1D array-like `float`
        :param zbar: The average observed redshift at which to calculate
        :type zbar: scalar or 1D array-like `float`
        :param sigma_z: The redshift uncertainty
        :type sigma_z: scalar `float`
        :kwarg index: Optionally provide an index for returning a 
        :class:`pandas.Series`. Default `None`
        :type index: :class:`pandas.Index` or :class:`pandas.MultiIndex`
        :return dvec: The inverse covariance matrix between the perpendicular
        and parallel directions
        :rtype dvec: :class:`numpy.ndarray` `float` with shape nx2, for input
        of length n (n = 1 for scalars)
        """
        if not hasattr(rpo, "__len__"):
            dvec = np.empty((1, 2))
        else:
            dvec = np.empty((len(rpo), 2))
        dvec[:,0] = rpt - self.mean_rpt(rpo, rlo, zbar, sigma_z)
        dvec[:,1] = rlt - self.mean_rlt(rpo, rlo, zbar, sigma_z)
        if index is not None:
            dvec = pd.DataFrame(dvec, index=index)
        return dvec
        

    def prob(self, rpt, rlt, rpo, rlo, zbar, sigma_z, *, index=None):
        """
        Get the probability of the true separations given the input observed
        separations, the average observed redshift, and the fractional redshift
        uncertainty. All parameters must be scalar or 1D array-like with the
        same size, except for :param:`sigma_z`, which can only be scalar

        :param rpt: The true perpendicular separation at which to find the
        probability
        :type rpt: scalar or 1D array-like `float`
        :param rlt: The true parallel separation at which to find the
        probability
        :type rlt: scalar or 1D array-like `float`
        :param rpo: The observed perpendicular separation at which to calculate
        :type rpo: scalar or 1D array-like `float`
        :param rlo: The observed parallel separation at which to calculate
        :type rlo: scalar or 1D array-like `float`
        :param zbar: The average observed redshift at which to calculate
        :type zbar: scalar or 1D array-like `float`
        :param sigma_z: The redshift uncertainty
        :type sigma_z: scalar `float`
        :kwarg index: If desired, can pass an index for returning a 
        :class:`pandas.Series` rather than a :class:`numpy.ndarray`. Default 
        `None`
        :type index: :class:`pandas.Index` or :class:`pandas.MultiIndex`
        :return p: The probability of the true separations given the set of
        observed separations and average observed redshifts as well as the
        redshift uncertainty
        :rtype p: scalar or 1D :class:`numpy.ndarray` or :class:`pandas.Series`
        `float`
        """
        icov = self.inverse_cov_matrix(rpo, rlo, zbar, sigma_z)
        det = self.det_cov_matrix(rpo, rlo, zbar, sigma_z)
        dvec = self.data_vector(rpt, rlt, rpo, rlo, zbar, sigma_z)
        p = (np.exp(-0.5 * (dvec[:,0]**2 * icov[:,0] + dvec[:,1]**2 * icov[:,1] + 
                            dvec[:,0] * dvec[:,1] * icov[:,2])) / 
             (2. * np.pi * np.sqrt(det)))
        if index is not None:
            p = pd.Series(p, index=index)
        return p


def double_einsum(a, b):
    """
    This is a helper function for doing :math:`\vec{a_i} \cdot \mathbf{b_i}
    \cdot \vec{a_i}` over all elements i in a and b.

    :param a: An array of vectors with shape (N,M)
    :type a: :class:`numpy.ndarray`
    :param b: An array of matrices with shape (N,M,M)
    :type b: class:`numpy.ndarray`
    :return: An array containing the chained dot product a.b.a for each
    element along the zeroth axis, with shape (N,)
    :rtype: :class:`numpy.ndarray`
    """
    return np.einsum("ik,ik->i", np.einsum("ij,ijk->ik", a, b), a)
