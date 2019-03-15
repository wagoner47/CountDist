from __future__ import print_function
from .utils import ndigits, init_logger, _initialize_cosmology
from astropy import cosmology
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
import contextlib
from abc import ABC, abstractmethod # implement python abstract base class
import types
import inspect

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
    Calculate the correlation coefficient between x and y. All inputs other than
    :param:`n` must be scalar or 1D array-like with the same shape

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
    :return: The correlation between x and y
    :rtype: scalar or 1D array-like `float`
    """
    return ((x - x_mean) * (y - y_mean)) / np.sqrt(x_var * y_var)



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

def prior_mean_y(theta):
    """
    The prior to use on the fitting function defined in :class:`MeanY`, which
    requires checking the relation between some parameters

    :param theta: The parameter values to be checked
    :type theta: array-like or `dict`
    :return: The value of the log prior probability, which is 0 for passing or
    -infinity if it fails
    :rtype: `float`

    """
    theta = __make_clean_dict__(theta, mean_y.params)
    if (0 < theta["alpha"] < theta["beta"] and theta["b"] > 0 and
        theta["c"] > 0 and theta["d"] > 0):
        return 0.0
    return -math.inf

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

class MeanR(object):
    params = [r"$a$", r"$b$", r"$c$"]
    __name__ = "mean_r"
    def __call__(self, rpo, rlo, a, b, c, *, index=None, rpo_scale=1.0,
                 rlo_scale=1.0, rpo_pivot=None, **kwargs):
        """For convenience, the best fitting function found for the mean of the
        correlation between :math:`R_\perp^T` and :math:`R_\parallel^T`. This
        function is actually constant in :param:`rlo`, but asks for it for
        consistency with the expected call signature of other potential fitting
        functions. The dependence on :param:`rpo` looks like an exponential of a
        quadratic of :math:`\ln R_\perp^O`. A pivot for :param:`rpo` can also
        optionally be provided, for better fitting.

        :param rpo: The observed perpendicular separations
        :type rpo: scalar or array-like `float`
        :param rlo: The observed parallel separations. This parameter is
        technically ignored in the function, but is included for call signature
        consistency.
        :type rlo: scalar or array-like `float`
        :param a: The constant on the squared term of the quadratic
        :type a: scalar `float`
        :param b: The constant on the linear term of the quadratic
        :type b: scalar `float`
        :param c: The constant term of the quadratic
        :type c: scalar `float`
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
        :key rpo_pivot: Optional pivot location for the observed perpendicular
        separation, used for stability in fitting. The value will be rescaled by
        :key:`rpo_scale`. If `None`, the scaled value will be set to 1,
        i.e. :key:`rpo_scale` will be used as the pivot. Default `None`
        :type rpo_pivot: `float` or `NoneType`
        :return f: The function evaluated at the separation(s)
        :rtype f: scalar, :class:`numpy.ndarray`, or :class:`pandas.Series`
        `float`
        """
        if rpo_pivot is None:
            log_x = np.log(rpo) - np.log(rpo_scale)
        else:
            log_x = np.log(rpo) - np.log(rpo_pivot)
        if isinstance(log_x, pd.Index):
            log_x = log_x.values
        f = np.exp(a * log_x**2 + b * log_x + c)
        if index is not None:
            f = pd.Series(f, index=index)
        return f
mean_r = MeanR()

def prior_mean_r(theta):
    """
    The non-flat prior to use for the fitting function defined in :class:`MeanR`

    :param theta: The values of the parameters being checked
    :type theta: array-like or `dict`
    :return: The value of the log prior probability, which is 0 for passing or
    -infinity if it fails
    :rtype: `float`
    """
    theta = __make_clean_dict__(theta, mean_r.params)
    if -1.0 < theat["c"] < 1.0:
        return 0.0
    return -math.inf

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

def _perp_mean_scale(rpo, rlo, zbar, sigma_z, cosmo):
    return rpo

def _perp_var_scale(rpo, rlo, zbar, sigma_z, cosmo):
    if hasattr(cosmo, "differential_comoving_distance"):
        dchi = cosmo.differential_comoving_distance(zbar).value
    else:
        dchi = cosmo.hubble_distance.value * cosmo.inv_efunc(zbar)
    return ((np.sqrt(0.5) * rpo * sigma_z * (1 + zbar) * dchi)
            / cosmo.comoving_distance(zbar).value)

def _par_mean_scale(rpo, rlo, zbar, sigma_z, cosmo):
    return rlo

def _par_var_scale(rpo, rlo, zbar, sigma_z, cosmo):
    if hasattr(cosmo, "differential_comoving_distance"):
        dchi = cosmo.differential_comoving_distance(zbar).value
    else:
        dchi = cosmo.hubble_distance.value * cosmo.inv_efunc(zbar)
    return (np.sqrt(2.0) * sigma_z * (1 + zbar) * dchi)

def _add_bin_column(seps, orig_col_name, bin_col_name, bin_size):
    seps[bin_col_name] = np.floor(seps.loc[:,orig_col_name] / bin_size).astype(
        int)

def _add_zbar(seps, cosmo=None):
    if "AVE_Z_OBS" in seps.columns:
        pass
    else:
        if "ZBAR" in seps.columns:
            seps.rename(columns={"ZBAR": "AVE_Z_OBS"})
        else:
            if cosmo is None:
                raise ValueError("Must have cosmology instance to obtain"
                                 " average redshift from average distance")
            if hasattr(cosmo, "inv_distance"):
                seps["AVE_Z_OBS"] = cosmo.inv_distance(seps["AVE_D_OBS"])
            else:
                seps["AVE_Z_OBS"] = cosmology.z_at_value(
                    cosmo.comoving_distance, seps["AVE_D_OBS"])

def _add_delta_column(seps, direction, scale_func, dcol_name, sigma_z, cosmo):
    tcol_name = "R_{}_T".format(direction)
    ocol_name = "R_{}_O".format(direction)
    scale = 1. / scale_func(seps.loc[:,"R_PERP_O"], seps.loc[:,"R_PAR_O"],
                            seps.loc[:,"AVE_Z_OBS"], sigma_z, cosmo)
    seps[dcol_name] = seps[tcol_name].sub(seps[ocol_name]).mul(scale)

def add_extra_columns(seps, perp_bin_size, par_bin_size, sigma_z, cosmo):
    """This function adds some of the extra data columns to the input DataFrame
    that are needed for grouping and generating the statistics. It does not add
    the column for the correlation, that is handled by a separate function.

    :param seps: The DataFrame of the separations which should already contain,
    at a minimum, columns 'R_PERP_O', 'R_PAR_O', 'R_PERP_T', and 'R_PAR_T'.
    The final needed column can be either 'AVE_D_OBS' or 'ZBAR'/'AVE_Z_OBS' for
    forward and backward compatability. Additional columns are ignored
    :type seps: :class:`pandas.DataFrame`
    :param perp_bin_size: The bin size to use for binning 'R_PERP_O', in the
    same units as 'R_PERP_O'
    :type perp_bin_size: `float`
    :param par_bin_size: The bin size to use for binning 'R_PAR_O', in the same
    units as 'R_PAR_O'
    :type par_bin_size: `float`
    :param sigma_z: The redshift error assumed for the separations
    :type sigma_z: `float`
    :param cosmo: An instance of a cosmology class. Compatible with any subclass
    of :class:`astropy.cosmology.Cosmology`, but works especially well with
    :class:`CatalogUtils.FastCosmology`
    :type cosmo: :class:`CatalogUtils.FastCosmology` or similar
    """
    logger = glogger.getChild(__name__)
    logger.debug("Add column RPO_BIN")
    _add_bin_column(seps, "R_PERP_O", "RPO_BIN", perp_bin_size)
    logger.debug("Add column RLO_BIN")
    _add_bin_column(seps, "R_PAR_O", "RLO_BIN", par_bin_size)
    logger.debug("Add column AVE_Z_OBS (if needed)")
    _add_zbar(seps, cosmo)
    logger.debug("Add column DELTA_R_PERP")
    _add_delta_column(seps, "PERP", _perp_mean_scale, "DELTA_R_PERP", sigma_z,
                      cosmo)
    logger.debug("Add column x")
    _add_delta_column(seps, "PERP", _perp_var_scale, "x", sigma_z, cosmo)
    logger.debug("Add column DELTA_R_PAR")
    _add_delta_column(seps, "PAR", _par_mean_scale, "DELTA_R_PAR", sigma_z,
                      cosmo)
    logger.debug("Add column y")
    _add_delta_column(seps, "PAR", _par_var_scale, "y", sigma_z, cosmo)

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
    stats = pd.concat([grouped["r"].agg(kstat, 1),
                       grouped["r"].agg(kstatvar, 1)],
                      keys=[("mean_r", "mean"), ("mean_r", "variance")], axis=1)
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
    extents = __make_clean_dict__(extents)
    theta = __make_clean_dict__(theta, list(extents.keys()))
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


class SingleFitterBase(ABC):
    """
    Abstract base for a single fitter, for fitting to a single set of data.
    """
    def __init__(self, data=None, index_names=None, col_names=None,
                 fitter_name=None, param_names=None, *, func=None, prior=None,
                 rpo_size=1.0, rlo_size=1.0, func_args=None, func_kwargs=None,
                 **kwargs):
        """
        Initialize the fitter object.

        :param data: A :class:`pandas.DataFrame` object with indices giving
        the observed perpendicular and parallel separations, and columns
        for the data and variance. This does not allow for correlations in
        the data, unfortunately. Note that while this is optional for use in
        context, the data (as well as :param:`index_names` and
        :param:`col_names`) must be provided before performing any fits.
        Calling the :func:`~countdist2.SingleFitterBase.plot` method without
        setting data gives a warning, but may still plot the model if
        separations are provided and the call is done in context. Default `None`
        :type data: :class:`pandas.DataFrame` or `NoneType`, optional
        :param index_names: A sequence of the index names, with the first
        element giving the name of the index level referring to the
        perpendicular separations and the second the index level name for the
        parallel separations (although the index levels need not be in that
        order). This is typically required for fitting (see
        :func:`~countdist2.SingleFitterBase.fit` method for specific fit type)
        or if :param:`data` is given, but is not needed when used in context.
        Default `None`
        :type index_names: `sequence`[`str`] or `NoneType`, optional
        :param col_names: A sequence of the column names, with the first
        element giving the name of the column containing the data and the
        second the column name for the variance (although the columns need not
        be in that order). This is always required for fitting or if
        :param:`data` is given, but is not needed when used in context.
        Default `None`
        :type col_names: `sequence`[`str`] or `NoneType`, optional
        :param fitter_name: A string for the name of this specific fitter.
        If `None`, the name is simply the name of the class. Otherwise,
        the :attr:`name` for the instance will be
        (`__class__.__name__`).(:param:`fitter_name`). Default `None`
        :type fitter_name: `str` or `NoneType`, optional
        :param param_names: A string or sequence of strings (depending on
        possible dimensionality of specific fit type) giving parameter names,
        with latex formatting as desired for plotting. For analytic fitters,
        a value of `None` here results in the parameter being named '$c$'.
        For non-analytic fitters, parameter names must be supplied when the
        fitting function is given. Default `None`
        :key func: A fitting function for non-analytic fitters. This parameter
        is ignored for analytic fitters, but must be given before fitting,
        plotting a model, or evaluating in context for all non-analytic fitters.
        The function must take the perpendicular separation as the first
        argument, parallel separation as the second argument, and parameters
        as individual positional arguments after that. Additional positional
        arguments must come after, and may be specified with :key:`func_args`,
        and keyword arguments may be specified with :key:`func_kwargs`. The
        function should return a `float` or `sequence` of `float`, and should
        be vectorized. Default `None`
        :type func: `Callable` or `NoneType`
        :key prior: The log prior probability function to assume for the data,
        or a dictionary with keys given by parameter names (from
        :param:`param_names`) and values as `sequence` of `float` for extents
        of flat priors, for which a function can be automatically generated.
        This is ignored for analytic fitters, but is required for fitting
        in any non-analytic fitters. The call signature should take the
        parameter values as a `sequence` or `dict` of `float` (with keys
        given by parameter names as in :param:`param_names` for `dict`), and
        return a `float`. Default `None`
        :type prior: `Callable`, `dict`(`str`, `sequence`[`float`]), or
        `NoneType`
        :key rpo_size: The size of observed perpendicular separation bins in
        the index of the data, for resizing bins in plots. Default 1.0
        :type rpo_size: `float`
        :key rlo_size: The size of observed parallel separation bins in the
        index of the data, for resizing bins in plots. Default 1.0
        :type rlo_size: `float`
        :key func_args: Additional positional arguments for the fitting function
        for non-analytic fitters, with `None` indicating no additional
        positional arguments are needed. Ignored for analytic fitters. Default
        `None`
        :type func_args: `sequence` or `NoneType`
        :key func_kwargs: Keyword arguments for the fitting function for
        non-analytic fitters, with `None` indicating no key word arguments
        are needed. Ignored for analytic fitters. Default `None`
        :type func_kwargs: `dict`(`str`, `any`) or `NoneType`
        """
        self._get_name(fitter_name)
        self.logger = init_logger(self.name)
        self._rpo_size = rpo_size
        self._rlo_size = rlo_size
        self.logger.debug("Set up data and variance")
        self.data = (data, index_names, col_names)
        self.logger.debug("Initialize best fit attributes")
        self._initialize_fit()
        self.logger.debug("__init__ complete")

    def _get_name(self, name):
        self.name = self.__class__.__name__
        if name is not None:
            self.name += "." + name

    def _initialize_fit(self):
        self._best_fit_params = None

    def __repr__(self):
        return ("{self.name!s}(ndim={self.ndim!r},"
                " best_fit={self._best_fit_params!r})".format(self=self))

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
        else:
            d["logger"] = init_logger(d["name"])
        if "c" in d:
            c = ChainConsumer()
            [c.add_chain(
                chain.chain, parameters=chain.parameters, name=chain.name,
                posterior=chain.posterior, color=chain.color,
                walkers=chain.walkers) for chain in d["c"][0]]
            for i, chain in enumerate(d["c"][0]):
                c.chains[i].config = chain.config
            c.config = d["c"][1]
            c.config_truth = d["c"][2]
            d["c"] = c
        if "c_walkers" in d:
            c = ChainConsumer()
            [c.add_chain(
                chain.chain, parameters=chain.parameters, name=chain.name,
                posterior=chain.posterior, color=chain.color,
                walkers=chain.walkers) for chain in d["c_walkers"][0]]
            for i, chain in enumerate(d["c_walkers"][0]):
                c.chains[i].config = chain.config
            c.config = d["c_walkers"][1]
            c.config_truth = d["c_walkers"][2]
            d["c_walkers"] = c
        # For backwards compatability
        if "_data" not in d:
            d["_data"] = d.pop("data", None)
        if "_rpo_size" not in d:
            d["_rpo_size"] = d.pop("rpo_size", 1.0)
        if "_rlo_size" not in d:
            d["_rlo_size"] = d.pop("rlo_size", 1.0)
        d["cosmo"] = d.get("cosmo", None)
        self.__dict__.update(d)

    @property
    def best_fit(self):
        """The best fit parameters"""
        return self._best_fit_params

    @property
    def data(self):
        """The underlying DataFrame. Setter must be used with tuple of
        (data, index_names, col_names)"""
        return self._data

    @data.setter
    def data(self, data_tuple):
        try:
            self._data, self.index_names, self.col_names = data_tuple
        except ValueError:
            raise ValueError("data setter requires tuple of data, index_names,"
                             " and col_names")
        else:
            if self._data is not None:
                if self.index_names is None:
                    raise ValueError("index_names must be given if data is"
                                     " given")
                if self.col_names is None:
                    raise ValueError("col_names must be given if data is given")
                self.rpo = pd.Series(
                    (self._data.index.get_level_values(self.index_names[0])
                     + 0.5) * self._rpo_size, index=self._data.index)
                self.rlo = pd.Series(
                    (self._data.index.get_level_values(self.index_names[1])
                     + 0.5) * self._rlo_size, index=self._data.index)
            else:
                self.rpo = None
                self.rlo = None

    @property
    def rpo_size(self):
        """Size of perpendicular separation bins"""
        return self._rpo_size

    @rpo_size.setter
    def rpo_size(self, value):
        if self.rpo is not None:
            self.rpo *= (value / self._rpo_size)
        self._rpo_size = value

    @property
    def rlo_size(self):
        """Size of parallel separation bins"""
        return self._rlo_size

    @rlo_size.setter
    def rlo_size(self, value):
        if self.rlo is not None:
            self.rlo *= (value / self._rlo_size)
        self._rlo_size = value

    @property
    def data_vs_rlo(self):
        """Get the data with perpendicular separations as the first index level
        and parallel separations as the second"""
        if self._data is None:
            return None
        if self.index_names[0] != self._data.index.names[0]:
            return self._data.swaplevel(0, 1, axis=0).sort_index()
        return self._data.copy()

    @property
    def data_vs_rpo(self):
        """Get the data with parallel separations as the first index level and
        perpendicular separations as the second"""
        if self._data is None:
            return None
        if self.index_names[1] != self._data.index.names[0]:
            return self._data.swaplevel(0, 1, axis=0).sort_index()
        return self._data.copy()

    @contextlib.contextmanager
    @abstractmethod
    def use(self):
        """Context manager to force the fitter to use best fit parameter values
        and samples/best fit parameter errors"""
        pass

    def f(self, *args, **kwargs):
        pass

    def model(self, rpo, rlo, index=None, **kwargs):
        """
        Get the best fit model at the given separations. The separations must
        either both be scalar or both be sequences with the same size.

        Keyword arguments are passed to the fitting function

        :param rpo: The observed perpendicular separation(s)
        :type rpo: `float` or `sequence`[`float`]
        :param rlo: The observed parallel separation(s)
        :type rlo: `float` or `sequence`[`float`]
        :param index: If not `None`, return evaluation as a
        :class:`pandas.Series` with this index. Default `None`
        :type index: :class:`pandas.Index`, :class:`pandas.MultiIndex`,
        `sequence`, or `NoneType`, optional
        :return: The model evaluated at the given separation pair(s)
        :rtype: `float`, :class:`numpy.ndarray`[`float`], or
        :class:`pandas.Series`[`float`]
        """
        if self._best_fit_params is None:
            raise AttributeError("Cannot evaluate best fit model without best"
                                 " fit parameters")
        return self._f(rpo, rlo, self._best_fit_params, index=index, **kwargs)

    @abstractmethod
    def model_with_errors(self, rpo, rlo, index=None, **kwargs):
        """
        Evaluate the model with the 68%/1 sigma confidence region at the given
        separation pair(s). Separations must both be scalar or both sequences
        with the same shape.

        Keyword arguments are passed to the fitting function

        :param rpo: The observed perpendicular separation(s)
        :type rpo: `float` or `sequence`[`float`]
        :param rlo: The observed parallel separation(s)
        :type rlo: `float` or `sequence`[`float`]
        :param index: If not `None`, use this as the index for the returned
        :class:`pandas.DataFrame`. Otherwise, use `Pandas` default indexing.
        Default `None`
        :type index: :class:`pandas.Index`, :class:`pandas.MultiIndex`,
        `seqeuence`, or `NoneType`, optional
        :return: The model and error region evaluated at the separation pair(s)
        :rtype: :class:`pandas.DataFrame`
        """
        pass

    @abstractmethod
    def fit(self):
        """
        Perform the fit of the data.
        """
        pass

    def plot(self, rpo_label, rlo_label, ylabel, bins, is_rpo,
             perp_bin_scale=1.0, par_bin_scale=1.0, perp_bin_size=1.0,
             par_bin_size=1.0, indices=None, index_names=None, exp=None,
             logx=False, logy=False, filename=None, figsize=None,
             display=False, text_size=22, with_fit=False, point_alpha=1.0,
             return_fig=True):
        """
        Plot the data and/or best fit model at the specified bins in
        perpendicular or parallel separation

        :param rpo_label: The label for perpendicular separtions. This will
        be used to label the x-axis if :param:`is_rpo` is `False` or the
        axes if :param:`is_rpo` is `True`
        :type rpo_label: `str`
        :param rlo_label: The label for parallel separations. This will be used
        to label either the x-axis or the axes opposite of :param:`rpo_label`
        :type rlo_label: `str`
        :param ylabel: The label on the y-axis
        :type ylabel: `str`
        :param bins: The bin(s) of fixed separation to include in the plot
        :type bins: `int` or `sequence`[`int`]
        :param is_rpo: If `True`, assume value(s) in :param:`bins` are drawn
        from the perpendicular separation index level. Otherwise, assume
        parallel separation bin(s) are specified
        :type is_rpo: `bool`
        :param perp_bin_scale: The scale of the perpendicular separation bins.
        Default 1.0
        :type perp_bin_scale: `float`, optional
        :param par_bin_scale: The scale of the parallel separation bins. Default
        1.0
        :type par_bin_scale: `float`, optional
        :param perp_bin_size: The size of perpendicular separation bins, ignored
        if data is set. Default 1.0
        :type perp_bin_size: `float`, optional
        :param par_bin_size: The size of parallel separation bins, ignored if
        data is set. Default 1.0
        :type par_bin_size: `float`, optional
        :param indices: Ignored if data is set. The perpendicular and parallel
        separation bins (as a :class:`pandas.MultiIndex` or a sequence of
        tuples to create one) at which to evaluate the model when plotting only
        the model, and from which the value(s) in :param:`bins` are drawn. In
        this case, the bin scale paramters should actually be the scale divided
        by the bin size. This is required when data is not set. Default `None`
        :type indices: :class:`pandas.MultiIndex` or `NoneType`, optional
        :param index_names: Ignored if data is set. The names of the index
        levels for :param:`indices`, with the first element giving the level
        name for perpendicular separations and the second for parallel
        separations. This is required when data is not set. Default `None`
        :type index_names: `sequence`[`str`] or `NoneType`, optional
        :param exp: The expected constant (or limiting behavior constant),
        if known. If `None`, no expected value is plotted. Default `None`
        :type exp: `float` or `NoneType`, optional
        :param logx: If `True`, use log scaling on the x-axis. Default `False`
        :type logx: `bool`, optional
        :param logy: If `True`, use log scaling on the y-axis. Default `False`
        :type logy: `bool`, optional
        :param filename: If not `None`, save the figure in a file at the given
        path. Otherwise, plot is not saved. Default `None`
        :param figsize: Figure size to use. If `None`, this is set to
        `(8, 0.75 + len(bins))`. Default `None`
        :type figsize: `tuple`(`float`, `float`) or `NoneType`, optional
        :param display: If `True`, display the plot when finished. Default
        `False`
        :type display: `bool`, optional
        :param text_size: Size of figure text. Default 22
        :type text_size: `float`
        :param with_fit: If `True`, try to include best fit on plot
        (requires fitting to be done or context to be used). If data is not
        set, this is ignored and best fit is always included (if possible).
        Default `False`
        :type with_fit: `bool`, optional
        :param point_alpha: The transparency to use on the points when plotting
        data. Useful when data is dense. This is ignored if data is not set.
        Smaller numbers are more transparent. Default 1.0
        :type point_alpha: `float`, optional
        :param return_fig: If `True`, return the figure object that was created.
        Default `True`
        :type return_fig: `bool`, optional
        :return fig: The figure that has been created, if :param:`return_fig`
        is `True`
        :rtype fig: :class:`matplotlib.figure.Figure`, optional
        """
        if is_rpo:
            rlabel = rpo_label
            xlabel = rlo_label
        else:
            rlabel = rlo_label
            xlabel = rpo_label
        if self._data is None:
            if indices is None or index_names is None:
                raise AttributeError("Cannot make plot with no data set and no"
                                     " separations given")
            if self._best_fit_params is None:
                raise AttributeError("Cannot make plot with no data set and no"
                                     " fit parameters")
            warnings.warn("Making plot without data")
            with_fit = True
            with_data = False
            if ((is_rpo and index_names[0] != indices.names[0]) or
                (not is_rpo and index_names[1] != indices.names[0])):
                data = indices.swaplevel(
                    0, 1).sort_values().to_frame().loc[bins]
            else:
                data = indices.to_frame().loc[bins]
            data += 0.5
            data *= [perp_bin_size, par_bin_size]
            data.set_index(data.columns, drop=False, append=False, inplace=True)
            if is_rpo:
                r_bin_size = perp_bin_size / perp_bin_scale
                x_bin_size = par_bin_size / par_bin_scale
            else:
                r_bin_size = par_bin_size / par_bin_scale
                x_bin_size = perp_bin_size / perp_bin_scale
        else:
            with_data = True
            if is_rpo:
                r_bin_size = self._rpo_size / perp_bin_scale
                x_bin_size = self._rlo_size / par_bin_scale
                data = self.data_vs_rlo.loc[bins]
            else:
                r_bin_size = self._rlo_size / par_bin_scale
                x_bin_size = self._rpo_size / perp_bin_scale
                data = self.data_vs_rpo.loc[bins]
            mod_kwargs = {}
        axis_label = r"${} = {{}} \pm {}$".format(
            smh.strip_dollars_and_double_braces(rlabel),
            smh.strip_dollar_signs(
                smh.pretty_print_number(0.5 * r_bin_size, 2)))
        if with_fit:
            if self._best_fit_params is None:
                warnings.warn("Ignoring with_fit option when no fit parameters"
                              " avilable")
                with_fit = False
            else:
                if isinstance(data.index, pd.MultiIndex):
                    all_r = data.index.get_level_values(
                        0).unique().sort_values()
                    all_x = data.index.get_level_values(
                        1).unique().sort_values()
                    mod_index = pd.MultiIndex.from_product(
                        [all_r, all_x], names=data.index.names)
                    all_r = mod_index.get_level_values(0)
                    all_x = mod_index.get_level_values(1)
                else:
                    all_x = data.index
                    all_r = np.repeat(bins, all_x.size)
                    mod_index = all_x
                if is_rpo:
                    model_args = (
                        (all_r + 0.5) * r_bin_size,
                        (all_x + 0.5) * x_bin_size)
                else:
                    model_args = (
                        (all_x + 0.5) * x_bin_size,
                        (all_r + 0.5) * r_bin_size)
                with_fill = True
                try:
                    mod = self.model_with_errors(
                        *model_args, index=mod_index, rpo_scale=perp_bin_scale,
                        rlo_scale=par_bin_scale)
                except AttributeError:
                    with_fill = False
                    mod = self.model(
                        *model_args, index=mod_index, rpo_scale=perp_bin_scale,
                        rlo_scale=par_bin_scale).to_frame(name=0.5)
                fill_alpha = point_alpha - 0.2
                if fill_alpha <= 0.0:
                    if fill_alpha <= -0.1:
                        fill_alpha = point_alpha / 10.0
                    fill_alpha = point_alpha - 0.1

        abins = np.atleast_1d(bins).flatten()
        if figsize is None:
            figsize = (8, 0.75 + abins.size)
        fig = plt.figure(figsize=figsize)
        grid = gridspec.GridSpec(abins.size, 1)
        full_ax = fig.add_subplot(grid[:])
        if logx:
            full_ax.set_xscale("log")
        for loc in ["top", "bottom", "left", "right"]:
            full_ax.spines[loc].set_color("none")
        full_ax.tick_params(
            labelcolor="w", which="both", top=False, bottom=False, left=False,
            right=False)
        full_ax.set_ylabel(ylabel)
        full_ax.set_xlabel(xlabel)
        for i, (r, r_val) in enumerate(zip(abins, (abins + 0.5) * r_bin_size)):
            ax = fig.add_subplot(grid[i], sharex=full_ax)
            if logy:
                ax.set_yscale("log")
            if exp is not None:
                ax.axhline(exp, c="k", lw=1)
            if with_data:
                line = ax.errorbar(
                    (data.loc[r].index + 0.5) * x_bin_size,
                    data.loc[r,self.col_names[0]],
                    yerr=np.sqrt(data.loc[r,self.col_names[1]]),
                    fmt="C0o", alpha=point_alpha)[0]
            if with_fit:
                if with_fill:
                    ax.fill_between(
                        (mod.loc[r].index + 0.5) * x_bin_size,
                        mod.loc[r,0.16], mod.loc[r,0.84], color="C1",
                        alpha=fill_alpha)
                fit, = ax.plot(
                    (mod.loc[r].index + 0.5) * x_bin_size,
                    mod.loc[r,0.5], "C1-")
            ax.legend(
                [line if with_data else fit],
                [axis_label.format(
                    smh.strip_dollar_signs(
                        smh.pretty_print_number(r_val, 2)))],
                loc=0, markerscale=0, handlelength=0, frameon=False)
            if ax.is_last_row():
                ax.tick_params(axis="x", which="both", direction="inout",
                               top=True, bottom=True)
            else:
                ax.tick_params(axis="x", which="both", direction="inout",
                               top=True, bottom=False, labelbottom=False)
        grid.tight_layout(fig)
        if filename is not None:
            fig.savefig(filename, bbox_inches="tight")
        if display:
            plt.show(fig)
        if return_fig:
            return fig
        return None


class SingleFitter(SingleFitterBase):
    """
    Non-analytic fitter of single set of data
    """
    __doc__ = SingleFitterBase.__doc__ + __doc__
    def __init__(self, data=None, index_names=None, col_names=None,
                 fitter_name=None, param_names=None, *, func=None, prior=None,
                 rpo_size=1.0, rlo_size=1.0, func_args=None, func_kwargs=None,
                 **kwargs):
        super(SingleFitter, self).__init__(
            data, index_names, col_names, fitter_name, param_names,
            rpo_size=rpo_size, rlo_size=rlo_size, **kwargs)
        self.set_fit_func(func, param_names, func_args, func_kwargs)
        self.set_prior_func(prior)

    def _initialize_fit(self):
        super(SingleFitter, self)._initialize_fit()
        self._samples = None
        self._nburnin = None

    def __setstate__(self, d):
        d["_f"] = d.get("_f", d["f"])
        d["_pr"] = d.get("_pr", d["pr"])
        super(SingleFitter, self).__setstate__(d)

    @property
    def nburnin(self):
        """Length of the burnin on MCMC"""
        return self._nburnin

    @nburnin.setter
    def nburnin(self, value):
        self._nburin = value

    @property
    def samples(self):
        """The MCMC chain samples without burn-in removed"""
        return self._samples

    @contextlib.contextmanager
    def use(self, params, samples=None):
        """

        :param params: Best fit parameters to use in this context as a
        dictionary keyed by the names in
        :attr:`~countdist2.SingleFitterBase.params` or as a sequence of values
        :type params: `dict`(`str`, `float`) or `sequence`[`float`]
        :param samples: If given, assumed MCMC chains to use for generating
        error regions. Default `None`
        :type samples: 2D or 3D :class:`numpy.ndarray`[`float`] or `NoneType`,
        optional
        """
        if isinstance(params, dict):
            params = np.array([params[key] for key in self.params])
        _params, self._best_fit_params = self._best_fit_params, params
        _samples, self._samples = self._samples, samples
        yield
        self._best_fit_params = _params
        self._samples = _samples

    def model_with_errors(self, rpo, rlo, index=None, **kwargs):
        if self._samples is None:
            raise AttributeError("Cannot evaluate model with errors if samples"
                                 " not available")
        if np.atleast_1d(rpo).size != np.atleast_1d(rlo).size:
            raise ValueError("Perpendicular and parallel separations must have"
                             " same size")
        if index is not None and len(index) != np.atleast_1d(rpo).size:
            raise ValueError("Index must have same size as separations if"
                             " given")
        nburnin = self._nburnin
        if nburnin is None:
            warnings.warn("Using nburnin = 0 (not set)")
            nburnin = 0
        samples = self._samples[:,nburnin:,:].reshape((-1, self.ndim))
        meval = map(
            lambda t: self._f(rpo, rlo, t, index=index, **kwargs), samples)
        if (not hasattr(rpo, "__len__") or len(rpo) == 1) and index is None:
            meval = pd.Series(list(meval))
            return meval.quantile([0.16, 0.5, 0.84])
        if index is None:
            index = np.arange(np.atleast_1d(rpo).size, dtype=int)
        meval = pd.DataFrame(
            dict(zip(range(len(samples)), meval)), index=index)
        return meval.quantile([0.16, 0.5, 0.84], axis="columns").T

    def set_fit_func(self, func, param_names=None, args=None, kwargs=None):
        """
        Set or reset the fitting function. If resetting, this also resets the
        fit quantities. Calling the method
        :func:`~countdist2.SingleFitterBase.f` from this instance now calls
        :param:`func` (without any parameters passed)

        :param func: The fitting function to use now. `None` is allowed, but
        not when fitting. This should take the perpendicular separation first,
        parallel separation second, parameters as positional arguments next,
        and finally any other positional arguments and key word arguments.
        Arguments and key word arguments that should be used every time the
        function is called may be supplied with :param:`args` and
        :param:`kwargs`, respectively
        :type func: `Callable` or `NoneType`
        :param param_names: The fitting function parameter names. If not given,
        will attempt to obtain the names from the function if passed. Default
        `None`
        :type param_names: `sequence`[`str`] or `NoneType`, optional
        :param args: Additional positional arguments to use every time the
        function is called from within the instance, if any. Default `None`
        :type args: `sequence`, optional
        :param kwargs: Key word arguments to be used every time the function
        is called within the instance, if any. If called with other key word
        arguments, those take precedence. Default `None`
        :type kwargs: `dict` or `NoneType`, optional
        """
        if func is None:
            self.logger.debug("Setting fitting function to None")
            self.params = param_names
            self.ndim = None if self.params is None else len(self.params)
            def f_wrapper(self, *args, **kwargs):
                pass
            self.f = types.MethodType(f_wrapper, self)
            self._f = None
        else:
            if param_names is None:
                if hasattr(func, "params"):
                    param_names = func.params
                else:
                    param_names = [
                        p.name for p in inspect.signature(
                            func).paramters.values() if
                        p.kind in [
                            p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD]][2:]
            self.params = param_names
            self.ndim = len(self.params)
            self.args = args if args is not None else []
            self.kwargs = kwargs if kwargs is not None else {}
            self.f = types.MethodType(func, self)
            def f_wrapper(self, x, y, theta, **kwargs):
                for key, value in self.kwargs:
                    kwargs[key] = kwargs.get(key, value)
                return self.f(x, y, *theta, *self.args, **kwargs)
            self._f = types.MethodType(f_wrapper, self)

    def set_prior_func(self, prior):
        """
        Set or reset the log prior probability function

        :param prior: The prior probability function, or a dictionary keyed
        by :attr:`~countdist2.SingleFitterBase.params` with extents for a
        flat prior
        :type prior: `Callable`, `dict`(`str`, `sequence`[`float`]), or
        `NoneType`
        """
        if prior is None:
            self.logger.debug("Setting prior to None")
            self.prior = None
            self._pr = None
        else:
            if isinstance(prior, dict):
                self.prior = prior
                for key in self.params:
                    self.prior[key] = self.prior.get(key, [-np.inf, np.inf])
                    if (not hasattr(self.prior[key], "__len__") or
                        len(self.prior[key]) != 2):
                        self.prior[key] = [-np.inf, np.inf]
                def prior_wrapper(self, theta):
                    return flat_prior(theta, self.prior)
                self._pr = types.MethodType(prior_wrapper, self)
            else:
                self.prior = None
                self._pr = types.MethodType(prior, self)

    def lnlike(self, theta):
        """
        Calculate the log likelihood of the parameters given the data

        :param theta: The parameter values at which to calculate
        :type theta: `sequence`[`float`]
        :return: The log likelihood value assuming a Gaussian likelihood
        :rtype: `float`
        """
        if self._data is None:
            raise AttributeError("Cannot evaluate likelihood function without"
                                 " data")
        if self._f is None:
            raise AttributeError("Cannot evaluate likelihood function without"
                                 " fitting function")
        fev = self._f(self.rpo, self.rlo, theta, index=self._data.index)
        return (-0.5 * np.sum(
            (fev - self._data.loc[:,self.col_names[0]])**2
            / self._data.loc[self.col_names[1]]))

    def lnprob(self, theta):
        """
        Get the log probability function, which is the log prior times the
        log likelihood

        :param theta: The parameter values at which to evaluate
        :type theta: `sequence`[`float`]
        :return: The value of the log probability function, assuming a Gaussian
        likelihood
        :rtype: `float`
        """
        if self._data is None:
            raise AttributeError("Cannot evaluate log proabaility without data")
        if self._pr is None:
            raise AttributeError("Cannot evaluate log proabaility without"
                                 " prior")
        if self._f is None:
            raise AttributeError("Cannot evaluate log proabaility without"
                                 " fitting function")
        lp = self._pr(theta)
        if not math.isfinite(lp):
            return -math.inf
        return lp + self.lnlike(theta)

    def fit_minimize(self, init_guess):
        """
        Run a fit of the data using :func:`scipy.optimize.minimize`

        :param init_guess: An initial guess for the starting point of the
        fit
        :type init_guess: `sequence`[`float`]
        :return res: The minimization result. See documenation for
        :func:`scipy.optimize.minimize`
        :rtype res: :class:`scipy.optimize.OptimizeResult`
        """
        nll = lambda theta: -self.lnprob(theta)
        res = minimize(nll, init_guess)
        self._best_fit_params = res.x
        return res

    def fit(self, nsteps, nburnin=None, init_guess=None, nwalkers=None,
            pool=None, sampler=None):
        """
        This fit method utilizes an MCMC. Provide a sampler object to continue
        sampling, or else create a new one

        .. |~| unicode:: U+00A0
           :trim:

        :param nsteps: The number of MCMC steps to take with each walker
        :type nsteps: `int`
        :param nburnin: The burn-in phase to omit. Setting to zero will result
        in a warning, as this may be unsafe. If not given this defaults to
        :attr:`~countdist2.SingleFitterBase.nburnin` if set or 0 otherwise.
        Default `None`
        :type nburnin: `int` or `NoneType`, optional
        :param init_guess: An initial guess for the walkers. This should be
        a 2D `sequence` with shape (:attr:`~countdist2.SingleFitter.ndim`, |~|
        :param:`nwalkers`) unless continuing a previous sampler. Default `None`
        :type init_guess: 2D `sequence`[`float`] or `NoneType`, optional
        :param nwalkers: The number of walkers to use. Must match the second
        dimension of :param:`init_guess` unless continuing a previous sampler.
        Also should be at least :math:`2 \times \textrm{ndim}`. Default `None`
        :type nwalkers: `int` or `NoneType`, optional
        :param pool: For parallelization, see documentation in
        :class:`emcee.EnsembleSampler` for details. Default `None`
        :type pool: `object` with `map` method or `NoneType`, optional
        :param sampler: An MCMC sampler instance such as
        :class:`emcee.EnsembleSampler`, for continuing a sample. Otherwise,
        a new sampler will be created. Default `None`
        :type sampler: :class:`emcee.EnsembleSampler` or similar
        :return new_sampler: A new MCMC sampler that is either a continuation
        of the input or run for the first time
        :rtype new_sampler: :class:`emcee.EnsembleSampler` or `type` of
        :param:`sampler`
        """
        if self._data is None:
            raise AttributeError("Cannot perform fit without data")
        if self._f is None:
            raise AttributeError("Cannot perform fit without fitting function")
        if nburnin is None and self._nburnin is None:
            warnings.warn("Using burn-in of 0 is ill-advised")
            nburnin = 0
        elif nburnin is None:
            nburnin = self._nburnin
        self._nburnin = nburnin

        if sampler is not None:
            if sampler.chain.shape[-1] != self.ndim:
                warnings.warn("Invalid sampler given, creating new sampler"
                              " instead")
                new_sampler = None
            else:
                new_sampler = copy.deepcopy(sampler)
        else:
            new_sampler = None
        if new_sampler is not None:
            if nburnin >= nsteps + new_sampler.chain.shape[1]:
                raise ValueError("Cannot have burn-in same or more than total"
                                 " number of steps")
            new_sampler.run_mcmc(None, nsteps)
        else:
            if nburnin >= nsteps:
                raise ValueError("Cannot have burn-in same or more than number"
                                 " of steps")
            if init_guess is None:
                raise ValueError("Must give initial guess for new sampler")
            if len(init_guess) != self.ndim:
                raise ValueError("Wrong size ({}) for axis 0 of init_guess for"
                                 " fitting function with {} dimensions".format(
                                     len(init_guess), self.ndim))
            if not hasattr(init_guess[0], "__len__"):
                raise ValueError("init_guess must be 2D")
            if nwalkers is not None and nwalkers != len(init_guess[0]):
                raise ValueError("Wrong size ({}) for axis 1 of init_guess for"
                                 " {} walkers".format(
                                     len(init_guess[0]), nwalkers))
            if nwalkers is None:
                nwalkers = len(init_guess[0])
            if nwalkers < 2 * self.ndim:
                raise ValueError("MCMC requires at least 2 * ndim walkers")
            new_sampler = emcee.EnsembleSampler(
                nwalkers, self.ndim,
                self.lnprob if self._pr is not None else self.lnlike,
                pool=pool)
            new_sampler.run_mcmc(init_guess, nsteps)
        self._samples = new_sampler.chain
        samples = new_sampler.chain[:,nburnin:,:].reshape((-1,self.ndim))
        lnprobs = new_sampler.lnprobability[:,nburnin:].flatten()
        self.c = ChainConsumer()
        self.c.add_chain(samples, parameters=self.params, posterior=lnprobs)
        self._best_fit_params = np.median(samples, axis=0)
        self.c_walkers = ChainConsumer()
        self.c_walkers.add_chain(
            new_sampler.flatchain, parameters=self.params,
            posterior=new_sampler.flatlnprobability, walkers=new_sampler.k)
        self.c_walkers = self.c_walkers.divide_chain()
        return new_sampler


class AnalyticSingleFitter(SingleFitterBase):
    """
    Analytic fitter for single parameter fit
    """
    __doc__ = SingleFitterBase.__doc__ + __doc__
    def __init__(self, data=None, index_names=None, col_names=None,
                 fitter_name=None, param_names=None, *, rpo_size=1.0,
                 rlo_size=1.0, **kwargs):
        """

        This fitter does not need any fitting functions, and the default
        parameter name is '$c$'
        """
        super(AnalyticSingleFitter, self).__init__(
            data, index_names, col_names, fitter_name, param_names,
            rpo_size=rpo_size, rlo_size=rlo_size, **kwargs)
        self.params = param_names
        if (self.params is None or not hasattr(self.params, "__len__")
            or len(self.params) != 1):
            self.params = [r"$c$"]

    def _initialize_fit(self):
        super(AnalyticSingleFitter, self)._initialize_fit()
        self._c = None
        self._c_err = None

    def __repr__(self):
        return ("{self.name!s}(ndim=1, c={self._c!r},"
                " c_err={self._c_err!r})".format(self=self))

    def __setstate__(self, d):
        d["_best_fit_params"] = d.get("_best_fit_params", np.array([d["_c"]]))
        super(AnalyticSingleFitter, self).__setstate__(d)

    @property
    def best_fit(self):
        if not hasattr(self, "_best_fit_params"):
            if self._c is None:
                self._best_fit_params = None
            else:
                self._best_fit_params = np.array([self._c])
        return super(AnalyticSingleFitter, self).best_fit

    @property
    def best_fit_err(self):
        """The error on the best fit"""
        return np.array([self._c_err]) if self._c_err is not None else None

    @contextlib.contextmanager
    def use(self, c, c_err=None):
        """

        :param c: The constant fit to use
        :type c: `float`
        :param c_err: If given, the error on the fit to assume. Default `None`
        :type c_err: `float` or `NoneType`, optional
        """
        c_arr = np.array([c]) if c is not None else None
        if not hasattr(self, "_best_fit_params"):
            self._best_fit_params = (np.array([self._c]) if self._c is not None
                                     else None)
        _c, self._c = self._c, c
        _c_err, self._c_err = self._c_err, c_err
        _params, self._best_fit_params = self._best_fit_params, c_arr
        yield
        self._c = _c
        self._c_err = _c_err
        self._best_fit_params = _params

    def _get_const(self):
        """
        Internal function for calculating fit constant
        """
        if self._data is None:
            raise AttributeError("Cannot find constant fit without data")
        self._c = (
            (self._data.loc[:,self.col_names[0]] /
             self._data.loc[:,self.col_names[1]]).sum()
            / (1. / self._data.loc[:,self.col_names[1]]).sum())
        self._best_fit_params = np.array([self._c])

    def _get_err(self):
        """
        Internal function for calculating error on fit constant
        """
        if self._data is None:
            raise AttributeError("Cannot find constant fit error without data")
        self._c_err = (1. / math.sqrt(
            (1. / self._data.loc[:,self.col_names[1]]).sum()))

    def fit(self):
        self.fit.__func__.__doc__ = SingleFitterBase.fit.__doc__
        self._get_const()
        self._get_err()

    def f(self, rpo, rlo, c, *, index=None, **kwargs):
        """
        Constant function

        :param rpo: Observed perpendicular separation. Unused other than to
        determine if a scalar or sequence should be output
        :type rpo: `float` or `sequence`[`float`]
        :param rlo: Observed parallel separation. Unused other than to determine
        if a scalar or sequence should be output
        :type rlo: `float` or `sequence`[`float`]
        :param c: The constant fit
        :type c: `float`
        :key index: If given, output should be a :class:`pandas.Series` with
        this as the index. Default `None`
        :type index: :class:`pandas.Index`, :class:`pandas.MultiIndex`,
        `sequence`, or `NoneType`
        :return: The constant function evaluated
        :rtype: `float`, :class:`numpy.ndarray`[`float`], or
        :class:`pandas.Series`[`float`]
        """
        if np.atleast_1d(rpo).size != np.atleast_1d(rlo).size:
            raise ValueError("Perpendicular and parallel separations must have"
                             " same size")
        if index is not None and np.atleast_1d(rpo).size != len(index):
            raise ValueError("Index must have same size as separations if"
                             " given")
        if not hasattr(rpo, "__len__") or len(rpo) == 1:
            if index is not None:
                return pd.Series([c], index=index)
            return c
        if index is not None:
            return pd.Series([c] * len(index), index=index)
        return np.full(np.atleast_1d(rpo).size, c)

    def _f(self, rpo, rlo, theta, **kwargs):
        return self.f(rpo, rlo, *theta, **kwargs)

    def model_with_errors(self, rpo, rlo, index=None):
        if self._c_err is None:
            raise AttributeError("Cannot evaluate model with errors if error"
                                 " not available")
        if np.atleast_1d(rpo).size != np.atleast_1d(rlo).size:
            raise ValueError("Perpendicular and parallel separations must have"
                             " same size")
        if index is not None and len(index) != np.atleast_1d(rpo).size:
            raise ValueError("Index must have same size as separations if"
                             " given")
        if not hasattr(rpo, "__len__") or len(rpo) == 1:
            if index is not None:
                return pd.DataFrame(
                    [self._c - self._c_err, self._c, self._c + self._c_err],
                    index=index, columns=[0.16, 0.5, 0.84])
            return pd.Series(
                [self._c - self._c_err, self._c, self._c + self._c_err],
                index=[0.16, 0.5, 0.84])
        if index is None:
            index = np.arange(np.atleast_1d(rpo).size, dtype=int)
        return pd.DataFrame(
            {0.16: self._c - self._c_err, 0.5: self._c,
             0.84: self._c + self._c_err}, index=index)


class ProbFitter(object):
    """
    Fit a bivariate Gaussian of 2 variables to the data. This is not a
    generalized fitter, it assumes the means and variances follow specific
    functions that are not set by the user but hardcoded within.
    """
    _fitter_types = ["mean_x", "var_x", "mean_y", "var_y", "mean_r"]
    def __init__(self, statistics=None, fitter_name=None, cosmo=None, *,
                 rpo_scale=1.0, rlo_scale=1.0, rpo_size=1.0, rlo_size=1.0,
                 mean_x_const=True, var_x_const=True, mean_y_const=False,
                 var_y_const=False, mean_r_const=True, **kwargs):
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
        :param cosmo: An instance of a cosmology class. This is compatible for
        any subclass of :class:`astropy.cosmology.Cosmology`, but especially
        for :class:`CatalogUtils.FastCosmology`. Other cosmology classes that
        implement functions for distance and the derivative of distance can
        also be used. This must be given for calculating the means and
        variances of the true separations or their correlation. Default `None`
        :type cosmo: :class:`CatalogUtils.FastCosmology` or similar or
        `NoneType`, optional
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
        :key mean_x_const: If `True`, assume 'mean_x' is fit as a constant.
        This can be changed later. Default `True`
        :type mean_x_const: `bool`
        :key var_x_const: As :key:`mean_x_const`, but for 'var_x'. Default
        `True`
        :type var_x_const: `bool`
        :key mean_y_const: As :key:`mean_x_const`, but for 'mean_y'. Default
        `False`
        :type mean_y_const: `bool`
        :key var_y_const: As :key:`mean_y_const`, but for 'var_y'. Default
        `False`
        :type var_y_const: `bool`
        :key mean_r_const: As :key:`mean_x_const`, but for 'mean_r'. Default
        `True`
        :type mean_r_const: `bool`

        Additional keyword arguments
        ----------------------------
        There are several other keyword arguments that can be given for cases
        when any/all of the fitters are not constant. All of them default to
        `None`, but they can be set or changed after initialziation as well.
        They are:

        :keyword mean_x_func: Fitting function for 'mean_x'
        :type mean_x_func: `callable`
        :keyword var_x_func: As for :keyword:`mean_x_func`, bur for 'var_x'
        :type var_x_func: `callable`
        :keyword mean_y_func: As for :keyword:`mean_x_func`, bur for 'mean_y'
        :type mean_y_func: `callable`
        :keyword var_y_func: As for :keyword:`mean_x_func`, bur for 'var_y'
        :type var_y_func: `callable`
        :keyword mean_r_func: As for :keyword:`mean_x_func`, bur for 'mean_r'
        :type mean_r_func: `callable`
        :keyword mean_x_prior: Prior likelihood function for 'mean_x', or
        dictionary of extents for a flat prior
        :type mean_x_prior: `callable` or `dict`(`str`, `float`)
        :keyword var_x_prior: As for :keyword:`mean_x_prior`, bur for 'var_x'
        :type var_x_prior: `callable` or `dict`(`str`, `float`)
        :keyword mean_y_prior: As for :keyword:`mean_x_prior`, bur for 'mean_y'
        :type mean_y_prior: `callable` or `dict`(`str`, `float`)
        :keyword var_y_prior: As for :keyword:`mean_x_prior`, bur for 'var_y'
        :type var_y_prior: `callable` or `dict`(`str`, `float`)
        :keyword mean_r_prior: As for :keyword:`mean_x_prior`, bur for 'mean_r'
        :type mean_r_prior: `callable` or `dict`(`str`, `float`)
        :keyword mean_x_params: Fitting function parameter names for 'mean_x'
        :type mean_x_params: `list`[`str`]
        :keyword var_x_params: As for :keyword:`mean_x_params`, bur for 'var_x'
        :type var_x_params: `list`[`str`]
        :keyword mean_y_params: As for :keyword:`mean_x_params`, bur for
        'mean_y'
        :type mean_y_params: `list`[`str`]
        :keyword var_y_params: As for :keyword:`mean_x_params`, bur for 'var_y'
        :type var_y_params: `list`[`str`]
        :keyword mean_r_params: As for :keyword:`mean_x_params`, bur for
        'mean_r'
        :type mean_r_params: `list`[`str`]
        """
        self._init_switcher = dict(
            mean_x=self.initialize_mean_x,
            var_x=self.initialize_var_x,
            mean_y=self.initialize_mean_y,
            var_y=self.initialize_var_y,
            mean_r=self.initialize_mean_r)
        self._fitter_names = dict.fromkeys(self.__class__._fitter_types)
        self._get_name(fitter_name)
        self.logger = init_logger(self.name)
        self._fitters = dict.fromkeys(self.__class__._fitter_types, None)
        self.logger.debug("Set bin sizes and separation scaling")
        self.rpo_size = rpo_size
        self.rlo_size = rlo_size
        self.rpo_scale = rpo_scale
        self.rlo_scale = rlo_scale
        self._const = dict(
            mean_x=mean_x_const, var_x=var_x_const, mean_y=mean_y_const,
            var_y=var_y_const, mean_r=mean_r_const)
        self.cosmo = cosmo
        self.logger.debug("Add statistics")
        self.add_stats(statistics)
        self.logger.debug("__init__ complete")

    def __repr__(self):
        return ("{self.name!s}(\n"
                "mean_x={mean_x!r},\n"
                "var_x={var_x!r},\n"
                "mean_y={mean_y!r},\n"
                "var_y={var_y!r}\n"
                ")".format(self=self, **self._fitters))

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
        if "_const" not in d:
            d["_const"] = dict.fromkeys(self.__class__._fitter_types)
            d["_const"]["mean_x"] = d.pop("_mean_x_const", True)
            d["_const"]["var_x"] = d.pop("_var_x_const", True)
            d["_const"]["mean_y"] = d.pop("_mean_y_const", False)
            d["_const"]["var_y"] = d.pop("_var_y_const", False)
            d["_const"]["mean_r"] = d.pop("_mean_r_const", True)
        d.pop("_mean_y_extents", None)
        d.pop("_var_y_extents", None)
        d.pop("_mean_r_extents", None)
        # backwards compatability
        d["cosmo"] = d.pop("cosmo", None)
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
    def mean_x_const(self):
        """
        The constant-ness of the 'mean_x' fitter

        :getter: .. function:: mean_x_const() -> bool
        :setter: .. function:: mean_x_const(constness : `bool`,
        func=`None` : `callable`, prior=`None` : `callable`,
        params=`None` : `list`[`str`])
        :type: `bool`
        """
        if self._const is None or "mean_x" not in self._const:
            if self._fitters["mean_x"] is None:
                return True
            return isinstance(self._fitters["mean_x"], AnalyticSingleFitter)
        return self._const["mean_x"]

    @mean_x_const.setter
    def mean_x_const(self, constness, func=None, prior=None, params=None):
        if self._const is None:
            self._const = dict.fromkeys(self.__class__._fitter_types, False)
            self._const["mean_y"] = True
            self._const["var_y"] = True
        if ("mean_x" not in self._const or not
            (constness ^ self._const["mean_x"])):
            self._const["mean_x"] = constness
            if self._fitters["mean_x"] is not None:
                data = self._fitters["mean_x"]._data
                if data is not None:
                    data.columns = pd.MultiIndex.from_product(
                        [["mean_x"], data.columns])
            else:
                data = None
            self.initialize_mean_x(
                data, mean_x_func=func, mean_x_prior=prior,
                mean_x_params=params)

    @property
    def var_x_const(self):
        """
        The constant-ness of the 'var_x' fitter

        :getter: .. function:: var_x_const() -> bool
        :setter: .. function:: var_x_const(constness : `bool`,
        func=`None` : `callable`, prior=`None` : `callable`,
        params=`None` : `list`[`str`])
        :type: `bool`
        """
        if self._const is None or "var_x" not in self._const:
            if self._fitters["var_x"] is None:
                return True
            return isinstance(self._fitters["var_x"], AnalyticSingleFitter)
        return self._const["var_x"]

    @var_x_const.setter
    def var_x_const(self, constness, func=None, prior=None, params=None):
        if self._const is None:
            self._const = dict.fromkeys(self.__class__._fitter_types, False)
            self._const["mean_y"] = True
            self._const["var_y"] = True
        if ("var_x" not in self._const or not
            (constness ^ self._const["var_x"])):
            self._const["var_x"] = constness
            if self._fitters["var_x"] is not None:
                data = self._fitters["var_x"]._data
                if data is not None:
                    data.columns = pd.MultiIndex.from_product(
                        [["var_x"], data.columns])
            else:
                data = None
            self.initialize_var_x(
                data, var_x_func=func, var_x_prior=prior,
                var_x_params=params)

    @property
    def mean_y_const(self):
        """
        The constant-ness of the 'mean_y' fitter

        :getter: .. function:: mean_y_const() -> bool
        :setter: .. function:: mean_y_const(constness : `bool`,
        func=`None` : `callable`, prior=`None` : `callable`,
        params=`None` : `list`[`str`])
        :type: `bool`
        """
        if self._const is None or "mean_y" not in self._const:
            if self._fitters["mean_y"] is None:
                return True
            return isinstance(self._fitters["mean_y"], AnalyticSingleFitter)
        return self._const["mean_y"]

    @mean_y_const.setter
    def mean_y_const(self, constness, func=None, prior=None, params=None):
        if self._const is None:
            self._const = dict.fromkeys(self.__class__._fitter_types, False)
            self._const["mean_y"] = True
            self._const["var_y"] = True
        if ("mean_y" not in self._const or not
            (constness ^ self._const["mean_y"])):
            self._const["mean_y"] = constness
            if self._fitters["mean_y"] is not None:
                data = self._fitters["mean_y"]._data
                if data is not None:
                    data.columns = pd.MultiIndex.from_product(
                        [["mean_y"], data.columns])
            else:
                data = None
            self.initialize_mean_y(
                data, mean_y_func=func, mean_y_prior=prior,
                mean_y_params=params)

    @property
    def var_y_const(self):
        """
        The constant-ness of the 'var_y' fitter

        :getter: .. function:: var_y_const() -> bool
        :setter: .. function:: var_y_const(constness : `bool`,
        func=`None` : `callable`, prior=`None` : `callable`,
        params=`None` : `list`[`str`])
        :type: `bool`
        """
        if self._const is None or "var_y" not in self._const:
            if self._fitters["var_y"] is None:
                return True
            return isinstance(self._fitters["var_y"], AnalyticSingleFitter)
        return self._const["var_y"]

    @var_y_const.setter
    def var_y_const(self, constness, func=None, prior=None, params=None):
        if self._const is None:
            self._const = dict.fromkeys(self.__class__._fitter_types, False)
            self._const["mean_y"] = True
            self._const["var_y"] = True
        if ("var_y" not in self._const or not
            (constness ^ self._const["var_y"])):
            self._const["var_y"] = constness
            if self._fitters["var_y"] is not None:
                data = self._fitters["var_y"]._data
                if data is not None:
                    data.columns = pd.MultiIndex.from_product(
                        [["var_y"], data.columns])
            else:
                data = None
            self.initialize_var_y(
                data, var_y_func=func, var_y_prior=prior,
                var_y_params=params)

    @property
    def mean_r_const(self):
        """
        The constant-ness of the 'mean_r' fitter

        :getter: .. function:: mean_r_const() -> bool
        :setter: .. function:: mean_r_const(constness : `bool`,
        func=`None` : `callable`, prior=`None` : `callable`,
        params=`None` : `list`[`str`])
        :type: `bool`
        """
        if self._const is None or "mean_r" not in self._const:
            if self._fitters["mean_r"] is None:
                return True
            return isinstance(self._fitters["mean_r"], AnalyticSingleFitter)
        return self._const["mean_r"]

    @mean_r_const.setter
    def mean_r_const(self, constness, func=None, prior=None, params=None):
        if self._const is None:
            self._const = dict.fromkeys(self.__class__._fitter_types, False)
            self._const["mean_y"] = True
            self._const["var_y"] = True
        if ("mean_r" not in self._const or not
            (constness ^ self._const["mean_r"])):
            self._const["mean_r"] = constness
            if self._fitters["mean_r"] is not None:
                data = self._fitters["mean_r"]._data
                if data is not None:
                    data.columns = pd.MultiIndex.from_product(
                        [["mean_r"], data.columns])
            else:
                data = None
            self.initialize_mean_r(
                data, mean_r_func=func, mean_r_prior=prior,
                mean_r_params=params)

    @property
    def rpo_bin(self):
        fitters_wdata = [f for f in list(self._fitters.values()) if f is
                         not None and f._data is not None]
        if len(fitters_wdata) == 0:
            return None
        return fitters_wdata[0].data.index.get_level_values("RPO_BIN")

    @property
    def rlo_bin(self):
        fitters_wdata = [f for f in list(self._fitters.values()) if f is
                         not None and f._data is not None]
        if len(fitters_wdata) == 0:
            return None
        return fitters_wdata[0].data.index.get_level_values("RLO_BIN")

    @property
    def stats_vs_rpo_rlo(self):
        names = [name for name in self._fitters.keys() if
                 self._fitters[name] is not None and self._fitters[name]._data
                 is not None]
        data = [self._fitters[name].data_vs_rlo for name in names]
        return pd.concat(data, axis=1, keys=names)

    @property
    def stats_vs_rlo_rpo(self):
        names = [name for name in self._fitters.keys() if
                 self._fitters[name] is not None and self._fitters[name]._data
                 is not None]
        data = [self._fitters[name].data_vs_rpo for name in names]
        return pd.concat(data, axis=1, keys=names)

    @property
    def stats_table(self):
        concat_keys = [name for name, fitter in self._fitters.items() if
                       fitter is not None and fitter._data is not None]
        stats = pd.concat([fitter.data for fitter in self._fitters.values() if
                           fitter is not None and fitter._data is not None],
                          axis=1, keys=concat_keys)
        return stats_df_to_stats_table(stats)

    def _get_name(self, fitter_name):
        self.name = ("{}.{}".format(self.__class__.__name__, fitter_name) if
                     fitter_name is not None else self.__class__.__name__)
        for name in self._fitter_names.keys():
            self._fitter_names[name] = ("{}_{}".format(fitter_name, name) if
                fitter_name is not None else name)

    def initialize_mean_x(self, stats=None, **kwargs):
        """
        Initialize the fitter for 'mean_x'

        :param stats: Data frame of statistics. If 'mean_x' is in the columns,
        it assumes that the 'mean_x' column has sub-columns 'mean' and
        'variance'. If 'mean_x' is not in the columns or this is `None`,
        initializes a fitter without data. Default `None`
        :type stats: :class:`pandas.DataFrame` or `NoneType`, optional
        :keyword mean_x_func: Fitting function if not constant. Default `None`
        :type mean_x_func: `callable`
        :keyword mean_x_prior: Prior likelihood or dictionary of extents for
        flat prior. Default `None`
        :type mean_x_prior: `callable` or `dict`(`str`, `float`)
        :keyword mean_x_params: Parameter names for fitting function. Default
        `None`
        :type mean_x_params: `list`[`str`]
        """
        self.logger.debug("init_mean_x")
        data = None if stats is None else (stats["mean_x"].copy() if "mean_x"
                                           in stats.columns else None)
        idx_names = ["RPO_BIN", "RLO_BIN"] if data is not None else None
        col_names = ["mean", "variance"] if data is not None else None
        if self._const["mean_x"]:
            self._fitters["mean_x"] = AnalyticSingleFitter(
                data, idx_names, col_names, self._fitter_names["mean_x"],
                rpo_size=self.rpo_size, rlo_size=self.rlo_size)
        else:
            self._fitters["mean_x"] = SingleFitter(
                data, idx_names, col_names, kwargs.pop("mean_x_func", None),
                kwargs.pop("mean_x_prior", None),
                kwargs.pop("mean_x_params", None), self._fitter_names["mean_x"],
                rpo_size=self.rpo_size, rlo_size=self.rlo_size,
                func_kwargs=dict(rpo_scale=self.rpo_scale,
                                 rlo_scale=self.rlo_scale))
        self.logger.debug("init_mean_x: {}".format(self._fitters["mean_x"]))

    def initialize_var_x(self, stats=None, **kwargs):
        """
        Initialize the fitter for 'var_x'

        :param stats: Data frame of statistics. If 'var_x' is in the columns,
        it assumes that the 'var_x' column has sub-columns 'mean' and
        'variance'. If 'var_x' is not in the columns or this is `None`,
        initializes a fitter without data. Default `None`
        :type stats: :class:`pandas.DataFrame` or `NoneType`, optional
        :keyword var_x_func: Fitting function if not constant. Default `None`
        :type var_x_func: `callable`
        :keyword var_x_prior: Prior likelihood or dictionary of extents for
        flat prior. Default `None`
        :type var_x_prior: `callable` or `dict`(`str`, `float`)
        :keyword var_x_params: Parameter names for fitting function. Default
        `None`
        :type var_x_params: `list`[`str`]
        """
        self.logger.debug("init_var_x")
        data = None if stats is None else (stats["var_x"].copy() if "var_x"
                                           in stats.columns else None)
        idx_names = ["RPO_BIN", "RLO_BIN"] if data is not None else None
        col_names = ["mean", "variance"] if data is not None else None
        if self._const["var_x"]:
            self._fitters["var_x"] = AnalyticSingleFitter(
                data, idx_names, col_names, self._fitter_names["var_x"],
                rpo_size=self.rpo_size, rlo_size=self.rlo_size)
        else:
            self._fitters["var_x"] = SingleFitter(
                data, idx_names, col_names, kwargs.pop("var_x_func", None),
                kwargs.pop("var_x_prior", None),
                kwargs.pop("var_x_params", None), self._fitter_names["var_x"],
                rpo_size=self.rpo_size, rlo_size=self.rlo_size,
                func_kwargs=dict(rpo_scale=self.rpo_scale,
                                 rlo_scale=self.rlo_scale))
        self.logger.debug("init_var_x: {}".format(self._fitters["var_x"]))

    def initialize_mean_y(self, stats=None, **kwargs):
        """
        Initialize the fitter for 'mean_y'

        :param stats: Data frame of statistics. If 'mean_y' is in the columns,
        it assumes that the 'mean_y' column has sub-columns 'mean' and
        'variance'. If 'mean_y' is not in the columns or this is `None`,
        initializes a fitter without data. Default `None`
        :type stats: :class:`pandas.DataFrame` or `NoneType`, optional
        :keyword mean_y_func: Fitting function if not constant. Default `None`
        :type mean_y_func: `callable`
        :keyword mean_y_prior: Prior likelihood or dictionary of extents for
        flat prior. Default `None`
        :type mean_y_prior: `callable` or `dict`(`str`, `float`)
        :keyword mean_y_params: Parameter names for fitting function. Default
        `None`
        :type mean_y_params: `list`[`str`]
        """
        self.logger.debug("init_mean_y")
        data = None if stats is None else (stats["mean_y"].copy() if "mean_y"
                                           in stats.columns else None)
        idx_names = ["RPO_BIN", "RLO_BIN"] if data is not None else None
        col_names = ["mean", "variance"] if data is not None else None
        if self._const["mean_y"]:
            self._fitters["mean_y"] = AnalyticSingleFitter(
                data, idx_names, col_names, self._fitter_names["mean_y"],
                rpo_size=self.rpo_size, rlo_size=self.rlo_size)
        else:
            self._fitters["mean_y"] = SingleFitter(
                data, idx_names, col_names, kwargs.pop("mean_y_func", None),
                kwargs.pop("mean_y_prior", None),
                kwargs.pop("mean_y_params", None), self._fitter_names["mean_y"],
                rpo_size=self.rpo_size, rlo_size=self.rlo_size,
                func_kwargs=dict(rpo_scale=self.rpo_scale,
                                 rlo_scale=self.rlo_scale))
        self.logger.debug("init_mean_y: {}".format(self._fitters["mean_y"]))

    def initialize_var_y(self, stats=None, **kwargs):
        """
        Initialize the fitter for 'var_y'

        :param stats: Data frame of statistics. If 'var_y' is in the columns,
        it assumes that the 'var_y' column has sub-columns 'mean' and
        'variance'. If 'var_y' is not in the columns or this is `None`,
        initializes a fitter without data. Default `None`
        :type stats: :class:`pandas.DataFrame` or `NoneType`, optional
        :keyword var_y_func: Fitting function if not constant. Default `None`
        :type var_y_func: `callable`
        :keyword var_y_prior: Prior likelihood or dictionary of extents for
        flat prior. Default `None`
        :type var_y_prior: `callable` or `dict`(`str`, `float`)
        :keyword var_y_params: Parameter names for fitting function. Default
        `None`
        :type var_y_params: `list`[`str`]
        """
        self.logger.debug("init_var_y")
        data = None if stats is None else (stats["var_y"].copy() if "var_y"
                                           in stats.columns else None)
        idx_names = ["RPO_BIN", "RLO_BIN"] if data is not None else None
        col_names = ["mean", "variance"] if data is not None else None
        if self._const["var_y"]:
            self._fitters["var_y"] = AnalyticSingleFitter(
                data, idx_names, col_names, self._fitter_names["var_y"],
                rpo_size=self.rpo_size, rlo_size=self.rlo_size)
        else:
            self._fitters["var_y"] = SingleFitter(
                data, idx_names, col_names, kwargs.pop("var_y_func", None),
                kwargs.pop("var_y_prior", None),
                kwargs.pop("var_y_params", None), self._fitter_names["var_y"],
                rpo_size=self.rpo_size, rlo_size=self.rlo_size,
                func_kwargs=dict(rpo_scale=self.rpo_scale,
                                 rlo_scale=self.rlo_scale))
        self.logger.debug("init_var_y: {}".format(self._fitters["var_y"]))

    def initialize_mean_r(self, stats=None, **kwargs):
        """
        Initialize the fitter for 'mean_r'

        :param stats: Data frame of statistics. If 'mean_r' is in the columns,
        it assumes that the 'mean_r' column has sub-columns 'mean' and
        'variance'. If 'mean_r' is not in the columns or this is `None`,
        initializes a fitter without data. Default `None`
        :type stats: :class:`pandas.DataFrame` or `NoneType`, optional
        :keyword mean_r_func: Fitting function if not constant. Default `None`
        :type mean_r_func: `callable`
        :keyword mean_r_prior: Prior likelihood or dictionary of extents for
        flat prior. Default `None`
        :type mean_r_prior: `callable` or `dict`(`str`, `float`)
        :keyword mean_r_params: Parameter names for fitting function. Default
        `None`
        :type mean_r_params: `list`[`str`]
        """
        self.logger.debug("init_mean_r")
        data = None if stats is None else (stats["mean_r"].copy() if "mean_r"
                                           in stats.columns else None)
        idx_names = ["RPO_BIN", "RLO_BIN"] if data is not None else None
        col_names = ["mean", "variance"] if data is not None else None
        if self._const["mean_r"]:
            self._fitters["mean_r"] = AnalyticSingleFitter(
                data, idx_names, col_names, self._fitter_names["mean_r"],
                rpo_size=self.rpo_size, rlo_size=self.rlo_size)
        else:
            self._fitters["mean_r"] = SingleFitter(
                data, idx_names, col_names, kwargs.pop("mean_r_func", None),
                kwargs.pop("mean_r_prior", None),
                kwargs.pop("mean_r_params", None), self._fitter_names["mean_r"],
                rpo_size=self.rpo_size, rlo_size=self.rlo_size,
                func_kwargs=dict(rpo_scale=self.rpo_scale,
                                 rlo_scale=self.rlo_scale))
        self.logger.debug("init_mean_r: {}".format(self._fitters["mean_r"]))

    def add_stats(self, stats_in=None, **kwargs):
        """Add statistics for initializing the fitters, with the columns as
        described in the __init__. This does the initialization for the
        appropriate fitters, if they haven't already been initialized. The
        values for :param:`**kwargs` are explained in :func:`initialize_mean_x`,
        :func:`initialize_var_x`, :func:`initialize_mean_y`,
        :func:`initialize_var_y`, and :func:`initialize_mean_r`, as well as
        the initialization function for :class:`ProbFitter`

        :param stats_in: The statistics to use for initializing fitters.
        Default `None`
        :type stats_in: :class:`pandas.DataFrame` or :class:`astropy.Table`
        or `NoneType`, optional
        """
        if isinstance(stats_in, Table):
            stats = stats_table_to_stats_df(stats_in)
        elif isinstance(stats_in, (pd.Series, pd.DataFrame)):
            stats = stats_in.copy(deep=True)
        else:
            stats = copy.deepcopy(stats_in)
        if stats is not None:
            self.logger.debug("Drop NAN columns")
            stats_filt = stats.dropna(axis=1)
        else:
            stats_filt = None
        for fit_type in self.__class__._fitter_types:
            if (self._fitters[fit_type] is None or
                (self._fitters[fit_type]._data is None and
                 (stats_filt is not None or (not stats_filt.empty and fit_type
                                             in stats_filt)))):
                self._init_switcher[fit_type](stats_filt, **kwargs)
        self.logger.debug(self.__repr__())

    @contextlib.contextmanager
    def use(self, *, mean_x_params=None, mean_x_err_samples=None,
            var_x_params=None, var_x_err_samples=None, mean_y_params=None,
            mean_y_err_samples=None, var_y_params=None,
            var_y_err_samples=None, mean_r_params=None,
            mean_r_err_samples=None):
        """
        A context manager for using set parameters rather than requiring a
        fit for the values. Note that for all instances, '*_err_samples' is
        either an optional assumed error for an :class:`AnalyticSingleFitter`
        instance or assumed MCMC chain from which to find quantiles for a
        :class:`SingleFitter` instance. The '*_err_samples' are not required,
        but '*_params' are required for any fitter the context manager should
        handle, and the corresponding '*_params' MUST be given for any
        '*_err_samples' that are provided. This function is likely a major
        slow-down unless multiple fitters are being managed, so it should be
        avoided if nothing is passed or if only one or two fitters are managed.

        :kwarg mean_x_params: Either the set of parameters (for
        :class:`SingleFitter`) or the constant (for
        :class:`AnalyticSingleFitter`) to be used by assumption for the mean_x
        fitter. Default `None`
        :type mean_x_params: `float`, or array-like or `dict` of `float`
        :kwarg mean_x_err_samples: Either a 2D array of values to assume for
        the MCMC samples (for :class:`SingleFitter`) or the error on the
        constant (for :class:`AnalyticSingleFitter`) to use by assumption
        for the mean_x fitter. Default `None`
        :type mean_x_err_samples: `float` or 2D ndarray of `float`
        :kwarg var_x_params: As :kwarg:`mean_x_params`, but for the var_x
        fitter. Default `None`
        :type var_x_params: `float`, or array-like or `dict` of `float`
        :kwarg var_x_err_samples: As :kwarg:`mean_x_err_samples`, but for the
        var_x fitter. Default `None`
        :type var_x_err_samples: `float` or 2D ndarray of `float`
        :kwarg mean_y_params: As :kwarg:`mean_x_params`, but for the mean_y
        fitter. Default `None`
        :type mean_y_params: `float`, or array-like or `dict` of `float`
        :kwarg mean_y_err_samples: As :kwarg:`mean_x_err_samples`, but for the
        mean_y fitter. Default `None`
        :type mean_y_err_samples: `float` or 2D ndarray of `float`
        :kwarg var_y_params: As :kwarg:`mean_x_params`, but for the var_y
        fitter. Default `None`
        :type var_y_params: `float`, or array-like or `dict` of `float`
        :kwarg var_y_err_samples: As :kwarg:`mean_x_err_samples`, but for the
        var_y fitter. Default `None`
        :type var_y_err_samples: `float` or 2D ndarray of `float`
        :kwarg mean_r_params: As :kwarg:`mean_x_params`, but for the mean_r
        fitter. Default `None`
        :type mean_r_params: `float`, or array-like or `dict` of `float`
        :kwarg mean_r_err_samples: As :kwarg:`mean_x_err_samples`, but for the
        mean_r fitter. Default `None`
        :type mean_r_err_samples: `float` or 2D ndarray of `float`
        """
        if self._const is None:
            self.__setstate__(self.__getstate__())
        param_kwarg_switcher = dict(
            mean_x=mean_x_params, var_x=var_x_params,
            mean_y=mean_y_params, var_y=var_y_params,
            mean_r=mean_r_params)
        err_kwarg_switcher = dict(
            mean_x=mean_x_err_samples, var_x=var_x_err_samples,
            mean_y=mean_y_err_samples, var_y=var_y_err_samples,
            mean_r=mean_r_err_samples)
        saved_fitters = copy.deepcopy(self._fitters)
        for fit_type in self.__class__._fitter_types:
            params = param_kwarg_switcher[fit_type]
            if params is not None:
                err = err_kwarg_switcher[fit_type]
                fitter = self._fitters[fit_type]
                if fitter is None:
                    if self._const[fit_type]:
                        fitter = AnalyticSingleFitter()
                        fitter._c = params
                        fitter._c_err = err
                    else:
                        fitter = SingleFitter()
                        fitter._best_fit_params = params
                        fitter._samples = err
                elif isinstance(fitter, SingleFitter):
                    fitter._best_fit_params = params
                    fitter._samples = err
                else:
                    fitter._c = params
                    fitter._c_err = err
                self._fitters[fit_type] = fitter
        yield
        self._fitters = copy.deepcopy(saved_fitters)

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
        if isinstance(rpo, pd.Index):
            rpo = rpo.values
        if isinstance(rlo, pd.Index):
            rlo = rlo.values
        if isinstance(zbar, pd.Index):
            zbar = zbar.values
        return (_perp_mean_scale(rpo, rlo, zbar, sigma_z, self.cosmo)
                * self.mean_x.model(rpo, rlo, index=index) + rpo)

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
        if self.cosmo is None:
            raise AttributeError("Must have cosmology set to calculate variance"
                                 " of true perpendicular separation")
        if isinstance(rpo, pd.Index):
            rpo = rpo.values
        if isinstance(rlo, pd.Index):
            rlo = rlo.values
        if isinstance(zbar, pd.Index):
            zbar = zbar.values
        return (_perp_var_scale(rpo, rlo, zbar, sigma_z, self.cosmo)**2
                * self.var_x.model(rpo, rlo, index=index))

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
        if isinstance(rpo, pd.Index):
            rpo = rpo.values
        if isinstance(rlo, pd.Index):
            rlo = rlo.values
        if isinstance(zbar, pd.Index):
            zbar = zbar.values
        return (_par_mean_scale(rpo, rlo, zbar, sigma_z, self.cosmo)
                * self.mean_y.model(rpo, rlo, index=index) + rlo)

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
        if self.cosmo is None:
            raise AttributeError("Must have cosmology set to calculate variance"
                                 " of true parallel separation")
        if isinstance(rpo, pd.Index):
            rpo = rpo.values
        if isinstance(rlo, pd.Index):
            rlo = rlo.values
        if isinstance(zbar, pd.Index):
            zbar = zbar.values
        return (_par_var_scale(rpo, rlo, zbar, sigma_z, self.cosmo)**2
                * self.var_y.model(rpo, rlo, index=index))

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
        if self.cosmo is None:
            raise AttributeError("Must have cosmology set to calculate"
                                 " covariance matrix of true separations")
        return (np.sqrt(self.var_rpt(rpo, rlo, zbar, sigma_z) *
                        self.var_rlt(rpo, rlo, zbar, sigma_z)) *
                self.mean_r.model(rpo, rlo, index=index))

    def draw_rpt_rlt(self, rpo, rlo, zbar, sigma_z, rlt_mag=True, *,
                     index=None, rstate=None):
        """
        Draw true separations given the observed separations and photo-z error

        :param rpo: The observed perpendicular separation at which to calculate
        :type rpo: scalar or 1D array-like `float`
        :param rlo: The observed parallel separation at which to calculate
        :type rlo: scalar or 1D array-like `float`
        :param zbar: The average observed redshift at which to calculate
        :type zbar: scalar or 1D array-like `float`
        :param sigma_z: The redshift uncertainty
        :type sigma_z: scalar `float`
        :param rlt_mag: If `True`, return the absolute value of the true
        parallel separations. This is usually desired. Default `True`
        :type rlt_mag: `bool`, optional
        :key index: Optionally provide an index for returning a
        :class:`pandas.Series`. Default `None`
        :type index: :class:`pandas.Index` or :class:`pandas.MultiIndex`
        :key rstate: Set the random state to this before drawing. If `None`,
        don't set the random state. See documentation for
        :func:`numpy.random.get_state` and :func:`numpy.random.set_state`
        for details on what this object should be. Default `None`
        :type rsate: `tuple` or `NoneType`, optional
        :return rpt: The drawn true perpendicular separation(s)
        :rtype rpt: `float`, :class:`numpy.ndarray`[`float`], or
        :class:`pandas.Series`[`float`]
        :return rlt: The drawn true parallel separation(s)
        :rtype rlt: `float`, :class:`numpy.ndarray`[`float`], or
        :class:`pandas.Series`[`float`]
        """
        if self.cosmo is None:
            raise AttributeError("Must have cosmology set to draw true"
                                 " separations")
        if (np.atleast_1d(rpo).size != np.atleast_1d(rlo).size or
            np.atleast_1d(rpo).size != np.atleast_1d(zbar).size):
            raise ValueError("Observed separation components must all have same"
                             " size")
        if index is not None and len(index) != np.atleast_1d(rpo).size:
            raise ValueError("Index must have same size as separations if"
                             " given")
        if rstate is not None:
            np.random.set_state(rstate)
        afunc = np.absolute if rlt_mag else lambda x: x
        u, v = np.random.rand(2, np.atleast_1d(rpo).size)
        u = np.sqrt(-2.0 * np.log(u))
        v *= 2.0 * np.pi
        delta_perp = u * np.cos(v)
        delta_par = (
            u + np.sin(
                v + np.arcsin(
                    np.sqrt(
                        self.var_rpt(rpo, rlo, zbar, sigma_z)
                        * self.var_rlt(rpo, rlo, zbar, sigma_z))
                    * self._fitters["mean_r"].model(rpo, rlo))))
        del u, v
        rpt = (np.sqrt(self.var_rpt(rpo, rlo, zbar, sigma_z, index=index))
               * (delta_perp + self.mean_rpt(rpo, rlo, zbar, sigma_z)))
        del delta_perp
        rlt = afunc(
            np.sqrt(self.var_rlt(rpo, rlo, zbar, sigma_z, index=index))
            * (delta_par + self.mean_rlt(rpo, rlo, zbar, sigma_z)))
        del delta_par
        return rpt, rlt

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
        if self.cosmo is None:
            raise AttributeError("Must have cosmology set to calculate the"
                                 " determinant of the covariance matrix")
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
        if self.cosmo is None:
            raise AttributeError("Must have cosmology set to find inverse"
                                 " covariance matrix")
        if not hasattr(rpo, "__len__"):
            icov = np.empty((1, 3))
        else:
            icov = np.empty((len(rpo), 3))
        inv_det = 1. / self.det_cov_matrix(rpo, rlo, zbar, sigma_z)
        icov[:,0] = self.var_rlt(rpo, rlo, zbar, sigma_z) * inv_det
        icov[:,1] = self.var_rpt(rpo, rlo, zbar, sigma_z) * inv_det
        icov[:,2] = -(2 * self.mean_r.model(rpo, rlo, index=index)
                      * np.sqrt(self.var_rpt(rpo, rlo, zbar, sigma_z)
                                * self.var_rlt(rpo, rlo, zbar, sigma_z))
                      * inv_det)
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
        if self.cosmo is None:
            raise AttributeError("Must have cosmology set to calculate"
                                 " probability")
        icov = self.inverse_cov_matrix(rpo, rlo, zbar, sigma_z)
        det = self.det_cov_matrix(rpo, rlo, zbar, sigma_z)
        dvec = self.data_vector(rpt, rlt, rpo, rlo, zbar, sigma_z)
        p = (np.exp(-0.5
                    * (dvec[:,0]**2 * icov[:,0]
                       + dvec[:,1]**2 * icov[:,1]
                       + dvec[:,0] * dvec[:,1] * icov[:,2]))
             / (2. * np.pi * np.sqrt(det)))
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
