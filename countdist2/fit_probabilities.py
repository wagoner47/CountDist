from __future__ import print_function
from .utils import ndigits, init_logger, _initialize_cosmology
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from chainconsumer import ChainConsumer
import CatalogUtils
from scipy.stats import kstat, kstatvar, describe
from scipy.optimize import minimize
from scipy.linalg import block_diag
import emcee
import numpy as np
import math
import pandas as pd
from astropy.table import Table
import pickle
import os, re, itertools, warnings

plt.rcParams["figure.facecolor"] = "white"
plt.style.use("seaborn-colorblind")

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


def mean_y(rpo, rlo, a, alpha, beta, s, **kwargs):
    """
    The mean of :math:`\frac{\Delta R_\parallel}{R_\parallel^O}`. This
    function looks like :math:`a x^\alpha y^\beta \exp[-y^2 / 2 s]`, where x
    is the perpendicular separation and y is the parallel separation. If
    :param:`rpo` or :param:`rlo` has a length, it will be assumed that they are
    the indices from a Series/DataFrame, and a Series will be returned.

    :param rpo: The observed perpendicular separation
    :type rpo: scalar or array-like `float`
    :param rlo: The observed parallel separation
    :type rlo: scalar or array-like `float`
    :param a: The amplitude of the function
    :type a: scalar `float`
    :param alpha: The power on the observed perpendicular separation
    :type alpha: scalar `float`
    :param beta: The power law on the observed parallel separation
    :type beta: scalar `float`
    :param s: The scale of the exponential term
    :type s: scalar `float`
    :param index: The index to add to the Series, if needed. Ignored for scalar
    output. If `None`, the index is made from :param:`rpo` and :param:`rlo`,
    with names ['RPO_BIN', 'RLO_BIN']. Default `None`
    :type index: :class:`pandas.Index`, :class:`pandas.MultiIndex`, or `None`,
    optional
    :key index: Optional index to use for returning a Series rather than an
    array for array-like :param:`rpo` and/or :param:`rlo`. Default `None`
    :type index: :class:`pandas.Index` or :class:`pandas.MultiIndex`, optional
    :return f: The function evaluated at the separation(s)
    :rtype f: scalar or 1D :class:`numpy.ndarray` or :class:`pandas.Series`
    `float`
    """
    if not hasattr(rpo, "__len__") and not hasattr(rlo, "__len__"):
        f = a * (rpo**alpha) * (rlo**beta) * math.exp(-0.5 * rlo**2 / s)
    else:
        index = kwargs.pop("index", None)
        if index is not None:
            f = pd.Series(a * (rpo**alpha) * (rlo**beta) * \
                          np.exp(-0.5 * rlo**2 / s), index=index)
        else:
            if hasattr(rpo, "values"):
                rpo = rpo.values
            if hasattr(rlo, "values"):
                rlo = rlo.values
            f = a * (rpo**alpha) * (rlo**beta) * np.exp(-0.5 * rlo**2 / s)
    return f


def var_y(rpo, rlo, b, s1, s2, rho, **kwargs):
    """
    The variance of :math:`\frac{\Delta R_\parallel}{\sqrt{2} \chi'(\bar{z})
    \sigma_z(\bar{z})}`. This function looks like :math:`1 - b \exp[-0.5 \vec{
    r}^T C^{-1} \vec{r}]`, where :math:`\vec{r}` is a vector of the observed
    perpendicular and parallel separations, and C looks like a covariance
    matrix if :param:`s1` and :param:`s2` are variances and :param:`rho` is
    the correlation coefficient. If :param:`rpo` or :param:`rlo` has a length,
    both will assumed to be indices from a Series/DataFrame, and a Series will
    be returned.

    :param rpo: The observed perpendicular separations
    :type rpo: scalar or array-like `float`
    :param rlo: The observed parallel separations
    :type rlo: scalar or array-like `float`
    :param b: The amplitude on the exponential term
    :type b: scalar `float`
    :param s1: The width of the exponential associated with the observed
    perpendicular separation
    :type s1: scalar `float`
    :param s2: The width of the exponential associated with the observed
    parallel separation
    :type s2: scalar `float`
    :param rho: The mixing of the perpendicular and parallel contriubtions to
    the exponential
    :type rho: scalar `float`
    :key index: Optional index to use for returning a Series rather than an
    array for array-like :param:`rpo` and/or :param:`rlo`. Default `None`
    :type index: :class:`pandas.Index` or :class:`pandas.MultiIndex`, optional
    :return f: The function evaluated at the separation(s)
    :rtype f: scalar or 1D :class:`numpy.ndarray` or :class:`pandas.Series`
    `float`
    """
    inv_weight = 1. / (s1 * s2 * (1 - rho**2))
    cinv = [x * inv_weight for x in [s2, s1, -2 * rho * math.sqrt(s1 * s2)]]
    if not hasattr(rpo, "__len__") and not hasattr(rlo, "__len__"):
        f = 1.0 - b * math.exp(-0.5 * (rpo**2 * cinv[0] + rlo**2 * cinv[1] + rpo
            * rlo * cinv[2]))
    else:
        index = kwargs.pop("index", None)
        if index is not None:
            f = 1.0 - b * np.exp(
                -0.5 * pd.Series(rpo**2 * cinv[0] + rlo**2 *
                                 cinv[1] + rpo * rlo * cinv[2], index=index))
        else:
            if hasattr(rpo, "values"):
                rpo = rpo.values
            if hasattr(rlo, "values"):
                rlo = rlo.values
            f = 1.0 - b * np.exp(-0.5 * (rpo**2 * cinv[0] + rlo**2 * cinv[1] +
                                         rpo * rlo * cinv[2]))
    return f


def prior_mean_y(theta):
    """
    The prior on the mean of the scaled difference in parallel separations.
    This is flat in the allowed regions (we only require that the scale of
    the exponential be positive).

    :param theta: The parameters at which to calculate the prior likelihood.
    This should be in the order [a, alpha, beta, s]
    :type theta: 1D array-like `float`
    :return: The value of the prior: 0 where allowed and negative infinity
    otherwise
    :rtype: `float`
    """
    a, alpha, beta, s = theta
    if s > 0.0:
        return 0.0
    return -math.inf


def prior_var_y(theta):
    """
    The prior on the variance of the scaled difference in parallel
    separations. This is flat in the allowed regions (we require that both
    scales are positive and that the correlation term be between -1 and 1)

    :param theta: The parameters at which to calculate the prior likelihood.
    This should be in the order [b, s1, s2, rho]
    :type theta: 1D array-like `float`
    :return: The value of the prior: 0 where allowed and negative infinity
    otherwise
    :rtype: `float`
    """
    b, s1, s2, rho = theta
    if s1 > 0.0 and s2 > 0.0 and -1.0 < rho < 1.0:
        return 0.0
    return -math.inf

def _check_pickleable(attrs):
    for attr in attrs:
        try:
            pickle.dumps(attr)
        except (pickle.PicklingError, TypeError):
            return False
    return True

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

def _add_bin_column(seps, orig_col_name, bin_col_name, bin_size, scale):
    seps[bin_col_name] = ((np.floor(seps.loc[:,orig_col_name] /
                                    bin_size) + 0.5) *
                          (bin_size / scale))

def _add_zbar(seps):
    seps["ZBAR"] = CatalogUtils.z_at_chi(seps["AVE_D_OBS"])

def _add_delta_column(seps, direction, scale_func, dcol_name, sigma_z):
    tcol_name = "R_{}_T".format(direction)
    ocol_name = "R_{}_O".format(direction)
    scale = 1. / scale_func(seps.loc[:,"R_PERP_O"], seps.loc[:,"R_PAR_O"],
                            seps.loc[:,"ZBAR"], sigma_z)
    seps[dcol_name] = seps[tcol_name].sub(seps[ocol_name]).mul(scale)

def add_extra_columns(seps, perp_bin_size, par_bin_size, perp_bin_scale,
                      par_bin_scale, sigma_z):
    """This function adds some of the extra data columns to the input DataFrame
    that are needed for grouping and generating the statistics. It does not
    add the column for the correlation, that is handled by a separate function.

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
    :param perp_bin_scale: The scaling to use for the 'R_PERP_O' bins, in the
    same units as 'R_PERP_O'
    :type perp_bin_scale: `float`
    :param par_bin_scale: The scaling to use for the 'R_PAR_O' bins, in the
    same units as 'R_PAR_O'
    :type par_bin_scale: `float`
    :param sigma_z: The redshift error assumed for the separations
    :type sigma_z: `float`
    """
    logger = glogger.getChild(__name__)
    logger.debug("Add column RPO_BIN")
    _add_bin_column(seps, "R_PERP_O", "RPO_BIN", perp_bin_size,
                    perp_bin_scale)
    logger.debug("Add column RLO_BIN")
    _add_bin_column(seps, "R_PAR_O", "RLO_BIN", par_bin_size, par_bin_scale)
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

    If the extra columns needed for the calculations have not been added yet
    via :function:`add_extra_columns`, an AttributeError will be raised. Please
    make sure to call that function first to have all needed columns

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
    """Get the sample mean and variance on sample mean of the correlation.
    The returned DataFrame will have columns 'mean' and 'variance', and indices
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
    stats = pd.concat([grouped["r"].agg([kstat, kstatvar], 2).rename(
                columns={"kstat": "mean", "kstatvar": "variance"})],
                      keys=["mean_r"], axis=1)
    return stats


# This class was borrowed/stolen (with slight modification) from emcee...
class _FitFunctionWrapper(object):
    """
    A hack to make dynamically set functions with `args` and/or `kwargs`
    pickleable
    """
    def __init__(self, f, args=None, kwargs=None):
        self.f = f
        self.func_name = f.__name__
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __call__(self, x, y, theta, **kwargs):
        try:
            return self.f(x, y, *theta, *self.args, **self.kwargs, **kwargs)
        except TypeError:
            print("TypeError when calling ", self.func_name, flush=True)
            print("\ttheta = ", theta, flush=True)
            print("\targs = ", self.args, flush=True)
            print("\tfixed kwargs = ", self.kwargs, flush=True)
            print("\tadditional kwargs = ", kwargs, flush=True)
            raise


class SingleFitter(object):
    """
    Fit a single function to a single set of data.
    """

    def __init__(self, data, index_names, col_names, func=None, prior=None,
                 param_names=None, fitter_name=None):
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
        :param prior: The prior to use for fitting the data. Must be set
        before doing a fit with the prior likelihood. Use
        :function:`SingleFitter.set_prior_func` to set or change this later.
        Default `None`
        :type prior: `function` or `None`, optional
        :param param_names: The names of the parameters for the fitting
        function. Must be set with function. Default `None`
        :type param_names: 1D array-like `str` or `None`, optional
        :param fitter_name: A name for the SingleFitter instance,
        for representation. If `None`, the name will be set to
        'SingleFitter'. Default `None`
        :type fitter_name: `str` or `None`, optional
        """
        self._get_name(fitter_name)
        self.logger = init_logger(self.name)
        self.logger.debug("Set up data and variance")
        self.logger.debug("Data columns: \n{}".format(data.columns))
        self.data = data
        self.index_names = index_names
        self.col_names = col_names
        self.logger.debug("Set fitting function and prior")
        self.set_fit_func(func, param_names)
        self.set_prior_func(prior)
        self.logger.debug("Initialize sampler and best fit parameters as None")
        self._best_fit_params = None
        self._samples = None
        self.logger.debug("__init__ complete")

    def __repr__(self):
        return "%s(ndim=%r, sampler=%r)" % (self.name, self.ndim, self.sampler)

    @property
    def nburnin(self):
        return self._nburnin

    @nburnin.setter
    def nburnin(self, value):
        self._nburnin = value
        if self.sampler is not None:
            samples = self.sampler.chain[:, value:, :].reshape((-1, self.ndim))
            self._best_fit_params = np.median(samples, axis=0)

    @property
    def best_fit(self):
        return self._best_fit_params

    @property
    def samples(self):
        return self._samples

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
                raise ValueError("Parameter names must be given when fitting "
                                 "function is specified")
            self.logger.debug("Setting fitting function and parameters")
            self.params = param_names
            self.ndim = len(param_names)
            self.f = _FitFunctionWrapper(func, args, kwargs)
        self.logger.debug("done")

    def set_prior_func(self, prior):
        """Set a (new) prior likelihood function

        :param prior: The prior probability function
        :type prior: `function` or `None`
        """
        if prior is None:
            self.logger.debug("Setting prior to None")
        else:
            self.logger.debug("Setting up prior function")
        self.pr = prior
        self.logger.debug("done")

    def lnlike(self, theta):
        if self.f is None:
            raise AttributeError("No fitting function set for likelihood")
        fev = self.f(self.data.index.get_level_values(self.index_names[0]),
                     self.data.index.get_level_values(self.index_names[1]),
                     theta)
        diff2 = self.data.loc[:,self.col_names[0]].sub(fev).pow(2)
        diffdiv = diff2.div(self.data.loc[:,self.col_names[1]])
        return -0.5 * diffdiv.sum()

    def lnprob(self, theta):
        if self.pr is None:
            raise AttributeError("No prior function set for lnprob")
        lp = self.pr(theta)
        if not np.isfinite(lp):
            return -np.inf
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

    def fit_mcmc(self, nsteps, nburnin, init_guess=None, nwalkers=None,
                 nthreads=1, sampler=None, with_init=True):
        """Fit the data using an MCMC (via `emcee`). If an MCMC has already
        been run and :param:`reset` is set to `False`, this will merely
        continue running the chains for another :param:`nsteps`, and all
        other optional parameters will be ignored. If no run has been done or
        :param:`reset` is set to `True`, this will start a new sampler,
        and possibly initialize using :function:`SingleFitter.fit_minimize`
        if :param:`with_init` is set to `True`. In all instances,
        :param:`nburnin` is required for setting the internal burn-in length
        for parameter estimation.

        :param nsteps: The number of MCMC steps to take with each walker
        :type nsteps: `int`
        :param nburnin: The length of the burn-in on the MCMC
        :type nburnin: `int`
        :param init_guess: The inital guess for the parameters. This is
        ignored if continuing chains. Otherwise, it is either used to
        initialize the walkers or passed to the minimizer for initialization,
        depending on :param:`with_init`. Default `None`
        :type init_guess: 1D array-like `float` or `None`, optional
        :param nwalkers: The number of MCMC walkers to use in `emcee`. This
        is ignored when continuing chains, but required otherwise. Default
        `None`
        :type nwalkers: `int` or `None`, optional
        :param nthreads: The number of threads to use for multiprocessing
        with `emcee`. If set to 1, multiprocessing will not be used. This
        parameter is ignored when continuing chains. It may be overridden if
        pickling will fail. Default 1
        :type nthreads: `int`, optional
        :param sampler: If given, this is a sampler that has already been
        initialized and possibly run. This will override the options for
        :param:`nwalkers` and :param:`nthreads` if they were given. Also,
        if a chain has already been run with the sampler, the options
        :param:`init_guess` and :param:`with_init` will be overridden.
        When not passed, a new sampler is created. The resulting sampler
        (either newly created or modified) is also returned. If number of
        dimensions for sampler is wrong, the sampler will be overwritten.
        Default `None`
        :type sampler: :class:`emcee.EnsembleSampler`, optional
        :param with_init: If `True`, use
        :function:`SingleFitter.fit_minimize` as a first pass fit before
        running the MCMC. Ignored when continuing chains. Default `True`
        :type with_init: `bool`, optional
        :return sampler: The EnsembleSampler for the MCMC run. This will either
        be one that was initialized here and run, or one that was updated
        (unless no update was needed for number of steps plus burn-in
        requested).
        :rtype sampler: :class:`emcee.EnsembleSampler`
        """
        attrs_needed = [self.lnlike, self.lnprob]
        if not _check_pickleable(attrs_needed):
            self.logger.debug("Setting number of threads to 1 because not "
                    "pickleable")
            nthreads = 1
        self.logger.debug("Set internal burn-in")
        self._nburnin = nburnin
        if sampler is None or sampler.chain.shape[-1] != self.ndim:
            self.logger.debug("Initializing sampler")
            sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.lnprob,
                                            threads=nthreads)
        if sampler.chain.shape[1] > 0:
            self.logger.debug("Continuing chains")
            nsteps_needed = (nsteps + nburnin) - sampler.chain.shape[1]
            if nsteps_needed <= 0:
                self.logger.debug("No further steps needed")
            else:
                sampler.run_mcmc(None, nsteps_needed)
        
        else:
            if init_guess is None or nwalkers is None:
                raise ValueError("Must give initial guess and number of "
                                 "walkers for new MCMC sampler")
            if len(init_guess) != self.ndim:
                raise ValueError("Wrong number of parameters in initial guess")
            if with_init:
                self.logger.debug("Initializing with minimizer")
                res = self.fit_minimize(init_guess)
                init_guess = res.x
            self.logger.debug("Setting up initial ball")
            p0 = [init_guess + 10 ** -4 * np.random.randn(self.ndim) for i in
                  range(nwalkers)]
            self.logger.debug("Running new MCMC")
            sampler.run_mcmc(p0, nsteps + nburnin)
        self.logger.debug("Set current best fit parameters")
        samples = sampler.chain[:, nburnin, :].reshape((-1, self.ndim))
        self.logger.debug("Set up Chain Consumer objects")
        c_walks = ChainConsumer()
        c_walks.add_chain(sampler.flatchain, parameters=self.params,
                          posterior=sampler.flatlnprobability,
                          walkers=nwalkers)
        self.c_walkers = c_walks.divide_chain()
        self._samples = sampler.chain[:, self._nburnin:, :].reshape(
            (-1, self.ndim))
        self._best_fit_params = np.median(self._samples, axis=0)
        lnprobs = sampler.lnprobability[:, self._nburnin:].flatten()
        self.c = ChainConsumer()
        self.c.add_chain(self._samples, parameters=self.params,
                         posterior=lnprobs)
        self.logger.debug("done")
        return sampler

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
        samples = self._samples
        meval = dict(zip(range(len(samples)), map(
                    lambda params: self.f(rpo, rlo, params, index=index),
                    samples)))
        if not hasattr(rpo, "__len__") and not hasattr(rlo, "__len__"):
            meval = pd.Series(meval)
            m = meval.quantile([0.16, 0.5, 0.84])
        else:
            meval = pd.DataFrame.from_dict(meval)
            m = meval.quantile(q=[0.16, 0.5, 0.84], axis="columns").T
        return m

    def plot(self, rpo_label, rlo_label, ylabel, bins, perp_bin_size, 
             par_bin_size, exp, is_rpo=False, logx=False, logy=False, 
             filename=None, figsize=None, display=False, text_size=22, 
             with_fit=False):
        """Plot the data (and optionally the best fit to the data) at a
        number of individual perpendicular or parallel separations.

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
        :param perp_bin_size: The size of the bins in the perpendicular
        direction, in the same units as the perpendicular separations
        :type perp_bin_size: `float`
        :param par_bin_size: The size of the bins in the parallel direction, in
        the same units as the parallel separations
        :type par_bin_size: `float`
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
        :return fig: The figure that has been created
        :rtype fig: :class:`matplotlib.figure.Figure`
        """
        if figsize is None:
            figsize = plt.rcParams["figure.figsize"]
        plt.rcParams["font.size"] = text_size

        if is_rpo:
            # x-axis will be RLO_BIN, bins are drawn from RPO_BIN
            r_bin_size = perp_bin_size
            x_bin_size = par_bin_size
            rlabel = rpo_label
            xlabel = rlo_label
            data = self.data.copy().loc[bins]
        else:
            # x-axis will be RPO_BIN, bins are drawn from RLO_BIN
            r_bin_size = par_bin_size
            x_bin_size = perp_bin_size
            rlabel = rlo_label
            xlabel = rpo_label
            data = self.data.swaplevel(0, 1, axis=0).loc[bins]

        axis_label = r"${} = {{}} \pm {}$".format(
            re.sub("\$", "", re.sub(r"([{}])", r"\1\1", rlabel)),
            round(0.5 * r_bin_size, 2 - ndigits(0.5 * r_bin_size)))
        
        if with_fit:
            if self._best_fit_params is None:
                warnings.warn("Ignoring with_fit option when no fit is done")
                with_fit = False
            else:
                mod = self.model_with_errors(
                    data.index.get_level_values(self.index_names[0]),
                    data.index.get_level_values(self.index_names[1]),
                    index=data.index)
        
        fig = plt.figure(figsize=figsize)
        if not hasattr(bins, "__len__"):
            # Case: single bin, don't need any subplots
            plt.xlabel(xlabel)
            plt.ylabel(ylabel, labelpad=(2 * plt.rcParams["font.size"]))
            if logx:
                plt.xscale("log")
            if logy:
                plt.yscale("log")
            plt.axhline(exp, c="k")
            line = plt.errorbar(data.index, data.loc[:,self.col_names[0]],
                                yerr=data.loc[:,self.col_names[1]].apply(
                    math.sqrt), fmt="C0o", alpha=0.6)[0]
            if with_fit:
                fit_fill = plt.fill_between(mod.index, mod.loc[:,0.16],
                                            mod.loc[:,0.84], color="C1",
                                            alpha=0.4)
                fit_line, = plt.plot(mod.index, mod.loc[:,0.5], "C1-")
            label_leg = plt.legend([line], [axis_label.format(
                        round(bins, 2 - ndigits(bins)))],
                                   loc=2, markerscale=0, frameon=False)
            if with_fit and with_combined_fit:
                plt.legend(
                    [(fit_fill, fit_line), (cfit_fill, cfit_line)],
                    [r"Best fit", r"Combined best fit"],
                    loc=1)
            plt.gca().add_artist(label_leg)
            plt.tight_layout()
        else:
            bins = np.reshape(bins, -1)
            grid = gridspec.GridSpec(bins.size, 1, hspace=0, left=0.2)
            full_ax = fig.add_subplot(grid[:])
            for loc in ["top", "bottom", "left", "right"]:
                full_ax.spines[loc].set_color("none")
            full_ax.tick_params(labelcolor="w", top="off", bottom="off", 
                    left="off", right="off", which="both")
            full_ax.set_ylabel(ylabel, labelpad=(2 * plt.rcParams["font.size"]))
            full_ax.set_xlabel(xlabel)
            for i, r in enumerate(bins):
                ax = fig.add_subplot(grid[i], sharex=full_ax)
                if logx:
                    ax.set_xscale("log")
                if logy:
                    ax.set_yscale("log")
                ax.axhline(exp, c="k")
                line = ax.errorbar(data.loc[r].index,
                                   data.loc[r,self.col_names[0]],
                                   yerr=data.loc[r,self.col_names[1]].apply(
                        math.sqrt), fmt="C0o", alpha=0.6)[0]
                if with_fit:
                    fit_fill = ax.fill_between(mod.loc[r].index,
                                               mod.loc[r,0.16],
                                               mod.loc[r,0.84],
                                               color="C1", alpha=0.4)
                    fit_line, = ax.plot(mod.loc[r].index, mod.loc[r,0.5], "C1-")
                label_leg = ax.legend([line], [axis_label.format(
                            round(r, 2 - ndigits(r)))], loc=2,
                                      markerscale=0, frameon=False)
                if with_fit and with_combined_fit:
                    ax.legend(
                        [(fit_fill, fit_line), (cfit_fill, cfit_line)],
                        [r"Best fit", r"Combined best fit"],
                        loc=1)
                ax.add_artist(label_leg)
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


class AnalyticSingleFitter(object):
    """
    Another single data model fitter, but this time for a model than can be
    fit analytically. This case is only valid (so far) for a constant
    function with uncorrelated errors (i.e. diagonal covariance matrix)
    """

    def __init__(self, data, index_names, col_names, fitter_name=None):
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
        """
        self._get_name(fitter_name)
        self.logger = init_logger(self.name)
        self.logger.debug("Setting up data and variance")
        self.logger.debug("Data columns: \n{}".format(data.columns))
        self.data = data
        self.index_names = index_names
        self.col_names = col_names
        self.logger.debug("__init__ complete")
        self._c = None
        self._c_err = None

    def __repr__(self):
        return "%s(c=%r, c_err=%r)" % (self.name, self._c, self._c_err)

    @property
    def best_fit(self):
        return np.array([self._c]) if self._c is not None else None

    @property
    def best_fit_err(self):
        return np.array([self._c_err]) if self._c_err is not None else None

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

    def plot(self, rpo_label, rlo_label, ylabel, bins, perp_bin_size,
             par_bin_size, exp, is_rpo=False, logx=False, logy=False, 
             filename=None, figsize=None, display=False, text_size=22, 
             with_fit=False):
        """Plot the data (and optionally the best fit to the data) at a
        number of individual perpendicular or parallel separations.

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
        :param perp_bin_size: The size of the bins in the perpendicular
        direction, in the same units as the perpendicular separations
        :type perp_bin_size: `float`
        :param par_bin_size: The size of the bins in the parallel direction, in
        the same units as the parallel separations
        :type par_bin_size: `float`
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
        :return fig: The figure that has been created
        :rtype fig: :class:`matplotlib.figure.Figure`
        """
        if figsize is None:
            figsize = plt.rcParams["figure.figsize"]
        plt.rcParams["font.size"] = text_size

        rpo = self.data.index.get_level_values(self.index_names[0])
        rlo = self.data.index.get_level_values(self.index_names[1])

        if is_rpo:
            # x-axis will be RLO_BIN, bins are drawn from RPO_BIN
            r_bin_size = perp_bin_size
            x_bin_size = par_bin_size
            rlabel = rpo_label
            xlabel = rlo_label
            data = self.data.copy().loc[bins]
        else:
            # x-axis will be RPO_BIN, bins are drawn from RLO_BIN
            r_bin_size = par_bin_size
            x_bin_size = perp_bin_size
            rlabel = rlo_label
            xlabel = rpo_label
            data = self.data.swaplevel(0, 1, axis=0).loc[bins]

        axis_label = r"${} = {{}} \pm {}$".format(
            re.sub("\$", "", re.sub(r"([{}])", r"\1\1", rlabel)),
            np.around(0.5 * r_bin_size, 2 - ndigits(0.5 * r_bin_size)))

        if with_fit:
            if (self._c is None or self._c_err is None):
                warnings.warn("Ignoring with_fit because no fit has been done")
                with_fit = False
            mod = self.model_with_errors(
                data.index.get_level_values(self.index_names[0]),
                data.index.get_level_values(self.index_names[1]),
                index=data.index)

        fig = plt.figure(figsize=figsize)
        if not hasattr(bins, "__len__"):
            # In this case, we don't need any subplots
            plt.xlabel(xlabel)
            plt.ylabel(ylabel, labelpad=(2 * plt.rcParams["font.size"]))
            if logx:
                plt.xscale("log")
            if logy:
                plt.yscale("log")
            plt.axhline(exp, c="k")
            line = plt.errorbar(data.index, data.loc[:,self.col_names[0]],
                                yerr=data.loc[:,self.col_names[1]].apply(
                    math.sqrt), fmt="C0o", alpha=0.6)[0]
            if with_fit:
                fit_fill = plt.fill_between(mod.index, mod.loc[:,0.16],
                                            mod.loc[:,0.84], color="C1",
                                            alpha=0.4)
                fit_line, = plt.plot(mod.index, mod.loc[:,0.5], "C1-")
            label_leg = plt.legend([line], [axis_label.format(
                        round(bins, 2 - ndigits(bins)))], loc=2,
                                   markerscale=0, frameon=False)
            if with_fit and with_combined_fit:
                plt.legend(
                    [(fit_fill, fit_line), (cfit_fill, cfit_line)],
                    [r"Best fit", r"Combined best fit"],
                    loc=1)
            plt.gca().add_artist(label_leg)
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
            for i, r in enumerate(bins):
                ax = fig.add_subplot(grid[i], sharex=full_ax)
                if logx:
                    ax.set_xscale("log")
                if logy:
                    ax.set_yscale("log")
                ax.axhline(exp, c="k")
                line = ax.errorbar(data.loc[r].index,
                                   data.loc[r,self.col_names[0]],
                                   yerr=data.loc[r,self.col_names[1]].apply(
                        math.sqrt), fmt="C0o", alpha=0.6)[0]
                if with_fit:
                    fit_fill = ax.fill_between(mod.loc[r].index,
                                               mod.loc[r,0.16],
                                               mod.loc[r,0.84], color="C1",
                                               alpha=0.4)
                    fit_line, = ax.plot(mod.loc[r].index, mod.loc[r,0.5], "C1-")
                label_leg = ax.legend([line], [axis_label.format(
                            round(r, 2 - ndigits(r)))], loc=2,
                                      markerscale=0, frameon=False)
                if with_fit and with_combined_fit:
                    ax.legend(
                        [(fit_fill, fit_line), (cfit_fill, cfit_line)],
                        [r"Best fit", r"Combined best fit"],
                        loc=1)
                ax.add_artist(label_leg)
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
    def __init__(self, statistics=None, fitter_name=None):
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
        self.logger.debug("Add statistics")
        self.add_stats(statistics)
        self.logger.debug("__init__ complete")

    def __repr__(self):
        return "{name}(mean_x={f[0]}, var_x={f[1]}, mean_y={f[2]}, var_y={f[3]}, mean_r={f[4]})".format(name=self.name, f=list(self._fitters.values()))

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
    def stats(self):
        names = [name for name in self._fitters.keys() if
                 self._fitters[name] is not None]
        data = [self._fitters[name].data for name in names]
        return pd.concat(data, axis=1, keys=names)

    @property
    def stats_table(self):
        stats = pd.DataFrame(columns=[], index=[])
        for name, fitter in self._fitters.items():
            if fitter is not None:
                if stats.empty:
                    stats = fitter.data.add_prefix("{}_".format(name))
                else:
                    stats = stats.join(fitter.data.add_prefix(
                            "{}_".format(name)))
        stats.reset_index(inplace=True)
        return Table.from_pandas(stats)

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
            ["mean", "variance"], self._fitter_names["mean_x"])
        self.logger.debug("init_mean_x: {}".format(self._fitters["mean_x"]))

    def initialize_var_x(self, stats):
        self.logger.debug("init_var_x")
        self._fitters["var_x"] = AnalyticSingleFitter(
            stats["var_x"].copy(), ["RPO_BIN", "RLO_BIN"],
            ["mean", "variance"], self._fitter_names["var_x"]
            )
        self.logger.debug("init_var_x: {}".format(self._fitters["var_x"]))

    def initialize_mean_y(self, stats):
        self.logger.debug("init_mean_y")
        self._fitters["mean_y"] = SingleFitter(
            stats["mean_y"].copy(), ["RPO_BIN", "RLO_BIN"],
            ["mean", "variance"], mean_y, prior_mean_y,
            [r"$a$", r"$\alpha$", r"$\beta$", r"$s$"],
            self._fitter_names["mean_y"]
            )
        self.logger.debug("init_mean_y: {}".format(self._fitters["mean_y"]))

    def initialize_var_y(self, stats):
        self.logger.debug("init_var_y")
        self._fitters["var_y"] = SingleFitter(
            stats["var_y"].copy(), ["RPO_BIN", "RLO_BIN"],
            ["mean", "variance"], var_y, prior_var_y,
            [r"$b$", r"$s_1$", r"$s_2$", r"$\rho$"],
            self._fitter_names["var_y"]
            )
        self.logger.debug("init_var_y: {}".format(self._fitters["var_y"]))

    def initialize_mean_r(self, stats):
        self.logger.debug("init_mean_r")
        self._fitters["mean_r"] = AnalyticSingleFitter(
            stats["mean_r"].copy(), ["RPO_BIN", "RLO_BIN"],
            ["mean", "variance"], self._fitter_names["mean_r"])
        self.logger.debug("init_mean_r: {}".format(self._fitters["mean_r"]))

    def add_stats(self, stats):
        """Add statistics for initializing the fitters, with the columns as
        described in the __init__. This does the initialization for the
        appropriate fitters, if they haven't already been initialized

        :param stats: The statistics to use for initializing fitters
        :type stats: :class:`pandas.DataFrame`
        """
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

    def mean_rpt(self, rpo, rlo, zbar, sigma_z):
        """
        Get the mean of the true perpendicular separation. All inputs must be
        scalar or 1D array-like with the same size, except
        :param:`sigma_z`, which can only be scalar

        :param rpo: The observed perpendicular separation at which to calculate
        :type rpo: scalar or 1D array-like `float`
        :param rlo: The observed parallel separation at which to calculate
        :type rlo: scalar or 1D array-like `float`
        :param zbar: The average observed redshift at which to calculate
        :type zbar: scalar or 1D array-like `float`
        :param sigma_z: The redshift uncertainty
        :type sigma_z: scalar `float`
        :return: The mean with the scaling undone
        :rtype: scalar or 1D :class:`numpy.ndarray` `float`
        """
        if hasattr(rpo, "__len__"):
            if not hasattr(rpo, "index"):
                index = pd.Index(len(rpo))
            else:
                index = rpo.index
        else:
            index = None
        return (_perp_mean_scale(rpo, rlo, zbar, sigma_z) * self.mean_x.model(
                rpo, rlo, index=index) + rpo)

    def var_rpt(self, rpo, rlo, zbar, sigma_z):
        """
        Get the variance of the true perpendicular separation. All inputs
        must be scalar or 1D array-like with the same size, except
        :param:`sigma_z`, which can only be scalar

        :param rpo: The observed perpendicular separation at which to calculate
        :type rpo: scalar or 1D array-like `float`
        :param rlo: The observed parallel separation at which to calculate
        :type rlo: scalar or 1D array-like `float`
        :param zbar: The average observed redshift at which to calculate
        :type zbar: scalar or 1D array-like `float`
        :param sigma_z: The redshift uncertainty
        :type sigma_z: scalar `float`
        :return: The variance with the scaling undone
        :rtype: scalar or 1D :class:`numpy.ndarray` `float`
        """
        if hasattr(rpo, "__len__"):
            if not hasattr(rpo, "index"):
                index = pd.Index(len(rpo))
            else:
                index = rpo.index
        else:
            index = None
        return (_perp_var_scale(rpo, rlo, zbar, sigma_z)**2 * self.var_x.model(
                rpo, rlo, index=index))

    def mean_rlt(self, rpo, rlo, zbar, sigma_z):
        """
        Get the mean of the true parallel separation. All inputs must be
        scalar or 1D array-like with the same size, except
        :param:`sigma_z`, which can only be scalar

        :param rpo: The observed perpendicular separation at which to calculate
        :type rpo: scalar or 1D array-like `float`
        :param rlo: The observed parallel separation at which to calculate
        :type rlo: scalar or 1D array-like `float`
        :param zbar: The average observed redshift at which to calculate
        :type zbar: scalar or 1D array-like `float`
        :param sigma_z: The redshift uncertainty
        :type sigma_z: scalar `float`
        :return: The mean with the scaling undone
        :rtype: scalar or 1D :class:`numpy.ndarray` `float`
        """
        if hasattr(rpo, "__len__"):
            if not hasattr(rpo, "index"):
                index = pd.Index(len(rpo))
            else:
                index = rpo.index
        else:
            index = None
        return (_par_mean_scale(rpo, rlo, zbar, sigma_z) * self.mean_y.model(
                rpo, rlo, index=index) + rlo)

    def var_rlt(self, rpo, rlo, zbar, sigma_z):
        """
        Get the variance of the true parallel separation. All inputs
        must be scalar or 1D array-like with the same size, except
        :param:`sigma_z`, which can only be scalar

        :param rpo: The observed perpendicular separation at which to calculate
        :type rpo: scalar or 1D array-like `float`
        :param rlo: The observed parallel separation at which to calculate
        :type rlo: scalar or 1D array-like `float`
        :param zbar: The average observed redshift at which to calculate
        :type zbar: scalar or 1D array-like `float`
        :param sigma_z: The redshift uncertainty
        :type sigma_z: scalar `float`
        :return: The variance with the scaling undone
        :rtype: scalar or 1D :class:`numpy.ndarray` `float`
        """
        if hasattr(rpo, "__len__"):
            if not hasattr(rpo, "index"):
                index = pd.Index(len(rpo))
            else:
                index = rpo.index
        else:
            index = None
        return (_par_var_scale(rpo, rlo, zbar, sigma_z)**2 * self.var_y.model(
                rpo, rlo, index=index))

    def cov_rpt_rlt(self, rpo, rlo, zbar, sigma_z):
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
        :return: The covariance between the perpendicular and parallel
        directions
        :rtype: scalar or 1D :class:`numpy.ndarray` or :class:`pandas.Series`
        `float`
        """
        if hasattr(rpo, "__len__"):
            if not hasattr(rpo, "index"):
                index = pd.Index(len(rpo))
            else:
                index = rpo.index
        else:
            index = None
        return (np.sqrt(self.var_rpt(rpo, rlo, zbar, sigma_z) *
                        self.var_rlt(rpo, rlo, zbar, sigma_z)) *
                self.mean_r.model(rpo, rlo, index=index))

    def det_cov_matrix(self, rpo, rlo, zbar, sigma_z):
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
        :return d: The determinant of the covariance matrix between the
        perpendicular and parallel directions
        :rtype d: scalar or 1D :class:`numpy.ndarray` or :class:`pandas.Series`
        `float`
        """
        scalar = False
        as_series = False
        if not hasattr(rpo, "__len__"):
            index = None
            scalar = True
        else:
            if hasattr(rpo, "index"):
                index = rpo.index
                as_series = True
            else:
                index = pd.Index(np.arange(len(rpo)))
        d = (self.var_rpt(rpo, rlo, zbar, sigma_z) *
             self.var_rlt(rpo, rlo, zbar, sigma_z) *
             (1. - self.mean_r.model(rpo, rlo, index=index)**2))
        if not scalar and not as_series:
            return d.values
        return d

    def inverse_cov_matrix(self, rpo, rlo, zbar, sigma_z):
        """
        Get the inverse covariance matrix of the true perpendicular and
        parallel separations. All inputs must be scalar or 1D array-like with
        the same size, except :param:`sigma_z`, which can only be scalar. The
        output will be a nx2x2 matrix where n is the length of the inputs, and
        is 1 for scalars.

        :param rpo: The observed perpendicular separation at which to calculate
        :type rpo: scalar or 1D array-like `float`
        :param rlo: The observed parallel separation at which to calculate
        :type rlo: scalar or 1D array-like `float`
        :param zbar: The average observed redshift at which to calculate
        :type zbar: scalar or 1D array-like `float`
        :param sigma_z: The redshift uncertainty
        :type sigma_z: scalar `float`
        :return icov: The inverse covariance matrix between the perpendicular
        and parallel directions
        :rtype icov: :class:`numpy.ndarray` `float` with shape nx2x2, for input
        of length n (n = 1 for scalars)
        """
        if not hasattr(rpo, "__len__"):
            icov = np.empty((1, 2, 2))
            index = None
        else:
            icov = np.empty((len(rpo), 2, 2))
            if hasattr(rpo, "index"):
                index = rpo.index
            else:
                index = pd.Index(np.arange(len(rpo)))
        inv_det = 1. / self.det_cov_matrix(rpo, rlo, zbar, sigma_z)
        icov[:,0,0] = self.var_rlt(rpo, rlo, zbar, sigmaz) * inv_det
        icov[:,0,1] = -(self.mean_r.model(rpo, rlo, index=index) *
                        np.sqrt(self.var_rpt(rpo, rlo, zbar, sigma_z) *
                                self.var_rlt(rpo, rlo, zbar, sigma_z)) *
                        inv_det)
        icov[:,1,0] = icov[:,0,1]
        icov[:,1,1] = self.var_rpt(rpo, rlo, zbar, sigmaz) * inv_det
        return icov


    def data_vector(self, rpt, rlt, rpo, rlo, zbar, sigma_z):
        """
        Get the "data vector" :math:`\vec{x} - \vec{\mu_x}` for the 2D data
        of the true parallel and perpendicular separations. The inputs must all
        be scalar or 1D array-like with the same size, except :param:`sigma_z`,
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
        return dvec
        

    def prob(self, rpt, rlt, rpo, rlo, zbar, sigma_z):
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
        :return p: The probability of the true separations given the set of
        observed separations and average observed redshifts as well as the
        redshift uncertainty
        :rtype p: scalar or 1D :class:`numpy.ndarray` or :class:`pandas.Series`
        `float`
        """
        icov = self.inverse_cov_matrix(rpo, rlo, zbar, sigma_z)
        det = self.det_cov_matrix(rpo, rlo, zbar, sigma_z)
        dvec = self.data_vector(rpt, rlt, rpo, rlo, zbar, sigma_z)
        if hasattr(rpo, "index"):
            index = rpo.index
        else:
            index = None
        if not hasattr(rpo, "__len__"):
            temp_exp = math.exp
            temp_pi = math.pi
            temp_sqrt = math.sqrt
        else:
            temp_exp = np.exp
            temp_pi = np.pi
            temp_sqrt = np.sqrt
        p = (temp_exp(-0.5 * double_einsum(dvec, icov)) /
             (2. * temp_pi * temp_sqrt(det)))
        if index is not None:
            p = pd.Series(p, index=index)
        return p


def double_einsum(a, b):
    """
    This is a helper function for doing
    :math:`\vec{a_i} \cdot \mathbf{b_i} \cdot \vec{a_i}` over all
    elements i in a and b.

    :param a: An array of vectors with shape (N,M)
    :type a: :class:`numpy.ndarray`
    :param b: An array of matrices with shape (N,M,M)
    :type b: class:`numpy.ndarray`
    :return: An array containing the chained dot product a.b.a for each
    element along the zeroth axis, with shape (N,)
    :rtype: :class:`numpy.ndarray`
    """
    return np.einsum("ik,ik->i", np.einsum("ij,ijk->ik", a, b), a)
