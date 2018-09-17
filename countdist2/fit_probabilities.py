from __future__ import print_function
from .utils import ndigits, init_logger, _initialize_cosmology
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from chainconsumer import ChainConsumer
import CatalogUtils
from scipy.stats import kstat, kstatvar, binned_statistic_2d
from scipy.optimize import minimize
import emcee
import numpy as np
import math
import pandas as pd
import pickle
import os, re, itertools

plt.rcParams["figure.facecolor"] = "white"


# Custom ValueError for too few groups (for catching during debugging)
class TooFewGroupsError(ValueError): pass


class CustomGroupBy(object):
    def __init__(self, keys, min_length=1):
        self.logger = init_logger(self.__class__.__name__)
        self.logger.debug("Get unique values and counts")
        self.key_values, self.keys_as_int, self.key_counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True)
        if min_length > 1:
            self.logger.debug("Filter based on number of elements")
            keep_keys = ((self.key_counts >= min_length))
            self.key_values = self.key_values[keep_keys]
            self.keys_as_int = self.keys_as_int[np.isin(self.keys_as_int,
                                                        np.where(keep_keys)[0])]
            self.key_counts = self.key_counts[keep_keys]
        self.n_keys = max(self.keys_as_int)
        self.set_indices()
        self.set_series_index(keys)

    def set_indices(self):
        self.indices = [[] for i in range(self.n_keys+1)]
        self.logger.debug("Set integer indices")
        for i, k in enumerate(self.keys_as_int):
            self.indices[k].append(i)
        self.indices = [np.array(elt) for elt in self.indices]
        self.logger.debug("Integer indices set")

    def set_series_index(self, keys):
        self.logger.debug("Set pandas index")
        if self.key_values.ndim > 1:
            self.logger.debug("Create MultiIndex")
            self.key_values = [tuple(key_val) for key_val in self.key_values]
            self.key_values = pd.MultiIndex.from_tuples(self.key_values,
                                                        names=keys.columns)
        else:
            self.logger.debug("Create Index")
            self.key_values = pd.Index(self.key_values, name=keys.name)
        self.logger.debug("Pandas index set")

    def apply(self, vector, function, *args, **kwargs):
        result = np.zeros(len(self.key_values))
        for k, idx in enumerate(self.indices):
            result[self.keys_as_int[k]] = function(
                vector.loc[self.key_values[idx]], *args, **kwargs)
        return pd.Series(result, index=self.key_values, name=function.__name__)


# Empty MultiIndex creation
def empty_multi_index(names):
    levels = [[] for i in range(len(names))]
    labels = [[] for i in range(len(names))]
    return pd.MultiIndex(levels=levels, labels=labels, names=names)


# Wrapper for kstat that also checks number of objects first
def kstat_check(arr, n, min_length):
    if len(arr) < min_length:
        return math.nan
    return kstat(arr, n)


# Wrapper for kstatvar that also checks number of objects first
def kstatvar_check(arr, n, min_length):
    if len(arr) < min_length:
        return math.nan
    return kstatvar(arr, n)


def get_array(x):
    """Get x as a 1D array"""
    try:
        arr_x = x.values
    except AttributeError:
        arr_x = np.atleast_1d(x).flatten()
    return arr_x


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


def mean_y(rpo, rlo, a, alpha, beta, s, index=None):
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
    :return f: The function evaluated at the separation(s)
    :rtype f: scalar or 1D :class:`pandas.Series` `float`
    """
    if not hasattr(rpo, "__len__") and not hasattr(rlo, "__len__"):
        f = a * (rpo**alpha) * (rlo**beta) * math.exp(-0.5 * rlo**2 / s)
    else:
        if index is not None:
            f = pd.Series(a * (rpo**alpha) * (rlo**beta) * \
                          np.exp(-0.5 * rlo**2 / s), index=index)
        else:
            f = a * (rpo**alpha) * (rlo**beta) * np.exp(-0.5 * rlo**2 / s)
    return f


def var_y(rpo, rlo, b, s1, s2, rho, index=None):
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
    :param index: The index to add to the Series, if needed. Ignored for scalar
    output. If `None`, the index is made from :param:`rpo` and :param:`rlo`,
    with names ['RPO_BIN', 'RLO_BIN']. Default `None`
    :type index: :class:`pandas.Index`, :class:`pandas.MultiIndex`, or `None`,
    optional
    :return f: The function evaluated at the separation(s)
    :rtype f: scalar or 1D :class:`pandas.Series` `float`
    """
    inv_weight = 1. / (s1 * s2 * (1 - rho**2))
    cinv = [x * inv_weight for x in [s2, s1, -2 * rho * math.sqrt(s1 * s2)]]
    if not hasattr(rpo, "__len__") and not hasattr(rlo, "__len__"):
        f = 1.0 - b * math.exp(-0.5 * (rpo**2 * cinv[0] + rlo**2 * cinv[1] + rpo
            * rlo * cinv[2]))
    else:
        if index is None:
            index = pd.MultiIndex.from_arrays([rpo, rlo], names=["RPO_BIN", 
                "RLO_BIN"])
        f = 1.0 - b * np.exp(-0.5 * pd.Series(rpo**2 * cinv[0] + rlo**2 *
            cinv[1] + rpo * rlo * cinv[2], index=index))
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


# This class was borrowed/stolen (with slight modification) from emcee...
class _FitFunctionWrapper(object):
    """
    A hack to make dynamically set functions with `args` and/or `kwargs`
    pickleable
    """
    def __init__(self, f, args=None, kwargs=None):
        self.f = f
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __call__(self, x, y, theta, extra_kwargs=None):
        extra_kwargs = {} if extra_kwargs is None else extra_kwargs
        return self.f(x, y, *theta, *self.args, **self.kwargs, **extra_kwargs)


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
        self.logger.debug("Set up separations and data")
        self.data = data
        self.index_names = index_names
        self.col_names = col_names
        self.data["ivar"] = 1. / self.data[col_names[1]]
        self.data["sigma"] = np.sqrt(self.data[col_names[1]])
        self.logger.debug("Set fitting function and prior")
        self.set_fit_func(func, param_names)
        self.set_prior_func(prior)
        self.logger.debug("Initialize sampler and best fit parameters as None")
        self.sampler = None
        self._best_fit_params = None
        self.logger.debug("__init__ complete")

    def __repr__(self):
        return "%s(%r, %r, %r, %r)" % (self.name, self.data, self.ndim,
                                       self.f, self.sampler)

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
        return-0.5 * self.data.loc[:,self.col_names[0]].sub(
            self.f(self.data.index.get_level_values(self.index_names[0]),
                   self.data.index.get_level_values(self.index_names[1]),
                   theta)).pow(2).div(
            self.data.loc[:,self.col_names[1]].apply(math.sqrt)).sum()

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
                 nthreads=1, reset=False, with_init=True):
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
        :param reset: If `True`, start a new sampler even if a previous one
        has been run. This makes the optional arguments (other than
        :param:`nthreads`) required. Default `False`
        :type reset: `bool`, optional
        :param with_init: If `True`, use
        :function:`SingleFitter.fit_minimize` as a first pass fit before
        running the MCMC. Ignored when continuing chains. Default `True`
        :type with_init: `bool`, optional
        """
        attrs_needed = [self.lnlike, self.lnprob]
        if not _check_pickleable(attrs_needed):
            self.logger.debug("Setting number of threads to 1 because not "
                    "pickleable")
            nthreads = 1
        if self.sampler is not None and not reset:
            self.logger.debug("Continuing chains")
            self.sampler.run_mcmc(None, nsteps)

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
            self.logger.debug("Initializing sampler")
            self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim,
                                                 self.lnprob, threads=nthreads)
            self.logger.debug("Running new MCMC")
            self.sampler.run_mcmc(p0, nsteps + nburnin)
        self.logger.debug("Set internal burn-in and current best fit "
                          "parameters")
        self._nburnin = nburnin
        samples = self.sampler.chain[:, nburnin, :].reshape((-1, self.ndim))
        self._best_fit_params = np.median(samples, axis=0)
        self.logger.debug("Set up Chain Consumer objects")
        c_walks = ChainConsumer()
        c_walks.add_chain(self.sampler.flatchain, parameters=self.params,
                          posterior=self.sampler.flatlnprobability,
                          walkers=nwalkers)
        self.c_walkers = c_walks.divide_chain()
        samples = self.sampler.chain[:, self._nburnin:, :].reshape((-1,
                                                                    self.ndim))
        lnprobs = self.sampler.lnprobability[:, self._nburnin:].flatten()
        self.c = ChainConsumer()
        self.c.add_chain(samples, parameters=self.params, posterior=lnprobs)
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
        m = self.f(rpo, rlo, self._best_fit_params, dict(index=index))
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
        :return m: The 16th, 50th, and 84th percentiles of the model at the
        points. The 0th axis is the percentiles, and the first axis is the
        flattened points, if not both scalar
        :rtype m: :class:`pandas.Series` or :class:`pandas.DataFrame` `float`
        """
        samples = self.sampler.chain[:, self._nburnin:, :].reshape((-1,
                                                                    self.ndim))
        meval = self.f(rpo, rlo, samples, dict(index=index))
        if hasattr(meval, "index"):
            m = meval.quantile(q=[0.16, 0.5, 0.84], axis="columns").T
        else:
            m = pd.Series(np.percentile(meval, [0.16, 0.5, 0.84]),
                                        index=pd.Index([0.16, 0.5, 0.84]))
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
                raise ValueError("with_fit=True, but no fit has been done")
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
                plt.fill_between(mod.index, mod.loc[:,0.16], mod.loc[:,0.84],
                                 color="C2", alpha=0.4)
                plt.plot(mod.index, mod.loc[:,0.5], "C2-")
            plt.legend([line], [axis_label.format(
                        round(bins, 2 - ndigits(bins)))], loc="best",
                       markerscale=0, frameon=False)
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
                    ax.fill_between(mod.loc[r].index, mod.loc[r,0.16],
                                    mod.loc[r,0.84], color="C2", alpha=0.4)
                    ax.plot(mod.loc[r].index, mod.loc[r,0.5], "C2-")
                ax.legend([line], [axis_label.format(
                            round(r, 2 - ndigits(r)))], loc="best",
                          markerscale=0, frameon=False)
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
        self.data = data
        self.index_names = index_names
        self.col_names = col_names
        self.logger.debug("__init__ complete")
        self._c = None
        self._c_err = None

    def __repr__(self):
        return "%s(%r, %r, %r, %r)" % (self.name, self.data, self._c,
                                       self._c_err)

    @property
    def best_fit(self):
        return np.array([self._c])

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
        if not hasattr(rpo, "__len__") and not hasattr(rlo, "__len__"):
            m = self._c
        else:
            if index is None:
                index = pd.MultiIndex.from_arrays([rpo, rlo], names=["RPO_BIN",
                    "RLO_BIN"])
            m = pd.Series(self._c, index=index)
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
        :return m: The (median - 1 sigma), median, and (median + 1 sigma)
        values of the model at the points. The 0th axis is the percentiles,
        and the first axis is the flattened points, if not both scalar
        :rtype m: :class:`pandas.Series` or :class:`pandas.DataFrame` `float`
        """
        if not hasattr(rpo, "__len__") and not hasattr(rlo, "__len__"):
            m = np.array([self._c - self._c_err, self._c, self._c + self._c_err])
        else:
            if index is None:
                if not hasattr(rpo, "__len__"):
                    index = pd.Index(rlo)
                elif not hasattr(rlo, "__len__"):
                    index = pd.Index(rpo)
                else:
                    index = pd.MultiIndex.from_arrays([rpo, rlo], 
                            names=["RPO_BIN", "RLO_BIN"])
            m = pd.DataFrame({0.16: self._c - self._c_err, 0.5: self._c, 0.84:
                self._c + self._c_err}, index=index)
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
                raise ValueError("with_fit=True, but no fit has been done")
            mod = self.model_with_errors(data.index.get_level_values(
                    self.index_names[0]), data.index.get_level_values(
                    self.index_names[1]), data.index)

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
                plt.fill_between(mod.index, mod.loc[:,0.16], mod.loc[:,0.84],
                                 color="C2", alpha=0.4)
                plt.plot(mod.index, mod.loc[:,0.5], "C2-")
            plt.legend([line], [axis_label.format(
                        round(bins, 2 - ndigits(bins)))], loc="best",
                       markerscale=0, frameon=False)
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
                    ax.fill_between(m.loc[r].index, m.loc[r,0.16],
                                    m.loc[r,0.84], color="C2", alpha=0.4)
                    ax.plot(m.loc[r].index, m.loc[r,0.5], "C2-")
                ax.legend([line], [axis_label.format(
                            round(r, 2 - ndigits(r)))], loc="best",
                          markerscale=0, frameon=False)
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
    
    def __init__(self, separations, perp_bin_size, par_bin_size,
                 perp_bin_scale, par_bin_scale, sigma_z, cosmo, min_counts=200,
                 fitter_name=None, **kwargs):
        """
        Initialize the fitter with separations
        :param separations: A DataFrame containing the true and observed
        separations
        :type separations: :class:`pandas.DataFrame`
        :param perp_bin_size: The bin size to use for the observed
        perpendicular separation, in Mpc
        :type perp_bin_size: `float`
        :param par_bin_size: The bin size to use for the observed parallel
        separation, in Mpc
        :type par_bin_size: `float`
        :param perp_bin_scale: The scale to use on observed perpendicular
        bins, in Mpc
        :type perp_bin_scale: `float`
        :param par_bin_scale: The scale to use on observed parallel bins, in Mpc
        :type par_bin_scale: `float`
        :param sigma_z: The fractional redshift error :math:`\frac{\sigma_z}{
        1 + z}` to assume for the data
        :type sigma_z: `float`
        :param cosmo: The path to a cosmology ini file, from which the
        cosmological functions can be set up
        :type cosmo: `str`
        :param min_counts: The minimum number of pairs needed in a bin to be
        included in the fit. Default 200
        :type min_counts: `int`, optional
        :param fitter_name: A name to use for this instance of the fitter. If
        `None`, the name will be set to 'ProbFitter'. Default `None`
        :type fitter_name: `str` or `None`, optional
        :keyword limits: A dictionary of limits to place on the separations
        being considered and/or the average observed LOS distance. All limits
        should be in Mpc. Default `None` (for no limits, all separations read)
        :type limits: `dict` or `None`
        """
        CatalogUtils.initialize(_initialize_cosmology(cosmo))
        self._stats_group_switcher = {
            "mean_x": dict(function=self._get_delta_column,
                           args=["PERP", self._perp_mean_scale]),
            "var_x": dict(function=self._get_delta_column,
                          args=["PERP", self._perp_var_scale]),
            "mean_y": dict(function=self._get_delta_column,
                           args=["PAR", self._par_mean_scale]),
            "var_y": dict(function=self._get_delta_column,
                          args=["PAR", self._par_var_scale]),
            "mean_r": dict(function=self._get_corr_column, args=[])
            }
        self._fitter_types = ["mean_x", "var_x", "mean_y", "var_y", "mean_r"]
        self._fitter_names = dict.fromkeys(self._fitter_types)
        self._get_name(fitter_name)
        self.logger = init_logger(self.name)
        self.sigma_z = sigma_z
        self.min_counts = min_counts
        self._fitters = dict.fromkeys(self._fitter_types)
        self.seps = separations.copy()
        self.logger.debug("Add extra data columns...")
        self._add_extra_columns(perp_bin_size, par_bin_size, perp_bin_scale,
                                par_bin_scale)
        self.logger.debug("Add extra data columns...[done]")
        self.logger.debug("__init__ complete")

    def __repr__(self):
        return "%s(%r, %r, %r)" % (self.name, self._fitters, self._has_r,
                                   self._has_r_stat)

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
    def stats(self):
        stats = pd.DataFrame(columns=[], index=[])
        for name, fitter in self._fitters.items():
            if fitter is not None:
                if stats.empty:
                    stats = fitter.data.add_prefix("{}_".format(name))
                else:
                    stats = stats.join(fitter.data.add_prefix(
                            "{}_".format(name)))
        return stats

    def _get_name(self, fitter_name):
        self.name = ("{}.{}".format(self.__class__.__name__, fitter_name) if
                     fitter_name is not None else self.__class__.__name__)
        for name in self._fitter_names.keys():
            self._fitter_names[name] = ("{}_{}".format(fitter_name, name) if
                fitter_name is not None else name)

    def _perp_mean_scale(self, rpo, rlo, zbar):
        return rpo

    def _perp_var_scale(self, rpo, rlo, zbar):
        return ((np.sqrt(0.5) * rpo * self.sigma_z * (1 + zbar) *
                 CatalogUtils.dr_dz(zbar)) / CatalogUtils.dist(zbar))

    def _par_mean_scale(self, rpo, rlo, zbar):
        return rlo

    def _par_var_scale(self, rpo, rlo, zbar):
        return (np.sqrt(2.0) * self.sigma_z * (1 + zbar) *
                CatalogUtils.dr_dz(zbar))

    def _add_bin_column(self, orig_col_name, bin_col_name, bin_size, scale):
        self.seps[bin_col_name] = np.floor(self.seps[orig_col_name] /
                                           bin_size).astype(int)

    def _add_zbar(self):
        self.seps["ZBAR"] = CatalogUtils.z_at_chi(self.seps["AVE_D_OBS"])

    def _add_extra_columns(self, perp_bin_size, par_bin_size, perp_bin_scale,
                           par_bin_scale):
        self.logger.debug("Add RPO_BIN")
        self._add_bin_column("R_PERP_O", "RPO_BIN", perp_bin_size,
                             perp_bin_scale)
        self.logger.debug("Add RLO_BIN")
        self._add_bin_column("R_PAR_O", "RLO_BIN", par_bin_size, par_bin_scale)
        self.logger.debug("Add ZBAR")
        self._add_zbar()

    def _get_delta_column(self, direction, scale_func):
        scale = 1. / scale_func(self.seps["R_PERP_O"], self.seps["R_PAR_O"],
                                self.seps["ZBAR"])
        tcol_name = "R_{}_T".format(direction)
        ocol_name = "R_{}_O".format(direction)
        delta = self.seps.loc[:,tcol_name].sub(self.seps.loc[:,ocol_name]).mul(
            scale)
        return pd.DataFrame.from_dict({"RPO_BIN": self.seps.loc[:,"RPO_BIN"],
                                       "RLO_BIN": self.seps.loc[:,"RLO_BIN"],
                                       "DELTA": delta})

    def _get_corr_column(self):
        rpo = self.seps["R_PERP_O"]
        rlo = self.seps["R_PAR_O"]
        zbar = self.seps["ZBAR"]
        self.logger.debug("Calculate mean R_PERP_T")
        x_mean = self.mean_rpt(rpo, rlo, zbar)
        self.logger.debug("Calculate mean R_PAR_T")
        y_mean = self.mean_rlt(rpo, rlo, zbar)
        self.logger.debug("Calculate var R_PERP_T")
        x_var = self.var_rpt(rpo, rlo, zbar)
        self.logger.debug("Calculate var R_PAR_T")
        y_var = self.var_rlt(rpo, rlo, zbar)
        self.logger.debug("Calculate correlations")
        r = corr_coeff(self.seps["R_PERP_T"], self.seps["R_PAR_T"],
                       x_mean, y_mean, x_var, y_var)
        return pd.DataFrame.from_dict({"RPO_BIN": self.data.loc[:,"RPO_BIN"],
                                       "RLO_BIN": self.data.loc[:,"RLO_BIN"],
                                       "r": r})

    def _get_stats(self, which):
        col = "r" if which == "mean_r" else "DELTA"
        n = 2 if "var" in which else 1
        col_getter_dict = self._stats_group_switcher[which]
        df_temp = col_getter_dict["function"](*col_getter_dict["args"])
        self.logger.info("Number of pairs = {}".format(len(df_temp)))
        stats = df_temp.groupby(["RPO_BIN", "RLO_BIN"]).filter(
            lambda x: len(x) >= self.min_counts).groupby(
            ["RPO_BIN", "RLO_BIN"]).agg([kstat, kstatvar], n).rename(
                columns={"kstat": "mean", "kstatvar": "variance"})
        self.logger.info("Number of unique groups = {}".format(len(stats)))
        if len(stats) <= 2:
            raise TooFewGroupsError(
                "Only {} bins with at least {} pairs. Consider decreasing "\
                    "the minimum number of pairs, increasing the bin size, "\
                    "or using a larger catalog".format(len(stats),
                                                       self.min_counts))
        if stats.columns.nlevels == 2:
            stats.columns = stats.columns.droplevel()
        return stats

    def initialize_mean_x(self):
        self._fitters["mean_x"] = AnalyticSingleFitter(
            self._get_stats("mean_x"), ["RPO_BIN", "RLO_BIN"],
            ["mean", "variance"], self._fitter_names["mean_x"]
            )

    def initialize_var_x(self):
        self._fitters["var_x"] = AnalyticSingleFitter(
            self._get_stats("var_x"), ["RPO_BIN", "RLO_BIN"],
            ["mean", "variance"], self._fitter_names["var_x"]
            )

    def initialize_mean_y(self):
        self._fitters["mean_y"] = SingleFitter(
            self._get_stats("mean_y"), ["RPO_BIN", "RLO_BIN"],
            ["mean", "variance"], mean_y, prior_mean_y,
            [r"$a$", r"$\alpha$", r"$\beta$", r"$s$"],
            self._fitter_names["mean_y"]
            )

    def initialize_var_y(self):
        self._fitters["var_y"] = SingleFitter(
            self._get_stats("var_y"), ["RPO_BIN", "RLO_BIN"],
            ["mean", "variance"], var_y, prior_var_y,
            [r"$b$", r"$s_1$", r"$s_2$", r"$\rho$"],
            self._fitter_names["var_y"]
            )

    def initialize_mean_r(self):
        if not np.all([self._fitters[f].best_fit is not None for f in
                       self._fitter_types[:-1]]):
            self._fitters["mean_r"] = None
        else:
            self._fitters["mean_r"] = AnalyticSingleFitter(
                self._get_stats("mean_r"), ["RPO_BIN", "RLO_BIN"],
                ["mean", "variance"], self._fitter_names["mean_r"]
                )

    def initialize_fitters(self):
        self.logger.debug("Initalize mean_x")
        self.initialize_mean_x()
        self.logger.debug("Initialize var_x")
        self.initialize_var_x()
        self.logger.debug("Initialize mean_y")
        self.initialize_mean_y()
        self.logger.debug("Initialize var_y")
        self.initialize_var_y()
        self.logger.debug("Initalize mean_r")
        self.initialize_mean_r()

    def mean_rpt(self, rpo, rlo, zbar):
        """
        Get the mean of the true perpendicular separation. All inputs must be
        scalar or 1D array-like with the same size

        :param rpo: The observed perpendicular separation at which to calculate
        :type rpo: scalar or 1D array-like `float`
        :param rlo: The observed parallel separation at which to calculate
        :type rlo: scalar or 1D array-like `float`
        :param zbar: The average observed redshift at which to calculate
        :type zbar: scalar or 1D array-like `float`
        :return: The mean with the scaling undone
        :rtype: scalar or 1D :class:`numpy.ndarray` `float`
        """
        if not hasattr(rpo, "index"):
            index = pd.Index(rpo.size)
        else:
            index = rpo.index
        return (self._perp_mean_scale(rpo, rlo, zbar) * self.mean_x.model(
            rpo, rlo, index=index) + rpo)

    def var_rpt(self, rpo, rlo, zbar):
        """
        Get the variance of the true perpendicular separation. All inputs
        must be scalar or 1D array-like with the same size

        :param rpo: The observed perpendicular separation at which to calculate
        :type rpo: scalar or 1D array-like `float`
        :param rlo: The observed parallel separation at which to calculate
        :type rlo: scalar or 1D array-like `float`
        :param zbar: The average observed redshift at which to calculate
        :type zbar: scalar or 1D array-like `float`
        :return: The variance with the scaling undone
        :rtype: scalar or 1D :class:`numpy.ndarray` `float`
        """
        if not hasattr(rpo, "index"):
            index = pd.Index(rpo.size)
        else:
            index = rpo.index
        return (self._perp_var_scale(rpo, rlo, zbar) ** 2 * self.var_x.model(
            rpo, rlo, index=index))

    def mean_rlt(self, rpo, rlo, zbar):
        """
        Get the mean of the true parallel separation. All inputs must be
        scalar or 1D array-like with the same size

        :param rpo: The observed perpendicular separation at which to calculate
        :type rpo: scalar or 1D array-like `float`
        :param rlo: The observed parallel separation at which to calculate
        :type rlo: scalar or 1D array-like `float`
        :param zbar: The average observed redshift at which to calculate
        :type zbar: scalar or 1D array-like `float`
        :return: The mean with the scaling undone
        :rtype: scalar or 1D :class:`numpy.ndarray` `float`
        """
        if not hasattr(rpo, "index"):
            index = pd.Index(rpo.size)
        else:
            index = rpo.index
        return (self._par_mean_scale(rpo, rlo, zbar) * self.mean_y.model(
            rpo, rlo, index=index) + rlo)

    def var_rlt(self, rpo, rlo, zbar):
        """
        Get the variance of the true parallel separation. All inputs
        must be scalar or 1D array-like with the same size

        :param rpo: The observed perpendicular separation at which to calculate
        :type rpo: scalar or 1D array-like `float`
        :param rlo: The observed parallel separation at which to calculate
        :type rlo: scalar or 1D array-like `float`
        :param zbar: The average observed redshift at which to calculate
        :type zbar: scalar or 1D array-like `float`
        :return: The variance with the scaling undone
        :rtype: scalar or 1D :class:`numpy.ndarray` `float`
        """
        if not hasattr(rpo, "index"):
            index = pd.Index(rpo.size)
        else:
            index = rpo.index
        return (self._par_var_scale(rpo, rlo, zbar) ** 2 * self.var_y.model(
            rpo, rlo, index=index))
