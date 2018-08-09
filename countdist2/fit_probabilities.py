from __future__ import print_function
from .utils import ndigits, init_logger
from .file_io import read_db_multiple
import CatalogUtils
from scipy.stats import kstat, kstatvar
from scipy.optimize import minimize
import emcee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

plt.rcParams["figure.facecolor"] = "white"


# Custom ValueError for too few groups (for catching during debugging)
class TooFewGroupsError(ValueError): pass


def get_array(x):
    """Get x as a 1D array"""
    try:
        arr_x = x.values
    except AttributeError:
        arr_x = np.atleast_1d(x).flatten()
    return arr_x


def corr_coef(x, y, x_mean, y_mean, x_var, y_var):
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
    :rtype r: scalar or 1D :class:`numpy.ndarray` `float`
    """
    r = ((x - x_mean) * (y - y_mean)) / (np.sqrt(x_var * y_var))
    if hasattr(r, "__len__"):
        r = np.asarray(r)
    return r


def mean_y(rpo, rlo, a, alpha, beta, s):
    """
    The mean of :math:`\frac{\Delta R_\parallel}{R_\parallel^O}`. This
    function looks like :math:`a x^\alpha y^\beta \exp[-y^2 / 2 s]`, where x
    is the perpendicular separation and y is the parallel separation. If
    :param:`rpo` and/or :param:`rlo` is array-like, they will be
    flattened before calculating. Notice that if :param:`rpo` and
    :param:`rlo` are both array-like with different lengths, they will be
    flattened, a meshgrid will be formed with :param:`rpo` first, and then
    they will be flattened again (for order of unzipping later).

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
    :return f: The function evaluated at the separation(s)
    :rtype f: scalar or 1D :class:`numpy.ndarray` `float`
    """
    rpo = get_array(rpo)
    rlo = get_array(rlo)
    if rpo.size != rlo.size:
        rpo, rlo = np.meshgrid(rpo, rlo)
        rpo = rpo.flatten()
        rlo = rlo.flatten()
    f = a * (rpo ** alpha) * (rlo ** beta) * np.exp(-0.5 * rlo ** 2 / s)
    if rpo.size == 1:
        f = f.item()
    return f


def var_y(rpo, rlo, b, s1, s2, rho):
    """
    The variance of :math:`\frac{\Delta R_\parallel}{\sqrt{2} \chi'(\bar{z})
    \sigma_z(\bar{z})}`. This function looks like :math:`1 - b \exp[-0.5 \vec{
    r}^T C^{-1} \vec{r}]`, where :math:`\vec{r}` is a vector of the observed
    perpendicular and parallel separations, and C looks like a covariance
    matrix if :param:`s1` and :param:`s2` are variances and :param:`rho` is
    the correlation coefficient. If :param:`rpo` and/or :param:`rlo` is
    array-like, they will be flattened before calculating. Notice that if
    :param:`rpo` and :param:`rlo` are both array-like with different lengths,
    they will be flattened, a meshgrid will be formed with :param:`rpo`
    first, and then they will be flattened again (for order of unzipping later).

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
    :return f: The function evaluated at the separation(s)
    :rtype f: scalar or 1D :class:`numpy.ndarray` `float`
    """
    rpo = get_array(rpo)
    rlo = get_array(rlo)
    if rpo.size != rlo.size:
        rpo, rlo = np.meshgrid(rpo, rlo)
        rpo = rpo.flatten()
        rlo = rlo.flatten()
    cinv = np.array([s2, s1, -2 * rho * np.sqrt(s1 * s2)]) / (s1 * s2(1 -
                                                                      rho ** 2))
    f = 1.0 - b
    np.exp(-0.5 * (rpo ** 2 * cinv[0] + rlo ** 2 * cinv[1] + rpo *
                   rlo * cinv[2]))
    if rpo.size == 1:
        f = f.item()
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
    return -np.inf


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
    return -np.inf


class SingleFitter(object):
    """
    Fit a single function to a single set of data.
    """

    def __init__(self, rpo, rlo, data, var, func=None, prior=None,
                 param_names=None, fitter_name=None):
        """Initialize the SingleFitter

        :param rpo: The observed perpendicular separation
        :type rpo: array-like `float`
        :param rlo: The observed parallel separation
        :type rlo: array-like `float`
        :param data: The vector of data to fit
        :type data: array-like `float`
        :param var: The variance (for diagonal covariance matrix) or covariance
        matrix of the data
        :type var: array-like `float`
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
        self.rpo = get_array(rpo)
        self.rlo = get_array(rlo)
        self.data = get_array(data)
        self.logger.debug("Set up inverse (co)variance and lnlike")
        self._set_var_lnlike(var)
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

    def _set_var_lnlike(self, var):
        if var.ndim == 1:
            self.logger.debug("Setting up 1D variance and likelihood")
            self.sigma = np.sqrt(var)
            self.ivar = 1. / var
            self.lnlike = lambda theta: self.lnlike_1d(theta)
        else:
            self.logger.debug("Setting up ND variance and likelihood")
            self.sigma = np.sqrt(np.diag(var))
            self.ivar = np.linalg.pinv(var)
            self.lnlike = lambda theta: self.lnlike_2d(theta)
        self.logger.debug("done")

    def set_fit_func(self, func, param_names):
        """Set a (new) fitting function with parameter names

        :param func: The fitting function to use
        :type func: `function` or `None`
        :param param_names: The names of the parameters for the fitting function
        :type param_names: 1D array-like `str` or `None`
        """
        if func is None:
            self.logger.debug("Setting fitting function to None")
            self.f = None
            self.params = None
            self.ndim = None
        else:
            if param_names is None:
                raise ValueError("Parameter names must be given when fitting "
                                 "function is specified")
            self.logger.debug("Setting fitting function and parameters")
            self.f = lambda x, y, params: func(x, y, *params)
            self.params = param_names
            self.ndim = len(param_names)
        self.logger.debug("done")

    def set_prior_func(self, prior):
        """Set a (new) prior likelihood function

        :param prior: The prior probability function
        :type prior: `function` or `None`
        """
        if prior is None:
            self.logger.debug("Setting prior to None")
            self.pr = None
        else:
            self.logger.debug("Setting up prior function")
            self.pr = lambda params: prior(params)
        self.logger.debug("done")

    def lnlike_1d(self, theta):
        if self.f is None:
            raise AttributeError("No fitting function set for likelihood")
        feval = self.f(self.rpo, self.rlo, theta)
        return -0.5 * np.sum((self.data - feval) ** 2 * self.ivar)

    def lnlike_2d(self, theta):
        if self.f is None:
            raise AttributeError("No fitting function set for likelihood")
        feval = self.f(self.rpo, self.rlo, theta)
        return -0.5 * np.dot((self.data - feval), np.dot(self.ivar,
                                                         (self.data - feval)))

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
        parameter is ignored when continuing chains. Default 1
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

    def model(self, rpo, rlo):
        """Get the best fit model at the given separations

        :param rpo: The observed perpendicular separation
        :type rpo: scalar or array-like `float`
        :param rlo: The observed parallel separation
        :type rlo: scalar or array-like `float`
        :return m: The value of the best fit model at the points. This will
        be 1D if :param:`rpo` and :param:`rlo` are both array-like, with both
        separations flattened
        :rtype m: scalar or 1D ndarray `float`
        """
        rpo = np.atleast_1d(rpo).flatten()
        rlo = np.atleast_1d(rlo).flatten()
        if rpo.size != rlo.size:
            rpo, rlo = np.meshgrid(rpo, rlo)
            rpo = rpo.flatten()
            rlo = rlo.flatten()
        m = self.f(rpo, rlo, self._best_fit_params)
        if rpo.size == 1:
            m = m.item()
        return m

    def model_with_errors(self, rpo, rlo):
        """Get the median and 68 percent contours of the model at the
        separations

        :param rpo: The observed perpendicular separation
        :type rpo: scalar or array-like `float`
        :param rlo: The observed parallel separation
        :type rlo: scalar or array-like `float`
        :return m: The 16th, 50th, and 84th percentiles of the model at the
        points. The 0th axis is the percentiles, and the first axis is the
        flattened points, if not both scalar
        :rtype m: 1D or 2D ndarray `float`
        """
        rpo = np.atleast_1d(rpo).flatten()
        rlo = np.atleast_1d(rlo).flatten()
        if rpo.size != rlo.size:
            rpo, rlo = np.meshgrid(rpo, rlo)
            rpo = rpo.flatten()
            rlo = rlo.flatten()
        samples = self.sampler.chain[:, self._nburnin:, :].reshape((-1,
                                                                    self.ndim))
        meval = map(lambda params: self.f(rpo, rlo, params), samples)
        m = np.percentile(meval, [16.0, 50.0, 84.0], axis=0)
        m = m.squeeze()
        return m

    def plot(self, rpo_label, rlo_label, ylabel, bins, bin_size,
             exp, is_rpo=False, logx=False, logy=False, filename=None,
             figsize=None, display=False, text_size=22, with_fit=False):
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
        :param bin_size: The size of the bins for the specified fixed
        separation, in the same units as the separation
        :type bin_size: `float`
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

        axis_label = r"${{}} \pm {}$".format(np.around(0.5 * bin_size,
                                                       2 - ndigits(0.5 *
                                                                   bin_size)))
        fig = plt.figure(figsize=figsize)
        bins = np.atleast_1d(bins).flatten()
        full_ax = fig.add_subplot(111, yticks=[], yticklabels=[], frameon=False)
        grid = plt.GridSpec(bins.size, 1, hspace=0, left=0.2)
        if logx:
            full_ax.set_xscale("log")
        if logy:
            full_ax.set_yscale("log")
        full_ax.set_ylabel(ylabel)
        for i, r in enumerate(bins):
            ax = fig.add_subplot(grid[i, :], sharex=full_ax)
            ax.axhline(exp, c="k")
            if is_rpo:
                idx = np.isclose(self.rpo, r, atol=0.25 * bin_size)
                x = self.rlo[idx]
                labeli = r"{}{}".format(rpo_label, axis_label.format(
                    np.around(r, 2 - ndigits(r))))
                xlabel = rlo_label
                if with_fit:
                    if self._best_fit_params is None:
                        raise AttributeError("Cannot plot best fit without "
                                             "fitting")
                    m = self.model_with_errors(r, x)
            else:
                idx = np.isclose(self.rlo, r, atol=0.25 * bin_size)
                x = self.rpo[idx]
                labeli = r"{}{}".format(rlo_label, axis_label.format(
                    np.around(r, 2 - ndigits(r))))
                xlabel = rpo_label
                if with_fit:
                    if self._best_fit_params is None:
                        raise AttributeError("Cannot plot best fit without "
                                             "fitting")
                    m = self.model_with_errors(x, r)
            d = self.data[idx]
            s = self.sigma[idx]
            line = ax.errorbar(x, d, yerr=s, fmt="C0o", alpha=0.6)[0]
            if with_fit:
                ax.fill_between(x, m[0], m[2], color="C2", alpha=0.4)
                ax.plot(x, m[1], "C2-")
            ax.legend([line], [labeli], loc="best", markerscale=0,
                      frameon=False)
            if ax.is_last_row():
                ax.set_xlabel(xlabel)
            else:
                ax.set_xticklabels("")
        if filename is not None:
            fig.savefig(filename, bbox_inches="tight")
        if display:
            fig.show()
        return fig


class AnalyticSingleFitter(object):
    """
    Another single data model fitter, but this time for a model than can be
    fit analytically. This case is only valid (so far) for a constant
    function with uncorrelated errors (i.e. diagonal covariance matrix)
    """

    def __init__(self, rpo, rlo, data, var, fitter_name=None):
        """
        Initialize the analytic fitter. Please note that all array-likes should
        have the same size
        :param rpo: The observed perpendicular separations
        :type rpo: 1D array-like `float`
        :param rlo: The observed parallel separations
        :type rlo: 1D array-like `float`
        :param data: The vector of data to be fit
        :type data: 1D array-like `float`
        :param var: The variance on the data
        :type var: 1D array-like `float`
        :param fitter_name: The name for this instance of the fitter. If
        `None`, the name will be set to 'AnalyticSingleFitter'. Default `None`
        :type fitter_name: `str` or `None`, optional
        """
        self._get_name(fitter_name)
        self.logger = init_logger(self.name)
        self.logger.debug("Setting up data and variance")
        rpo = get_array(rpo)
        rlo = get_array(rlo)
        if rpo.size != rlo.size:
            raise ValueError("Separation arrays must have same size")
        data = get_array(data)
        if rpo.size != data.size:
            raise ValueError("Data array must have same size as separations")
        if np.allclose(np.diag(np.diag(var)), var):
            var = np.diag(var)
        else:
            var = get_array(var)
        if data.size != var.size:
            raise ValueError("Variance must have same size as data")
        self.rpo = rpo
        self.rlo = rlo
        self.data = data
        self.ivar = 1. / var
        self.sigma = np.sqrt(var)
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
        self._c = np.sum(self.data * self.ivar) / np.sum(self.ivar)

    def _get_err(self):
        self._c_err = np.sqrt(1. / np.sum(self.ivar))

    def fit(self):
        self._get_const()
        self._get_err()

    def model(self, rpo, rlo):
        """Get the best fit model at the given separations

        :param rpo: The observed perpendicular separation
        :type rpo: scalar or array-like `float`
        :param rlo: The observed parallel separation
        :type rlo: scalar or array-like `float`
        :return m: The value of the best fit model at the points. This will
        be 1D if :param:`rpo` and :param:`rlo` are both array-like, with both
        separations flattened
        :rtype m: scalar or 1D ndarray `float`
        """
        rpo = get_array(rpo)
        rlo = get_array(rlo)
        if rpo.size != rlo.size:
            rpo, rlo = np.meshgrid(rpo, rlo)
            rpo = rpo.flatten()
        m = np.full(rpo.size, self._c)
        if rpo.size == 1:
            m = m.item()
        return m

    def model_with_errors(self, rpo, rlo):
        """Get the median and 1 sigma error region of the model at the
        separations

        :param rpo: The observed perpendicular separation
        :type rpo: scalar or array-like `float`
        :param rlo: The observed parallel separation
        :type rlo: scalar or array-like `float`
        :return m: The (median - 1 sigma), median, and (median + 1 sigma)
        values of the model at the points. The 0th axis is the percentiles,
        and the first axis is the flattened points, if not both scalar
        :rtype m: 1D or 2D ndarray `float`
        """
        rpo = get_array(rpo)
        rlo = get_array(rlo)
        if rpo.size != rlo.size:
            rpo, rlo = np.meshgrid(rpo, rlo)
            rpo = rpo.flatten()
        m = np.empty((3, rpo.size), dtype=float)
        m[0, :] = self._c - self._c_err
        m[1, :] = self._c
        m[2, :] = self._c + self._c_err
        m = m.squeeze()
        return m

    def plot(self, rpo_label, rlo_label, ylabel, bins, bin_size,
             exp, is_rpo=False, logx=False, logy=False, filename=None,
             figsize=None, display=False, text_size=22, with_fit=False):
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
        :param bin_size: The size of the bins for the specified fixed
        separation, in the same units as the separation
        :type bin_size: `float`
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

        axis_label = r"${{}} \pm {}$".format(np.around(0.5 * bin_size,
                                                       2 - ndigits(0.5 *
                                                                   bin_size)))
        fig = plt.figure(figsize=figsize)
        bins = np.atleast_1d(bins).flatten()
        full_ax = fig.add_subplot(111, yticks=[], yticklabels=[], frameon=False)
        grid = plt.GridSpec(bins.size, 1, hspace=0, left=0.2)
        if logx:
            full_ax.set_xscale("log")
        if logy:
            full_ax.set_yscale("log")
        full_ax.set_ylabel(ylabel)
        for i, r in enumerate(bins):
            ax = fig.add_subplot(grid[i, :], sharex=full_ax)
            ax.axhline(exp, c="k")
            if is_rpo:
                idx = np.isclose(self.rpo, r, atol=0.25 * bin_size)
                x = self.rlo[idx]
                labeli = r"{}{}".format(rpo_label, axis_label.format(
                    np.around(r, 2 - ndigits(r))))
                xlabel = rlo_label
                if with_fit:
                    if self._c is None or self._c_err is None:
                        raise AttributeError("Cannot plot best fit without "
                                             "fitting")
                    m = self.model_with_errors(r, x)
            else:
                idx = np.isclose(self.rlo, r, atol=0.25 * bin_size)
                x = self.rpo[idx]
                labeli = r"{}{}".format(rlo_label, axis_label.format(
                    np.around(r, 2 - ndigits(r))))
                xlabel = rpo_label
                if with_fit:
                    if self._c is None or self._c_err is None:
                        raise AttributeError("Cannot plot best fit without "
                                             "fitting")
                    m = self.model_with_errors(x, r)
            d = self.data[idx]
            s = self.sigma[idx]
            line = ax.errorbar(x, d, yerr=s, fmt="C0o", alpha=0.6)[0]
            if with_fit:
                ax.fill_between(x, m[0], m[2], color="C2", alpha=0.4)
                ax.plot(x, m[1], "C2-")
            ax.legend([line], [labeli], loc="best", markerscale=0,
                      frameon=False)
            if ax.is_last_row():
                ax.set_xlabel(xlabel)
            else:
                ax.set_xticklabels("")
        if filename is not None:
            fig.savefig(filename, bbox_inches="tight")
        if display:
            fig.show()
        return fig


class ProbFitter(object):
    """
    Fit a bivariate Gaussian of 2 variables to the data. This is not a
    generalized fitter, it assumes the means and variances follow specific
    functions that are not set by the user but hardcoded within.
    """

    def __init__(self, db_file, table_name, perp_bin_size, par_bin_size,
                 perp_bin_scale, par_bin_scale, sigma_z, cosmo, min_counts=200,
                 fitter_name=None, **kwargs):
        """
        Initialize the fitter with separations

        :param db_file: The file path to the database containing the
        separations. Assumed to be in the format that is output by the
        distance functions within this module.
        :type db_file: `str`
        :param table_name: The name of the table within the database
        :type table_name: `str`
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
        :param cosmo: The cosmology to use for calculating distance functions
        :type cosmo: :class:`astropy.cosmology.FLRW` or subclass
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
        CatalogUtils.initialize(cosmo, 0.0, 2.0)
        self._get_name(fitter_name)
        self.logger = init_logger(self.name)
        self.sigma_z = sigma_z
        self.min_counts = min_counts
        self._has_r = False
        self._has_r_stat = False
        self._fitters = dict.fromkeys(["mean_x", "var_x", "mean_y", "var_y",
                                       "mean_r"])
        self.logger.debug("Read data...")
        self.seps = read_db_multiple(db_file, table_name, limits=kwargs.pop(
            "limits", None))
        self.logger.debug("Read data...[done]")
        self.logger.debug("Add extra data columns...")
        self._add_extra_columns(perp_bin_size, par_bin_size, perp_bin_scale,
                                par_bin_scale)
        self.logger.debug("Add extra data columns...[done]")
        self.logger.debug("Generate statistics...")
        self._get_stats()
        self.rpo = self.stats.index.get_level_values("RPO_BIN")
        self.rlo = self.stats.index.get_level_values("RLO_BIN")
        self.logger.debug("Generate statistics...[done]")
        self.logger.debug("Set up fitters...")
        self.logger.debug("mean_x")
        self._fitters["mean_x"] = AnalyticSingleFitter(self.rpo, self.rlo,
                                                       self.stats["m_x"],
                                                       self.stats["var_m_x"],
                                                       "mean_x")
        self.logger.debug("var_x")
        self._fitters["var_x"] = AnalyticSingleFitter(self.rpo, self.rlo,
                                                      self.stats["v_x"],
                                                      self.stats["var_v_x"],
                                                      "var_x")
        self.logger.debug("mean_y")
        self._fitters["mean_y"] = SingleFitter(self.rpo, self.rlo,
                                               self.stats["m_y"], self.stats[
                                                   "var_m_y"], mean_y,
                                               prior_mean_y, [r"$a$",
                                                              r"$\alpha$",
                                                              r"$\beta$",
                                                              r"$s$"], "mean_y")
        self.logger.debug("var_y")
        self._fitters["var_y"] = SingleFitter(self.rpo, self.rlo, self.stats[
            "v_y"], self.stats["var_v_y"], var_y, prior_var_y, [r"$b$",
                                                                r"$s_1$",
                                                                r"$s_2$",
                                                                r"$\rho$"],
                                              "var_y")
        self.logger.debug("mean_r")
        self._fitters["mean_r"] = None
        self.logger.debug("Set up fitters...[done]")
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

    def _get_name(self, fitter_name):
        self.name = ("{}.{}".format(self.__class__.__name__, fitter_name) if
                     fitter_name is not None else self.__class__.__name__)

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

    def _add_bin_column(selfself, orig_col_name, bin_col_name, bin_size, scale):
        self.seps[bin_col_name] = ((np.floor(self.seps[orig_col_name] /
                                             bin_size) + 0.5) * (bin_size /
                                                                 scale))

    def _add_zbar(self):
        self.seps["ZBAR"] = CatalogUtils.z_at_chi(self.seps["AVE_OBS_LOS"])

    def _add_delta_column(self, dirname, new_col_name, scale_func):
        scale = 1. / scale_func(self.seps["R_PERP_O"], self.seps["R_PAR_O"],
                                self.seps["ZBAR"])
        tcol_name = "R_{}_T".format(dirname)
        ocol_name = "R_{}_O".format(dirname)
        self.seps[new_col_name] = ((self.seps[tcol_name] - self.seps[
            ocol_name]) * scale)

    def _add_extra_columns(self, perp_bin_size, par_bin_size, perp_bin_scale,
                           par_bin_scale):
        self.logger.debug("Add RPO_BIN")
        self._add_bin_column("R_PERP_O", "RPO_BIN", perp_bin_size,
                             perp_bin_scale)
        self.logger.debug("Add RLO_BIN")
        self._add_bin_column("R_PAR_O", "RLO_BIN", par_bin_size, par_bin_scale)
        self.logger.debug("Add ZBAR")
        self._add_zbar()
        self.logger.debug("Add DELTA_R_PERP")
        self._add_delta_column("PERP", "DELTA_R_PERP", self._perp_mean_scale)
        self.logger.debug("Add DELTA_R_PAR")
        self._add_delta_column("PAR", "DELTA_R_PAR", self._par_mean_scale)
        self.logger.debug("Add x")
        self._add_delta_column("PERP", "x", self._perp_var_scale)
        self.logger.debug("Add y")
        self._add_delta_column("PAR", "y", self._par_var_scale)

    def _group_seps(self):
        grouped_all = self.seps.groupby(["RPO_BIN", "RLO_BIN"])
        filtered = grouped_all.filter(lambda x: len(x) >= self.min_counts)
        grouped = filtered.groupby(["RPO_BIN", "RLO_BIN"])
        return grouped

    def _get_stats(self):
        self.logger.debug("Group data")
        grouped = self._group_seps()
        if len(grouped) <= 2:  ## Is this good enough, or should I require more?
            raise TooFewGroupsError("Only {} groups with at least {} pairs. " \
                                    "consider decreasing the minimum number " \
                                    "of pairs or using a larger " \
                                    "catalog".format(len(grouped),
                                                     self.min_counts))
        self.stats = pd.DataFrame()
        self.logger.debug("Sample mean of DELTA_R_PERP")
        self.stats["m_x"] = grouped["DELTA_R_PERP"].agg(kstat, n=1)
        self.logger.debug("Variance on sample mean of DELTA_R_PERP")
        self.stats["var_m_x"] = grouped["DELTA_R_PERP"].agg(kstatvar, n=1)
        self.logger.debug("Sample variance of x")
        self.stats["v_x"] = grouped["x"].agg(kstat, n=2)
        self.logger.debug("Variance on sample variance of x")
        self.stats["var_v_x"] = grouped["x"].agg(kstatvar, n=2)
        self.logger.debug("Sample mean of DELTA_R_PAR")
        self.stats["m_y"] = grouped["DELTA_R_PAR"].agg(kstat, n=1)
        self.logger.debug("Variance on sample mean of DELTA_R_PAR")
        self.stats["var_m_y"] = grouped["DELTA_R_PAR"].agg(kstatvar, n=1)
        self.logger.debug("Sample variance of y")
        self.stats["v_y"] = grouped["y"].agg(kstat, n=2)
        self.logger.debug("Variance on sample variance of y")
        self.stats["var_v_y"] = grouped["y"].agg(kstatvar, n=2)

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
        rpo = get_array(rpo)
        rlo = get_array(rlo)
        zbar = get_array(zbar)
        return (self._perp_mean_scale(rpo, rlo, zbar) * self.mean_x.model(
            rpo, rlo) + rpo)

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
        rpo = get_array(rpo)
        rlo = get_array(rlo)
        zbar = get_array(zbar)
        return (self._perp_var_scale(rpo, rlo, zbar) ** 2 * self.var_x.model(
            rpo, rlo))

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
        rpo = get_array(rpo)
        rlo = get_array(rlo)
        zbar = get_array(zbar)
        return (self._par_mean_scale(rpo, rlo, zbar) * self.mean_y.model(
            rpo, rlo) + rlo)

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
        rpo = get_array(rpo)
        rlo = get_array(rlo)
        zbar = get_array(zbar)
        return (self._par_var_scale(rpo, rlo, zbar) ** 2 * self.var_y.model(
            rpo, rlo))

    def _add_corr_column(self):
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
        self.seps["r"] = corr_coef(self.seps["R_PERP_T"], self.seps[
            "R_PAR_T"], x_mean, y_mean, x_var, y_var)
        self._has_r = True

    def _add_corr_stat(self):
        if not self._has_r:
            self.logger.debug("Get correlations...")
            self._add_corr_column()
            self.logger.debug("Get correlations...[done]")
        self.logger.debug("Regroup data")
        grouped = self._group_seps()
        self.logger.debug("Sample mean of r")
        self.stats["m_r"] = grouped["r"].agg(kstat, n=1)
        self.logger.debug("Variance on sample mean of r")
        self.stats["var_m_r"] = grouped["r"].agg(kstatvar, n=1)
        self._has_r_stat = True

    def init_corr_fitter(self):
        if not self._has_r_stat:
            self.logger.debug("Generate correlation statistics...")
            self._add_corr_stat()
            self.logger.debug("Generate correlation statistics...[done]")
        self._fitters["mean_r"] = SingleFitter(self.rpo, self.rlo,
                                               self.stats["m_r"], self.stats[
                                                   "var_m_r"],
                                               fitter_name="mean_r")
