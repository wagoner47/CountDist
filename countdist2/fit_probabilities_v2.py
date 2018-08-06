from __future__ import print_function
import os, CatalogUtils, emcee
from scipy.stats import kstat, kstatvar
from scipy.optimize import minimize
from scipy import special
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from .utils import MyConfigObj, init_logger
from . import file_io, utils
from itertools import product

logger = init_logger("{}.{}".format(__package__, os.path.splitext(
            os.path.basename(__file__))[0]))


# Wrapper class and Exception for it
class FunctionCallError(Exception):
    """
    Exception to be raised when an error occurs in the __call__ method of 
    _function_wrapper

    Attributes
    ----------
    :attribute message: The prepended message
    :type message: `str`
    :attribute params: The parameter values at which the function was being 
    evaluated at the time of the error
    :type x: 1D array-like
    :attribute args: The positional arguments at the time of the error
    :type args: 1D array-like
    :attribute kwargs: The keyword arguments at the time of the error
    :type kwargs: `dict`
    :attribute type: Type of Exception originally raised
    :type type: `str`
    :attribute tb_message: The original TraceBack message
    :type tb_message: `str`
    
    """
    def __init__(self, message, params, args, kwargs, exc):
        self.type = type(exc).__name__
        self.tb_message = exc.args[0]
        self.params = params
        self.args = args
        self.kwargs = kwargs
        self.message = message
    
    def __str__(self):
        msg = self.message
        msg += "\n\tparams: " + str(self.params)
        msg += "\n\targs: " + str(self.args)
        msg += "\n\tkwargs: " + str(self.kwargs)
        msg += "\nError: " + self.type + "\n" + self.tb_message
        return msg


class _function_wrapper(object):
    """
    Adopted from emcee
    
    Attributes
    ----------
    :attribute f: The function being wrapped
    :type f: `function`
    :attribute args: The positional arguments of the function, other than 
    parameters 
    and position
    :type args: 1D array-like
    :attribute kwargs: The keyword arguments of the function
    :type kwargs: `dict`
    
    """
    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs
    
    def __call__(self, params, index):
        try:
            return self.f(*params, index, *self.args, **self.kwargs)
        except Exception as e:
            raise FunctionCallError("Error occured while calling wrapped "\
                                        "function", params, args, kwargs, e)
    
    def call_chain(self, params_chain, index):
        """Call the function at a chain of parameter values
        
        Parameters
        ----------
        :param params_chain: The set of parameters at which to evaluate
        :type params_chain: 2D array-like
        :param index: The location(s) at which to evaluate the function
        :type index: :class:`pandas.MultiIndex` or `tuple`
        
        Returns
        -------
        :return f: The function evaluated at each step in the parameter chain, 
        as a dictionary with keys given by the step number
        :rtype f: `dict`
        """
        f = {}
        for i, params in enumerate(params_chain):
            f[i] = self.__call__(params, index)
        return f

# Stats functions
def sample_mean(x):
    """Get the sample mean of x
    
    Parameters
    ----------
    :param x: The sample for which to estimate the sample mean
    :type x: array-like
    
    Returns
    -------
    :return: The sample mean
    :rtype: `float`
    """
    return kstat(x, n=1)


def sample_var(x):
    """Get the sample variance of x
    
    Parameters
    ----------
    :param x: The sample for which to estimate the sample variance
    :type x: array-like
    
    Returns
    -------
    :return: The sample variance
    :rtype: `float`
    """
    return kstat(x, n=2)



def var_sample_mean(x):
    """Get the variance on the sample mean of x
    
    Parameters
    ----------
    :param x: The sample for which to estimate the variance on the sample mean
    :type x: array-like
    
    Returns
    -------
    :return: The variance on the sample mean
    :rtype: `float`
    """
    return kstatvar(x, n=1)


def var_sample_var(x):
    """Get the variance on the sample variance of x
    
    Parameters
    ----------
    :param x: The sample for which to estimate the variance on the sample 
    variance
    :type x: array-like
    
    Returns
    -------
    :return: The variance on the sample variance
    :rtype: `float`
    """
    return kstatvar(x, n=2)


def sample_corr(xy, x_mean, x_var, y_mean, y_var):
    """Get the sample correlation between x and y. The means and variances are 
    passed so that they may be calculated using any function desired.
    
    Parameters
    ----------
    :param xy: The product of the two sets of values to correlate
    :type xy: array-like
    :param x_mean: The mean of :param:`x`
    :type x_mean: `float`
    :param x_var: The variance of :param:`x`
    :type x_var: `float`
    :param y_mean: The mean of :param:`y`
    :type y_mean: `float`
    :param y_var: The variance of :param:`y`
    :type y_var: `float`
    
    Returns
    -------
    :return: The sample correlation
    :rtype: `float`
    """
    n = len(xy)
    return ((np.sum(xy) - (n * x_mean * y_mean)) / 
            ((n - 1) * np.sqrt(x_var * y_var)))


# Fitting functions (the forms of these are checked before-hand)
def mean_x(c, index, ret_scalar=False, index_list=["RPO_BIN", "RLO_BIN"]):
    """The mean of x, which is the perpendicular direction scaled variable. 
    This function is a constant
    
    Parameters
    ----------
    :param c: The constant (the value of the function everywhere)
    :type c: `float`
    :param index: The MultiIndex at which to evaluate, or a tuple to be turned 
    into a length-1 MultiIndex
    :type index: :class:`pandas.MultiIndex` or `tuple`
    :param ret_scalar: If `True`, return a scalar rather than a Series. Default 
    `False`
    :type ret_scalar: `bool`, optional
    :param index_list: List of names for the MultiIndex levels, for setting the 
    MultiIndex. Ignored if a MultiIndex is passed. Default 
    ['RPO_BIN', 'RLO_BIN'] (for :math:`R_\perp^O` and :math:`R_\parallel^O`, 
    respectively)
    :type index_list: 1D array-like, optional
    
    Returns
    -------
    :return f: The mean function evaluated at the index/indices provided, as a 
    Series unless :param:`ret_scalar` is `True`
    :rtype f: :class:`pandas.Series` or `float`
    """
    if ret_scalar:
        f = c
    else:
        if isinstance(index, tuple):
            index = pd.MultiIndex((index,), names=index_list)
        f = pd.Series(c, index=index)
    return f


def var_x(c, index, ret_scalar=False, index_list=["RPO_BIN", "RLO_BIN"]):
    """The variance of x, which is the perpendicular direction scaled variable. 
    This function is a constant
    
    Parameters
    ----------
    :param c: The constant (the value of the function everywhere)
    :type c: `float`
    :param index: The MultiIndex at which to evaluate, or a tuple to be turned 
    into a length-1 MultiIndex
    :type index: :class:`pandas.MultiIndex` or `tuple`
    :param ret_scalar: If `True`, return a scalar rather than a Series. Default 
    `False`
    :type ret_scalar: `bool`, optional
    :param index_list: List of names for the MultiIndex levels, for setting the 
    MultiIndex. Ignored if a MultiIndex is passed. Default 
    ['RPO_BIN', 'RLO_BIN'] (for :math:`R_\perp^O` and :math:`R_\parallel^O`, 
    respectively)
    :type index_list: 1D array-like, optional
    
    Returns
    -------
    :return f: The variance function evaluated at the index/indices provided, 
    as a Series unless :param:`ret_scalar` is `True`
    :rtype f: :class:`pandas.Series` or `float`
    """
    if ret_scalar:
        f = c
    else:
        if isinstance(index, tuple):
            index = pd.MultiIndex((index,), names=index_list)
        f = pd.Series(c, index=index)
    return f


def mean_y(a, alpha, beta, s, index, ret_scalar=False, 
           index_list=["RPO_BIN", "RLO_BIN"]):
    """The mean of y, which is the parallel direction scaled variable. 
    This function is a product of power laws times a Gaussian:
    
    .. math::
        
        \mu(y) = a R_\perp^\alpha R_\parallel^\beta 
                      \operatorname{e}^{-\frac{R_\parallel^2}{2 s}}
    
    Parameters
    ----------
    :param a: The amplitude
    :type a: `float`
    :param alpha: The power law index on the observed perpendicular separation
    :type alpha: `float`
    :param beta: The power law index on the observed parallel separation
    :type beta: `float`
    :param s: The scale for the Gaussian term
    :type s: `float`
    :param index: The MultiIndex at which to evaluate, or a tuple to be turned 
    into a length-1 MultiIndex. The order should be perpendicular separation 
    as the first level or item and parallel separation as the second
    :type index: :class:`pandas.MultiIndex` or `tuple`
    :param ret_scalar: If `True`, return a scalar rather than a Series. Default 
    `False`
    :type ret_scalar: `bool`, optional
    :param index_list: List of names for the MultiIndex levels, for setting the 
    MultiIndex. Ignored if a MultiIndex is passed. Default 
    ['RPO_BIN', 'RLO_BIN'] (for :math:`R_\perp^O` and :math:`R_\parallel^O`, 
    respectively)
    :type index_list: 1D array-like, optional
    
    Returns
    -------
    :return f: The mean function evaluated at the index/indices provided, as a 
    Series unless :param:`ret_scalar` is `True`
    :rtype f: :class:`pandas.Series` or `float`
    """
    if isinstance(index, tuple):
        rpo = index[0]
        rlo = index[1]
        index = pd.MultiIndex((index,), names=index_list)
    else:
        rpo = index.get_level_values(0).values
        rlo = index.get_level_values(1).values
    f = a * rpo**alpha * rlo**beta * np.exp(-0.5 * rlo**2 / s)
    if not ret_scalar:
        f = pd.Series(f, index=index)
    return f


def var_y(a, b, s1, s2, rho, index, ret_scalar=False, 
          index_list=["RPO_BIN", "RLO_BIN"]):
    """The variance of y, which is the parallel direction scaled variable. 
    This function is a constant minus a 2D Gaussian term with correlation
    
    .. math::
        
        \sigma_y^2 = a - b \operatorname{exp}\left[-\frac{1}{2} 
             \vec{r}^{\,\,\intercal} \cdot \mathbf{C}^{-1} \cdot \vec{r}\right]
    
    where 
    
    .. math::
        
        \vec{r} = \begin{bmatrix} R_\perp \\ R_\parallel \end{bmatrix}
    
    and 
    
    .. math::
        
        \mathbf{C} = \begin{bmatrix} s_1 & \rho \sqrt{s_1 s_2} \\
                                     \rho \sqrt{s_1 s_2} & s_2 \end{bmatrix}
    
    Parameters
    ----------
    :param a: The amplitude
    :type a: `float`
    :param alpha: The power law index on the observed perpendicular separation
    :type alpha: `float`
    :param beta: The power law index on the observed parallel separation
    :type beta: `float`
    :param s: The scale for the Gaussian term
    :type s: `float`
    :param index: The MultiIndex at which to evaluate, or a tuple to be turned 
    into a length-1 MultiIndex. The order should be perpendicular separation 
    as the first level or item and parallel separation as the second
    :type index: :class:`pandas.MultiIndex` or `tuple`
    :param ret_scalar: If `True`, return a scalar rather than a Series. Default 
    `False`
    :type ret_scalar: `bool`, optional
    :param index_list: List of names for the MultiIndex levels, for setting the 
    MultiIndex. Ignored if a MultiIndex is passed. Default 
    ['RPO_BIN', 'RLO_BIN'] (for :math:`R_\perp^O` and :math:`R_\parallel^O`, 
    respectively)
    :type index_list: 1D array-like, optional
    
    Returns
    -------
    :return f: The mean function evaluated at the index/indices provided, as a 
    Series unless :param:`ret_scalar` is `True`
    :rtype f: :class:`pandas.Series` or `float`
    """
    if isinstance(index, tuple):
        rpo = index[0]
        rlo = index[1]
        index = pd.MultiIndex((index,), names=index_list)
    else:
        rpo = index.get_level_values(0).values
        rlo = index.get_level_values(1).values
    # Inverse of 2x2 matrix is analytic, so define inverted matrix
    cinv = (np.array([[s2, -rho * np.sqrt(s1 * s2)], 
                      [-rho * np.sqrt(s1 * s2), s1]]) / 
            (s1 * s2 * (1 - rho**2)))
    f = a - b * np.exp(-0.5 * (x**2 * cinv[0,0] + y**2 * cinv[1,1] +
                               x * y * (cinv[0,1] + cinv[1,0])))
    if not ret_scalar:
        f = pd.Series(f, index=index)
    return f


# Priors (these have been determined based on the above fitting functions)
def prior_mean_x(theta):
    """The prior on the mean of x, which is the perpendicular direction.
    
    Note: This is a flat prior, the same as having no prior at all
    
    Parameters
    ----------
    :param theta: The parameter values being checked
    :type theta: 1D array-like
    
    Returns
    -------
    :return: The log-prior likelihood for these parameter values, in this case 
    always 0
    :rtype: `float`
    """
    return 0.0


def prior_var_x(theta):
    """The prior on the variance of x, which is the perpendicular direction.
    
    Note: This is a flat prior on any :math:`c > 0`
    
    Parameters
    ----------
    :param theta: The parameter values being checked
    :type theta: 1D array-like
    
    Returns
    -------
    :return: The log-prior likelihood for these parameter values, in this case 
    either 0 or :math:`-\infty`
    :rtype: `float`
    """
    if theta[0] > 0:
        return 0.0
    return -np.inf


def prior_mean_y(theta):
    """The prior on the mean of y, which is the parallel direction.
    
    Parameters
    ----------
    :param theta: The parameter values being checked
    :type theta: 1D array-like
    
    Returns
    -------
    :return: The log-prior likelihood for these parameter values, in this case 
    either 0 or :math:`-\infty`
    :rtype: `float`
    """
    if theta[-1] > 0:
        return 0.0


def prior_var_x(theta):
    """The prior on the variance of x, which is the perpendicular direction.
    
    Parameters
    ----------
    :param theta: The parameter values being checked
    :type theta: 1D array-like
    
    Returns
    -------
    :return: The log-prior likelihood for these parameter values, in this case 
    either 0 or :math:`-\infty`
    :rtype: `float`
    """
    if theta[-3] > 0 and theta[-2] > 0 and theta[-1]**2 < 1:
        return 0.0
    return -np.inf


# General fitter class
class Fitter2D(object):
    """
    Run fits and plot the results for a set of data described by a function of 
    2 variables x and y
    
    Attributes
    ----------
    :attribute index: The locations in (x, y) at which the fit should be performed
    :type index: :class:`pandas.MultiIndex`
    :attribute data: The data to be fitted
    :type data: :class:`pandas.Series`
    :attribute icov: The inverse covariance matrix for the data
    :type icov: :class:`pandas.DataFrame`
    :attribute sampler: The EnsembleSampler for the MCMC fit
    :type sampler: :class:`emcee.EnsembleSampler`
    :attribute param_names: The list of parameter names
    :type param_names: 1D array-like
    :attribute nsteps: The number of MCMC steps that were run. Available only after 
    a call has been made to fit_mcmc
    :type nsteps: `int`
    :attribute nburnin: The number of steps for the burn-in on the MCMC. Available 
    only after a call has been made to fit_mcmc
    :type nburnin: `int`
    
    Special Methods
    ---------------
    .. method:: func(
    """
