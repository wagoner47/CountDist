from __future__ import print_function
import os
import countdist2
import CatalogUtils
from scipy.stats import kstat, kstatvar
from scipy.optimize import minimize
from scipy import special
import emcee
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from .utils import MyConfigObj
from itertools import product


def setup(params_file):
    """Setup some of the global options we will need for the fits
    
    Parameters
    ----------
    :param params_file: Location of parameter file with file paths for loading and saving data
    :type params_file: `str`
    """
    global SIGMAR_EFF, SIGMAP_RP, DELTA_RLO, DELTA_RLT, DELTA_LN_RPO, DELTA_LN_RPT, DELTA_RPO, MAKE_PLOTS, SHOW_PLOTS
    global DB_FILE, FIT_SAVE_DIR, PLT_SAVE_DIR
    global TYPE
    config = MyConfigObj(params_file)
    TYPE = config["fit_params"]["type"].upper()
    if not TYPE in ["DD", "DR", "RD", "RR"]:
        raise ValueError("Invalid paircount type ({}). Choices are DD, DR, RD, RR".format(TYPE))
    catalog_meta_data = Table.read(config["run_params"]["ifname"]).meta
    SIGMAR_EFF = meta["SIGMAR"]
    cosmo_config = MyConfigObj(config["fit_params"]["cosmo_file"])
    cosmo = FlatLambdaCDM(100.0 * cosmo_config["cosmological_parameters"].as_float("h0"), config["cosmological_parameters"].as_float("omega_m"))
    sigmaz = config["fit_params"].as_float("sigmaz")
    zmin = config["run_params"].as_float("zp_min")
    zmax = config["run_params"].as_float("zp_max")
    CatalogUtils.initialize(cosmo, zmin, zmax)
    SIGMAP_RP = CatalogUtils.dr_dz(meta["ZEFF"]) * sigmaz * (1.0 + meta["ZEFF"]) / CatalogUtils.dist(meta["ZEFF"])
    DELTA_RLO = SIGMAR_EFF * config["fit_params"].as_float("delta_rlo")
    DELTA_RLT = SIGMAR_EFF * config["fit_params"].as_float("delta_rlt")
    DELTA_LN_RPO = SIGMAP_RP * config["fit_params"].as_float("delta_ln_rpo")
    DELTA_LN_RPT = SIGMAP_RP * config["fit_params"].as_float("delta_ln_rpt")
    DELTA_RPO = config["fit_params"].as_float("delta_rpo")
    MAKE_PLOTS = config["fit_params"].as_bool("make_plots")
    SHOW_PLOTS = config["fit_params"].as_bool("show_plots")
    DB_FILE = config["run_params"]["db_file"]
    FIT_SAVE_DIR = config["fit_params"]["fit_dir"]
    if MAKE_PLOTS:
        PLT_SAVE_DIR = config["fit_params"]["plt_dir"]


def ndigits(x):
    """A quick function to determine how many digits x has
    
    Parameters
    ----------
    :param x: The number to check
    :type x: `int` or `float`
    
    Returns
    -------
    :return: The number of digits in x
    :rtype: `int`
    """
    if x / 10.0 == x:
        return 0
    return int(np.floor(np.log10(np.abs(x))))
ndigits = np.vectorize(ndigits)


def lnlike(theta, df, col_dict, index_names, func):
    """A general likelihood function that takes the fitting function as input. Note this does not have the option for a non-diagonal covariance matrix
    
    Parameters
    ----------
    :param theta: The parameters at which to evaluate the log-likelihood
    :type theta: 1D array-like
    :param df: The DataFrame containing the data and errors to fit
    :type df: :class:`pandas.DataFrame`
    :param col_dict: A dictionary of the column names to use, where the data column should be key 'm' and the variances should be key 'var'
    :type col_dict: `dict`
    :param index_names: The name(s) of the index levels, where the first element should be the index name of the first variable in the fitting function, the second should be the second variable, etc.
    :type index_names: 1D array-like `str`
    :param func: The function to fit to the data
    :type func: callable
    
    Returns
    -------
    :return: The log-likelihood of the model with the given parameters with the data, using the function provided
    :rtype: float
    """
    f = func(df.index, index_names, *theta)
    return -0.5 * (df[col_dict["m"]] - f).pow(2).div(df[col_dict["var"]]).sum()


def lnprob(theta, df, col_dict, index_names, func, lnprior):
    """A general probability function that takes the fitting function and the function for the log-prior as inputs. Note this does not have the option for a non-diagonal covariance matrix
    
    Parameters
    ----------
    :param theta: The parameters at which to evaluate the log-probability
    :type theta: 1D array-like
    :param df: The DataFrame containing the data and errors to fit
    :type df: :class:`pandas.DataFrame`
    :param col_dict: A dictionary of the column names to use, where the data column should be key 'm' and the variances should be key 'var'
    :type col_dict: `dict`
    :param index_names: The name(s) of the index levels, where the first element should be the index name of the first variable in the fitting function, the second should be the second variable, etc.
    :type index_names: 1D array-like `str`
    :param func: The function to fit to the data
    :type func: callable
    :param lnprior: The function for the log-prior
    :type lnprior: callable
    
    Returns
    -------
    :return: The log-probability of the model with the given parameters with the data, using the function provided, and including the prior
    :rtype: float
    """
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, df, col_dict, index_names, func)


def fit_first_pass(init_guess, df, col_dict, index_names, func):
    """Do a first pass of fitting the function given to the data using scipy.optimize.minimize
    
    Parameters
    ----------
    :param init_guess: The initial guess for the parameters to the function
    :type init_guess: 1D array-like
    :param df: The DataFrame containing the data and errors to fit
    :type df: :class:`pandas.DataFrame`
    :param col_dict: A dictionary of the column names to use, where the data column should be key 'm' and the variances should be key 'var'
    :type col_dict: `dict`
    :param index_names: The name(s) of the index levels, where the first element should be the index name of the first variable in the fitting function, the second should be the second variable, etc.
    :type index_names: 1D array-like `str`
    :param func: The function to fit to the data
    :type func: callable
    
    Returns
    -------
    :return res.x: The result of using scipy.optimize.minimize to fit the function to the data
    :rtype res.x: 1D array float
    """
    nll = lambda *args: -lnlike(*args, df, col_dict, index_names, func)
    res = minimize(nll, init_guess)
    return res.x


def fit_mcmc(init_guess, df, col_dict, index_names, func, lnprior, nwalkers, nsteps, nburnin, nthreads=None):
    """Run an MCMC on the data to fit with the given function
    
    Parameters
    ----------
    :param init_guess: The initial guess for the parameters to the function
    :type init_guess: 1D array-like
    :param df: The DataFrame containing the data and errors to fit
    :type df: :class:`pandas.DataFrame`
    :param col_dict: A dictionary of the column names to use, where the data column should be key 'm' and the variances should be key 'var'
    :type col_dict: `dict`
    :param index_names: The name(s) of the index levels, where the first element should be the index name of the first variable in the fitting function, the second should be the second variable, etc.
    :type index_names: 1D array-like `str`
    :param func: The function to fit to the data
    :type func: callable
    :param lnprior: The function for the log-prior
    :type lnprior: callable
    :param nwalkers: The number of walkers to call the sampler with
    :type nwalkers: `int`
    :param nsteps: The number of steps to run that are kept. Note the total number of steps will be nsteps + nburnin
    :type nsteps: `int`
    :param nburnin: The number of steps to use as a burnin. Note the total number of steps will be nsteps + nburnin
    :type nburnin: `int`
    :param nthreads: The number of threads to use for multi-threaded emcee. If `None`, multi-threading not used. Default `None`
    :type nthreads: `int`
    
    Returns
    -------
    :return sampler: The sampler object
    :rtype sampler: :class:`emcee.EnsembleSampler`
    """
    start = fit_first_pass(init_guess, df, col_dict, index_names, func)
    ndim = len(init_guess)
    np.random.seed(0)
    pos = [start + 10.**(-4 + ndigits(start)) * np.random.randn(ndim) for i in range(nwalkers)]
    if threads is None:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(df, col_dict, index_names, func, lnprior))
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(df, col_dict, index_names, func, lnprior), threads=nthreads)
    sampler.run_mcmc(pos, nsteps + nburnin);
    return sampler


def r_perp_t_mean(index, index_names, x0, a, alpha):
    """This is the fitting function for the mean of the true perpendicular separation with a power law:
    
    .. math::
       
       a \left(\frac{x}{x0}\right)^\alpha
    
    Parameters
    ----------
    :param index: The values at which to calculate the mean
    :type index: :class:`pandas.Index` of :class:`pandas.MultiIndex`
    :param index_names: The name(s) of the index levels, where the first element should be used as x
    :type index_names: array-like `str`
    :param pivot: The pivots to use for x
    :type pivot: 1D array-like float
    :param a: The amplitude parameter
    :type a: float
    :param alpha: The power on x
    :type alpha: float
    
    Returns
    -------
    :return f: The value of the fitting function evaluated on index for the given parameters, with the same index as given
    :rtype f: :class:`pandas.Series`
    """
    try:
        x = index.get_level_values(index_names[0]).values
    except AttributeError:
        x = index.values
    f = pd.Series((a * (x / x0)**alpha), index=index)
    return f


def get_init_guess_mean_perp(index, index_names, df, col_dict):
    """Generate an initial guess for the parameters to :function:`r_perp_t_mean` for the given index. Also determines the pivot points to use for the index or indices being used
    
    Parameters
    ----------
    :param index: The values at which to calculate the mean
    :type index: :class:`pandas.Index` or :class:`pandas.MultiIndex`
    :param index_names: The name(s) of the index levels, where the first element should be used as x
    :type index_names: array-like `str`
    :param df: The DataFrame containing the data and errors to fit
    :type df: :class:`pandas.DataFrame`
    :param col_dict: A dictionary of the column names to use, where the data column should be key 'm' and the variances should be key 'var'
    :type col_dict: `dict`
    
    Returns
    -------
    :return init_guess: An initial guess for the parameters a and alpha
    :rtype init_guess: 1D array float
    :return x0: Pivot for x
    :rtype pivot: `float`
    """
    twod = True
    try:
        x = index.get_level_values(index_names[0]).values
    except AttributeError:
        twod = False
        x = index.values
    unique_x = np.unique(x)
    x0 = np.median(unique_x)
    nx = unique_x.size
    nx_odd = (nx%2 == 1)
    if nx_odd:
        x_around_mid = [unique_x[int((nx - 1) / 2) - 1], unique_x[int((nx - 1) / 2) + 1]]
        if twod:
            a = df[col_dict["m"]].loc[(x0, slice(None))].mean()
        else:
            a = df[col_dict["m"]].loc[x0]
    else:
        x_around_mid = [unique_x[int(nx / 2) - 1], unique_x[int(nx / 2) + 1]]
        if twod:
            a = np.mean([df[col_dict["m"]].loc[(x_around_mid[0], slice(None))].mean(), df[col_dict["m"]].loc[(x_around_mid[1], slice(None))].mean()])
        else:
            a = df[col_dict["m"]].loc[x_around_mid].mean()
    if twod:
        alpha = (df[col_dict["m"]].loc[(x_around_mid[1], slice(None))] - df[col_dict["m"]].loc[(x_around_mid[0], slice(None))]).mean() / (x_around_mid[1] - x_around_mid[0])
    else:
        alpha = (df[col_dict["m"]].loc[x_around_mid[1]] - df[col_dict["m"]].loc[x_around_mid[0]]) / (x_around_mid[1] - x_around_mid[0])
    init_guess = np.array([a, alpha])
    return init_guess, x0       


def fit_mean_perp(
