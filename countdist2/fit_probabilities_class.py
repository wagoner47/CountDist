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
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer


plt.rcParams["font.size"] = 20.0


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


class GeneralFitter(object):
    """A class containing some of the general functions to be used for the
    fitting pipeline
    """
    def __init__(self, df, index_names, col_dict, func, nwalkers, nsteps,
                 nburnin, paramater_names, other_args=None, nthreads=None, state=None,
                 fit_type="dd", fit_str="fit", fit_save_dir=None,
                 make_plots=True,
                 show_plots=True, plt_save_dir=None):
        """Initialize the fitter

        Parameters
        ----------
        :param df: The DataFrame containing the data and errors to fit
        :type df: :class:`pandas.DataFrame`
        :param index_names: The name(s) of the index levels, where the first
        element should be the index name of the first variable in the fitting
        function, the second should be the second variable, etc.
        :type index_names: 1D array-like `str`
        :param col_dict: A dictionary of the column names to use, where the data
        column should be key 'm' and the variances should be key 'var'
        :type col_dict: `dict`
        :param func: The function to fit to the data. The function must take
        an index (or multi-index) as the first argument, an array-like of
        index names as the second argument, and parameters as the last
        arguments, with remaining positional arguments in the middle.
        :type func: callable
        :param nwalkers: The number of walkers to call the sampler with
        :type nwalkers: `int`
        :param nsteps: The number of steps to run that are kept. Note the total
        number of steps will be nsteps + nburnin
        :type nsteps: `int`
        :param nburnin: The number of steps to use as a burnin. Note the total
        number of steps will be nsteps + nburnin
        :type nburnin: `int`
        :param parameter_names: The names of the parameters (including
        formatting marks for plots). This will be used to set the number of
        dimensions
        :type parameter_names: 1D array-like `str`
        :param other_args: The other positional arguments to be passed to
        :param:`func`, if any. If `None`, the code assumes there are no other
        arguments. Defualt `None`
        :type other_args: `tuple`
        :param nthreads: The number of threads to use for multi-threaded emcee.
        If `None`, multi-threading not used. Default `None`
        :type nthreads: `int`
        :param state: The random state to set for the random number
        generator. If `None`, the internal random state is seeded by 0.
        Default `None`
        :type state: `tuple(str, ndarray of 624 uints, int, int, float)`
        :param fit_type: The type of the fit to be done, with choices of
        'dd', 'rr', or 'dr' (not case-sensitive). Default 'dd'
        :type fit_type: `str`
        :param fit_str: A string to represent what the fit is (e.g.
        'delta_r_parallel', 'r_perp_t', etc.) for file naming. Default 'fit'
        :type fit_str: `str`
        :param fit_save_dir: The directory in which to save the fit samples
        and likelihoods. If the burn-in is changed, these files will be
        overwritten. If `None`, the current working directory is used.
        Default `None`
        :type fit_save_dir: `str`
        :param make_plots: Whether to make plots for the fits. Default `True`
        :type make_plots: `bool`
        :param show_plots: Whether to show the plots (only if
        :param:`make_plots` is `True`). Default `True`
        :type show_plots: `bool`
        :param plt_save_dir: The directory in which to save the fit plots,
        if they are made. New plots will be saved (if making plots) if the
        burn-in is changed. If `None`, the current working directory is used. Default `None`
        :type plt_save_dir: `str`
        """
        self.parameter_names = parameter_names
        self.ndim = len(paramater_names)
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.nburnin = nburnin
        if state is not None:
            np.random.set_state(state)
        else:
            np.random.seed(0)
        self.data = df[col_dict["m"]]
        self.data_var = df[col_dict["var"]]
        if other_args is not None:
            self.func = lambda params: func(df.index, index_names,
                                            *other_args, *params)
        else:
            self.func = lambda params: func(df.index, index_names, *params)
        if nthreads is not None:
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                                 self._lnprob, threads=nthreads)
        else:
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprob)
        if not fit_type.lower() in ["dd", "rr", "dr"]:
            raise ValueError("Invalid fit type: {}".format(fit_type))
        self.fit_type = fit_type.lower()
        self.file_base = "{}_{}".format(self.fit_type, fit_str)
        if fit_save_dir is None:
            self.fit_save_dir = os.getcwd()
        else:
            self.fit_save_dir = os.path.normpath(fit_save_dir)
        self.make_plots = make_plots
        self.show_plots = show_plots
        if plt_save_dir is None:
            self.plt_save_dir = os.getcwd()
        else:
            self.plt_save_dir = os.path.normpath(plt_save_dir)


    def _lnlike(self, theta):
        """General likelihood function assuming a diagonal covariance matrix

        Parameters
        ----------
        :param theta: The parameters at which to evaluate the log-likelihood
        :type theta: 1D array-like

        Returns
        -------
        :return: The log-likelihood of the model (for this class instance)
        given the parameter values
        :rtype: float
        """
        f = self.func(theta)
        return -0.5 * self.data.sub(f).pow(2).div(self.data_var).sum()


    def _lnprior(self, theta):
        """A general function for priors in the fitting. This prior is
        basically the same as having no prior, or a flat prior over all
        values of all parameters. This should be updated in subclasses when
        needed

        Parameters
        ----------
        :param theta: The parameters at which to evaluate the log-prior
        :type theta: 1D array-like

        Returns
        -------
        :return: The log-prior of the parameter values
        :rtype: float
        """
        return 0.0


    def _lnprob(self, theta):
        """General probability function assuming a diagonal covariance
        matrix, including a prior set for the class

        Parameters
        ----------
        :param theta: The parameters at which to evaluate the log-probability
        :type theta: 1D array-like

        Returns
        -------
        :return: The log-probability of the model (for this class instance)
        given the parameter values
        :rtype: float
        """
        lp = self._lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._lnlike(theta)


    def _fit_first_pass(self, init_guess):
        """Do a first pass fitting of the model to the data using
        scipy.optimize.minimize

        Parameters
        ----------
        :param init_guess: An initial guess for the parameters
        :type init_guess: 1D array-like `float`

        Returns
        -------
        :return res.x: The best fit according to scipy.optimize.minimize
        :rtype res.x: 1D ndarray `float`
        """
        nll = lambda theta: -self._lnlike(theta)
        res = minimize(nll, init_guess)
        return res.x


    def _fit_mcmc(self, init_guess):
        """Run an MCMC on the data for the model

        Parameters
        ----------
        :param init_guess: An initial guess for the parameters
        :type init_guess: 1D array-like `float`
        """
        pos = [init_guess + 10.**(-4 + ndigits(init_guess)) *
               np.random.randn(self.ndim) for i in range(self.nwalkers)]
        self.sampler.run_mcmc(pos, nsteps + nburnin);
        self.all_lnprobs = pd.DataFrame(self.sampler.lnprobability.T,
                                        index=pd.Index(np.arange(self.nsteps
                                                                 +
                                                                 self.nburnin, dtype=int), name="Step"), columns=pd.Index(np.arange(self.nwalkers, dtype=int), name="Walker"))
        self.lnprobs = pd.Series(self.sampler.lnprobability[
                                 :,nburnin:].flatten())
        self.samples = pd.DataFrame(self.sampler.samples[:,nburnin:,
                                    :].reshape((-1, self.ndim)),
                                    columns=self.parameter_names)
        self.all_lnprobs.to_pickle(os.path.join(self.fit_save_dir,
                                                self.file_base +
                                                "full_probs.pickle.gz"))
        self.lnprobs.to_pickle(os.path.join(self.fit_save_dir,
                                                self.file_base +
                                                "flat_probs.pickle.gz"))
        self.samples.to_pickle(os.path.join(self.fit_save_dir, self.file_base
                                            + "flat_samples.pickle.gz"))


    def configure_plots(self, **kwargs):
        plt.rc(**kwargs)


    def configure_chain_consumer(self, *args, **kwargs):
        self.cc_args = args
        self.cc_kwargs = kwargs


    def run_fit(self, init_guess):
        """An automated function for running both an initial pass fit as well as an MCMC

        Parameters
        ----------
        :param init_guess: An initial guess for the parameters
        :type init_guess: 1D array-like `float`
        """
        start = self._fit_first_pass(init_guess)
        self._fit_mcmc(start)
        if self.make_plots:
            c_full = ChainConsumer()
            [c_full.add_chain(sample, parameters=self.parameter_names,
                              posterior=prob) for sample, prob in zip(
                self.sampler.samples, self.sampler.lnprobability)];
            c_full.configure(*self.cc_args, **self.cc_kwargs)
            c_full.plotter.plot_walks(display=self.show_plots,
                                      filename=os.path.join(
                                          self.plt_save_dir, self.file_base +
                                                             "_walks.png"))
            c = ChainConsumer()
            c.add_chain(self.samples.values, parameters=self.parameter_names,
                        posterior=self.lnprobs.values)
            c.configure(*self.cc_args, **self.cc_kwargs)
            c.plotter.plot(display=self.show_plots, filtname=os.path.join(
                self.plt_save_dir, self.file_base + "_corner.png"))


    def change_nburnin(self, nburnin, init_guess):
        """Re-run the MCMC with a new burn-in phase (in case previous burn-in was not long enough, for instance)

        Parameters
        ----------
        :param nburnin: The new burn-in length to use
        :type nburnin: `int`
        :param init_guess: The initial guess for the parameters
        :type init_guess: 1D array-like `float`
        """
        self.nburnin = nburnin
        self.run_fit(init_guess)


    def get_mle_params(self):
        """Get the best fit parameters and their 68% confidence limits for
        the current fit

        Returns
        -------
        :return mle_params: The maximum likelihood parameter estimates and
        their 68% confidence levels
        :rtype mle_params: :class:`pandas.DataFrame`
        """
        mle_params = self.samples.quantile([0.16, 0.5, 0.84]).T
        mle_params[0.16] = mle_params[0.5] - mle_params[0.16]
        mle_params[0.84] = mle_params[0.84] - mle_params[0.5]
        return mle_params


    def get_fit_intervals(self):
        """Get the mean, 68%, and 95% confidence levels for the best fit
        function. Note that here, the confidence regions are not the values
        to be added to/subtracted from the mean

        Returns
        -------
        :return func_quantiles: The mean and the lower and upper limits for
        the 68% and 95% quantiles of the fitting function, indexed by the
        quantile
        :rtype func_quantiles: :class:`pandas.Series`
        """
        func_samples = pd.Series(data=np.array([self.func(theta) for theta in
                                  self.samples.values]))
        func_quantiles = func_samples.quantile([0.025, 0.16, 0.5, 0.84,
                                                0.975]).T
        return func_quantiles


    def plot_fit(self, xlabel, ylabel, title=None, bins_to_plot=None):
        """Plot the best fit of the model with the data in a few bins

        Parameters
        ----------
        :param xlabel: The label to use on the x-axis
        :type xlabel: `str`
        :param ylabel: The label to use on the y-axis
        :type ylabel: `str`
        :param title: The title(s) to put on the plot or the subpanels of the plot (should be related to the bin). Default `None`
        :type title: `str` or 1D array-like `str`
        :param bins_to_plot: Argument for pandas loc to select which bins to
        plot if the dataframe has a multiindex. If `None`, the dataframe
        should only have a single index. Default `None`
        :type bins_to_plot: `tuple` or 1D array-like of `tuple`
        """
        func_quantiles = self.get_fit_intervals()
        if bins_to_plot is not None:
            if len(bins_to_plot) == 2 and not hasattr(bins_to_plot[0],
                                                      "__len__"):
                # Case: single slice panel
                nrows = 2
                ncols = 1
                x_level = np.where(np.isin(bins_to_plot, slice(None),
                                           invert=True))[0][0]
                data_plot = self.data.loc[bins_to_plot]
                data_err_plot = self.data_var.loc[bins_to_plot].apply(np.sqrt)
                func_plot = func_quantiles.loc(axis=0)[bins_to_plot]
                x = data_plot.index.get_level_values(x_level)
            else:
                # Case: multiple slices panel
                nrows = 2
                ncols = len(bins_to_plot)
                x_level = np.where(np.isin(bins_to_plot[0], slice(None),
                                           invert=True))[0][0]
                data_plot = self.data.loc[bins_to_plot]
                data_err_plot = self.data_var.loc[bins_to_plot].apply(np.sqrt)
                x = data_plot.index.get_level_values(x_level).unique()
        else:
            # Case: single index level
            nrows = 2
            ncols = 1
            data_plot = self.data
            data_err_plot = self.data_var.apply(np.sqrt)
            func_plot = func_quantiles
            x = data_plot.index

        chi = data_plot.sub(func_plot[0.5]).div(data_err_plot)
        xlim = [x.min() - 10.**(ndigits(x.min()) - 1), x.max() + 10.**(
            ndigits(x.max()) - 1)]
        ymin = min(data_plot.sub(data_err_plot).min(), func_plot[0.025].min())
        ymax = max(data_plot.add(data_err_plot).max(), func_plot[0.975].max())
        ylim = [ymin - 10.**(ndigits(ymin) - 1), ymax + 10.**(ndigits(ymax) +
                                                              1)]
        yclim = [chi.min() - 10.**(ndigits(chi.min()) - 1), chi.max() +
                 10.**(ndigits(chi.max()) - 1)]

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex="column",
                               sharey="row", gridspec_kw={"height_ratios": [
                3,1], "left": 0.2, "wspace": 0.0, "hspace": 0.0, "bottom": 0.2})
        ax[0,0].set_ylabel(ylabel)
        ax[1,0].set_ylabel(r"$\chi$")
        ax[0,0].set_ylim(ylim)
        ax[1,0].set_ylim(yclim)
        [axi.set_xlabel(xlabel) for axi in ax[1,:]]
        [axi.set_xlim(xlim) for axi in ax[1,:]]
        if ncols > 1:
            for i, bini in enumerate(bins_to_plot):
                if title is not None:
                    ax[0,i].set_title(title[i])
                ax[0,i].errorbar(x, data_plot.loc[bini],
                                 yerr=data_err_plot.loc[bini], fmt="C0o",
                                 alpha=0.6)
                ax[0,i].fill_between(x, func_plot[0.025].loc[bini], func_plot[
                    0.975].loc[bini], "C1", alpha=0.4)
                ax[0,i].fill_between(x, func_plot[0.16].loc[bini], func_plot[
                    0.84].loc[bini], "C1", alpha=0.5)
                ax[0,i].plot(x, func_plot[0.5].loc[bini], "C1-")
                ax[1,i].plot(x, chi.loc[bini], "C0o", alpha=0.6)
        else:
            if title is not None:
                ax[0,0].set_title(title)
                ax[0,0].errorbar(x, data_plot, yerr=data_err_plot, fmt="C0o",
                                 alpha=0.6)
                ax[0,0].fill_between(x, func_plot[0.025], func_plot[0.975],
                                     "C1", alpha=0.4)
                ax[0,0].fill_between(x, func_plot[0.16], func_plot[0.84],
                                     "C1", alpha=0.5)
                ax[0,0].plot(x, func_plot[0.5], "C1-")
                ax[1,0].plot(x, chi, "C0o", alpha=0.6)
        fig.savefig(os.path.join(self.plt_save_dir, self.file_base +
                                 "_fit_panels.png"))
        if self.show_plots:
            plt.show()
        plt.close()