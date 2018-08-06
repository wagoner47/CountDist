from __future__ import print_function
import os
import countdist2
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from chainconsumer import ChainConsumer
import numpy as np
from scipy.special import kstat, kstatvar

plt.rcParams["font.size"] = 20.0
plt.rcParams["figure.facecolor"] = "w"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"].insert(0, "cm")
plt.rcParams["text.usetex"] = True


def silverman_bin_width(data):
    """Get the bin width using Silverman's rule of thumb:
    
    .. math::
       
       h \approx 1.06 \hat{\sigma} n^{-1/5}
    
    Parameters
    ----------
    :param data: The data for which to get a bin width
    :type data: 1D array-like
    
    Returns
    -------
    :return: The estimated bin width
    :rtype: float
    """
    return 1.06 * np.sqrt(kstat(data, 2)) * float(len(data))**-0.2


def get_hist_xlims(grouped_data, bin_widths=None):
    """Get the x limits such that the groups within the given data can be histogrammed on the same axes (or in an animation). This is meant primarily to be called by :function:`get_hist_xylims`
    
    Parameters
    ----------
    :param grouped_data: The data to be histogrammed, already grouped. Grouped data should be a Series rather than a DataFrame
    :type grouped_data: :class:`pandas.GroupBy`
    :param bin_widths: The widths to use in the histogram. If `None`, will be set via Silverman's rule of thumb. Otherwise, should be a Series indexed by the group keys of :param:`grouped_data`. Default `None`
    :type bin_widths: :class:`pandas.Series`
    
    Returns
    -------
    :return xlim: The limits in x such that all groups may be plot together
    :rtype xlim: list
    :return bin_widths: The bin widths, especially if calculated via Silverman's rule of thumb
    :rtype bin_widths: :class:`pandas.Series`
    """
    if bin_widths is None:
        # This gets the widths for each group
        bin_widths = grouped_data.agg(silverman_bin_width)
    # A container for the bin minima and maxima
    bin_groups = pd.DataFrame(index=bin_widths.index, columns=["min", "max"])
    for name, group in grouped_data:
        h = bin_widths.loc[name]
        bin_groups["min"].loc[name] = np.floor(group / h).min() * h
        bin_groups["max"].loc[name] = (np.floor(group / h).max() + 1.0) * h
    xlim = [bin_groups["min"].min(), bin_groups["max"].max()]
    return xlim, bin_widths


def get_hist_ylims(grouped_data, xlim, bin_widths):
    """Get the y limits such that the groups within the given data can be histogrammed on the same axes (or in an animation). This is meant primarily to be called by :function:`get_hist_xylims`
    
    Parameters
    ----------
    :param grouped_data: The data to be histogrammed, already grouped. Grouped data should be a Series rather than a DataFrame
    :type grouped_data: :class:`pandas.GroupBy`
    :param xlim: The limits in the x-direction (same for each group)
    :type xlim: 1D array-like
    :param bin_widths: The widths to use in the histogram
    :type bin_widths: :class:`pandas.Series`
    
    Returns
    -------
    :return ylim: The limits in y such that all groups may be plot together
    :rtype ylim: list
    """
    # A container to hold this histogram minima and maxima
    hist_groups = pd.DataFrame(index=bin_widths.index, columns=["min", "max"])
    for name, group in grouped_data:
        h = bin_widths.loc[name]
        bins = np.arange(xlim[0], xlim[1], h)
        if bins[-1] < xlim[1]:
            bins = np.append(bins, xlim[1])
        bin_area = np.array([u - l for l, u in zip(bins[:-1], bins[1:])])
        counts = np.histogram(group, bins=bins)[0]
        hist = counts / (counts * bin_area).sum()
        errs = np.sqrt(counts * (1 - (counts / counts.sum()))) / np.sqrt((counts * bin_area).sum())
        hist_groups["min"].loc[name] = (hist - errs).min()
        hist_groups["max"].loc[name] = (hist + errs).max()
    ylim = [hist_groups["min"].min(), hist_groups["max"].max()]
    return ylim


def get_hist_xylims(grouped_data, bin_widths=None):
    """Get the x and y limits such that the groups within the given data can be histogrammed on the same axes (or in an animation)
    
    Parameters
    ----------
    :param grouped_data: The data to be histogrammed, already grouped. Grouped data should be a Series rather than a DataFrame
    :type grouped_data: :class:`pandas.GroupBy`
    :param bin_widths: The widths to use in the histogram. If `None`, will be set via Silverman's rule of thumb. Otherwise, should be a Series indexed by the group keys of :param:`grouped_data`. Default `None`
    :type bin_widths: :class:`pandas.Series`
    
    Returns
    -------
    :return xlim: The limits in x such that all groups may be plot together
    :rtype xlim: list
    :return ylim: The limits in y such that all groups may be plot together
    :rtype ylim: list
    :return bin_widths: The bin widths, especially if calculated via Silverman's rule of thumb
    :rtype bin_widths: :class:`pandas.Series`
    """
    xlim, bin_widths = get_hist_xlims(grouped_data, bin_widths)
    ylim = get_hist_ylims(grouped_data, xlim, bin_widths)
    return xlim, ylim, bin_widths


def plot_hist_pair(grouped_data, grouped_by
