from __future__ import print_function
from astropy.table import Table
import subprocess
import os, sys
from .utils import MyConfigObj, init_logger, _initialize_cosmology
import logging
import calculate_distances as _calculate_distances
import math
import pandas as pd
import numpy as np
import astropy.cosmology
import CatalogUtils
import time
from datetime import timedelta
from scipy import sparse


def _read_catalog(file_name, has_true, has_obs, use_true, use_obs, dtcol=None,
                  docol=None, ztcol=None, zocol=None, read_col_names=None):
    logger = init_logger(__name__)
    file_name = pathlib.Path(file_name).resolve()
    is_fits = "fit" in file_name.suffix
    logger.debug("read data")
    if not is_fits:
        if read_col_names is None:
            raise ValueError("Must give column names in catalog file if file"
                             " type is ASCII")
        data = Table.read(file_name, format="ascii", names=read_col_names)
    else:
        data = Table.read(file_name)
    logger.debug("remove extra columns")
    names = ["RA", "DEC"]
    if use_true:
        if not has_true:
            raise ValueError("Cannot use true redshifts/distances if catalog"
                             " does not have true redshifts/distances")
        if dtcol is None:
            raise ValueError("Cannot use true redshifts/distances if true"
                             " distance column name is not provided")
        if dtcol not in data.colnames:
            raise ValueError("Missing column {} for true distances in catalog"
                             " read from file {}".format(dtcol, str(file_name)))
        print("Using column {} for true distances".format(dtcol))
        names.append(dtcol)
        if ztcol is None:
            ztcol = dtcol.replace("D", "Z")
        if ztcol not in data.colnames:
            raise ValueError("Must provide true redshift column name if"
                             " using true distances and redshift column is"
                             " not tagged the same as distance column")
        names.append(ztcol)
    if use_obs:
        if not has_obs:
            raise ValueError("Cannot use observed redshifts/distances if"
                             " catalog does not contain them")
        if docol is None:
            raise ValueError("Cannot use observed redshifts/distances if"
                             " observed distance column name is not provided")
        if docol not in data.colnames:
            raise ValueError("Missing column {} for observed distances in"
                             " catalog read from file {}".format(
                                 docol, str(file_name)))
        names.append(docol)
        if zocol is None:
            zocol = docol.replace("D", "Z")
        if zocol not in data.colnames:
            raise ValueError("Must provide observed redshift column name if"
                             " using observed distances and redshift column is"
                             " not tagged the same as distance column")
        names.append(zocol)
    data.keep_columns(names)
    logger.debug("replace unneeded columns or rename kept columns")
    nan_col = np.full(len(data), np.nan)
    if not use_true:
        data["D_TRUE"] = nan_col
        data["Z_TRUE"] = nan_col
    else:
        data.rename_column(dtcol, "D_TRUE")
        data.rename_column(ztcol, "Z_TRUE")
    if not use_obs:
        data["D_OBS"] = nan_col
        data["Z_OBS"] = nan_col
    else:
        data.rename_column(docol, "D_OBS")
        data.rename_column(zocol, "Z_OBS")
    return np.array(data)


def calculate_separations(perp_binner, par_binner, use_true, use_obs, cat1,
                          cat2=None, as_type="dataframe"):
    """
    Calculate the separations for all pairs within limits.

    Call the routine for calculating separations. Use the bin specifications
    given to provide the minimum and maximum observed perpendicular and parallel
    separations. If :param:`cat2` is `None`, call the routine with the flag
    for auto-correlating set to `True`.

    Parameters
    ----------
    :param perp_binner: The bin specifications in the perpendicular direction.
    Only the bin minimum and maximum need be set.
    :type perp_binner: :class:`BinSpecification`
    :param par_binner: The bin specifications in the parallel direction.
    Only the bin minimum and maximum need be set.
    :type par_binner: :class:`BinSpecification`
    :param use_true: Flag specifying that true separations should be calculated
    :type use_true: `bool`
    :param use_obs: Flag specifying that observed separations should be
    calculated
    :type use_obs: `bool`
    :param cat1: The first (or only for auto-correlating) catalog to use in
    calculating separations. It should have either of the two dtypes listed,
    although NaN values can be used for either the true or the observed
    distances and redshifts if calculating only true or only observed
    separations.
    :type cat1: :class:`astropy.table.Table`, :class:`astropy.table.QTable`,
    :class:`pandas.DataFrame`, or :class:`numpy.ndarray`, dtype={
    [('RA', float), ('DEC', float), ('D_TRUE', float), ('D_OBS', float),
    ('Z_TRUE', float), ('Z_OBS', float)], [('nx', float), ('ny', float),
    ('nz', float), ('D_TRUE', float), ('D_OBS', float), ('Z_TRUE', float),
    ('Z_OBS', float)]}
    :param cat2: The second catalog to use in calculating separations, if not
    calculating for auto-correlation. Please note that providing the same
    catalog will result in double counting, as no checking is done to verify
    that the catalogs are different. Default `None`
    :type cat2: same as :param:`cat1` or `NoneType`, optional
    :param as_type: What type of object to return, of :class:`pandas.DataFrame`
    ('dataframe'), :class:`astropy.table.Table` ('table'), or structured
    :class:`numpy.ndarray` ('array'). Default 'dataframe'
    :type as_type: `str` {'dataframe', 'table', 'array'}, optional
    (case-insensitive)

    Returns
    -------
    :return seps: The separations calculated, converted to the appropriate type.
    The dtype of the return is [('R_PERP_T', float), ('R_PAR_T', float),
    ('R_PERP_O', float), ('R_PAR_O', float), ('AVE_Z_OBS', float),
    ('ID1', uint), ('ID2', uint)]. Note that columns 'R_PERP_T' and 'R_PAR_T'
    will be NaN if :param:`use_true` is `False`, and columns 'R_PERP_O',
    'R_PAR_O', and 'AVE_Z_OBS' will be NaN if :param:`use_obs` is `False`
    :rtype seps: :class:`pandas.DataFrame` (:param:`as_type`='dataframe',
    default), :class:`astropy.table.Table` (:param:`as_type`='table'), or
    :class:`numpy.ndarray` (:param:`as_type`='array')
    """
    logger = init_logger(__name__)
    if as_type.lower() not in ['dataframe', 'table', 'array']:
        raise ValueError("Invalid return type ({}): valid return type"
                         " specifiers are {'dataframe', 'table',"
                         " 'array'}".format(as_type))
    as_type = as_type.lower()
    logger.debug("convert cat1")
    if isinstance(cat1, pd.DataFrame):
        cat1 = cat1.to_records(index=False)
    else:
        cat1 = np.asarray(cat1)
    if cat2 is None:
        logger.debug("run auto-correlation for separations")
        seps = _calculate_distances.get_separations(
            cat1, cat1, perp_binner, par_binner, use_true, use_obs, True)
    else:
        logger.debug("convert cat2")
        if isinstance(cat2, pd.DataFrame):
            cat2 = cat2.to_records(index=False)
        else:
            cat2 = np.asarray(cat2)
        logger.debug("run cross-correlation for separations")
        seps = _calculate_distances.get_separations(
            cat1, cat2, perp_binner, par_binner, use_true, use_obs, False)
    logger.debug("convert separations")
    if as_type == 'dataframe':
        seps = pd.DataFrame.from_records(seps)
    elif as_type == 'table':
        seps = Table(seps)
    else:
        pass
    return seps


def _get_distance_redshif_colnames(params_file, *, dtkey, dokey, ztkey, zokey):
    try:
        dtcol = params_in[dtkey]
    except KeyError:
        dtcol = None
    try:
        docol = params_in[dokey]
    except KeyError:
        docol = None
    try:
        ztcol = params_in[ztkey]
    except KeyError:
        ztcol = None
    try:
        zocol = params_in[zocol]
    except KeyError:
        zocol = None
    return dtcol, docol, ztcol, zocol


def calculate_separations_from_params(params_file, as_type="dataframe"):
    """
    Calculate the separations with parameters given in a parameter file.

    This function runs the executable for finding the separations between
    galaxies. The only input is the location of the parameter file, which will
    first be read in here. The input catalogs file will be checked for type
    metadata and will be converted to an ascii file n the same directory with
    a new paramter file being created to give the correct file. The data is
    then automatically stored in a database, and can be read in using the
    :function:`countdist2.read_db` or :function:`countdist2.read_db_multiple`
    functions. Please note that the catalogs should be FITS files, and the
    same catalog may safely be used for both inputs for an auto-correlation.

    Parameters
    ----------
    :param params_file: The parameter file to be used. Note that a temporary
    copy will be made with the appropriate input file readable by the
    executable, but the temporary file will be removed after the code has run.
    :type params_file: `str` or :class:`os.PathLike`
    :param as_type: The return type to use, of :class:`pandas.DataFrame`
    ('dataframe'), :class:`astropy.table.Table` ('table'), or structured
    :class:`numpy.ndarray` ('array'). Default 'dataframe'
    :type as_type: `str` {'dataframe', 'table', 'array'}, optional

    Returns
    -------
    :return seps: The separations calculated from the catalogs in the
    parameter file. The type is specified by :param:`as_type`, but the
    dtype is [('R_PERP_T', float), ('R_PAR_T', float), ('R_PERP_O', float),
    ('R_PAR_O', float), ('AVE_Z_OBS', float), ('ID1', uint), ('ID2', uint)].
    Please note that columns 'R_PERP_T' and 'R_PAR_T' will be NaN if not
    calculating true separations, and columns 'R_PERP_O', 'R_PAR_O', and
    'AVE_Z_OBS' will be NaN if not calculating observed separations
    :rtype seps: :class:`pandas.DataFrame` (:param:`as_type`='dataframe',
    default), :class:`astropy.table.Table` (:param:`as_type`='table'), or
    :class:`numpy.ndarray` (:param:`as_type`='array')
    """
    logger = init_logger(__name__)
    logger.debug("read parameter file")
    params_in = MyConfigObj(params_file, file_error=True)
    try:
        params_in = params_in["run_params"]
    except KeyError:
        pass
    dtcol1, docol1, ztcol1, zocol1 = _get_distance_redshift_colnames(
        params_in, dtkey="dtcol1", dokey="docol1", ztkey="ztcol1",
        zokey="zocol1")
    try:
        read_colnames1 = params_in.as_list("read_colnames1")
    except KeyError:
        read_colnames1 = None
    logger.debug("read cat1")
    cat1 = _read_catalog(
        params_in["ifname1"], params_in.as_bool("has_true1"),
        params_in.as_bool("has_obs1"), params_in.as_bool("use_true"),
        params_in.as_bool("use_obs"), dtcol1, docol1, ztcol1, zocol1,
        read_colnames1)
    if params_in["ifname2"] != params_in["ifname1"]:
        dtcol2, docol2, ztcol2, zocol2 = _get_distance_redshift_colnames(
            params_in, dtkey="dtcol2", dokey="docol2", ztkey="ztcol2",
            zokey="zocol2")
        try:
            read_colnames2 = params_in.as_list("read_colnames2")
        except KeyError:
            read_colnames2 = None
        logger.debug("read cat2")
        cat2 = _read_catalog(
            params_in["ifname2"], params_in.as_bool("has_true2"),
            params_in.as_bool("has_obs2"), params_in.as_bool("use_true"),
            params_in.as_bool("use_obs"), dtcol2, docol2, ztcol2, zocol2,
            read_colnames2)
    else:
        cat2 = None
    logger.debug("set up bin specifications")
    rpo_bins = _calculate_distances.BinSpecifier()
    rpo_bins.bin_min = params_in.as_float("rp_min")
    rpo_bins.bin_max = params_in.as_float("rp_max")
    rlo_bins = _calculate_distances.BinSpecifier()
    rlo_bins.bin_min = params_in.as_float("rl_min")
    rlo_bins.bin_max = params_in.as_float("rl_max")
    logger.debug("run calculation")
    seps = calculate_separations(
        rpo_bins, rlo_bins, params_in.as_bool("use_true"),
        params_in.as_bool("use_obs"), cat1, cat2, as_type)
    return seps


def get_observed_pair_counts(rpo_binner, rlo_binner, zbar_binner, cat1,
                             cat2=None):
    """
    Get the observed pair counts between the two catalogs

    Find the observed pair counts between :param:`cat1` and :param:`cat2`,
    or just within :param:`cat1` as an auto-correlation if :param:`cat2` is
    `None`. Please make sure that :param:`rpo_binner`, :param:`rlo_binner`,
    and :param:`zbar_binner` are fully set instances of :class:`BinSpecifier`

    Parameters
    ----------
    :param rpo_binner: The bin specifications in observed perpendicular
    separation
    :type rpo_binner: :class:`BinSpecifier`
    :param rlo_binner: The bin specificiations in observed parallel separation
    :type rlo_binner: :class:`BinSpecifier`
    :param zbar_binner: The bin specification in average observed redshift
    :type zbar_binner: :class:`BinSpecifier`
    :param cat1: The first (or only for auto-correlation) catalog to use for
    finding pairs. The catalog may have either of the dtypes listed below,
    although the columns for 'D_TRUE' and 'Z_TRUE' may be NaN
    :type cat1: :class:`astropy.table.Table`, :class:`astropy.table.QTable`,
    :class:`pandas.DataFrame`, or :class:`numpy.ndarray`, dtype={
    [('RA', float), ('DEC', float), ('D_TRUE', float), ('D_OBS', float),
    ('Z_TRUE', float), ('Z_OBS', float)], [('nx', float), ('ny', float),
    ('nz', float), ('D_TRUE', float), ('D_OBS', float), ('Z_TRUE', float),
    ('Z_OBS', float)]}
    :param cat2: The second catalog to use for finding pairs, if
    cross-correlating. Please note that providing the same catalog will result
    in double counting, as no checking is done to verify that the catalogs are
    different. Default `None`
    :type cat2: same as :param:`cat1` or `NoneType`, optional

    Returns
    -------
    :return: The counter object for this set of pair counts. The binning
    information can be recalled using calls to 'nn.*_bin_info' (r_perp, r_par,
    or zbar), the 3D array of counts can be obtained with 'nn.counts', and
    the total number of pairs processed is available via 'nn.n_tot'
    :rtype: :class:`NNCounts3D`
    """
    logger = init_logger(__name__)
    logger.debug("convert cat1")
    if isinstance(cat1, pd.DataFrame):
        cat1 = cat1.to_records(index=False)
    else:
        cat1 = np.asarray(cat1)
    if cat2 is None:
        logger.debug("run auto-correlation for pair counts")
        return _calculate_distances.get_obs_pair_counts(
            cat1, cat1, rpo_binner, rlo_binner, zbar_binner, True)
    logger.debug("convert cat2")
    if isinstance(cat2, pd.DataFrame):
        cat2 = cat2.to_records(index=False)
    else:
        cat2 = np.asarray(cat2)
    logger.debug("run cross-correlation for pair counts")
    return _calculate_distances.get_obs_pair_counts(
        cat1, cat2, rpo_binner, rlo_binner, zbar_binner, False)

def get_observed_pair_counts_from_params(params_file):
    """
    Get the observed pair counts with options given in parameter file

    This function gets the observed pair counts for the files specified in
    :param:`params_file`. Please see :func:`get_observed_pair_counts` for
    more details

    Parameters
    ----------
    :param params_file: The parameter file from which to get the other details
    for the pair counting
    :type params_file: `str` or :class:`os.PathLike`

    Returns
    -------
    :return: The counter object for this set of pair counts. The binning
    information can be recalled using calls to 'nn.*_bin_info' (r_perp, r_par,
    or zbar), the 3D array of counts can be obtained with 'nn.counts', and
    the total number of pairs processed is available via 'nn.n_tot'
    :rtype: :class:`NNCounts3D`
    """
    logger = init_logger(__name__)
    logger.debug("read parameter file")
    params_in = MyConfigObj(params_file, file_error=True)
    try:
        params_in = params_in["run_params"]
    except KeyError:
        pass
    dtcol1, docol1, ztcol1, zocol1 = _get_distance_redshift_colnames(
        params_in, dtkey="dtcol1", dokey="docol1", ztkey="ztcol1",
        zokey="zocol1")
    try:
        read_colnames1 = params_in.as_list("read_colnames1")
    except KeyError:
        read_colnames1 = None
    logger.debug("read cat1")
    cat1 = _read_catalog(
        params_in["ifname1"], params_in.as_bool("has_true1"),
        params_in.as_bool("has_obs1"), False, True, dtcol1, docol1,
        ztcol1, zocol1, read_colnames1)
    if params_in["ifname2"] != params_in["ifname1"]:
        dtcol2, docol2, ztcol2, zocol2 = _get_distance_redshift_colnames(
            params_in, dtkey="dtcol2", dokey="docol2", ztkey="ztcol2",
            zokey="zocol2")
        try:
            read_colnames2 = params_in.as_list("read_colnames2")
        except KeyError:
            read_colnames2 = None
        logger.debug("read cat2")
        cat2 = _read_catalog(
            params_in["ifname2"], params_in.as_bool("has_true2"),
            params_in.as_bool("has_obs2"), False, True, dtcol2, docol2,
            ztcol2, zocol2, read_colnames2)
    else:
        cat2 = None

    logger.debug("set up binning")
    if "nbins" in params_in["rpo_bins"]:
        rpo_bins = _calculate_distances.BinSpecifier(
            params_in["rpo_bins"].as_float("bin_min"),
            params_in["rpo_bins"].as_float("bin_max"),
            params_in["rpo_bins"].as_int("nbins"),
            params_in["rpo_bins"].as_bool("log_binning"))
    else:
        rpo_bins = _calculate_distances.BinSpecifier(
            params_in["rpo_bins"].as_float("bin_min"),
            params_in["rpo_bins"].as_float("bin_max"),
            params_in["rpo_bins"].as_float("bin_size"),
            params_in["rpo_bins"].as_bool("log_binning"))
    if "nbins" in params_in["rlo_bins"]:
        rlo_bins = _calculate_distances.BinSpecifier(
            params_in["rlo_bins"].as_float("bin_min"),
            params_in["rlo_bins"].as_float("bin_max"),
            params_in["rlo_bins"].as_int("nbins"),
            params_in["rlo_bins"].as_bool("log_binning"))
    else:
        rlo_bins = _calculate_distances.BinSpecifier(
            params_in["rlo_bins"].as_float("bin_min"),
            params_in["rlo_bins"].as_float("bin_max"),
            params_in["rlo_bins"].as_float("bin_size"),
            params_in["rlo_bins"].as_bool("log_binning"))
    if "nbins" in params_in["zbar_bins"]:
        zbar_bins = _calculate_distances.BinSpecifier(
            params_in["zbar_bins"].as_float("bin_min"),
            params_in["zbar_bins"].as_float("bin_max"),
            params_in["zbar_bins"].as_int("nbins"),
            params_in["zbar_bins"].as_bool("log_binning"))
    else:
        zbar_bins = _calculate_distances.BinSpecifier(
            params_in["zbar_bins"].as_float("bin_min"),
            params_in["zbar_bins"].as_float("bin_max"),
            params_in["zbar_bins"].as_float("bin_size"),
            params_in["zbar_bins"].as_bool("log_binning"))

    logger.debug("get pair counts")
    return get_observed_pair_counts(
        rpo_bins, rlo_bins, zbar_bins, cat1, cat2)


class ExpectedNNCounts2D(object):
    """
    This class stores the results of convolving an observed pair count with
    a probability of true separations given observed to get expected true pair
    counts.
    """
    def _on_bin_update(self):
        self._n_tot = 0
        self._n_real = 0
        try:
            self._x_max = self._perp_binner.nbins
        except (AttributeError, TypeError):
            self._x_max = 0
        try:
            self._y_max = self._par_binner.nbins
        except (AttributeError, TypeError):
            self._y_max = 0
        self._shape = (self._x_max, self._y_max)
        self._counts = []
        self._ave_counts = np.zeros(self._shape)
        self._cov_counts = np.zeros(
            (self._x_max, self._y_max, self._x_max, self._y_max))

    def _update_counts(self):
        self._n_real = len(self._counts)
        self._ave_counts = (
            np.sum([c.tocsr() for c in self._counts]).toarray() / self._n_real)
        if self._n_real > 1:
            diff = [c.toarray() - self._ave_counts for c in self._counts]
            self._cov_counts = (
                np.einsum("aij,akl->ijkl", diff, diff)
                / (self._n_real * (self._n_real - 1)))

    def _append_counts(self, new_counts):
        try:
            self._counts.extend(new_counts)
        except TypeError:
            self._counts.append(new_counts)
        self._update_counts()

    def _add_count(self, x_index, y_index, add_count=1):
        x_index = np.atleast_1d(x_index).flatten()
        y_index = np.atleast_1d(y_index).flatten()
        if x_index.size != y_index.size:
            raise ValueError("Cannot add counts with mismatching index sizes")
        if not hasattr(add_count, "__len__"):
            add_count = np.array([add_count] * x_index.size)
        if len(add_count) != x_index.size:
            raise ValueError("Cannot add different number of counts than"
                             " number of indices provided")
        self._counts.append(
            sparse.coo_matrix(
                (add_count, (x_index, y_index)), shape=self._shape))
        self._update_counts()

    def __init__(self, other=None, *, perp_binning=None, par_binning=None):
        """
        There are two different constructors that can be used. They cannot be
        used simultaneously. There is also an empty constructor possible if
        nothing is given. The other two signatures are listed below.

        Copy constructor call signature:
        :code:`ExpectedNNCounts2D(other : ExpectedNNCounts2D)`

        Parameters
        ----------
        :param other: Another instance of :class:`ExpectedNNCounts2D` to
        create a copy of
        :type other: :class:`ExpectedNNCounts2D`

        Basic constructor call signature:
        :code:`ExpectedNNCounts2D
        (perp_binning : BinSpecifier, par_binning : BinSpecifier)`

        Parameters
        ----------
        :kwarg perp_binning: The binning scheme in perpendicular separations
        to use in the pair counting
        :type perp_binning: :class:`BinSpecifier`
        :kwarg par_binning: The binning scheme in parallel separations to use
        in the pair counting
        :type par_binning: :class:`BinSpecifier`
        """
        if (other is not None and
            (perp_binning is not None and par_binning is not None)):
            raise ValueError("ExpectedNNCounts2D copy constructor and basic"
                             " constructor are mutually exclusive")
        if other is not None:
            self._perp_binner = other._perp_binner
            self._par_binner = other._par_binner
            self._on_bin_update()
            self._n_tot = other._n_tot
            self._update_counts(other._counts)
        elif perp_binning is not None or par_binning is not None:
            if perp_binning is None or par_binning is None:
                raise ValueError("ExpectedNNCounts2D basic constructor requires"
                                 " both perpendicular and parallel binning")
            self._perp_binner = perp_binning
            self._par_binner = par_binning
            self._on_bin_update()
        else:
            self._perp_binner = None
            self._par_binner = None
            self._on_bin_update()

    def __repr__(self):
        return ("ExpectedNNCounts1D(bins={:r})".format(self._binner))

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._on_bin_update()
        self._counts = state["_counts"]
        self._n_tot = state["_n_tot"]
        self._n_real = state["_n_real"]

    @property
    def perp_bins(self):
        """
        The bin specifications for separations in the perpendicular direction
        """
        return _calculate_distances.BinSpecifier(self._perp_binner)

    @property
    def par_bins(self):
        """
        The bin specifications for separations in the parallel direction
        """
        return _calculate_distances.BinSpecifier(self._par_binner)

    @property
    def n_tot(self):
        """
        Total number of pairs processed in getting the pair counts (per
        realization)
        """
        return self._n_tot

    @property
    def n_real(self):
        """
        The number of realizations contributing to the average
        """
        return self._n_real

    @property
    def counts(self):
        """
        Current 2D array of pair counts
        """
        return self._ave_counts

    @property
    def normalized_counts(self):
        """
        Get the pair counts normalized by the total number of pairs
        """
        n_tot = max(1, self._n_tot)
        return self._ave_counts / n_tot

    @property
    def variance(self):
        """
        Calculate the covariance of the counts. All zero if n_real < 2
        """
        return self._cov_counts

    @property
    def normalized_variance(self):
        """
        Get the covariance of the normalized counts. All zero if n_real < 2
        """
        n_tot = max(1, self._n_tot)
        return self._cov_counts / n_tot**2

    def update_perp_binning(self, new_binning, prefer_old=True):
        """
        Update the perpendicular separation binning scheme. Note that this
        deletes any currently stored counts or total.

        Parameters
        ----------
        :param new_binning: The new binning specification to use
        :type new_binning: :class:`BinSpecifier`
        :param prefer_old: Whether to prefer the previous bin specifications
        for parameters that are set in both the current and new binning
        schemes. Default `True`
        :type prefer_old: `bool`, optional
        """
        if prefer_old:
            self._perp_binner.fill(new_binning)
        else:
            self._perp_binner.update(new_binning)
        self._on_bin_update()

    def update_par_binning(self, new_binning, prefer_old=True):
        """
        Update the parallel separation binning scheme. Note that this
        deletes any currently stored counts or total.

        Parameters
        ----------
        :param new_binning: The new binning specification to use
        :type new_binning: :class:`BinSpecifier`
        :param prefer_old: Whether to prefer the previous bin specifications
        for parameters that are set in both the current and new binning
        schemes. Default `True`
        :type prefer_old: `bool`, optional
        """
        if prefer_old:
            self._par_binner.fill(new_binning)
        else:
            self._par_binner.update(new_binning)
        self._on_bin_update()

    def get_bin(self, r_perp, r_par):
        """
        Get the bin index corresponding to the given separation components

        Parameters
        ----------
        :param r_perp: The perpendicular component of the separation(s)
        :type r_perp: scalar or array-like `float`
        :param r_par: The parallel component of the separation(s)
        :type r_par: scalar or array-like `float`

        Returns
        -------
        :return: The index/indices for each separation, with -1
        corresponding to separations outside of the bins in a dimension. Each
        element is a tuple
        :rtype: tuple or ndarray of tuple of `int`
        """
        perp_bin = self._perp_binner.assign_bin(r_perp)
        par_bin = self._par_binner.assign_bin(r_par)
        return np.array(list(zip(perp_bin, par_bin)))

    def assign_bins(self, r_perp, r_par):
        """
        Assign bins to an entier realization

        This calls the :func:`assign_bin` method, but also increments the n_real
        counter to indicate another realization has been accumulated

        Parameters
        ----------
        :param r_perp: The perpendendicular separations to add for this
        realization
        :type r_perp: array-like `float`
        :param r_par: The parallel separations to add for this realization
        :type r_par: array-like `float`
        """
        if not hasattr(r_perp, "__len__"):
            raise TypeError("Non-array-like r_perp is non-sensical for an"
                            " entire realization")
        if not hasattr(r_par, "__len__"):
            raise TypeError("Non-array-like r_par is non-sensical for an"
                            " entire realization")
        if len(r_perp) != len(r_par):
            raise ValueError("Single realization has different number of"
                             " r_perp than r_par")
        if self._n_tot != 0 and len(r_perp) != self._n_tot:
            raise ValueError("Different number of total pairs in new"
                             " realization does not make sense")
        if self._n_tot == 0:
            self._n_tot = len(r_perp)
        all_indices = self.get_bin(r_perp, r_par)
        all_indices = all_indices[np.all(all_indices > -1, axis=1)]
        indices, counts = np.unique(all_indices, axis=0, return_counts=True)
        self._add_count(indices[:,0], indices[:,1], counts)

    def __getitem__(self, indexer):
        counts_arr = self._get_counts_array()
        return counts_arr[indexer]

    def append(self, other):
        """
        Add another realization by appending individual instances

        Parameters
        ----------
        :param other: Another realization or set of realizations
        :type other: :class:`ExpectedNNCounts2D`

        Returns
        -------
        :return: Returns itself for using equals, but also updates in place
        :rtype: :class:`ExpectedNNCounts2D`
        """
        if not hasattr(other, "__len__"):
            if other._perp_binner != self._perp_binner:
                raise ValueError("Cannot combine ExpectedNNCounts2D instances"
                                 " with different perpendicular binning"
                                 " schemes")
            if other._par_binner != self._par_binner:
                raise ValueError("Cannot combine ExpectedNNCounts2D instances"
                                 " with different parallel binning schemes")
            if other._n_real >= 1:
                self._n_tot += other._n_tot
                self._append_counts(other._counts)
            return self
        [self.append(this_other) for this_other in other]
        return self

    # def __add__(self, other):
    #     if other._perp_binner != self._perp_binner:
    #         raise ValueError("Cannot add ExpectedNNCounts2D instances with"
    #                          " different perpendicular binning schemes")
    #     if other._par_binner != self._par_binner:
    #         raise ValueError("Cannot add ExpectedNNCounts2D instances"
    #                          " with different parallel binning schemes")
    #     if other._n_real != self._n_real:
    #         raise ValueError("Covariance matrix for addition of"
    #                          " ExpectedNNCounts2D instances with different"
    #                          " number of realizations does not make sense")
    #     new_nn = ExpectedNNCounts2D(self)
    #     new_nn._n_tot *= other._n_tot
    #     new_counts = [
    #         other._n_tot * sc + new_nn._n_tot * oc for sc, oc in zip(
    #             new_nn._counts, other._counts)]
    #     new_nn._counts = new_counts
    #     new_nn._update_counts()
    #     return new_nn
    # 
    # def __iadd__(self, other):
    #     self = self.__add__(other)
    #     return self
    # 
    # def __radd__(self, other):
    #     if other == 0:
    #         return self
    #     else:
    #         return self.__add__(other)
    # 
    # def __neg__(self):
    #     new_nn = ExpectedNNCounts2D(self)
    #     neg_counts = [-c for c in new_nn._counts]
    #     new_nn._counts = neg_counts
    #     new_nn._update_counts()
    #     return new_nn
    # 
    # def __sub__(self, other):
    #     return self.__add__(-other)
    # 
    # def __isub__(self, other):
    #     self = self.__sub__(other)
    #     return self
    # 
    # def __mul__(self, other):
    #     new_nn = ExpectedNNCounts2D(self)
    #     if hasattr(other, "_counts") and hasattr(other, "_n_tot"):
    #         if other._perp_binner != self._perp_binner:
    #             raise ValueError("Cannot multiply ExpectedNNCounts2D instances"
    #                              " with different perpendicular binning"
    #                              " schemes")
    #         if other._par_binner != self._par_binner:
    #             raise ValueError("Cannot multiply ExpectedNNCounts2D instances"
    #                              " with different parallel binning schemes")
    #         if other._n_real != self._n_real:
    #             raise ValueError("Covariance matrix for multiplication of"
    #                              " ExpectedNNCounts2D instances with different"
    #                              " number of realizations does not make sense")
    #         new_nn._n_tot *= other._n_tot
    #         mul_counts = [sc * oc for sc, oc in zip(
    #             new_nn._counts, other._counts)]
    #     else:
    #         mul_counts = [other * c for c in new_nn._counts]
    #     new_nn._counts = mul_counts
    #     new_nn._update_counts()
    #     return new_nn
    # 
    # def __imul__(self, other):
    #     self = self.__mul__(other)
    #     return self
    # 
    # def __truediv__(self, other):
    #     new_nn = ExpectedNNCounts2D(self)
    #     if hasattr(other, "_counts") and hasattr(other, "_n_tot"):
    #         if other._perp_binner != self._perp_binner:
    #             raise ValueError("Cannot divide ExpectedNNCounts2D instances"
    #                              " with different perpendicular binning"
    #                              " schemes")
    #         if other._par_binner != self._par_binner:
    #             raise ValueError("Cannot divide ExpectedNNCounts2D instances"
    #                              " with different parallel binning schemes")
    #         if other._n_real != self._n_real:
    #             raise ValueError("Covariance matrix for division of"
    #                              " ExpectedNNCounts2D instances with different"
    #                              " number of realizations does not make sense")
    #         new_nn._n_tot /= other._n_tot
    #         div_counts = []
    #         for sc, oc in zip(new_nn._counts, other._counts):
    #             dc = sparse.dok_matrix(oc.shape)
    #             dc[oc.nonzero()] = (sc.todok()[oc.nonzero()]
    #                                 / oc.todok()[oc.nonzero()])
    #             div_counts.append(dc.tocoo())
    #     else:
    #         div_counts = [c / other for c in new_nn._counts]
    #     new_nn._counts = div_counts
    #     new_nn._update_counts()
    #     return new_nn
    # 
    # def __itruediv__(self, other):
    #     self = self.__truediv__(other)
    #     return self
    # 
    # def __pow__(self, other):
    #     new_nn = ExpectedNNCounts2D(self)
    #     new_nn._n_tot **= other
    #     pow_counts = []
    #     for c in new_nn._counts:
    #         pc = c.tocoo()
    #         pc.data **= other
    #         pow_counts.append(pc)
    #     new_nn._counts = pow_counts
    #     new_nn._update_counts()
    #     return new_nn


def draw_true_seps(prob, rpo, rlo, zbar, sigmaz):
    """
    Draw true separations from the probability

    Given the observed separations :param:`rpo`, :param:`rlo`, and
    :param:`zbar`, as well as the probability contained in :param:`prob`,
    draw true separations

    Parameters
    ----------
    :param prob: The probability of true separations given observed
    :type prob: :class:`ProbFitter`
    :param rpo: The observed perpendicular separation
    :type rpo: scalar or array-like `float`
    :param rlo: The observed parallel separation
    :type rlo: scalar or array-like `float`
    :param zbar: The average observed redshift
    :type zbar: scalar or array-like `float`
    :param sigmaz: The redshift error associated with the observed separations
    :type sigmaz: scalar `float`

    Returns
    -------
    :return rpt: The true perpendicular separation
    :rtype rpt: scalar or :class:`numpy.ndarray` `float`
    :return rlt: The true parallel separation
    :rtype rlt: scalar or :class:`numpy.ndarray` `float`
    """
    if not hasattr(rpo, "__len__"):
        npairs = 1
    else:
        npairs = len(rpo)
    delta_perp, delta_parallel = np.random.randn(npairs, 2)
    rho = prob.mean_r.model(rpo, rlo)
    delta_perp *= np.sqrt(1. - rho**2)
    delta_par += rho * delta_perp
    del rho
    rpt = (np.sqrt(prob.var_rpt(rpo, rlo, zbar, sigmaz, index=None))
           * (delta_perp + prob.mean_rpt(rpo, rlo, zbar, sigmaz, index=None)))
    del delta_perp
    rlt = (np.sqrt(prob.var_rlt(rpo, rlo, zbar, sigmaz, index=None))
           * (delta_par + prob.mean_rlt(rpo, rlo, zbar, sigmaz, index=None)))
    del delta_par
    return rpt, rlt


def convolve_pair_counts(nn_3d, prob_nn, perp_binner, par_binner, sigmaz,
                         n_real=1):
    """
    Convolve the pair counts in :param:`nn_3d` with the probability
    :param:`prob_nn` by doing :param:`n_real` realizations of a Monte Carlo
    simulation of the pair counts. If :param:`n_real` is more than one,
    calculate both the mean and variance of the realizations.

    Parameters
    ----------
    :param nn_3d: The observed pair counts in bins of observed perpendicular
    separation, observed parallel separation, and average observed redshift.
    :type nn_3d: :class:`NNCounts3d`
    :param prob_nn: The corresponding probability fitter for this pair count,
    with fits already done (or use the context manager when calling this
    function)
    :type prob_nn: :class:`ProbFitter`
    :param perp_binner: The binning specifications in true perpendicular
    separation to use
    :type perp_binner: :class:`BinSpecifier`
    :param par_binner: The binning specifications in true parallel separation
    to use
    :type par_binner: :class:`BinSpecifier`
    :param sigmaz: The redshift error associated with observed separations
    :type sigmaz: scalar `float`
    :param n_real: The number of realizations of MC simulations to perform.
    Default 1
    :type n_real: `int`, optional

    Returns
    -------
    :return nn_2d: The estimated true pair counts from the MC realizations. If
    doing only one realization, the expected counts will be equal to the counts
    from the single realization, and the variance will be `None`. For more than
    one realization, the expected counts are an average of the realizations,
    and the variance is calculated as the variance on the mean
    :rtype nn_2d: :class:`ExpectedNNCounts2D`
    """
    if n_real == 1:
        # Case: 1 realization only, just draw positions
        nn_2d = ExpectedNNCounts2D(
            perp_binning=perp_binner, par_binning=par_binner)
        rpo = (nn_3d.r_perp_bin_info.bin_size * np.random.random(nn_3d.shape[0])
               + nn_3d.r_perp_bin_info.bin_min)
        rlo = (nn_3d.r_par_bin_info.bin_size * np.random.random(nn_3d.shape[1])
               + nn_3d.r_par_bin_info.bin_min)
        zbo = (nn_3d.zbar_bin_info.bin_size * np.random.random(nn_3d.shape[2])
               + nn_3d.zbar_bin_info.bin_min)
        rpt, rlt = draw_true_seps(prob_nn, ro, rlo, zbo, sigmaz)
        nn_2d.assign_bins(rpt, rlt)
        return nn_2d
    else:
        nn_2d = ExpectedNNCounts2D(
            perp_binning=perp_binner, par_binning=par_binner)
        nn_2d.append(
            [convolve_pair_counts(
                nn_3d, prob_nn, perp_binner, par_binner, sigmaz) for _ in
             range(n_real)])
        return nn_2d
