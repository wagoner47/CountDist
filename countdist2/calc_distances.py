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
import pathlib
import multiprocessing


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
    logger.debug("rename columns as needed")
    if use_true:
        if dtcol != "D_TRUE":
            data.reaname_column(dtcol, "D_TRUE")
        if ztcol != "Z_TRUE":
            data.rename_column(ztcol, "Z_TRUE")
    if use_obs:
        if docol != "D_OBS":
            data.rename_column(docol, "D_OBS")
        if zocol != "Z_OBS":
            data.rename_column(zocol, "Z_OBS")
    logger.debug("return data as structured ndarray")
    return data.as_array()


def _keep_fields_structured_array(struc_arr, fieldnames_to_keep):
    return struc_arr[
        [name for name in struc_arr.dtype.names if name in fieldnames_to_keep]]


def _convert_catalog_to_structured_array(cat, use_true, use_obs, dtcol="D_TRUE",
                                         docol="D_OBS", ztcol="Z_TRUE",
                                         zocol="Z_OBS"):
    logger = init_logger(__name__)
    keep_cols = ["RA", "DEC"]
    if use_true:
        keep_cols.extend([dtcol, ztcol])
    if use_obs:
        keep_cols.extend([docol, zocol])
    keep_cols = np.array(keep_cols)
    if isinstance(cat, pd.DataFrame):
        if not np.all(np.isin(keep_cols, cat.columns)):
            raise ValueError("Input catalog missing required columns"
                             " {}".format(keep_cols[np.isin(
                                 keep_cols, cat.columns, invert=True)]))
        return cat.loc[:,keep_cols].to_records(index=False)
    if not hasattr(cat, "dtype") or not hasattr(cat.dtype, "names"):
        raise TypeError("Invalid type for cat: {}".format(type(cat)))
    if not np.all(np.isin(keep_cols, cat.dtype.names)):
        raise ValueError("Input catalog missing required columns"
                         " {}".format(np.array(keep_cols)[
                             np.isin(keep_cols, cat.dtype.names, invert=True)]))
    if np.array_equal(np.sort(cat.dtype.names), np.sort(keep_cols)):
        return cat.as_array() if isinstance(cat, Table) else cat
    return _keep_fields_structured_array(
        cat.as_array() if isinstance(cat, Table) else cat, keep_cols)


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
    cat1_arr = _convert_catalog_to_structured_array(cat1, use_true, use_obs)
    if cat2 is None:
        logger.debug("run auto-correlation for separations")
        seps = _calculate_distances.get_auto_separations(
            cat1_arr, perp_binner, par_binner)
    else:
        logger.debug("convert cat2")
        cat2_arr = _convert_catalog_to_structured_array(cat2, use_true, use_obs)
        logger.debug("run cross-correlation for separations")
        seps = _calculate_distances.get_cross_separations(
            cat1_arr, cat2_arr, perp_binner, par_binner)
    logger.debug("convert separations")
    if as_type == 'dataframe':
        seps = pd.DataFrame.from_records(seps)
    elif as_type == 'table':
        seps = Table(seps)
    else:
        pass
    return seps


def _get_distance_redshift_colnames(params_in, *, dtkey, dokey, ztkey, zokey):
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
        zocol = params_in[zokey]
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
    if as_type.lower() not in ["dataframe", "table", "array"]:
        raise ValueError("Invalid choice for return type ('{}'): valid choices"
                         " are (case-insensitive) 'dataframe', 'table', or"
                         " 'array'".format(as_type))
    rtype = as_type.lower()
    logger = init_logger(__name__)
    logger.debug("read parameter file")
    params_in = MyConfigObj(params_file, file_error=True)
    try:
        params_in = params_in["run_params"]
    except KeyError:
        pass
    logger.debug("set up bin specifications")
    rpo_bins = _calculate_distances.BinSpecifier(
        params_in.as_float("rp_min"), params_in.as_float("rp_max"), 1, False)
    rlo_bins = _calculate_distances.BinSpecifier(
        params_in.as_float("rl_min"), params_in.as_float("rl_max"), 1, False)
    logger.debug("get first catalog column names")
    dtcol1, docol1, ztcol1, zocol1 = _get_distance_redshift_colnames(
        params_in, dtkey="dtcol1", dokey="docol1", ztkey="ztcol1",
        zokey="zocol1")
    try:
        read_colnames1 = params_in.as_list("read_colnames1")
    except KeyError:
        read_colnames1 = None
    logger.debug("read first catalog")
    cat1 = _read_catalog(
        params_in["ifname1"], params_in.as_bool("has_true1"),
        params_in.as_bool("has_obs1"), params_in.as_bool("use_true"),
        params_in.as_bool("use_obs"), dtcol1, docol1, ztcol1, zocol1,
        read_colnames1)
    if params_in["ifname2"] == params_in["ifname1"]:
        logger.debug("calculate auto-correlation separations")
        return calculate_separations(
            rpo_bins, rlo_bins, params_in.as_bool("use_true"),
            params_in.as_bool("use_obs"), cat1, as_type=as_type)
    logger.debug("get second catalog column names")
    dtcol2, docol2, ztcol2, zocol2 = _get_distance_redshift_colnames(
        params_in, dtkey="dtcol2", dokey="docol2", ztkey="ztcol2",
        zokey="zocol2")
    try:
        read_colnames2 = params_in.as_list("read_colnames2")
    except KeyError:
        read_colnames2 = None
    logger.debug("read second catalog")
    cat2 = _read_catalog(
        params_in["ifname2"], params_in.as_bool("has_true2"),
        params_in.as_bool("has_obs2"), params_in.as_bool("use_true"),
        params_in.as_bool("use_obs"), dtcol2, docol2, ztcol2, zocol2,
        read_colnames2)
    logger.debug("calculate cross-correlation separations")
    return calculate_separations(
        rpo_bins, rlo_bins, params_in.as_bool("use_true"),
        params_in.as_bool("use_obs"), cat1, cat2, as_type)


def get_3d_pair_counts(rpo_binner, rlo_binner, zbar_binner, cat1, cat2=None,
                       use_true=False):
    """
    Get the pair counts between the two catalogs in 3D bins

    Find the 3D binned pair counts between :param:`cat1` and :param:`cat2`,
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
    :class:`pandas.DataFrame`, or :class:`numpy.ndarray`, dtype=
    [('RA', float), ('DEC', float), ('D_TRUE', float), ('D_OBS', float),
    ('Z_TRUE', float), ('Z_OBS', float)]
    :param cat2: The second catalog to use for finding pairs, if
    cross-correlating. Please note that providing the same catalog will result
    in double counting, as no checking is done to verify that the catalogs are
    different. Default `None`
    :type cat2: same as :param:`cat1` or `NoneType`, optional
    :param use_true: If `True`, use true positions for separations. Default
    `False`
    :type use_true: `bool`, optional

    Returns
    -------
    :return nn: The counter object for this set of pair counts. The binning
    information can be recalled using calls to 'nn.*_bin_info' (r_perp, r_par,
    or zbar), the 3D array of counts can be obtained with 'nn.counts', and
    the total number of pairs processed is available via 'nn.n_tot'
    :rtype nn: :class:`NNCounts3D`
    """
    logger = init_logger(__name__)
    logger.debug("initalize NNCounts3D object")
    nn = _calculate_distances.NNCounts3D(rpo_binner, rlo_binner, zbar_binner)
    logger.debug("convert first catalog")
    cat1_arr = _convert_catalog_to_structure_array(
        cat1, use_true, (not use_true))
    if cat2 is None:
        logger.debug("run auto-correlation for pair counts")
        nn.process_auto(cat1_arr)
    else:
        logger.debug("convert cat2")
        cat2_arr = _convert_catalog_to_structured_array(
            cat2, use_true, (not use_true))
        logger.debug("run cross-correlation for pair counts")
        nn.process_cross(cat1_arr, cat2_arr)
    return nn

def get_3d_pair_counts_from_params(params_file):
    """
    Get the pair counts in 3D bins with options given in parameter file

    This function gets the 3D binned pair counts for the files specified in
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
    the total number of pairs processed is available via 'nn.ntot'
    :rtype: :class:`NNCounts3D`
    """
    logger = init_logger(__name__)
    logger.debug("read parameter file")
    params_in = MyConfigObj(params_file, file_error=True)
    try:
        params_in = params_in["run_params"]
    except KeyError:
        pass
    use_true = params_in.as_bool("use_true")

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

    logger.debug("get first catalog column names")
    dtcol1, docol1, ztcol1, zocol1 = _get_distance_redshift_colnames(
        params_in, dtkey="dtcol1", dokey="docol1", ztkey="ztcol1",
        zokey="zocol1")
    try:
        read_colnames1 = params_in.as_list("read_colnames1")
    except KeyError:
        read_colnames1 = None
    logger.debug("read first catalog")
    cat1 = _read_catalog(
        params_in["ifname1"], params_in.as_bool("has_true1"),
        params_in.as_bool("has_obs1"), use_true, (not use_true), dtcol1, docol1,
        ztcol1, zocol1, read_colnames1)
    if params_in["ifname2"] == params_in["ifname1"]:
        logger.debug("process and return auto-correlation counts")
        return get_observed_pair_counts(
            rpo_bins, rlo_bins, zbar_bins, cat1, use_true=use_true)
    logger.debug("get second catalog column names")
    dtcol2, docol2, ztcol2, zocol2 = _get_distance_redshift_colnames(
        params_in, dtkey="dtcol2", dokey="docol2", ztkey="ztcol2",
        zokey="zocol2")
    try:
        read_colnames2 = params_in.as_list("read_colnames2")
    except KeyError:
        read_colnames2 = None
    logger.debug("read second catalog")
    cat2 = _read_catalog(
        params_in["ifname2"], params_in.as_bool("has_true2"),
        params_in.as_bool("has_obs2"), use_true, (not use_true), dtcol2, docol2,
        ztcol2, zocol2, read_colnames2)
    logger.debug("process and return cross-correlaion counts")
    return get_observed_pair_counts(
        rpo_bins, rlo_bins, zbar_bins, cat1, cat2, use_true)


class DummyLock(object):
    def __init__(self):
        pass
    def acquire(self):
        pass
    def release(self):
        pass


def make_single_realization(nn_3d, prob, perp_binner, par_binner, sigmaz,
                            lock=None, rstate=None, rlt_mag=True):
    """
    Make a single Monte Carlo realization of the true pair counts in bins
    specified by the BinSpecifier objects given the 3D pair counts. A
    lock object may be given if running in parallel to keep from running
    the multi-threaded 'process_separation' method in parallel.

    Parameters
    ----------
    :param nn_3d: The observed pair counts in bins of observed perpendicular
    separation, observed parallel separation, and average observed redshift.
    :type nn_3d: :class:`NNCounts3d`
    :param prob: The corresponding probability fitter for this pair count,
    with fits already done (or use the context manager when calling this
    function)
    :type prob: :class:`ProbFitter`
    :param perp_binner: The binning specifications in true perpendicular
    separation to use
    :type perp_binner: :class:`BinSpecifier`
    :param par_binner: The binning specifications in true parallel separation
    to use
    :type par_binner: :class:`BinSpecifier`
    :param sigmaz: The redshift error associated with observed separations
    :type sigmaz: scalar `float`
    :param lock: A lock instance to create a parallel critical section. Default
    `None`
    :type lock: :class:`threading.Lock`, :class:`multiprocessing.Lock`, or
    `NoneType`, optional
    :param seed: A random seed to set, if any. Default `None` will not set any
    seed (rather than setting to 0)
    :type seed: `int` or `NoneType`
    :param rlt_mag: Return the absolute value (if `True`) of the drawn true
    parallel separations so that they cannot be negative. Default `True`
    :type rlt_mag: `bool`, optional
    """
    if seed is not None:
        np.random.seed(0)
        np.random.seed(seed)
    if lock is None:
        lock = DummyLock()
    nn_2d = _calculate_distances.ExpectedNNCounts2D(
        perp_binner, par_binner, nn_3d.ntot)
    rpo_mins = nn_3d.rperp_bins.lower_bin_edges
    rpo_widths = nn_3d.rperp_bins.bin_widths
    rlo_mins = nn_3d.rpar_bins.lower_bin_edges
    rlo_widths = nn_3d.rpar_bins.bin_widths
    zbo_mins = nn_3d.zbar_bins.lower_bin_edges
    zbo_widths = nn_3d.zbar_bins.bin_widths
    is_first = True
    for (i, j, k), c in np.ndenumerate(nn_3d.counts):
        if c > 0:
            rpo = rpo_widths[i] * np.random.rand(c) + rpo_mins[i]
            rlo = rlo_widths[j] * np.random.rand(c) + rlo_mins[j]
            zbo = zbo_widths[k] * np.random.rand(c) + zbo_mins[k]
            rpt, rlt = prob.draw_rpt_rlt(rpo, rlo, zbo, sigmaz, rlt_mag)
            logger.debug("rpt: {}".format(
                (rpt - rpo) / np.sqrt(prob.var_rpt(rpo, rlo, zbo, sigmaz))))
            logger.debug("rlt: {}".format(
                (rlt - rlo) / np.sqrt(prob.var_rlt(rpo, rlo, zbo, sigmaz))))
            lock.acquire()
            nn_2d.process_separation(rpt, rlt, is_first)
            lock.release()
            is_first = False
    nn_2d.update()
    return nn_2d


def convolve_pair_counts(nn_3d, prob, perp_binner, par_binner, sigmaz,
                         n_real=1, n_process=1, rlt_mag=True):
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
    :param prob: The corresponding probability fitter for this pair count,
    with fits already done (or use the context manager when calling this
    function)
    :type prob: :class:`ProbFitter`
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
    :param n_processes: Number of processes to use for parallelization. Default
    1
    :type n_processes: `int`, optional
    :param rlt_mag: Return the absolute value (if `True`) of the drawn true
    parallel separations so that they cannot be negative. Default `True`
    :type rlt_mag: `bool`, optional

    Returns
    -------
    :return nn_2d: The estimated true pair counts from the MC realizations. If
    doing only one realization, the expected counts will be equal to the counts
    from the single realization, and the variance will be `None`. For more than
    one realization, the expected counts are an average of the realizations,
    and the variance is calculated as the variance on the mean
    :rtype nn_2d: :class:`ExpectedNNCounts2D`
    """
    logger = init_logger(__name__)
    if n_real == 1:
        return make_single_realization(
            nn_3d, prob, perp_binner, par_binner, sigmaz, DummyLock(), rlt_mag)
    nn_2d = _calculate_distances.ExpectedNNCounts2D(perp_binner, par_binner)
    if n_processes == 1:
        nn_2d.append_real(
            [make_single_realization(
                nn_3d, prob, perp_binner, par_binner, sigmaz, DummyLock(),
                rlt_mag) for _ in range(n_real)])
    else:
        lock = multiprocessing.Lock()
        def call_wrapper(i, nn=nn_3d, p=prob, pb=perp_binner, lb=par_binner,
                         s=sigmaz, l=lock, m=rlt_mag):
            return make_single_realization(nn, p, pb, lb, s, l, i, m)
        with multiprocessing.Pool(n_processes) as pool:
            nn_2d.append_real(pool.map(call_wrapper, range(n_real)))
    nn_2d.update()
    return nn_2d
