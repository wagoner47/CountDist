from __future__ import print_function

import functools
import multiprocessing
import os
import pathlib
import typing

import calculate_distances as _calculate_distances
import numpy as np
import pandas as pd
from astropy.table import Table

from .fit_probabilities import ProbFitter
from .utils import MyConfigObj, init_logger

structured_array = typing.NewType("numpy structured array", typing.List[
    typing.Mapping[
        typing.Any,
        typing.Tuple[
            np.dtype, typing.Union[int, typing.Tuple[int, ...]]]]])


def _read_catalog(file_name: typing.Union[str, os.PathLike], has_true: bool,
                  has_obs: bool, use_true: bool, use_obs: bool,
                  dtcol: typing.Optional[str] = None,
                  docol: typing.Optional[str] = None,
                  ztcol: typing.Optional[str] = None,
                  zocol: typing.Optional[str] = None,
                  read_col_names: typing.Optional[
                      typing.Sequence[str]] = None) -> structured_array:
    """
    Read a catalog from a file, and only keep needed columns

    :param file_name: Path to catalog file to read, which must be either FITS
    or ASCII type file
    :type file_name: `str` or :class:`os.PathLike`
    :param has_true: Flag specifying that catalog has true distance and
    redshift (when `True`) or not (when `False`)
    :type has_true: `bool`
    :param has_obs: Flag specifying that catalog has observed distance and
    redshift (when `True`) or not (when `False`)
    :type has_obs: `bool`
    :param use_true: Flag specifying that true distance and redshift should
    be kept in returned catalog (when `True`) or not (when `False`)
    :type use_true: `bool`
    :param use_obs: Flag specifying that observed distance and redshift should
    be kept in returned catalog (when `True`) or not (when `False`)
    :type use_obs: `bool`
    :param dtcol: Name of true distance column in catalog file. Must be
    specified if :param:`use_true` is `True`. Default `None`
    :type dtcol: `str` or `NoneType`, optional
    :param docol: Name of observed distance column in catalog file. Must be
    specified if :param:`use_obs` is `True`. Default `None`
    :type docol: `str` or `NoneType`, optional
    :param ztcol: Name of true redshift column in catalog file. If `None`
    and :param:`use_true` is `True`, defaults to `dtcol.replace('D', 'Z')`.
    Default `None`
    :type ztcol: `str` or `NoneType`, optional
    :param zocol: Name of observed redshift column in catalog file. If `None`
    and :param:`use_obs` is `True`, defaults to `docol.replace('D', 'Z')`.
    Default `None`
    :type zocol: `str` or `NoneType`, optional
    :param read_col_names: Column names when reading catalog from an ASCII
    file. Ignored for FITS file types, but required otherwise. Default `None`
    :type read_col_names: Sequence[`str`] or `NoneType`, optional

    :return: Catalog with unneeded columns removed, and distance/redshift
    columns renamed as needed.
    :rtype: :class:`numpy.recarray`

    :raises ValueError: If :param:`read_col_names` is `None` when the file
    type is not FITS
    :raises ValueError: If :param:`use_true` is `True` but :param:`has_true`
    is `False`
    :raises ValueError: If :param:`use_obs` is `True` but :param:`has_obs`
    is `False`
    :raises ValueError: If :param:`use_true` is `True` but :param:`dtcol` is
    `None`
    :raises ValueError: If :param:`use_obs` is `True` but :param:`docol` is
    `None`
    :raises ValueError: If any of :param:`dtcol`, :param:`docol`,
    :param:`ztcol`, or :param:`zocol` (or default replacements for the last
    two) are not in the column names of the catalog read from the file.
    """
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
    logger.debug("Rename distance and redshift columns if needed")
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
        if ztcol is None:
            ztcol = dtcol.replace("D", "Z")
        if ztcol not in data.colnames:
            raise ValueError("Must provide true redshift column name if"
                             " using true distances and redshift column is"
                             " not tagged the same as distance column")
        if dtcol != "D_TRUE":
            data.rename_column(dtcol, "D_TRUE")
            dtcol = "D_TRUE"
        if ztcol != "Z_TRUE":
            data.rename_column(ztcol, "Z_TRUE")
            ztcol = "Z_TRUE"
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
        if zocol is None:
            zocol = docol.replace("D", "Z")
        if zocol not in data.colnames:
            raise ValueError("Must provide observed redshift column name if"
                             " using observed distances and redshift column is"
                             " not tagged the same as distance column")
        if docol != "D_OBS":
            data.rename_column(docol, "D_OBS")
            docol = "D_OBS"
        if zocol != "Z_OBS":
            data.rename_column(zocol, "Z_OBS")
            zocol = "Z_OBS"
    logger.debug("Remove extra columns and convert catalog to structured array")
    return _convert_catalog_to_structured_array(data, use_true, use_obs,
                                                dtcol, docol, ztcol, zocol)


def _keep_fields_structured_array(
        struc_arr: typing.Union[np.recarray, structured_array],
        fieldnames_to_keep: typing.Sequence[str]) -> \
        typing.Union[np.recarray, structured_array]:
    """
    Remove extra columns from a structured array or recarray

    :param struc_arr: Input array which must be structured or have records
    :type struc_arr: :class:`numpy.recarray` or structured
    :class:`numpy.ndarray`
    :param fieldnames_to_keep: List of fields to keep in the array
    :type fieldnames_to_keep: Sequence[`str`]

    :return: The array with only the fields given in
    :param:`fieldnames_to_keep`, with the same type (other than reduced
    fields) as :param:`struc_arr`
    :rtype: :class:`numpy.recarray` or structured :class:`numpy.ndarray`
    """
    return struc_arr[
        [name for name in struc_arr.dtype.names if name in fieldnames_to_keep]]


def _convert_catalog_to_structured_array(cat: typing.Union[pd.DataFrame,
                                                           Table,
                                                           np.recarray,
                                                          structured_array],
                                         use_true: bool, use_obs: bool,
                                         dtcol: str = "D_TRUE",
                                         docol: str = "D_OBS",
                                         ztcol: str = "Z_TRUE",
                                         zocol: str = "Z_OBS") -> \
        typing.Union[np.recarray, structured_array]:
    """
    Convert a catalog of any type to a structured or recarray with columns
    not needed removed. The kept columns include 'RA' and 'DEC', as well as
    any of the true/observed distance/redshift columns needed given
    :param:`use_true` and :param:`use_obs`

    :param cat: Input catalog
    :type cat: :class:`pandas.DataFrame`, :class:`astropy.table.Table`,
    :class:`numpy.recarray`, or structured :class:`numpy.ndarray`
    :param use_true: Specifies to keep true distance and redshift (if `True`)
    or not (if `False`)
    :type use_true: `bool`
    :param use_obs: Specifies to keep observed distance and redshift (if
    `True`) or not (if `False`)
    :type use_obs: `bool`
    :param dtcol: Name of true distance column in input catalog. Only needed
    if :param:`use_true` is `True`. Default 'D_TRUE'
    :type dtcol: `str`, optional
    :param docol: Name of observed distance column in input catalog. Only
    needed if :param:`use_obs` is `True`. Default 'D_OBS'
    :type docol: `str`, optional
    :param ztcol: Name of true redshift column in input catalog. Only needed
    if :param:`use_true` is `True`. Default 'Z_TRUE'
    :type ztcol: `str`, optional
    :param zocol: Name of observed redshift column in input catalog. Only
    needed if :param:`use_obs` is `True`. Default `Z_OBS`
    :type zocol: `str`, optional

    :return: The catalog with extra columns stripped, converted to a
    structured array (if input catalog is a structured array or a
    :class:`astropy.table.Table`) or a recarray (if input catalog is a
    recarray or a :class:`pandas.DataFrame`)
    :rtype: :class:`numpy.recarray` or structured :class:`numpy.ndarray`

    :raises ValueError: If any of the needed columns are missing
    :raises TypeError: If input catalog is not a :class:`pandas.DataFrame` or
    does not have a dtype with named fields
    """
    logger = init_logger(__name__)
    keep_cols = ["RA", "DEC"]
    if use_true:
        keep_cols.extend([dtcol, ztcol])
    if use_obs:
        keep_cols.extend([docol, zocol])
    keep_cols = np.array(keep_cols)
    logger.debug("Remove extra columns")
    if isinstance(cat, pd.DataFrame):
        logger.debug("case: pandas.DataFrame")
        if not np.all(np.isin(keep_cols, cat.columns)):
            logger.debug("Have one or more needed columns missing")
            raise ValueError("Input catalog missing required columns"
                             " {}".format(keep_cols[np.isin(
                keep_cols, cat.columns, invert=True)]))
        logger.debug("Return DataFrame as numpy.recarray with only needed "
                     "columns kept")
        return cat.loc[:, keep_cols].to_records(index=False)
    if not hasattr(cat, "dtype") or not hasattr(cat.dtype, "names"):
        logger.debug("No recipe to convert anything that is not a "
                     "pandas.DataFrame, numpy.recarray, or astropy.table.Table")
        raise TypeError("Invalid type for cat: {}".format(type(cat)))
    logger.debug(
        "case: either a numpy.recarray or an astropy.table.Table was passed")
    if not np.all(np.isin(keep_cols, cat.dtype.names)):
        logger.debug("Have one or more needed columns missing")
        raise ValueError("Input catalog missing required columns"
                         " {}".format(np.array(keep_cols)[
                                          np.isin(keep_cols, cat.dtype.names,
                                                  invert=True)]))
    if np.array_equal(np.sort(cat.dtype.names), np.sort(keep_cols)):
        logger.debug("Don't need to remove any columns")
        return cat.as_array() if isinstance(cat, Table) else cat
    logger.debug("Remove extra columns and return as numpy.recarray")
    return _keep_fields_structured_array(
        cat.as_array() if isinstance(cat, Table) else cat, keep_cols)


def calculate_separations(
        perp_binner: _calculate_distances.BinSpecifier,
        par_binner: _calculate_distances.BinSpecifier,
        use_true: bool,
        use_obs: bool,
        cat1: typing.Union[
            typing.Type[Table], pd.DataFrame, np.recarray, structured_array],
        cat2: typing.Optional[
            typing.Union[
                typing.Type[Table],
                pd.DataFrame,
                np.recarray,
                structured_array]] = None,
        as_type: str = "dataframe") -> typing.Union[pd.DataFrame, Table,
                                                    structured_array]:
    """
    Calculate the separations for all pairs within limits.

    Call the routine for calculating separations. Use the bin specifications
    given to provide the minimum and maximum observed perpendicular and parallel
    separations. If :param:`cat2` is `None`, call the routine with the flag
    for auto-correlating set to `True`.

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

    :return seps: The separations calculated, converted to the appropriate type.
    The dtype of the return is [('R_PERP_T', float), ('R_PAR_T', float),
    ('R_PERP_O', float), ('R_PAR_O', float), ('AVE_Z_OBS', float),
    ('ID1', uint), ('ID2', uint)]. Note that columns 'R_PERP_T' and 'R_PAR_T'
    will be NaN if :param:`use_true` is `False`, and columns 'R_PERP_O',
    'R_PAR_O', and 'AVE_Z_OBS' will be NaN if :param:`use_obs` is `False`
    :rtype seps: :class:`pandas.DataFrame` (:param:`as_type`='dataframe',
    default), :class:`astropy.table.Table` (:param:`as_type`='table'), or
    :class:`numpy.ndarray` (:param:`as_type`='array')
    
    :raises ValueError: If an invaliid string is given for :param:`as_type`
    """
    logger = init_logger(__name__)
    if as_type.lower() not in ['dataframe', 'table', 'array']:
        raise ValueError("Invalid return type ({}): valid return type"
                         " specifiers are ['dataframe', 'table',"
                         " 'array']".format(as_type))
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


def calculate_separations_from_params(
        params_file: typing.Union[str, os.PathLike],
        as_type: str = "dataframe") -> \
        typing.Union[pd.DataFrame, Table, structured_array]:
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

    :param params_file: The parameter file to be used. Note that a temporary
    copy will be made with the appropriate input file readable by the
    executable, but the temporary file will be removed after the code has run.
    :type params_file: `str` or :class:`os.PathLike`
    :param as_type: The return type to use, of :class:`pandas.DataFrame`
    ('dataframe'), :class:`astropy.table.Table` ('table'), or structured
    :class:`numpy.ndarray` ('array'). Default 'dataframe'
    :type as_type: `str` {'dataframe', 'table', 'array'}, optional

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


def get_pair_counts(binners: typing.Sequence[_calculate_distances.BinSpecifier],
                    cat1: typing.Union[typing.Type[Table],
                                       pd.DataFrame, np.recarray,
                                       structured_array],
                    cat2: typing.Optional[typing.Union[typing.Type[
                                                           Table],
                                                       pd.DataFrame,
                                                       np.recarray,
                                                       structured_array]] =
                    None,
                    use_true: bool = False) -> typing.Union[
                        _calculate_distances.NNCounts3D,
                        _calculate_distances.NNCounts2D,
                        _calculate_distances.NNCounts1D]:
    """
    Get the pair counts within a catalog or between catalogs in 1, 2,
    or 3 dimensional bins

    :param binners: Set of bin specifications for the desired dimensionality.
    Currently, this must either be 1, 2, or 3 objects, and an error is thrown
    if the length is anything else
    :type binners: Sequence[:class:`~countdist2.BinSpecifier`] (of length 1,
    2, or 3)
    :param cat1: First catalog for cross-correlation pair counts, or the only
    catalog for auto-correlation pair counts. The distance and redshift
    columns must be named 'D_TRUE', 'D_OBS', 'Z_TRUE', and 'Z_OBS' (for any
    of those columns in the catalog)
    :type cat1: :class:`pandas.DataFrame`, :class:`astropy.table.Table`,
    :class:`astropy.table.QTable`, :class:`numpy.recarray`, or structured
    :class:`numpy.ndarray`
    :param cat2: Second catalog for cross-correlation pair counts. The
    distance and redshift columns must be named 'D_TRUE', 'D_OBS', 'Z_TRUE',
    and 'Z_OBS' (for any of those columns in the catalog). If `None`, compute
    pair counts for an auto-correlation. Default `None`
    :type cat2: :class:`pandas.DataFrame`, :class:`astropy.table.Table`,
    :class:`astropy.table.QTable`, :class:`numpy.recarray`, structured
    :class:`numpy.ndarray` or `NoneType`, optional
    :param use_true: If `True`, use the true distance (and possibly redshift)
    for calculating separations, or use observed distance (and redshift) if
    `False`. Default `False`
    :type use_true: `bool`, optional

    :return nn: The pair count object with pairs processed. Which NNCountsND
    type it is matches the length of :param:`binners`
    :rtype nn: :class:`~countdist2.NNCounts1D`,
    :class:`~countdist2.NNCounts2D`, or :class:`~countdist2.NNCounts3D`

    :raises ValueError: If the length of :param:`binners` is not 1, 2, or 3
    """
    logger = init_logger(__name__)
    binners = list(binners)
    logger.debug("Initialize the correct NNCountsND object")
    if len(binners) == 1:
        logger.debug("NNCounts1D")
        nn = _calculate_distances.NNCounts1D(binners)
    elif len(binners) == 2:
        logger.debug("NNCounts2D")
        nn = _calculate_distances.NNCounts2D(binners)
    elif len(binners) == 3:
        logger.debug("NNCounts3D")
        nn = _calculate_distances.NNCounts3D(binners)
    else:
        logger.debug("Invalid dimensionality")
        raise ValueError("Cannot do pair counts in {} dimensions".format(len(
            binners)))
    logger.debug("Convert first catalog")
    cat1_arr = _convert_catalog_to_structured_array(cat1, use_true,
                                                    not use_true)
    if cat2 is None:
        logger.debug("Auto pair counts")
        nn.process_auto(cat1_arr)
    else:
        logger.debug("Convert second catalog")
        cat2_arr = _convert_catalog_to_structured_array(cat2, use_true,
                                                        not use_true)
        logger.debug("Cross pair counts")
        nn.process_cross(cat2_arr)
    return nn


def get_3d_pair_counts(rpo_binner: _calculate_distances.BinSpecifier,
                       rlo_binner: _calculate_distances.BinSpecifier,
                       zbar_binner:
                       _calculate_distances.BinSpecifier,
                       cat1: typing.Union[typing.Type[Table],
                                          pd.DataFrame, np.recarray,
                                          structured_array],
                       cat2: typing.Optional[
                           typing.Union[
                               typing.Type[Table],
                               pd.DataFrame,
                               np.recarray,
                               structured_array]] = None,
                       use_true: bool = False) -> \
        _calculate_distances.NNCounts3D:
    """
    Get the pair counts between the two catalogs in 3D bins

    Find the 3D binned pair counts between :param:`cat1` and :param:`cat2`,
    or just within :param:`cat1` as an auto-correlation if :param:`cat2` is
    `None`. Please make sure that :param:`rpo_binner`, :param:`rlo_binner`,
    and :param:`zbar_binner` are fully set instances of :class:`BinSpecifier. 
    This function is basically a wrapper for :func:`get_pair_counts` with 
    3-dimensional binning specified`

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

    :return: The counter object for this set of pair counts. The binning
    information can be recalled using calls to 'nn.*_bin_info' (rperp, rpar,
    or zbar), the 3D array of counts can be obtained with 'nn.counts', and
    the total number of pairs processed is available via 'nn.n_tot'
    :rtype: :class:`NNCounts3D`
    """
    return get_pair_counts([rpo_binner, rlo_binner, zbar_binner], cat1, cat2,
                           use_true)


def get_3d_pair_counts_from_params(params_file: typing.Union[str,
                                                             os.PathLike]) -> \
        _calculate_distances.NNCounts3D:
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
        return get_3d_pair_counts(
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
    return get_3d_pair_counts(
        rpo_bins, rlo_bins, zbar_bins, cat1, cat2, use_true)

    # :param perp_binner: The binning specifications in true perpendicular
    # separation to use
    # :type perp_binner: :class:`BinSpecifier`
    # :param par_binner: The binning specifications in true parallel separation
    # to use
    # :type par_binner: :class:`BinSpecifier`


def make_single_realization(
        nn_3d: _calculate_distances.NNCounts3D,
        prob: ProbFitter,
        binners: typing.Sequence[_calculate_distances.BinSpecifier],
        # perp_binner: _calculate_distances.BinSpecifier,
        # par_binner: _calculate_distances.BinSpecifier,
        sigmaz: float,
        rlt_mag: bool = True,
        rstate: typing.Optional[
            typing.Union[
                int,
                typing.Tuple[
                    str, typing.List[np.uint, 624], int, int, float]]] = None) -> \
        typing.Union[_calculate_distances.ExpectedNNCounts1D,
                     _calculate_distances.ExpectedNNCounts2D]:
    """
    Make a single Monte Carlo realization of the true pair counts in bins
    specified by the BinSpecifier objects given the 3D pair counts. A
    lock object may be given if running in parallel to keep from running
    the multi-threaded 'process_separation' method in parallel.

    :param nn_3d: The observed pair counts in bins of observed perpendicular
    separation, observed parallel separation, and average observed redshift.
    :type nn_3d: :class:`NNCounts3d`
    :param prob: The corresponding probability fitter for this pair count,
    with fits already done (or use the context manager when calling this
    function)
    :type prob: :class:`ProbFitter`
    :param binners: BinSpecifier object(s) for the new binning. Can only be
    length 1 or 2. Output type is determined by the length of this parameter.
    :type binners: Sequence[:class:`~countdist2.BinSpecifier`] of length 1 or 2
    :param sigmaz: The redshift error associated with observed separations
    :type sigmaz: scalar `float`
    :param rlt_mag: Return the absolute value (if `True`) of the drawn true
    parallel separations so that they cannot be negative. Default `True`
    :type rlt_mag: `bool`, optional
    :param rstate: A random seed or state to set, if any. Default `None` will
    not alter the current state
    :type rstate: `int`, `tuple`(`str`, :class:`numpy.ndarray`[`uint`, 624],
    `int`, `int`, `float`) or `NoneType`, optional

    :return enn: The expected pair counts in 1D or 2D bins for a single
    realization, with mean calculated
    :rtype enn: :class:`~countdist2.ExpectedNNCounts1D` or
    :class:`~countdist2.ExpectedNNCounts2D`
    """
    logger = init_logger(__name__)
    binners = list(binners)
    logger.debug("Initialize ExpectedNNCountsND object")
    if len(binners) == 1:
        logger.debug("case: 1D")
        enn = _calculate_distances.ExpectedNNCounts1D(binners, nn_3d.ntot)
        in_1d = True
    elif len(binners) == 2:
        logger.debug("case: 2D")
        enn = _calculate_distances.ExpectedNNCounts2D(binners, nn_3d.ntot)
        in_1d = False
    else:
        logger.debug("Invalid dimensionality")
        raise ValueError("Cannot find expected pair counts in {} "
                         "dimensionts".format(len(binners)))
    logger.debug("Set random state")
    if rstate is not None:
        if isinstance(rstate, int):
            np.random.seed(0)
            np.random.seed(rstate)
        else:
            np.random.set_state(rstate)
    is_first = True
    logger.debug("Begin loop")
    for (i, j, k), c in np.ndenumerate(nn_3d.counts):
        if c > 0:
            rpt, rlt = prob.draw_rpt_rlt(
                nn_3d.rperp_bins.bin_widths[i] * np.random.rand(c)
                + nn_3d.rperp_bins.lower_bin_edges[i],
                nn_3d.rpar_bins.bin_widths[j] * np.random.rand(c)
                + nn_3d.rpar_bins.lower_bin_edges[j],
                nn_3d.zbar_bins.bin_widths[k] * np.random.rand(c)
                + nn_3d.zbar_bins.lower_bin_edges[k],
                sigmaz,
                rlt_mag)
            if in_1d:
                args = (np.sqrt(rpt**2 + rlt**2), is_first)
            else:
                args = (rpt, rlt, is_first)
            with multiprocessing.Lock():
                enn.process_separation(*args)
            is_first = False
    logger.debug("Calculate mean")
    enn.update()
    return enn


def convolve_pair_counts(
        nn_3d: _calculate_distances.NNCounts3D,
        prob: ProbFitter,
        binners: typing.Sequence[_calculate_distances.BinSpecifier],
        sigmaz: float,
        n_real: int = 1,
        n_process: int = 1,
        rlt_mag: bool = True,
        rstate: typing.Optional[
            typing.Union[
                int,
                typing.Tuple[
                    str, typing.List[
                        np.uint, 624], int, int, float]]] = None) -> \
        typing.Union[_calculate_distances.ExpectedNNCounts1D,
                     _calculate_distances.ExpectedNNCounts2D]:
    """
    Convolve the pair counts in :param:`nn_3d` with the probability
    :param:`prob_nn` by doing :param:`n_real` realizations of a Monte Carlo
    simulation of the pair counts. If :param:`n_real` is more than one,
    calculate both the mean and variance of the realizations.

    :param nn_3d: The observed pair counts in bins of observed perpendicular
    separation, observed parallel separation, and average observed redshift.
    :type nn_3d: :class:`NNCounts3d`
    :param prob: The corresponding probability fitter for this pair count,
    with fits already done (or use the context manager when calling this
    function)
    :type prob: :class:`ProbFitter`
    :param binners: BinSpecifier object(s) for the new binning. Can only be
    length 1 or 2. Output type is determined by the length of this parameter.
    :type binners: Sequence[:class:`~countdist2.BinSpecifier`] of length 1 or 2
    :param sigmaz: The redshift error associated with observed separations
    :type sigmaz: scalar `float`
    :param n_real: The number of realizations of MC simulations to perform.
    Default 1
    :type n_real: `int`, optional
    :param n_process: Number of processes to use for parallelization. Default
    1
    :type n_process: `int`, optional
    :param rlt_mag: Return the absolute value (if `True`) of the drawn true
    parallel separations so that they cannot be negative. Default `True`
    :type rlt_mag: `bool`, optional
    :param rstate: An initial random seed or state to set, if any. All random
    states of the sub-processes will be set from this initial state. Default
    `None` will not alter the current state
    :type rstate: `int`, `tuple`(`str`, :class:`numpy.ndarray`[`uint`, 624],
    `int`, `int`, `float`) or `NoneType`, optional

    :return enn: The estimated true pair counts from the MC realizations. If
    doing only one realization, the expected counts will be equal to the counts
    from the single realization, and the variance will be `None`. For more than
    one realization, the expected counts are an average of the realizations,
    and the variance is calculated as the variance on the mean
    :rtype enn: :class:`~countdist2.ExpectedNNCounts1D` or
    :class:`~countdist2.ExpectedNNCounts2D`
    """
    if n_real == 1:
        return make_single_realization(
            nn_3d, prob, binners, sigmaz, rlt_mag, rstate)
    binners = list(binners)
    if len(binners) == 1:
        enn = _calculate_distances.ExpectedNNCounts1D(binners, nn_3d.ntot)
    elif len(binners) == 2:
        enn = _calculate_distances.ExpectedNNCounts2D(binners, nn_3d.ntot)
    else:
        raise ValueError("Cannot get expected pair counts in {} "
                         "dimensions".format(len(binners)))
    with multiprocessing.Pool(n_process) as pool:
        enn.append_real(
            pool.map(
                functools.partial(
                    nn_3d, prob, binners, sigmaz, rlt_mag),
                range(1, n_real + 1)))
    enn.update()
    return enn
