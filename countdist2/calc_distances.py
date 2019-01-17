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


def _read_catalog(file_name, has_true, has_obs, dtcol=None, docol=None,
                  ztcol=None, zocol=None):
    if has_obs:
        print("Using column {} for true distances".format(dtcol))
    if "fit" in os.path.splitext(file_name)[1]:
        data = Table.read(file_name)
    else:
        names = ["RA", "DEC"]
        if has_true:
            if dtcol is None:
                raise ValueError("True distance column name must be given if"
                                 " 'has_true' is True")
            names.append(dtcol)
            if ztcol is None:
                ztcol = dtcol.replace("D", "Z")
            names.append(ztcol)
        if has_obs:
            if docol is None:
                raise ValueError("Observed distance column name must be given"
                                 " if 'has_obs' is True")
            names.append(docol)
            if zocol is None:
                zocol = docol.replace("D", "Z")
            names.append(zocol)
        data = Table.read(file_name, format="ascii", names=names)
    if not has_true:
        if dtcol is None:
            dtcol = "D_TRUE"
        if ztcol is None:
            ztcol = dtcol.replace("D", "Z")
        data[dtcol] = np.nan
        data[ztcol] = np.nan
    if not has_obs:
        if docol is None:
            docol = "D_OBS"
        if zocol is None:
            zocol = docol.replace("D", "Z")
        data[docol] = np.nan
        data[zocol] = np.nan
    cat = _calculate_distances.fill_catalog_vector(
        data["RA"], data["DEC"], data[dtcol], data[docol], data[ztcol],
        data[zocol])
    return cat


def calculate_survey_volume(cosmo_file, cat_file, zcol, map_file):
    """Calculate the volume of the survey from the catalog in the given
    cosmology, assuming the range of redshifts is constant over the entire
    area. The survey area is obtained from the HEALPix map given.

    Parameters
    ----------
    :param cosmo_file: The path to a file containing the cosmological
    parameters to use for calculations
    :type cosmo_file: `str`
    :param cat_file: The path to a file containing the galaxy positions
    :type cat_file: `str`
    :param zcol: The name of the column from the catalog to use for the redshift
    :type zcol: `str`
    :param map_file: The path to a file containing the HEALPix map for the
    survey
    :type map_file: `str`

    Returns
    -------
    :return survey_volume: The volume of the survey in Mpc^3
    :rtype survey_volume: `float`
    """
    cosmo = _initialize_cosmology(cosmo_file)
    survey_area = CatalogUtils.calculate_survey_area(map_file)
    z = Table.read(cat_file)[zcol]
    zmin = z.min()
    zmax = z.max()
    spline_z_min = min(0.0, zmin)
    spline_z_max = max(2.0, zmax)
    CatalogUtils.initialize(cosmo, zmin=spline_z_min, zmax=spline_z_max)
    survey_volume = CatalogUtils.vol(zmax, zmin, survey_area)
    return survey_volume


def _get_survey_volume(params):
    """This function is meant to be used only by the code, to check if the
    survey volume is already calculated in the parameter file or if it needs
    to be calculated

    Parameters
    ----------
    :param params: The dictionary of the parameters read from the config file
    :type params: `dict`

    Returns
    -------
    :return volume: The survey volume, either read from the parameters or
    calculated
    :rtype volume: `float`
    """
    try:
        volume = params.as_float("survey_volume")
    except:
        volume = calculate_survey_volume(
            params["cosmo_file"],
            params["ifname2"],
            params["zcol"],
            params["survey_volume"]
            )
    return volume


def run_calc(params_file):
    """This function runs the executable for finding the separations between
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
    :type params_file: string
    """
    logger = init_logger(__name__)
    logger.info("Reading parameter file")
    params_in = MyConfigObj(params_file, file_error=True)
    try:
        params_in = params_in["run_params"]
    except KeyError:
        pass
    if "ztcol1" in params_in:
        ztcol1 = params_in["ztcol1"]
    else:
        if params_in.as_bool("has_true1"):
            ztcol1 = params_in["dtcol1"].replace("D", "Z")
        else:
            ztcol1 = None
    if "zocol1" in params_in:
        zocol1 = params_in["zocol1"]
    else:
        if params_in.as_bool("has_obs1"):
            zocol1 = params_in["docol1"].replace("D", "Z")
        else:
            zocol1 = None
    cat1 = _read_catalog(
        params_in["ifname1"], params_in.as_bool("has_true1"),
        params_in.as_bool("has_obs1"),
        params_in["dtcol1"] if params_in.as_bool("has_true1") else None,
        params_in["docol1"] if params_in.as_bool("has_obs1") else None,
        ztcol1, zocol1)
    if params_in["ifname2"] != params_in["ifname1"]:
        if "ztcol2" in params_in:
            ztcol2 = params_in["ztcol2"]
        else:
            if params_in.as_bool("has_true2"):
                ztcol2 = params_in["dtcol2"].replace("D", "Z")
            else:
                ztcol2 = None
        if "zocol2" in params_in:
            zocol2 = params_in["zocol2"]
        else:
            if params_in.as_bool("has_obs2"):
                zocol2 = params_in["docol2"].replace("D", "Z")
            else:
                zocol2 = None
        cat2 = _read_catalog(
            params_in["ifname2"], params_in.as_bool("has_true2"),
            params_in.as_bool("has_obs2"),
            params_in["dtcol2"] if params_in.as_bool("has_true2") else None,
            params_in["docol2"] if params_in.as_bool("has_obs2") else None,
            ztcol2, zocol2)
        is_auto = False
    else:
        cat2 = cat1
        is_auto = True
    logger.info("Running calculation")
    seps_out = _calculate_distances.get_separations(
        cat1, cat2, params_in.as_float("rp_min"),
        params_in.as_float("rp_max"), params_in.as_float("rl_min"),
        params_in.as_float("rl_max"), params_in.as_bool("use_true"),
        params_in.as_bool("use_obs"), is_auto)
    logger.info("Converting result to DataFrame")
    seps_result = pd.DataFrame.from_dict(
        {"ID1": seps_out.id1, "ID2": seps_out.id2})
    if params_in.as_bool("use_true") and params_in.as_bool("use_obs"):
        seps_result = pd.DataFrame.from_dict(
            {"R_PERP_T": seps_out.r_perp_t, "R_PAR_T": seps_out.r_par_t,
             "R_PERP_O": seps_out.r_perp_o, "R_PAR_O": seps_out.r_par_o,
             "AVE_Z_OBS": seps_out.ave_z_obs}).join(seps_result)
    elif params_in.as_bool("use_true"):
        seps_result = pd.DataFrame.from_dict(
            {"R_PERP": seps_out.r_perp_t, "R_PAR": seps_out.r_par_t}).join(
                seps_result)
    elif params_in.as_bool("use_obs"):
        seps_result = pd.DataFrame.from_dict(
            {"R_PERP": seps_out.r_perp_o, "R_PAR": seps_out.r_par_o,
             "AVE_Z_OBS": seps_out.ave_ro}).join(seps_result)
    else:
        # Should never get here, but check just in case
        raise ValueError("Must use at least true or observed distances, or"
                         " both")
    return seps_result

def get_observed_pair_counts(rpo_binner, rlo_binner, zbar_binner, params_file):
    """This function gets the observed pair counts for the files specified in
    :param:`params_file`. Please note that the parameters '*_binner' should be
    instances of :class:`BinSpecifier` with values set.

    Parameters
    ----------
    :param rpo_binner: The bin specifier for observed perpendicular separations
    :type rpo_binner: :class:`BinSpecifier`
    :param rlo_binner: The bin specifier for observed parallel separations
    :type rlo_binner: :class`BinSpecifier`
    :param zbar_binner: The bin specifier for average observed redshifts
    :type zbar_binner: :class:`BinSpecifier`
    :param params_file: The parameter file from which to get the other details
    for the pair counting
    :type params_file: `str` or :class:`os.PathLike`

    Returns
    -------
    :return nn: The counter object for this set of pair counts. The binning
    information can be recalled using calls to 'nn.*_bin_info' (r_perp, r_par,
    or zbar), the 3D array of counts can be obtained with 'nn.counts', and
    the total number of pairs processed is available via 'nn.n_tot'
    :rtype nn: :class:`NNCounts3D`
    """
    logger = init_logger(__name__)
    if not isinstance(rpo_binner, _calculate_distances.BinSpecifier):
        err_msg = ("Incorrect type for rpo_binner: {}; rpo_binner must be of"
                   " type BinSpecifier".format(type(rpo_binner)))
        logger.error(err_msg)
        raise TypeError(err_msg)
    if rpo_binner.nbins == 0:
        err_msg = ("Values not set for rpo_binner; please make sure bin"
                   " specifications are provided for all bin specifiers")
        logger.error(err_msg)
        raise ValueError(err_msg)
    if not isinstance(rlo_binner, _calculate_distances.BinSpecifier):
        err_msg = ("Incorrect type for rlo_binner: {}; rlo_binner must be of"
                   " type BinSpecifier".format(type(rlo_binner)))
        logger.error(err_msg)
        raise TypeError(err_msg)
    if rlo_binner.nbins == 0:
        err_msg = ("Values not set for rlo_binner; please make sure bin"
                   " specifications are provided for all bin specifiers")
        logger.error(err_msg)
        raise ValueError(err_msg)
    if not isinstance(zbar_binner, _calculate_distances.BinSpecifier):
        err_msg = ("Incorrect type for zbar_binner: {}; zbar_binner must be of"
                   " type BinSpecifier".format(type(zbar_binner)))
        logger.error(err_msg)
        raise TypeError(err_msg)
    if zbar_binner.nbins == 0:
        err_msg = ("Values not set for zbar_binner; please make sure bin"
                   " specifications are provided for all bin specifiers")
        logger.error(err_msg)
        raise ValueError(err_msg)
    logger.info("Reading parameter file")
    params_in = MyConfigObj(params_file, file_error=True)
    try:
        params_in = params_in["run_params"]
    except KeyError:
        pass
    if params_in.as_bool("has_true1"):
        dtcol1 = params_in["dtcol1"]
        if "ztcol1" in params_in:
            ztcol1 = params_in["ztcol1"]
        else:
            ztcol1 = dtcol1.replace("D", "Z")
    else:
        dtcol1 = None
        ztcol1 = None
    if params_in.as_bool("has_obs1"):
        docol1 = params_in["docol1"]
        if "zocol1" in params_in:
            zocol1 = params_in["zocol1"]
        else:
            zocol1 = docol1.replace("D", "Z")
    else:
        docol1 = None
        zocol1 = None
    logger.info("Reading first catalog")
    cat1 = _read_catalog(
        params_in["ifname1"], params_in.as_bool("has_true"),
        params_in.as_bool("has_obs"), dtcol1, docol1, ztcol1, zocol1)
    if params_in["ifname2"] != params_in["ifname1"]:
        if params_in.as_bool("has_true2"):
            dtcol2 = params_in["dtcol2"]
            if "ztcol2" in params_in:
                ztcol2 = params_in["ztcol2"]
            else:
                ztcol2 = dtcol2.replace("D", "Z")
        else:
            dtcol2 = None
            ztcol2 = None
        if params_in.as_bool("has_obs2"):
            docol2 = params_in["docol2"]
            if "zocol2" in params_in:
                zocol2 = params_in["zocol2"]
            else:
                zocol2 = docol2.replace("D", "Z")
        else:
            docol2 = None
            zocol2 = None
        logger.info("Reading second catalog")
        cat2 = _read_catalog(
            params_in["ifname2"], params_in.as_bool("has_true2"),
            params_in.as_bool("has_obs2"), dtcol2, docol2, ztcol2, zocol2)
        is_auto = False
    else:
        logger.info("Performing auto pair count")
        cat2 = cat1
        is_auto = True

    logger.info("Getting pair counts")
    start = time.monotonic()
    nn = _calculate_distances.get_obs_pair_counts(
        cat1, cat2, rpo_binner, rlo_binner, zbar_binner, is_auto)
    stop = time.monotonic()
    logger.info("Finished; elapsed time = {} sec".format(
        timedelta(seconds=(stop - start)).total_seconds()))
    return nn
