from __future__ import print_function
from astropy.table import Table
import subprocess
import os, sys
from .utils import MyConfigObj, init_logger
import logging
import calculate_distances as _calculate_distances
import math
import pandas as pd
import numpy as np
import astropy.cosmology
import CatalogUtils


def pylevel_to_cpplevel(logger):
    """This function translates the python logging level filter from the logger
    object to the equivalent level in my C++ code, which is numbered in reverse
    order. It is more a convenience than anything, and should probably not be
    called by the user.

    Parameters
    ----------
    :param logger: A Logger instance with a level set.
    :type logger: :class:`logging.Logger`

    Returns
    -------
    :return: The integer for the corresponding level in the C++ code.
    :rtype: `int`
    """
    cpplevel_mapper = {logging.CRITICAL: 10, logging.ERROR: 20, logging.WARNING:
            30, logging.INFO: 40, logging.DEBUG: 50}
    return cpplevel_mapper[logger.getEffectiveLevel()]


def _read_catalog(file_name, has_true, has_obs, dtcol=None, docol=None):
    if has_obs:
        print("Using column {} for true distances".format(dtcol))
    if "fit" in os.path.splitext(file_name)[1]:
        data = Table.read(file_name)
    else:
        names = ["RA", "DEC"]
        if has_true:
            names.append(dtcol)
        if has_obs:
            names.append(docol)
        data = Table.read(file_name, format="ascii", names=names)
    if not has_true:
        if dtcol is None:
            dtcol = "D_TRUE"
        data[dtcol] = np.nan
    if not has_obs:
        if docol is None:
            docol = "D_OBS"
        data[docol] = np.nan
    cat = _calculate_distances.fill_catalog_vector(data["RA"], data["DEC"], data[dtcol], data[docol])
    return cat


def _initialize_cosmology(cosmo_file):
    """A helper function to chose the correct cosmology to initialize based
    on the parameters in the cosmology parameter file

    Parameters
    ----------
    :param cosmo_file: The path to a file containing the cosmological
    parameters
    :type cosmo_file: `str`

    Returns
    -------
    :return cosmo: The cosmology instance of the correct type from
    `astropy.cosmology`
    :rtype cosmo: An instance of one of the subclasses of
    :class:`astropy.cosmology.FLRW`  
    """
    cosmo_mapper = {
        "FlatLambda": astropy.cosmology.FlatLambdaCDM,
        "Flatw0": astropy.cosmology.FlatwCDM,
        "Flatw0wa": astropy.cosmology.Flatw0waCDM,
        "Lambda": astropy.cosmology.LambdaCDM,
        "w0": astropy.cosmology.wCDM,
        "w0wa": astropy.cosmology.w0waCDM
        }
    cosmol_params = MyConfigObj(cosmo_file, file_error=True)
    cosmol_params = cosmol_params["cosmological_parameters"]
    cosmol_kwargs = dict(
        H0=(100.0 * cosmol_params.as_float("h0")),
        Om0=cosmol_params.as_float("omega_m")
        )
    if "omega_b" in cosmol_params:
        cosmol_kwargs["Ob0"] = cosmol_params.as_float("omega_b")
    if math.isclose(cosmol_params.as_float("omega_k"), 0.0):
        func_name = "Flat"
    else:
        func_name = ""
        cosmol_kwargs["Ode0"] = (1.0 - cosmol_params.as_float("omega_m") -
                                 cosmol_params.as_float("omega_k"))
    if math.isclose(cosmol_params.as_float("w"), -1.0):
        func_name += "Lambda"
    else:
        func_name += "w0"
        cosmol_kwargs["w0"] = cosmol_params.as_float("w")
        if not math.isclose(cosmol_params.as_float("wa"), 0.0):
            func_name += "wa"
            cosmol_kwargs["wa"] = cosmol_params.as_float("wa")
    cosmo = cosmo_mapper[func_name](**cosmol_kwargs)
    return cosmo


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
    survey_volume = _get_survey_volume(params_in)
    cat1 = _read_catalog(params_in["ifname1"], params_in.as_bool("has_true1"), params_in.as_bool("has_obs1"), params_in["dtcol1"] if params_in.as_bool("has_true1") else None, params_in["docol1"] if params_in.as_bool("has_obs1") else None)
    if params_in["ifname2"] != params_in["ifname1"]:
        cat2 = _read_catalog(params_in["ifname2"], params_in.as_bool("has_true2"), params_in.as_bool("has_obs2"), params_in["dtcol2"] if params_in.as_bool("has_true2") else None, params_in["docol2"] if params_in.as_bool("has_obs2") else None)
        is_auto = False
    else:
        cat2 = cat1
        is_auto = True
    logger.info("Running calculation")
    seps_out = _calculate_distances.get_separations(cat1, cat2, survey_volume, params_in.as_float("rp_min"), params_in.as_float("rp_max"), params_in.as_float("rl_min"), params_in.as_float("rl_max"), params_in.as_bool("use_true"), params_in.as_bool("use_obs"), is_auto)
    logger.info("Converting result to DataFrame")
    seps_result = pd.DataFrame.from_dict({"ID1": seps_out.id1, "ID2": seps_out.id2})
    if params_in.as_bool("use_true") and params_in.as_bool("use_obs"):
        seps_result = pd.DataFrame.from_dict({"R_PERP_T": seps_out.r_perp_t, "R_PAR_T": seps_out.r_par_t, "R_PERP_O": seps_out.r_perp_o, "R_PAR_O": seps_out.r_par_o, "AVE_D_OBS": seps_out.ave_r_obs}).join(seps_result)
    elif params_in.as_bool("use_true"):
        seps_result = pd.DataFrame.from_dict({"R_PERP": seps_out.r_perp_t, "R_PAR": seps_out.r_par_t}).join(seps_result)
    elif params_in.as_bool("use_obs"):
        seps_result = pd.DataFrame.from_dict({"R_PERP": seps_out.r_perp_o, "R_PAR": seps_out.r_par_o, "AVE_D_OBS": seps_out.ave_ro}).join(seps_result)
    else:
        # Should never get here, but check just in case
        raise ValueError("Must use at least true or observed distances, or both")
    return seps_result

# def pair_counts_perp(rp_min, rp_max, nbins, log_bins=False, load_dir=None):
#     """Get the perpendicular pair counts (i.e. histogram of perpendicular
# separations) for
#     comparing to other pair counting codes. This should allow for checking the
#     performance of the separations (rather than just that saving doesn't
# change
#     them) and could potentially be used for verification of the convolution
#
#     Parameters
#     ----------
#     :param rp_min: The minimum separation to include in the histogram
#     :type rp_min: float
#     :param rp_max: The maximum separation to include in the histogram
#     :type rp_max: float
#     :param nbins: The number of separation bins to histogram in
#     :type nbins: int
#     :param log_bins: This flag says to use logarithmic binning (for TreeCorr
#     comparison). Default False
#     :type log_bins: bool, optional
#     :param load_dir: The directory from which to read the data. If None, will
#     attempt to read from the current directory. Default None
#     :type load_dir: str
#
#     Returns
#     -------
#     :return pc_table: The pair counts as an astropy QTable. The columns are
#     'r_perp' (the bin centers), 'DD_TRUE', and 'DD_OBS'
#     :rtype pc_table: `astropy.table.QTable`
#     """
#     pc_table = QTable()
#     if log_bins:
#         rp_edges, delta = np.linspace(np.log(rp_min), np.log(rp_max),
#                 num=(nbins+1), retstep=True)
#         rp_center = np.exp(rp_edges[:-1] + (0.5 * delta))
#         rp_edges = np.exp(rp_edges)
#     else:
#         rp_edges, delta = np.linspace(rp_min, rp_max, num=(nbins+1),
#                 retstep=True)
#         rp_center = rp_edges[:-1] + (0.5 * delta)
#     pc_table["r_perp"] = rp_center
#
#     # Note: use very large max value for max's to get all true pairs
#     fac = 10
#     data = read_files(0.0, fac*rp_max, 0.0, 1.0e6, load_dir)
#     if data["r_perp_o"].max() < rp_max:
#         raise ValueError("Can not reach desired max observed separation")
#     while data["r_perp_t"].max() < rp_max and data["r_perp_o"].max() < (
# fac*rp_max):
#         fac *= 10
#         data = read_files(0.0, fac*rp_max, 0.0, 1.0e6, load_dir)
#     if data["r_perp_t"].max() < rp_max:
#         raise ValueError("Can not reach desired max true separation")
#     pc_table["DD_TRUE"] = np.histogram(data["r_perp_t"], bins=rp_edges)[0]
#
#     pc_table["DD_OBS"] = np.histogram(data["r_perp_o"], bins=rp_edges)[0]
#
#     return pc_table
#
#
# def pair_counts_perp_par(rp_min, rp_max, rl_min, rl_max, np_bins, nl_bins,
#         log_bins=False, load_dir=None):
#     """Get the perpendicular and parallel pair counts (i.e. histogram of
#     perpendicular separations) for comparing to other pair counting codes.
# This
#     should allow for checking the performance of the separations (rather than
#     just that saving doesn't change them) and could potentially be used for
#     verification of the convolution
#
#     Parameters
#     ----------
#     :param rp_min: The minimum perpendicular separation to include in the
#     histogram
#     :type rp_min: float
#     :param rp_max: The maximum perpendicular separation to include in the
#     histogram
#     :type rp_max: float
#     :param rl_min: The minimum parallel separation to include in the histogram
#     :type rl_min: float
#     :param rl_max: The maximum parallel separation to include in the histogram
#     :type rl_max: float
#     :param np_bins: The number of separation bins to histogram in the
#     perpendicular direction
#     :type np_bins: int
#     :param nl_bins: The number of separation bins to histogram in the
#     parallel direction
#     :type nl_bins: int
#     :param log_bins: This flag says to use logarithmic binning (for TreeCorr
#     comparison). Default False
#     :type log_bins: bool, optional
#     :param load_dir: The directory from which to read the data. If None, will
#     attempt to read from the current directory. Default None
#     :type load_dir: str
#
#     Returns
#     -------
#     :return pc_table: The pair counts as an astropy QTable. The columns are
#     'r_perp' (the perpendicular bin centers), 'r_par' (the parallel bin
#     centers), 'DD_TRUE', and 'DD_OBS'. Note that the pair counts are flattened
#     to 1D columns
#     :rtype pc_table: `astropy.table.QTable`
#     """
#     pc_table = QTable()
#     if log_bins:
#         rp_edges, delta = np.linspace(np.log(rp_min), np.log(rp_max),
#                 num=(np_bins+1), retstep=True)
#         rp_center = np.exp(rp_edges[:-1] + (0.5 * delta))
#         rp_edges = np.exp(rp_edges)
#         rl_edges, delta = np.linspace(np.log(rl_min), np.log(rl_max),
#                 num=(nl_bins+1), retstep=True)
#         rl_center = np.exp(rl_edges[:-1] + (0.5 * delta))
#         rl_edges = np.exp(rl_edges)
#     else:
#         rp_edges, delta = np.linspace(rp_min, rp_max, num=(np_bins+1),
#                 retstep=True)
#         rp_center = rp_edges[:-1] + (0.5 * delta)
#         rl_edges, delta = np.linspace(rl_min, rl_max, num=(nl_bins+1),
#                 retstep=True)
#         rl_center = rl_edges[:-1] + (0.5 * delta)
#     pc_table["r_perp"] = np.tile(rp_center, nl_bins)
#     pc_table["r_par"] = np.repeat(rl_center, np_bins)
#
#     # Note: use very large max value for max's to get all true pairs
#     pfac = 10
#     lfac = 10
#     data = read_files(0.0, pfac*rp_max, 0.0, lfac*rl_max, load_dir)
#     if data["r_perp_o"].max() < rp_max:
#         raise ValueError("Can not reach desired max observed perpendicular "\
#                 "separation")
#     if data["r_par_o"].max() < rl_max:
#         raise ValueError("Can not reach desired max observed parallel "\
#                 "separation")
#     perp_cond = ((data["r_perp_t"].max() < rp_max) and (data[
# "r_perp_o"].max() \
#             < (pfac * rp_max)))
#     par_cond = ((data["r_par_t"].max() < rl_max) and (data["r_par_o"].max()
#  < \
#             (lfac * rl_max)))
#     while perp_cond or par_cond:
#         if perp_cond:
#             pfac *= 10
#         if par_cond:
#             lfac *= 10
#         data = read_files(0.0, pfac*rp_max, 0.0, lfac*rl_max, load_dir)
#         perp_cond = ((data["r_perp_t"].max() < rp_max) and
#                 (data["r_perp_o"].max() < (pfac * rp_max)))
#         par_cond = ((data["r_par_t"].max() < rl_max) and
#                 (data["r_par_o"].max() < (lfac * rl_max)))
#     if data["r_perp_t"].max() < rp_max:
#         raise ValueError("Can not reach desired max true perpendicular "\
#                 "separation")
#     if data["r_par_t"].max() < rl_max:
#         raise ValueError("Can not reach desired max true parallel "\
#                 "separation")
#     pc_table["DD_TRUE"] = np.ravel(np.histogram2d(data["r_par_t"], \
#             data["r_perp_t"], bins=[rl_edges, rp_edges])[0])
#     pc_table["DD_OBS"] = np.ravel(np.histogram2d(data["r_par_o"], \
#             data["r_perp_o"], bins=[rl_edges, rp_edges])[0])
#
#     return pc_table
