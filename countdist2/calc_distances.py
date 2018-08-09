from __future__ import print_function
from astropy.table import QTable
import subprocess
import os, sys
from .utils import MyConfigObj, init_logger


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
    logger.info("Reading parameter file and creating temporary parameter file")
    params_in = MyConfigObj(params_file, file_error=True)
    try:
        params_in = params_in["run_params"]
    except KeyError:
        pass
    temp_params_fname = "{}_ascii{}".format(*os.path.splitext(params_file))
    params_out = MyConfigObj(temp_params_fname)
    params_out["rp_min"] = params_in.as_float("rp_min")
    params_out["rp_max"] = params_in.as_float("rp_max")
    params_out["rl_min"] = params_in.as_float("rl_min")
    params_out["rl_max"] = params_in.as_float("rl_max")
    params_out["use_true"] = params_in.as_bool("use_true")
    params_out["use_obs"] = params_in.as_bool("use_obs")
    params_out["has_true1"] = params_in.as_bool("has_true1")
    params_out["has_obs1"] = params_in.as_bool("has_obs1")
    params_out["has_true2"] = params_in.as_bool("has_true2")
    params_out["has_obs2"] = params_in.as_bool("has_obs2")
    params_out["table_name"] = params_in["table_name"]
    params_out["meta_name1"] = params_in["meta_name1"]
    params_out["meta_name2"] = params_in["meta_name2"]
    params_out["table_name"] = params_in["table_name"]
    if "db_file" in params_in:
        params_out["db_file"] = params_in["db_file"]
    else:
        params_out["db_file"] = os.path.join(os.getcwd(), "seps_db.sqlite3")
    os.makedirs(os.path.dirname(params_out["db_file"]), exist_ok=True)
    if params_in.as_bool("has_true1"):
        print("Using column '{}' for true distance in catalog 1".format(
            params_in["dtcol1"]))
    if params_in.as_bool("has_true2"):
        print("Using column '{}' for true distance in catalog 2".format(
            params_in["dtcol2"]))

    params_out["ifname1"] = os.path.splitext(params_in["ifname1"])[0] + ".txt"
    data_in = QTable.read(params_in["ifname1"])
    params_out["SIGMA_R_EFF1"] = data_in.meta["SIGMAR"]
    params_out["Z_EFF1"] = data_in.meta["ZEFF"]
    params_out["SIGMA_Z1"] = data_in.meta["SIGMAZ"]
    include_names = ["RA", "DEC"]
    if params_in.as_bool("has_true1"):
        include_names.append(params_in["dtcol1"])
    if params_in.as_bool("has_obs1"):
        include_names.append(params_in["docol1"])
    data_in.write(params_out["ifname1"], format="ascii.no_header",
                  include_names=include_names, overwrite=True)
    del data_in

    params_out["ifname2"] = os.path.splitext(params_in["ifname2"])[0] + ".txt"
    data_in = QTable.read(params_in["ifname2"])
    params_out["SIGMA_R_EFF2"] = data_in.meta["SIGMAR"]
    params_out["Z_EFF2"] = data_in.meta["ZEFF"]
    params_out["SIGMA_Z2"] = data_in.meta["SIGMAZ"]
    include_names = ["RA", "DEC"]
    if params_in.as_bool("has_true2"):
        include_names.append(params_in["dtcol2"])
    if params_in.as_bool("has_obs2"):
        include_names.append(params_in["docol2"])
    data_in.write(params_out["ifname2"], format="ascii.no_header",
                  include_names=include_names, overwrite=True)
    del data_in
    params_out["is_auto"] = (params_out["ifname1"] == params_out["ifname2"])

    params_out.write()

    logger.info("Running executable")
    sys.stdout.flush()
    command = "run {}".format(temp_params_fname)
    subprocess.check_call(command, shell=True)
    os.remove(temp_params_fname)
    os.remove(params_out["ifname1"])
    os.remove(params_out["ifname2"])

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
