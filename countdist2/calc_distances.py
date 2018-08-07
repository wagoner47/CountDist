from __future__ import print_function
import numpy as np
from astropy.table import QTable
import subprocess
import os, sys
# from .file_io import read_files, resave
from .utils import MyConfigObj
from glob import glob


def run_calc(params_file):
    """This function runs the executable for finding the separations between
    galaxies. The only input is the location of the parameter file, which will
    first be read in here. The input catalog file will be checked for type
    compatability, and if it is a FITS file, will be converted to an ascii file
    in the same directory with a new paramter file being created to give the
    correct file. The data is then automatically stored in files, and can be
    read in using the :func:`countdist2.read_files` function.

    Parameters
    ----------
    :param params_file: The parameter file to be used. Note that a temporary
    copy will be made with the appropriate input file readable by the 
    executable, but the temporary file will be removed after the code has run.
    :type params_file: string
    """
    print("Reading parameter file and creating temporary parameter file")
    params_in = MyConfigObj(params_file, file_error=True)
    params_in = params_in["run_params"]
    temp_params_fname = "{}_ascii{}".format(*os.path.splitext(params_file))
    params_out = MyConfigObj(temp_params_fname)
    params_out["rp_min"] = params_in.as_float("rp_min")
    params_out["rp_max"] = params_in.as_float("rp_max")
    params_out["rl_min"] = params_in.as_float("rl_min")
    params_out["rl_max"] = params_in.as_float("rl_max")
    print("Using column '{}' for true distance".format(params_in["dtcol"]))
    if ".fit" in os.path.splitext(params_in["ifname"])[1].lower():
        params_out["ifname"] = os.path.splitext(params_in["ifname"])[0] + ".txt"
        data_in = QTable.read(params_in["ifname"])
        dtcol = params_in["dtcol"]
        docol = params_in["docol"]
        data_in.write(params_out["ifname"], format="ascii.no_header",
                include_names=["RA", "DEC", dtcol, docol], overwrite=True)
        del data_in
        rm1_when_done = True
    else:
        params_out["ifname"] = params_in["ifname"]
        rm1_when_done = False
    rm2_when_done = False
    if "ifname2" in params_in:
        if ".fit" in os.path.splitext(params_in["ifname2"])[1].lower():
            params_out["ifname2"] = os.path.splitext(
                    params_in["ifname2"])[1] + ".txt"
            data_in = QTable.read(params_in["ifname2"])
            if "dtcol2" in params_in:
                dtcol = params_in["dtcol2"]
            if "docol2" in params_in:
                docol = params_in["docol2"]
            data_in.write(params_out["ifname2"], format="ascii.no_header",
                    include_names=["RA", "DEC", dtcol, docol], overwrite=True)
            del data_in
            rm2_when_done = True
        else:
            params_out["ifname2"] = params_in["ifname2"]
    if "db_file" in params_in:
        params_out["db_file"] = params_in["db_file"]
    else:
        params_out["db_file"] = os.path.join(os.getcwd(), "seps_db.sqlite3")
    params_out.write()
    if not os.path.exists(os.path.dirname(params_out["db_file"])):
        os.makedirs(os.path.dirname(params_out["db_file"]))

    print("Running executable")
    sys.stdout.flush()
    command = "run {}".format(temp_params_fname)
    subprocess.check_call(command, shell=True)
    os.remove(temp_params_fname)

    if rm1_when_done:
        os.remove(params_out["ifname"])
    if rm2_when_done:
        os.remove(params_out["ifname2"])

# def pair_counts_perp(rp_min, rp_max, nbins, log_bins=False, load_dir=None):
#     """Get the perpendicular pair counts (i.e. histogram of perpendicular separations) for
#     comparing to other pair counting codes. This should allow for checking the
#     performance of the separations (rather than just that saving doesn't change
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
#     while data["r_perp_t"].max() < rp_max and data["r_perp_o"].max() < (fac*rp_max):
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
#     perpendicular separations) for comparing to other pair counting codes. This
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
#     perp_cond = ((data["r_perp_t"].max() < rp_max) and (data["r_perp_o"].max() \
#             < (pfac * rp_max)))
#     par_cond = ((data["r_par_t"].max() < rl_max) and (data["r_par_o"].max() < \
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
