from __future__ import print_function
import os, sys
import numpy as np
from astropy.table import QTable
import treecorr as tc
from helper_util import timer, my_setup_function, my_teardown_function
from countdist2 import read_files, run_calc, MyConfigObj
from countdist2.calc_distances import pair_counts_perp, pair_counts_perp_par
from nose import with_setup

test_dir = os.path.join(os.environ["CCOUNTDIST"], "tests/test_results/zp0.65-0.7")
test_data = os.path.join(os.environ["CCOUNTDIST"], "tests/test_data", \
        "HALOGEN_mock0000_rm-like_1.0-04_sigmaz0.02_zp0.65-0.7_small.txt")
params_dir = os.path.join(os.environ["CCOUNTDIST"], "tests")
cwd = os.getcwd()


@with_setup(my_setup_function, my_teardown_function)
@timer
def test_pc_perp_too_big():
    """Test getting pair counts when using a larger maximum r_perp than
    available. This test will only pass if the proper error message is given.
    """
    run_calc(os.path.join(params_dir, "my_test_params_small.ini"))
    rp_min = 60.0
    rp_max = 101.0  # This is bigger than my rp_max
    nbins = 50
    with np.testing.assert_raises_regex(ValueError, "Can not reach desired max"\
            " observed separation"):
        pair_counts_perp(rp_min, rp_max, nbins, log_bins=True,
                load_dir=os.path.join(test_dir, "small_range"))
    rp_max = 100.0  # This is at my rp_max saved
    with np.testing.assert_raises_regex(ValueError, "Can not reach desired max"\
            " true separation"):
        pair_counts_perp(rp_min, rp_max, nbins, log_bins=True,
                load_dir=os.path.join(test_dir, "small_range"))


@with_setup(my_setup_function, my_teardown_function)
@timer
def test_pc_perp():
    run_calc(os.path.join(params_dir, "my_test_params_large.ini"))
    rp_min = 60.0
    rp_max = 200.0
    nbins = 50
    tc_config = {"min_sep": rp_min, "max_sep": rp_max, "nbins": nbins,
            "bin_slop": 0}
    tcat = tc.Catalog(test_data, ra_col=1, dec_col=2, ra_units="deg",
            dec_units="deg", r_col=3)
    ocat = tc.Catalog(test_data, ra_col=1, dec_col=2, ra_units="deg",
            dec_units="deg", r_col=4)
    tt = tc.NNCorrelation(tc_config)
    oo = tc.NNCorrelation(tc_config)
    tt.process(tcat, metric="Rperp")
    oo.process(ocat, metric="Rperp")
    pc_test = pair_counts_perp(rp_min, rp_max, nbins, log_bins=True,
            load_dir=os.path.join(test_dir, "large_range"))
    np.testing.assert_allclose(pc_test["r_perp"], tt.rnom, err_msg="Pair "\
            "counts not at the same separations")
    np.testing.assert_array_equal(pc_test["DD_OBS"], oo.npairs.astype(int),
            err_msg="Observed pair counts not the same")
    np.testing.assert_array_equal(pc_test["DD_TRUE"], tt.npairs.astype(int),
            err_msg="True pair counts not the same")


@with_setup(my_setup_function, my_teardown_function)
@timer
def test_pc_perp_par_too_big():
    """Test getting pair counts when using a larger maximum separation than
    available. This test will only pass if the proper error message is given.
    """
    run_calc(os.path.join(params_dir, "my_test_params_small.ini"))
    rp_min = 60.0
    rp_max = 101.0  # This is bigger than my rp_max saved
    np_bins = 50
    rl_min = 60.0
    rl_max = 101.0  # This is bigger than my rl_max saved
    nl_bins = 50
    with np.testing.assert_raises_regex(ValueError, "Can not reach desired max"\
            " observed perpendicular separation"):
        pair_counts_perp_par(rp_min, rp_max, np_bins, rl_min, rl_min + 10,
                nl_bins, True, os.path.join(test_dir, "small_range"))
    with np.testing.assert_raises_regex(ValueError, "Can not reach desired max"\
            " observed parallel separation"):
        pair_counts_perp_par(rp_min, rp_min + 10, np_bins, rl_min, rl_max,
                nl_bins, True, os.path.join(test_dir, "small_range"))
    rp_max = 100.0  # This is at my rp_max saved
    rl_max = 100.0  # This is at my rl_max saved
    with np.testing.assert_raises_regex(ValueError, "Can not reach desired max"\
            " true perpendicular separation"):
        pair_counts_perp(rp_min, rp_max, np_bins, rl_min, rl_min + 10, nl_bins,
                os.path.join(test_dir, "small_range"))
    with np.testing.assert_raises_regex(ValueError, "Can not reach desired max"\
            " true parallel separation"):
        pair_counts_perp(rp_min, rp_min + 10, np_bins, rl_min, rl_max, nl_bins,
                os.path.join(test_dir, "small_range"))


@with_setup(my_setup_function, my_teardown_function)
@timer
def test_pc_perp_par():
    rp_min = 60.0
    rp_max = 200.0
    np_bins = 50
    rl_min = 60.0
    rl_max = 200.0
    nl_bins = 50
    rl_edges, delta = np.linspace(np.log(rl_min), np.log(rl_max),
            num=(nl_bins+1), retstep=True)
    rl_center = np.exp(rl_edges[:-1] + (0.5 * delta))
    rl_edges = np.exp(rl_edges)
    tc_config = {"min_sep": rp_min, "max_sep": rp_max, "nbins": np_bins,
            "bin_slop": 0}
    tcat = tc.Catalog(test_data, ra_col=1, dec_col=2, ra_units="deg",
            dec_units="deg", r_col=3)
    ocat = tc.Catalog(test_data, ra_col=1, dec_col=2, ra_units="deg",
            dec_units="deg", r_col=4)
    pc_test = pair_counts_perp_par(rp_min, rp_max, np_bins, rl_min, rl_max,
            nl_bins, True, os.path.join(test_dir, "large_range"))
    pc_grouped = pc_test.group_by("r_par")
    tt = tc.NNCorrelation(tc_config)
    oo = tc.NNCorrelation(tc_config)
    for rl_min, rl_max, (i, rli) in zip(rl_edges[:-1], rl_edges[1:],
            enumerate(rl_center)):
        tt.process(tcat, metric="Rperp", min_rpar=rl_min, max_rpar=rl_max)
        oo.process(ocat, metric="Rperp", min_rpar=rl_min, max_rpar=rl_max)
        np.testing.assert_allclose(pc_grouped.groups[i]["r_perp"], tt.rnom,
                err_msg="Pair counts not at the same perpendicular separations"\
                        " for r_par = {}".format(rli))
        np.testing.assert_array_equal(pc_grouped.groups[i]["DD_OBS"],
                oo.npairs.astype(int), err_msg="Observed pair counts not "\
                        "the same for r_par = {}".format(rli))
        np.testing.assert_array_equal(pc_grouped.groups[i]["DD_TRUE"],
                tt.npairs.astype(int), err_msg="True pair counts not the "\
                        "same for r_par = {}".format(rli))
