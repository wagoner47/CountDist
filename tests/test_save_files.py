from __future__ import print_function
import os, sys
import numpy as np
from astropy.table import QTable, vstack, Column, unique
import subprocess
from helper_util import timer, my_setup_function, my_teardown_function
from glob import glob
from countdist2.file_io import DELTAP, DELTAL
from countdist2 import resave, read_files
from nose import with_setup
from nose.tools import nottest

run_dir = os.environ["CCOUNTDIST"]
test_dir = os.path.join(run_dir, "tests")
cwd = os.getcwd()


@with_setup(my_setup_function, my_teardown_function)
@timer
def test_run_path():
    assert os.path.exists(os.path.join(run_dir, "run")), "Can not find "\
            "run executable"



@with_setup(my_setup_function, my_teardown_function)
@timer
def test_run_no_params():
    os.chdir(run_dir)
    with np.testing.assert_raises(subprocess.CalledProcessError):
        subprocess.check_call("./run", shell=True)
    os.chdir(cwd)


@with_setup(my_setup_function, my_teardown_function)
@timer
def test_run_flag_no_params():
    os.chdir(run_dir)
    with np.testing.assert_raises(subprocess.CalledProcessError):
        subprocess.check_call("./run -t", shell=True)
        subprocess.check_call("./run --test", shell=True)
    os.chdir(cwd)



@with_setup(my_setup_function, my_teardown_function)
@timer
def test_run_code():
    os.chdir(run_dir)
    command = "./run {}".format(os.path.join(test_dir, "my_test_params.ini"))
    subprocess.check_call(command, shell=True)
    os.chdir(cwd)
    assert not os.path.exists(os.path.join(test_dir, \
            "test_results/zp0.65-0.7/temp")), "Temporary files not deleted"


#@nottest
@with_setup(my_setup_function, my_teardown_function)
@timer
def test_sep_files():
    os.chdir(run_dir)
    command = "./run {} --test".format(os.path.join(test_dir, "my_test_params.ini"))
    subprocess.check_call(command, shell=True)
    os.chdir(cwd)
    temp_files = glob(os.path.join(test_dir, 
            "test_results/zp0.65-0.7/temp/temp_file*.txt"))
    data_temp = vstack([QTable.read(tempi, format="ascii.commented_header",
            delimiter=" ", comment="# ") for tempi in temp_files])
    grid_files = glob(os.path.join(test_dir, "test_results/zp0.65-0.7/file*.txt"))
    nlines = 0
    for gridi in grid_files:
        nlines += sum(1 for line in open(gridi) if gridi[0] != "#")
    data_grid = vstack([QTable.read(filei, format="ascii.commented_header", 
            delimiter=" ", comment="# ") for filei in
            glob(os.path.join(test_dir, "test_results/zp0.65-0.7/file*.txt"))])
    np.testing.assert_equal(len(data_grid), len(data_temp), err_msg="Number "\
            "of pairs not the same in grid files")
    col_names_temp = data_temp.colnames
    col_names_grid = data_grid.colnames
    np.testing.assert_array_equal(col_names_grid, col_names_temp, err_msg="Column "\
            "names are not the same in grid files")
    data_temp.add_column(Column(np.trunc(data_temp["r_perp_o"] /
        DELTAP).astype(int)), name="rp_idx")
    data_temp.add_column(Column(np.trunc(data_temp["r_par_o"] /
        DELTAL).astype(int)), name="rl_idx")
    gdata_temp = data_temp.group_by(["rp_idx", "rl_idx"])
    data_grid.add_column(Column(np.trunc(data_grid["r_perp_o"] /
        DELTAP).astype(int)), name="rp_idx")
    data_grid.add_column(Column(np.trunc(data_grid["r_par_o"] /
        DELTAL).astype(int)), name="rl_idx")
    gdata_grid = data_grid.group_by(["rp_idx", "rl_idx"])
    for key, (gnum, group) in zip(gdata_grid.groups.keys, enumerate(gdata_grid.groups)):
        for col in col_names_temp:
            np.testing.assert_allclose(group[col], gdata_temp.groups[gnum][col],
                    err_msg="Column {} not the same in grid files".format(col))


@with_setup(my_setup_function, my_teardown_function)
@timer
def test_grid_fits_files():
    os.chdir(run_dir)
    command = "./run {} --test".format(os.path.join(test_dir,
        "my_test_params.ini"))
    subprocess.check_call(command, shell=True)
    os.chdir(cwd)
    temp_files = glob(os.path.join(test_dir, 
            "test_results/zp0.65-0.7/temp/temp_file*.txt"))
    resave(os.path.join(test_dir, "test_results"))
    data_temp = vstack([QTable.read(tempi, format="ascii.commented_header",
            delimiter=" ", comment="# ") for tempi in temp_files])
    data_grid = vstack([QTable.read(filei) for filei in
            glob(os.path.join(test_dir, "test_results/zp0.65-0.7/file*.fits"))])
    np.testing.assert_equal(len(data_grid), len(data_temp), err_msg="Number "\
            "of pairs not the same in grid fits files")
    col_names_temp = data_temp.colnames
    col_names_grid = data_grid.colnames
    np.testing.assert_array_equal(col_names_grid, col_names_temp, err_msg="Column "\
            "names are not the same in grid fits files")
    data_temp.add_column(Column(np.trunc(data_temp["r_perp_o"] /
        DELTAP).astype(int)), name="rp_idx")
    data_temp.add_column(Column(np.trunc(data_temp["r_par_o"] /
        DELTAL).astype(int)), name="rl_idx")
    gdata_temp = data_temp.group_by(["rp_idx", "rl_idx"])
    data_grid.add_column(Column(np.trunc(data_grid["r_perp_o"] /
        DELTAP).astype(int)), name="rp_idx")
    data_grid.add_column(Column(np.trunc(data_grid["r_par_o"] /
        DELTAL).astype(int)), name="rl_idx")
    gdata_grid = data_grid.group_by(["rp_idx", "rl_idx"])
    for key, (gnum, group) in zip(gdata_grid.groups.keys, enumerate(gdata_grid.groups)):
        for col in col_names_temp:
            np.testing.assert_allclose(group[col], gdata_temp.groups[gnum][col],
                    err_msg="Column {} not the same in grid fits files".format(col))


@with_setup(my_setup_function, my_teardown_function)
@timer
def test_read_grid_files():
    os.chdir(run_dir)
    command = "./run {} --test".format(os.path.join(test_dir,
        "my_test_params.ini"))
    subprocess.check_call(command, shell=True)
    os.chdir(cwd)
    temp_files = glob(os.path.join(test_dir, 
            "test_results/zp0.65-0.7/temp/temp_file*.txt"))
    resave(os.path.join(test_dir, "test_results/zp0.65-0.7"))
    data_temp = vstack([QTable.read(tempi, format="ascii.commented_header",
            delimiter=" ", comment="# ") for tempi in temp_files])
    rp_min = data_temp["r_perp_o"].min()
    rp_max = data_temp["r_perp_o"].max()
    rl_min = data_temp["r_par_o"].min()
    rl_max = data_temp["r_par_o"].max()
    data_grid = read_files(rp_min, rp_max, rl_min, rl_max,
        os.path.join(test_dir, "test_results"))
    np.testing.assert_equal(len(data_grid), len(data_temp), err_msg="Number "\
            "of pairs not the same in grid files read")
    col_names_temp = data_temp.colnames
    col_names_grid = data_grid.colnames
    np.testing.assert_array_equal(col_names_grid, col_names_temp, err_msg="Column "\
            "names are not the same in grid files read")
    data_temp.add_column(Column(np.trunc(data_temp["r_perp_o"] /
        DELTAP).astype(int)), name="rp_idx")
    data_temp.add_column(Column(np.trunc(data_temp["r_par_o"] /
        DELTAL).astype(int)), name="rl_idx")
    gdata_temp = data_temp.group_by(["rp_idx", "rl_idx"])
    data_grid.add_column(Column(np.trunc(data_grid["r_perp_o"] /
        DELTAP).astype(int)), name="rp_idx")
    data_grid.add_column(Column(np.trunc(data_grid["r_par_o"] /
        DELTAL).astype(int)), name="rl_idx")
    gdata_grid = data_grid.group_by(["rp_idx", "rl_idx"])
    for key, (gnum, group) in zip(gdata_grid.groups.keys, enumerate(gdata_grid.groups)):
        for col in col_names_temp:
            np.testing.assert_allclose(group[col], gdata_temp.groups[gnum][col],
                    err_msg="Column {} not the same in grid files read".format(col))
