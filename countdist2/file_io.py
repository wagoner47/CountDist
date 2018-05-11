from __future__ import print_function
import numpy as np
from astropy.table import QTable, vstack, Column
import pandas as pd
import os
# import gc
from glob import glob
import re
import sqlite3


def read_files(rpo_min, rpo_max, rlo_min, rlo_max, db_file=None):
    """This function is for reading the files saved using
    :func:`write_files`, accounting for the gridding done in that
    function. The user specifies the range in observed separations desired,
    and this function automatically reads the data from the files
    corresponding to those ranges and then cuts the resulting data to only
    include the ranges specified. Please note that the min and max
    separations must be given in the same units as the stored data for the
    file numbering to be properly reproduced.
    
    Parameters
    ----------
    :param rpo_min: The minimum observed perpendicular separation to read
    :type rpo_min: float
    :param rpo_max: The maximum observed perpendicular separation to read
    :type rpo_max: float
    :param rlo_min: The minimum observed parallel separation to read
    :type rlo_min: float
    :param rlo_max: The maximum observed parallel separation to read
    :type rlo_max: float
    :param db_file: The database file name with the data table. If None, will
    try to use the default 'seps_db.sqlite3' in the current working directory.
    Default None
    :type db_file: str or None, optional
    
    Returns
    -------
    :return data: A table containing the separations (and potentially
    positions of pairs) within the ranges specified
    :rtype data: astropy.table.QTable
    """
    if db_file is None:
        db_file = os.path.join(os.getcwd(), "seps_db.sqlite3")
    db = sqlite3.connect(db_file)
    data = QTable.from_pandas(pd.read_sql_query("SELECT * FROM SEPARATIONS "\
            "WHERE R_PERP_O >= {} AND R_PERP_O <= {} "\
            "AND R_PAR_O >= {} AND R_PAR_O <= {}".format(rpo_min, rpo_max,
                rlo_min, rlo_max), db))
    data["ID1"] = data["ID1"].astype(np.uint64)
    data["ID2"] = data["ID2"].astype(np.uint64)
    return data


def resave(dir_name):
    """This function takes the grid files that have been saved as ascii files
    and converts them to FITS files, deleting each ascii file only when the
    corresponding FITS file has been saved. This is to make the saved files
    smaller.
    
    Parameters
    ----------
    :param dir_name: The directory in which the grid files are saved.
    :type dir_name: str

    Notes
    -----
    This function does not return anything. It is merely used to transfer the
    data from ascii files to FITS files, which should be smaller in memory.
    """
    for fname in glob(os.path.join(dir_name, "file_*.txt")):
        with open(fname, "r") as fin:
            fline = fin.readline()
        if fline.startswith("#"):
            data = QTable.read(fname, format="ascii.commented_header",
                    delimiter=" ", comment="# ")
        else:
            data = QTable.read(fname, format="ascii.no_header",
                    names=["r_perp_t", "r_par_t", "r_perp_o", "r_par_o", "id1",
                        "ra1", "dec1", "dt1", "do1", "id2", "ra2", "dec2",
                        "dt2", "do2"], delimiter=" ", comment="# ")
        data.write("{}.fits".format(os.path.splitext(fname)[0]), overwrite=True)
        os.remove(fname)
        del data
