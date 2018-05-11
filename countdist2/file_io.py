from __future__ import print_function
import numpy as np
import pandas as pd
import os
import sqlite3


DB_COLS = np.array(["R_PERP_O", "R_PAR_O", "R_PERP_T", "R_PAR_T", "ID1", "ID2"])


def make_query(**kwargs):
    """This function makes a query to select items from the table SEPARATIONS.
    Keyword arguments allow the user to specify columns to select or limits to
    use on the query. If nothing is provided, everything will be read from the
    table.

    Keyword Arguments
    -----------------
    :kwarg cols: A column or list of columns to select from the table. Can be
    passed as column names, or numbers referring to column indexes from the
    default order (see below). If `None`, all columns are read. Default `None`
    :type cols: scalar or array-like of `str` or `int`, or `None` 
    :kwarg rpo: Limits to use on the observed perpendicular separations (in
    units of Mpc). If `None`, no limits will be placed on this separation. Default
    `None`
    :type rpo: array-like `float` or `None`
    :kwarg rlo: Limits to use on the observed parallel separations (in units of
    Mpc). If `None`, no limits will be placed on this separation. Default `None`
    :type rlo: array-like `float` or `None`
    :kwarg rpt: Limits to use on the true perpendicular separations (in units of
    Mpc). If `None`, no limits will be placed on this separation. Default `None`
    :type rpt: array-like `float` or `None`
    :kwarg rlt: Limits to use on the true parallel separations (in units of
    Mpc). If `None`, no limits will be placed on this separation. Default `None`
    :type rlt: array-like `float` or `None`

    Returns
    -------
    :return query: A string for the query to be performed, including any
    specified columns to select and any where statements for limits
    :rtype query: `str`

    Notes
    -----
    As mentioned above, columns may also be specified via indices. The default
    order (and indices) is as follows:

    - 0 = 'R_PERP_O': observed perpendicular separation in Mpc
    - 1 = 'R_PAR_O': observed parallel separation in Mpc, always positive
    - 2 = 'R_PERP_T': true perpendicular separation in Mpc
    - 3 = 'R_PAR_T': true parallel separation in Mpc, signed based on
      configuration relative to observed
    - 4 = 'ID1': ID of the first object in the pair
    - 5 = 'ID2': ID of the second object in the pair
    """
    query = "SELECT "

    # Get columns to select
    cols = kwargs.pop("cols", None)
    if cols is not None:
        # Check user-input columns for valid entries
        cols = np.atleast_1d(cols).flatten()
        if cols.dtype.type == np.str_:
            # Make sure none are repeated, and all are upper case
            cols = np.unique(np.char.upper(cols))
            if cols.size > DB_COLS.size:
                # Throw an error because more columns specified than available
                raise ValueError("Too many columns ({}) to select: {} columns "\
                        "available".format(cols.size, DB_COLS.size))
            # Check for any columns given that don't exist
            bad_cols = np.isin(cols, DB_COLS, assume_unique=True, invert=True)
            if np.any(bad_cols):
                raise ValueError("Invalid column(s): {}".format(cols[bad_cols]))
        elif cols.dtype in [int, float]:
            # Make sure none are repeated, all indices are ints
            col_idx = np.unique(cols.astype(int))
            if col_idx.size > DB_COLS.size:
                # Throw an error because more columns specified than available
                raise ValueError("Too many columns ({}) to select: {} columns "\
                        "available".format(col_idx.size, DB_COLS.size))
            # Check for invalid indices
            bad_idx = np.isin(col_idx, np.arange(DB_COLS.size),
                    assume_unique=True, invert=True)
            if np.any(bad_idx):
                raise ValueError("Invalid column indices: "\
                        "{}".format(col_idx[bad_idx]))
            cols = DB_COLS[col_idx]
        else:
            # What type is this?
            raise ValueError("Unknown type for cols: {}".format(cols.dtype))
        if cols.size == DB_COLS.size:
            cols = np.array(["*"])
    else:
        cols = np.array(["*"])
    for col in cols[:-1]:
        query += "{}, ".format(col)
    query += "{} FROM SEPARATIONS".format(col[-1])

    # Get any limits to use
    rpo = kwargs.pop("rpo", None)
    rlo = kwargs.pop("rlo", None)
    rpt = kwargs.pop("rpt", None)
    rlt = kwargs.pop("rlt", None)
    lims = np.array([np.asarray(limi).astype(float) if limi is not None else
        np.full(2, np.nan) for limi in [rpo, rlo, rpt, rlt]])
    if np.any(np.isfinite(lims)):
        query += " WHERE"
        first = True  # Flag to let us know if this is the first set of limits
        for col, col_lims in zip(DB_COLS[:-2], lims):
            # Set up the string for this column where based on if it's the first
            # where entry
            if first:
                col_str = " "
            else:
                col_str = " AND "
            if np.all(np.isfinite(col_lims)):
                # Use a between statement
                first = False
                col_str += "{} BETWEEN {} AND {}".format(col, col_lims[0],
                        col_lims[1])
            elif np.any(np.isfinite(col_lims)):
                # Must figure out which one is finite
                first = False
                if np.isfinite(col_lims[0]):
                    col_str += "{} >= {}".format(col, col_lims[0])
                else:
                    col_str += "{} < {}".format(col, col_lims[1])
            else:
                # None are finite, so just reset col_str
                col_str = ""
            query += col_str

    # Finally, return our string for the query
    return query


def read_db(db_file, **kwargs):
    """This function reads data from the database in the file :param:`db_file`,
    with options given via the keyword arguments

    Parameters
    ----------
    :param db_file: The database file from which to read
    :type db_file: `str`

    Keyword Arguments
    -----------------
    :kwarg cols: A column or list of columns to select from the table. Can be
    passed as column names, or numbers referring to column indexes from the
    default order (see below). If `None`, all columns are read. Default `None`
    :type cols: scalar or array-like of `str` or `int`, or `None` 
    :kwarg rpo: Limits to use on the observed perpendicular separations (in
    units of Mpc). If `None`, no limits will be placed on this separation. Default
    `None`
    :type rpo: array-like `float` or `None`
    :kwarg rlo: Limits to use on the observed parallel separations (in units of
    Mpc). If `None`, no limits will be placed on this separation. Default `None`
    :type rlo: array-like `float` or `None`
    :kwarg rpt: Limits to use on the true perpendicular separations (in units of
    Mpc). If `None`, no limits will be placed on this separation. Default `None`
    :type rpt: array-like `float` or `None`
    :kwarg rlt: Limits to use on the true parallel separations (in units of
    Mpc). If `None`, no limits will be placed on this separation. Default `None`
    :type rlt: array-like `float` or `None`

    Returns
    -------
    :return: A pandas dataframe containing the results of the query
    :rtype: :class:`pandas.DataFrame`

    Notes
    -----
    As mentioned above, columns may also be specified via indices. The default
    order (and indices) is as follows:

    - 0 = 'R_PERP_O': observed perpendicular separation in Mpc
    - 1 = 'R_PAR_O': observed parallel separation in Mpc, always positive
    - 2 = 'R_PERP_T': true perpendicular separation in Mpc
    - 3 = 'R_PAR_T': true parallel separation in Mpc, signed based on
      configuration relative to observed
    - 4 = 'ID1': ID of the first object in the pair
    - 5 = 'ID2': ID of the second object in the pair
    """
    query = make_query(**kwargs)
    conn = sqlite3.connect(db_file)
    return pd.read_sql(query, conn)


def read_db_multiple(db_file, **kwargs):
    """This function allows the user to concatenate multiple reads of the same
    database file to combine calls with various limits for instance.

    Parameters
    ----------
    :param db_file: The database file from which to read
    :type db_file: `str`

    Keyword Arguments
    -----------------
    :kwarg cols: A column or list of columns to select from the table. Can be
    passed as column names, or numbers referring to column indexes from the
    default order (see below). If `None`, all columns are read. Default `None`
    :type cols: scalar or array-like of `str` or `int`, or `None` 
    :kwarg rpo: Limits to use on the observed perpendicular separations (in
    units of Mpc). If `None`, no limits will be placed on this separation in any
    query. Individual entries may also be `None`, so that no limits are used for
    just that query. Default `None`
    :type rpo: 2D array-like `float` or `None`
    :kwarg rlo: Limits to use on the observed parallel separations (in units of
    Mpc). If `None`, no limits will be placed on this separation in any query.
    Individual entries may also be `None`, so that no limits are used for just
    that query. Default `None`
    :type rlo: 2D array-like `float` or `None`
    :kwarg rpt: Limits to use on the true perpendicular separations (in units of
    Mpc). If `None`, no limits will be placed on this separation in any query.
    Individual entries may also be `None`, so that no limits are used for just
    that query. Default `None`
    :type rpt: array-like `float` or `None`
    :kwarg rlt: Limits to use on the true parallel separations (in units of
    Mpc). If `None`, no limits will be placed on this separation in any query.
    Individual entries may also be `None`, so that no limits are used for just
    that query. Default `None`
    :type rlt: array-like `float` or `None`

    Returns
    -------
    :return: Concatenated pandas dataframe for each query
    :rtype: :class:`pandas.DataFrame`

    Notes
    -----
    As mentioned above, columns may also be specified via indices. The default
    order (and indices) is as follows:

    - 0 = 'R_PERP_O': observed perpendicular separation in Mpc
    - 1 = 'R_PAR_O': observed parallel separation in Mpc, always positive
    - 2 = 'R_PERP_T': true perpendicular separation in Mpc
    - 3 = 'R_PAR_T': true parallel separation in Mpc, signed based on
      configuration relative to observed
    - 4 = 'ID1': ID of the first object in the pair
    - 5 = 'ID2': ID of the second object in the pair
    """
    rpo = kwargs.pop("rpo", None)
    rlo = kwargs.pop("rlo", None)
    rpt = kwargs.pop("rpt", None)
    rlt = kwargs.pop("rlt", None)
    all_lims = np.asarray([rpo, rlo, rpt, rlt])
    # Get an array for whether each set of limits is given or None
    lims_not_none = np.array([limi is not None for limi in all_lims])
    if not np.any(lims_not_none):
        # In this case, the user gave all limits as None, so only one query is
        # needed
        return read_db(db_file, **kwargs, rpo=rpo, rlo=rlo, rpt=rpt, rlt=rlt)
    else:
        # At least one set of limits is not None
        first_not_none = all_lims[lims_not_none][0]
        if len(first_not_none) == 2 and not np.any([hasattr(el, "__len__") for el
            in first_not_none]):
            # In this case, the user only provided limits for a single query
            return read_db(db_file, **kwargs, rpo=rpo, rlo=rlo, rpt=rpt,
                    rlt=rlt)
        else:
            # Need to get limits for each query
            shape2d = (len(first_not_none), 2)
            for i in np.where(~lims_not_none)[0]:
                # Set any None limits to nans
                all_lims[i] = np.full(shape2d, np.nan)
            for i in np.where(lims_not_none)[0]:
                # Set limits for each query for columns that are not None always
                for j, lims in enumerate(all_lims[i]):
                    ## Looping over query number and limit for this query for
                    ## column i
                    if lims is not None:
                        # Set limits for column i and query j
                        all_lims[i][j] = np.asarray(lims).astype(float)
                    else:
                        # Limits are None for column i in query j: set as nans
                        all_lims[i][j] = np.full(2, np.nan)
                    # Make sure all_lims for column i is an array
                    all_lims[i] = np.asarray(all_lims[i])
            # Run queries and concatenate before returning
            return pd.concat([read_db(db_file, **kwargs, rpo=all_lims[0, i],
                rlo=all_lims[1, i], rpt=all_lims[2, i], rlt=all_lims[3, i]) for
                i in range(shape2d[0])])
