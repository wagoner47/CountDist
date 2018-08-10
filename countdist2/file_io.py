from __future__ import print_function
import numpy as np
import pandas as pd
import sqlite3

B_DB_COLS = np.array(["R_PERP_O", "R_PAR_O", "R_PERP_T", "R_PAR_T",
                      "AVE_LOS_OBS"])
S_DB_COLS = np.array(["R_PERP", "R_PAR"])


def set_col_select(cols, valid_cols):
    """A convenience function for creating the SELECT clause for the query.
    This is not meant to be called on its own, and assumes that at least one
    name from :param:`valid_cols` is present in :param:`cols` without
    checking to make sure that all names in :param:`cols` are valid together.
    It only includes in the statement those names from :param:`cols` that are
    valid

    Parameters
    ----------
    :param cols: The column names to possible select
    :type cols: 1D array-like of `str`
    :param valid_cols: The valid possible column names
    :type valid_cols: 1D array-like of `str`

    Returns
    -------
    :return select_str: A string for the SELECT clause for the query
    :rtype select_str: `str`
    """
    select_str = "SELECT "
    if cols.size == valid_cols.size:
        # Case: selecting all columns
        select_str += "*"
        return select_str
    # If we make it here, only some of the columns are being selected. Find
    # the valid ones
    use_cols = cols[np.in1d(cols, valid_cols, assume_unique=True)]
    # Now loop over all valid column names
    for i, col in enumerate(use_cols):
        select_str += col
        if i < use_cols.size - 1:
            # Not the last column, so need a comma and another space
            select_str += ", "
        else:
            # Last one: do nothing
            pass
    # Return the string
    return select_str


def set_query_limits(keys_to_use, limits_dict):
    """A convenience function for creating the string for the WHERE part of
    the query (for column limits). This is not meant to be called on its own,
    and does not check to make sure the keys being used are valid keys

    Parameters
    ----------
    :param keys_to_use: The keys from the limits dictionary that are valid to
    use in the criteria
    :type keys_to_use: 1D array-like of `str`
    :param limits_dict: The dictionary containing the limits that may be
    given. At least one limit should be non-trivial
    :type limits_dict: `dict`

    Returns
    -------
    :return where_str: A string of the condition to add to the query
    :rtype where_str: `str`
    """
    where_str = " WHERE"
    first = True  ## Flag to tell us if this is the first condition
    # Loop over keys
    for key in keys_to_use:
        lim = np.sort(limits_dict[key])
        if np.any(np.isfinite(lim)):
            # Case: at least 1 non-trivial limit. Start adding to the string
            # and change the first flag to false (even if already false)
            if first:
                col = " "  ## Only a space to start the first
            else:
                col = " AND "  ## Join with AND if not first
            first = False
            if np.all(np.isfinite(lim)):
                # Case: both limits non-trivial -- use between
                col += "{} BETWEEN {} AND {}".format(key, lim[0], lim[1])
            else:
                # Case: only 1 non-trivial limit
                if np.isfinite(lim[0]):
                    # Case: only non-trivial min
                    col += "{} >= {}".format(key, lim[0])
                else:
                    # Case: only non-trivial max
                    col += "{} < {}".format(key, lim[1])
            # Add to condition string
            where_str += col
        else:
            # Case: no non-trivial limits for this key. Skip it
            pass
    # Return the completed condition string
    return where_str


def make_query(table_name, **kwargs):
    """This function makes a query to select items from the table
    SEPARATIONS.
    Keyword arguments allow the user to specify columns to select or limits to
    use on the query. If nothing is provided, everything will be read from the
    table.

    Parameters
    ----------
    :param table_name: The name of the table to read within the database
    :type table_name: `str`

    Keyword Arguments
    -----------------
    :kwarg cols: A column name or list of column names to select from the
    table. See the notes below for allowed column names. If `None`,
    all columns are read. Default `None`
    :type cols: scalar or array-like of `str`, or `None`
    :kwarg limits: A dictionary containing the limits to use (if any) for
    each column. Keys should be given by valid column names (see notes). Any
    column not included will not have limits. The values for the limits
    should be a 2-element array-like to specify the min and max, but an
    infinity for either will say that only a min or a max is to be specified.
    If `None`, no limits will be used on any column. Default `None`
    :type limits: `dict` or `None`

    Returns
    -------
    :return query: A string for the query to be performed, including any
    specified columns to select and any where statements for limits
    :rtype query: `str`

    Notes
    -----
    The valid columns are different depending on whether both true and
    observed separations are included in the database. For both, the columns
    are:

    - 'R_PERP_O': The observed perpendicular separation in Mpc
    - 'R_PAR_O': The observed parallel separation in Mpc, with a value that
    is always non-negative
    - 'R_PERP_T': The true perpendicular separation in Mpc
    - 'R_PAR_T': The true parallel separation in Mpc, with a sign based on
    the relative orientation between true and observed
    - 'AVE_OBS_LOS': The average observed LOS distance to the pair of
    galaxies in Mpc

    If only true or observed (and not both) separations are available,
    the column names are:

    - 'R_PERP': The perpendicular separation in Mpc
    - 'R_PAR': The parallel separation in Mpc, with a value that is always
    non-negative
    """
    # Get columns to select
    cols = kwargs.pop("cols", None)
    if cols is not None:
        # Case: at least some column names have been given
        # We will make sure this is a 1D array, with no repeated names and
        # everything uppercase
        cols = np.unique(np.char.upper(cols))
        # Check user-input columns for valid entries
        if np.any(np.in1d(S_DB_COLS, cols, assume_unique=True)) and np.any(
                np.in1d(B_DB_COLS, cols, assume_unique=True)):
            # Case: User provided at least one column from both types. This
            # is invalid, and should throw an error
            raise ValueError("Invalid column name combination: cannot have "
                             "columns from both database types")
        elif np.any(np.in1d(S_DB_COLS, cols, assume_unique=True)):
            # Case: only true or observed. Check against S_DB_COLS
            select_str = set_col_select(cols, S_DB_COLS)
        elif np.any(np.in1d(B_DB_COLS, cols, assume_unique=True)):
            # Case: both true and observed. Check against B_DB_COLS
            select_str = set_col_select(cols, B_DB_COLS)
        else:
            # Case: no column names of either database given. Throw an error
            # because we don't have any valid columns
            raise ValueError("No valid column names given for any database")
    else:
        # Case: No columns are given means all columns are selected
        select_str = "SELECT *"

    # Now check for any limits
    limits = kwargs.pop("limits", None)
    if limits is None:
        # Case: no limits
        where_str = ""
    else:
        # Case: at least some limits given. First check if all of the values
        # are infinity
        if np.all(np.isinf(list(limits.values()))):
            where_str = ""
        # Now check for valid column name combinations
        limit_keys = np.char.upper(list(limits.keys()))
        if np.count_nonzero(np.unique(limit_keys, return_counts=True)[1] > 1) \
                > 0:
            # Case: at least one key has been used twice. Throw an error
            raise ValueError("At least one column name repeated in limits")
        if np.any(np.in1d(S_DB_COLS, limit_keys, assume_unique=True)) and \
                np.any(np.in1d(B_DB_COLS, limit_keys, assume_unique=True)):
            # Case: some of each type given. Throw an error
            raise ValueError("Invalid combination of column names in limits. "
                             "Cannot mix column names from both database types")
        elif np.any(np.in1d(S_DB_COLS, limit_keys, assume_unique=True)):
            # Case: Single separation type. Use only the keys that are in
            # S_DB_COLS
            use_keys = limit_keys[np.in1d(limit_keys, S_DB_COLS,
                                          assume_unique=True)]
            where_str = set_query_limits(use_keys, limits)
        elif np.any(np.in1d(B_DB_COLS, limit_keys, assume_unique=True)):
            # Case: Both separation types. Use only the keys that are in
            # B_DB_COLS
            use_keys = limit_keys[np.in1d(limit_keys, B_DB_COLS,
                                          assume_unique=True)]
            where_str = set_query_limits(use_keys, limits)
        else:
            # Case: no valid column names in limits -- assume no limits
            where_str = ""

    # Now construct the query and return
    query = "{} FROM {}{}".format(select_str, table_name, where_str)
    return query


def read_db(db_file, table_name, **kwargs):
    """This function reads data from the database in the file :param:`db_file`,
    with options given via the keyword arguments

    Parameters
    ----------
    :param db_file: The database file from which to read
    :type db_file: `str`
    :param table_name: The name of the table to read
    :type table_name: `str`

    Keyword Arguments
    -----------------
    :kwarg cols: A column name or list of column names to select from the
    table. See the notes below for allowed column names. If `None`,
    all columns are read. Default `None`
    :type cols: scalar or array-like of `str`, or `None`
    :kwarg limits: A dictionary containing the limits to use (if any) for
    each column. Keys should be given by valid column names (see notes). Any
    column not included will not have limits. The values for the limits
    should be a 2-element array-like to specify the min and max, but an
    infinity for either will say that only a min or a max is to be specified.
    If `None`, no limits will be used on any column. Default `None`
    :type limits: `dict` or `None`

    Returns
    -------
    :return results: A pandas dataframe containing the results of the query
    :rtype results: :class:`pandas.DataFrame`

    Notes
    -----
    The valid columns are different depending on whether both true and
    observed separations are included in the database. For both, the columns
    are:

    - 'R_PERP_O': The observed perpendicular separation in Mpc
    - 'R_PAR_O': The observed parallel separation in Mpc, with a value that
    is always non-negative
    - 'R_PERP_T': The true perpendicular separation in Mpc
    - 'R_PAR_T': The true parallel separation in Mpc, with a sign based on
    the relative orientation between true and observed
    - 'AVE_OBS_LOS': The average observed LOS distance to the pair of
    galaxies in Mpc

    If only true or observed (and not both) separations are available,
    the column names are:

    - 'R_PERP': The perpendicular separation in Mpc
    - 'R_PAR': The parallel separation in Mpc, with a value that is always
    non-negative
    """
    query = make_query(table_name, **kwargs)
    conn = sqlite3.connect(db_file)
    results = pd.read_sql(query, conn)
    conn.close()
    return results


def read_db_multiple(db_file, table_name, limits=None, cols=None):
    """This function allows the user to concatenate multiple reads of the same
    database file to combine calls with various limits for instance. However,
    it assumes all queries are to be made on the same table within the database

    Parameters
    ----------
    :param db_file: The database file from which to read
    :type db_file: `str`
    :param table_name: The name of the table to read
    :type table_name: `str`
    :param limits: A dictionary containing the limits to use on each column,
    if any. The values should all be array-like of shape (N, 2), where N is
    the number of queries being made. Any column with no limits for any query
    does not need to be included, but columns with limits on only some
    queries should have +/- infinity for the limits not to be used, as these
    will be interpreted as no limit. This also holds for columns where only a
    minimum or maximum is to be used.
    :type limits: `dict`
    :param cols: A column name or list of column names to select from the
    table. See the notes below for allowed column names. If `None`,
    all columns are read. Default `None`
    :type cols: scalar or array-like of `str`, or `None`

    Returns
    -------
    :return: Concatenated pandas dataframe for each query
    :rtype: :class:`pandas.DataFrame`

    Notes
    -----
    The valid columns are different depending on whether both true and
    observed separations are included in the database. For both, the columns
    are:

    - 'R_PERP_O': The observed perpendicular separation in Mpc
    - 'R_PAR_O': The observed parallel separation in Mpc, with a value that
    is always non-negative
    - 'R_PERP_T': The true perpendicular separation in Mpc
    - 'R_PAR_T': The true parallel separation in Mpc, with a sign based on
    the relative orientation between true and observed
    - 'AVE_OBS_LOS': The average observed LOS distance to the pair of
    galaxies in Mpc

    If only true or observed (and not both) separations are available,
    the column names are:

    - 'R_PERP': The perpendicular separation in Mpc
    - 'R_PAR': The parallel separation in Mpc, with a value that is always
    non-negative
    """
    if limits is None:
        return read_db(db_file, table_name, cols=cols)
    if np.all(np.isinf(list(limits.values()))):
        # Case: no non-trivial limits, so don't need multiple queries
        return read_db(db_file, table_name, cols=cols)
    nqueries = np.atleast_2d(list(limits.values())[0]).shape[0]
    if nqueries == 1:
        # Case: only one set of limits given, so don't need multiple queries
        limi.fromkeys(limits.keys())
        for key in limi:
            limi[key] = np.squeeze(limits[key])
        return read_db(db_file, table_name, cols=cols, limits=limi)
    # If we made it here, we have more than one query
    limits_list = [limi.fromkeys(limits.keys()) for i in range(nqueries)]
    for i in range(nqueries):
        for key in limits:
            limits_list[i][key] = limits[key][i]
    return pd.concat([read_db(db_file, table_name, cols=cols, limits=limi)
                      for limi in limits_list])


def read_metadata(db_file, data=True, random=True):
    """Read the metadata for the catalogs from the database file. The
    metadata for the data or random catalogs can be optionally read by
    setting each to True or False (default is that both are read). The
    metadata is then returned as a dictionary for each, or None for catalogs
    that weren't selected.

    Parameters
    ----------
    :param db_file: The path to the database file to read
    :type db_file: `str`
    :param data: Flag to read (`True`) or not read (`False`) the metadata for the data catalog. Default `True`
    :type data: `bool`, optional
    :param random: Flag to read (`True`) or not read (`False`) the metadata for
    the random catalog. Default `True`
    :type random: `bool`, optional

    Returns
    -------
    :return data_meta: A dictionary containing the metadata for the data
    catalog, or `None` if :param:`data` is set to `False`
    :rtype data_meta: `dict` or `None`
    :return rand_meta: A dictionary containing the metadata for the random
    catalog, or `None` if :param:`random` is set to `False`
    :rtype rand_meta: `dict` or `None`
    """
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    if data:
        try:
            c.execute("SELECT * FROM DataMeta")
            r = c.fetchone()
            data_meta = dict.fromkeys(r.keys())
            for key in data_meta:
                data_meta[key] = r[key]
        except sqlite3.OperationalError:
            data_meta = None
    else:
        data_meta = None
    if random:
        try:
            c.execute("SELECT * FROM RandomMeta")
            r = c.fetchone()
            rand_meta = dict.fromkeys(r.keys())
            for key in rand_meta:
                rand_meta[key] = r[key]
        except sqlite3.OperationalError:
            rand_meta = None
    else:
        rand_meta = None
    conn.close()

    return data_meta, rand_meta
