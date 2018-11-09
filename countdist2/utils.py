from __future__ import print_function
import os
from configobj import ConfigObj
import logging
import numpy as np
import math
import astropy.cosmology

cosmo_mapper = {
    "FlatLambda": astropy.cosmology.FlatLambdaCDM,
    "Flatw0": astropy.cosmology.FlatwCDM,
    "Flatw0wa": astropy.cosmology.Flatw0waCDM,
    "Lambda": astropy.cosmology.LambdaCDM,
    "w0": astropy.cosmology.wCDM,
    "w0wa": astropy.cosmology.w0waCDM
    }

class MyConfigObj(ConfigObj):
    def __init__(self, infile=None, options=None, configspec=None,
                 encoding=None, interpolation=True, raise_errors=False,
                 list_values=True, create_empty=False, file_error=False,
                 stringify=True, indent_type=None, default_encoding=None,
                 unrepr=False, write_empty_values=False, _inspec=False):
        super(MyConfigObj, self).__init__(infile, options, configspec, encoding,
                                          interpolation, raise_errors,
                                          list_values, create_empty, file_error,
                                          stringify, indent_type,
                                          default_encoding, unrepr,
                                          write_empty_values, _inspec)
    
    def _write_line(self, indent_string, entry, this_entry, comment):
        if not self.unrepr:
            val = super(MyConfigObj, self)._decode_element(
                super(MyConfigObj, self)._quote(this_entry))
        else:
            val = repr(this_entry)
        
        return "%s%s%s%s%s" % (indent_string,
                               super(MyConfigObj, self)._decode_element(
                                   super(MyConfigObj, self)._quote(entry,
                                                                   multiline=False)),
                               super(MyConfigObj, self)._a_to_u("= "), val,
                               super(MyConfigObj, self)._decode_element(
                                   comment))


def iterable_len(iterable):
    """Get the length of any iterable

    Parameters
    ----------
    :param iterable: Any type of iterable for which to get the sum
    :type iterable: array-like or iterable

    Returns
    -------
    :return: The length of the iterable
    :rtype: `int`
    """
    return math.fsum(1 for _ in iterable)

def ndigits(x):
    """Determine how many digits are in x. Not the same as significant digits: 
    if x < 1, the number of digits will only reflect the first non-zero 
    decimal place.
    
    Parameters
    ----------
    :param x: The number(s) to check for digits
    :type x: `int` or `float`, scalar or array-like
    
    Returns
    -------
    :return: The number of digits in x, or in each element of x for array-like
    :rtype: scalar or ndarray of `int`
    """
    if not hasattr(x, "__len__"):
        x = math.fabs(x)
        if math.isclose(x / 10.0, x):
            return 1
        else:
            return int(math.floor(math.log10(x)))
    else:
        return np.array([ndigits(thisx) for thisx in x])


def init_logger(name=None):
    """Initialize a new logger with the given name and logging level.
    
    Parameters
    ----------
    :param name: The name for the logger. If None, this will be the root 
    logger. Default None
    :type name: `str`, optional
    
    Returns
    -------
    :return logger: The logger object with format and level set
    :rtype logger: :class:`logging.Logger`
    """
    pkg_opts = os.path.join(os.path.dirname(__file__), os.pardir,
        "package_options.ini")
    config = MyConfigObj(pkg_opts)
    fmt = '%(asctime)s %(levelname)s - %(name)s.%(funcName)s (%(lineno)d): %(' \
          'message)s'
    dtfmt = '%m/%d/%Y %H:%M:%S'
    if name is None:
        name = ""
    level = config["level"]
    if level == "None":
        level = "INFO"
    else:
        try:
            assert level.upper() in ["NOTSET", "DEBUG", "INFO", "WARNING", 
                                     "ERROR", "CRITICAL"]
        except AssertionError:
            raise ValueError("Invalid level: {}".format(level))
        finally:
            level = level.upper()
    logging.basicConfig(format=fmt, datefmt=dtfmt)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

def _cosmology_setup(params):
    """A helper function for setting up the mapper name and key word arguments
    for the various possible cosmologies.

    :param params: The cosmological parameters section from the cosmology ini
    file
    :type params: :class:`MyConfigObj`

    :return cosmo_func_name: The string that maps onto the correct
    astropy.cosmology class
    :rtype cosmo_func_name: `str`
    :return ckwargs: A dictionary containing the key word arguments for
    initializing the correct astropy.cosmology class
    :rtype ckwargs: `dict`
    """
    ckwargs = dict(
        H0=(100.0 * params.as_float("h0")),
        Om0=params.as_float("omega_m")
        )
    if "omega_b" in params:
        ckwargs["Ob0"] = params.as_float("omega_b")
    
    if math.isclose(math.fabs(params.as_float("omega_k")), 0.0):
        cosmo_func_name = "Flat"
    else:
        cosmo_func_name = ""
        ckwargs["Ode0"] = (1.0 - params.as_float("omega_m") -
                           params.as_float("omega_k"))

    if math.isclose(params.as_float("w"), -1.0):
        cosmo_func_name += "Lambda"
    else:
        cosmo_func_name += "w0"
        ckwargs["w0"] = params.as_float("w")
        if not math.isclose(math.fabs(params.as_float("wa")), 0.0):
            cosmo_func_name += "wa"
            ckwargs["wa"] = params.as_float("wa")

    return cosmo_func_name, ckwargs

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
    cosmol_params = MyConfigObj(cosmo_file, file_error=True)
    cosmo_func_name, cosmol_kwargs = _cosmology_setup(
        cosmol_params["cosmological_parameters"])
    cosmo = cosmo_mapper[cosmo_func_name](**cosmol_kwargs)
    return cosmo
