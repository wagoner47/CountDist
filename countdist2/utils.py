from __future__ import print_function
import os
from configobj import ConfigObj
import logging


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
    :return digits: The number of digits in x, or in each element of x for 
    array-like
    :rtype digits: `int`, scalar or ndarray
    """
    x = np.atleast_1d(x)
    digits = np.empty(x.shape, dtype=int)
    # Zeros must be separated because log10(0) = -inf: catch them by finding
    # where x / 10 is the same as x
    zeros = ((x / 10.0 == x))
    # Zero has 1 digit
    digits[zeros] = 1
    # The rest can be obtained using log10 and floor
    digits[~zeros] = np.floor(np.log10(np.abs(x[~zeros]))).astype(int)
    if x.size == 1:
        # Convert back to scalar if x was a scalar
        digits = digits.item()
    # Return the number of digits
    return digits


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
    config = MyConfigObj(os.path.join(os.pardir, "package_options.ini"))
    fmt = '%(asctime)s %(levelno)s - %(name)s.%(funcName)s (%(lineno)d): %(message)s'
    dtfmt = '%m/%d/%Y %H:%M:%S'
    if name is None:
        name = ""
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
