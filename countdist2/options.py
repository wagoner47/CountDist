from __future__ import print_function
import os
from .utils import MyConfigObj

def set_logging_level(level="NOTSET"):
    """Set the package wide logging level. If this is called after an
    instance of a class is created, the level for that instance will not be
    updated, so this should be called before running anything else

    :param level: The logging level to set, as a case-insensitive string.
    This must be one of 'NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', or
    'CRITICAL', if set, although the checking is not done here. Default 'NOTSET'
    :type level: `str`, optional
    """
    pkg_opts = os.path.join(os.path.dirname(__file__), os.pardir,
                            "package_options.ini")
    config = MyConfigObj(pkg_opts)
    config["level"] = level
    config.write()
