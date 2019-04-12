from __future__ import print_function

import functools
from datetime import datetime


def timer(f):
    """A decorator to put before each test function to have it output how
    long it took to run the test.
    """

    @functools.wraps(f)
    def f2(*args, **kwargs):
        fname = repr(f).split()[1]
        print(fname)
        start = datetime.now()
        result = f(*args, **kwargs)
        end = datetime.now()
        print("Time for {} = {}\n".format(fname, end - start))
        return result

    return f2


def my_setup_function():
    print("\n")


def my_teardown_function():
    pass
