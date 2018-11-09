from __future__ import print_function
from .utils import ndigits, iterable_len
import re
import numpy as np
import math

def delatexify(string):
    """This function takes an input string (or iterable of strings) and strips
    the parts that make it 'pretty' for LaTex: any dollar signs, curly braces,
    or backslashes are removed, leaving an unformatted version. This is useful
    for converting between names of things for plotting and names in a function.

    :param string: The string from which to remove formatting, or an iterable of
    such

    :type string: `str` or any array-like or iterable of `str`

    :return: The formatted string, or a list of such

    :rtype: `str` or list of `str`
    """
    if isinstance(string, str):
        return re.sub(r"[\${}\\]", r"", string)
    else:
        return [delatexify(s) for s in string]

def strip_dollar_signs(string):
    """This function takes an input string (or iterable of strings) and strips
    the dollar signs. This is useful for inserting a LaTeX formatted string into
    another LaTeX formatted string.

    :param string: The string from which to remove dollar signs, or an iterable
    of such

    :type string: `str` or any array-like or iterable of `str`

    :return: The stripped string, or a list of them

    :rtype: `str` or list of `str`
    """
    if isinstance(string, str):
        return re.sub(r"\$", r"", string)
    else:
        return [strip_dollar_signs(s) for s in string]

def double_curly_braces(string):
    """This function takes an input string (or iterable of strings) and doubles
    any curly braces present. This is useful when using new-style string format
    syntax in a string with LaTeX formatting.

    :param string: The string in which to double the curly braces, or an
    iterable of such

    :type string: `str` or any array-like or iterable of `str`

    :return: The stripped string, or a list of them

    :rtype: `str` or list of `str`
    """
    if isinstance(string, str):
        return re.sub(r"([{}])", r"\1\1", string)
    else:
        return [double_curly_braces(s) for s in string]

def strip_dollars_and_double_braces(string):
    """This is a convenience function for chaining
    :function:`strip_dollar_signs` and :function:`double_curly_braces`.

    :param string: The string to reformat, or an iterable of such

    :type string: `str` or any array-like or iterable of `str`

    :return: The stripped string, or a list of them

    :rtype: `str` or list of `str`
    """
    return double_curly_braces(strip_dollar_signs(string))

def numeric(string):
    """Convert a string to a number, either an int or a float

    :param string: The string to convert, or an iterable of such

    :type string: `str` or any array-like or iterable of `str`

    :return: The numeric form of the string, or a list of numbers

    :rtype: scalar or list of `int` or `float`
    """
    if isinstance(string, str):
        try:
            return int(string)
        except ValueError:
            return float(string)
    else:
        return [numeric(s) for s in string]

def stringify(number):
    """Convert any number to a string with LaTeX formatting

    :param number: The number to be converted, or an iterable of such

    :type number: `int` or `float`, or any array-like or iterable of these

    :return string: The LaTeX-formatted string reprsentation of the number, or a
    list of such strings

    :rtype string: `str` or list of `str`
    """
    if isinstance(number, (int, float)):
        raw_string = str(number)
        if "e" in raw_string:
            string = r"${parts[0]} \times 10^{{{parts[1]}}}$".format(
                parts=numeric(raw_string.split("e")))
        else:
            string = r"${}$".format(raw_string)
        return string
    else:
        return [stringify(n) for n in number]

def round_to_significant_figures(number, sig_figs=1):
    """Round a number to the specified number of significant figures

    :param number: The number to be rounded, or any iterable of such

    :type number: `int` or `float`, or any array-like or iterable of these

    :param sig_figs: The number of significant figures to which to
    round. Default 1

    :type sig_figs: `int`, optional

    :return: The rounded number, or a list of them

    :rtype: `int` or `float` or list of `int` or `float`
    """
    if isinstance(number, (int, float)):
        is_integer = (isinstance(number, int) or
                      ("e" in str(number) and isinstance(
                    numeric(str(number).split("e")[0]), int)))
        if math.isclose(math.fabs(number), 0):
            return number
        abs_of_number = math.fabs(number)
        sign_of_number = number / abs_of_number
        digits_of_number = ndigits(abs_of_number)
        if digits_of_number > 0:
            digits_of_number -= 1
        rescaled_number = abs_of_number * 10**(-digits_of_number + sig_figs - 1)
        int_part = int(rescaled_number)
        if int((rescaled_number - int_part) * 10) >= 5:
            int_part += 1
        rounded_number = int_part * 10**(digits_of_number - sig_figs + 1)
        if is_integer:
            return sign_of_number * int(rounded_number)
        return sign_of_number * rounded_number
    else:
        return [round_to_significant_figures(n, sig_figs) for n in number]

def pretty_print_number(number, sig_figs=1):
    """Print a number with LaTeX formatting rounded to some number of digits.

    :param number: The number to be formatted, or an iterable of such

    :type number: `int` or `float`, or any array-like or iterable of `int` or
    `float`

    :param sig_figs: The number of significant figures to which to
    round. Default 1

    :type sig_figs: `int`, optional

    :return: The string format of the number after rounding, or a list of them

    :rtype: `str` or list of `str`
    """
    return stringify(round_to_significant_figures(number, sig_figs))
