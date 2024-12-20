# Human friendly input/output in Python.
#
# Author: Peter Odding <peter@peterodding.com>
# Last Change: September 17, 2021
# URL: https://humanfriendly.readthedocs.io
# https://github.com/xolox/python-humanfriendly/blob/master/humanfriendly/__init__.py
"""Copied from the main module of the `humanfriendly` package."""

# Standard library modules.
import collections
import re

# Named tuples to define units of size.
SizeUnit = collections.namedtuple("SizeUnit", "divider, symbol, name")
CombinedUnit = collections.namedtuple("CombinedUnit", "decimal, binary")

# Common disk size units in binary (base-2) and decimal (base-10) multiples.
disk_size_units = (
    CombinedUnit(
        SizeUnit(1000**1, "KB", "kilobyte"), SizeUnit(1024**1, "KiB", "kibibyte")
    ),
    CombinedUnit(
        SizeUnit(1000**2, "MB", "megabyte"), SizeUnit(1024**2, "MiB", "mebibyte")
    ),
    CombinedUnit(
        SizeUnit(1000**3, "GB", "gigabyte"), SizeUnit(1024**3, "GiB", "gibibyte")
    ),
    CombinedUnit(
        SizeUnit(1000**4, "TB", "terabyte"), SizeUnit(1024**4, "TiB", "tebibyte")
    ),
    CombinedUnit(
        SizeUnit(1000**5, "PB", "petabyte"), SizeUnit(1024**5, "PiB", "pebibyte")
    ),
    CombinedUnit(
        SizeUnit(1000**6, "EB", "exabyte"), SizeUnit(1024**6, "EiB", "exbibyte")
    ),
    CombinedUnit(
        SizeUnit(1000**7, "ZB", "zettabyte"), SizeUnit(1024**7, "ZiB", "zebibyte")
    ),
    CombinedUnit(
        SizeUnit(1000**8, "YB", "yottabyte"), SizeUnit(1024**8, "YiB", "yobibyte")
    ),
)


def round_number(count, keep_width=False):
    """Round a floating point number to two decimal places in a human friendly format.

    :param count: The number to format.
    :param keep_width: :data:`True` if trailing zeros should not be stripped,
                       :data:`False` if they can be stripped.
    :returns: The formatted number as a string. If no decimal places are
              required to represent the number, they will be omitted.

    The main purpose of this function is to be used by functions like
    :func:`format_length()`, :func:`format_size()` and
    :func:`format_timespan()`.

    Here are some examples:

    >>> from humanfriendly import round_number
    >>> round_number(1)
    '1'
    >>> round_number(math.pi)
    '3.14'
    >>> round_number(5.001)
    '5'
    """
    text = "%.2f" % float(count)
    if not keep_width:
        text = re.sub("0+$", "", text)
        text = re.sub(r"\.$", "", text)
    return text


def pluralize_raw(count, singular, plural=None):
    """Select the singular or plural form of a word based on a count.

    :param count: The count (a number).
    :param singular: The singular form of the word (a string).
    :param plural: The plural form of the word (a string or :data:`None`).
    :returns: The singular or plural form of the word (a string).

    When the given count is exactly 1.0 the singular form of the word is
    selected, in all other cases the plural form of the word is selected.

    If the plural form of the word is not provided it is obtained by
    concatenating the singular form of the word with the letter "s". Of course
    this will not always be correct, which is why you have the option to
    specify both forms.
    """
    if not plural:
        plural = singular + "s"
    return singular if float(count) == 1.0 else plural


def pluralize(count, singular, plural=None):
    """Combine a count with the singular or plural form of a word.

    :param count: The count (a number).
    :param singular: The singular form of the word (a string).
    :param plural: The plural form of the word (a string or :data:`None`).
    :returns: The count and singular or plural word concatenated (a string).

    See :func:`pluralize_raw()` for the logic underneath :func:`pluralize()`.
    """
    return "%s %s" % (count, pluralize_raw(count, singular, plural))


def format_size(num_bytes, keep_width=False, binary=False):
    """Format a byte count as a human readable file size.

    :param num_bytes: The size to format in bytes (an integer).
    :param keep_width: :data:`True` if trailing zeros should not be stripped,
                       :data:`False` if they can be stripped.
    :param binary: :data:`True` to use binary multiples of bytes (base-2),
                   :data:`False` to use decimal multiples of bytes (base-10).
    :returns: The corresponding human readable file size (a string).

    This function knows how to format sizes in bytes, kilobytes, megabytes,
    gigabytes, terabytes and petabytes. Some examples:

    >>> from humanfriendly import format_size
    >>> format_size(0)
    '0 bytes'
    >>> format_size(1)
    '1 byte'
    >>> format_size(5)
    '5 bytes'
    > format_size(1000)
    '1 KB'
    > format_size(1024, binary=True)
    '1 KiB'
    >>> format_size(1000 ** 3 * 4)
    '4 GB'
    """
    for unit in reversed(disk_size_units):
        if num_bytes >= unit.binary.divider and binary:
            number = round_number(
                float(num_bytes) / unit.binary.divider, keep_width=keep_width
            )
            return pluralize(number, unit.binary.symbol, unit.binary.symbol)
        elif num_bytes >= unit.decimal.divider and not binary:
            number = round_number(
                float(num_bytes) / unit.decimal.divider, keep_width=keep_width
            )
            return pluralize(number, unit.decimal.symbol, unit.decimal.symbol)
    return pluralize(num_bytes, "byte")
