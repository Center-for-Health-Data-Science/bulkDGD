#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    tableio.py
#
#    One place where the package decides how a table is written.
#
#    Copyright (C) 2026 Valentina Sora
#                       <sora.valentina1@gmail.com>
#
#    This program is free software: you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public
#    License along with this program.
#    If not, see <http://www.gnu.org/licenses/>.


#######################################################################


__doc__ = \
    """One place where the package decides how a table is written.

    WHY IT IS ONE PLACE. Every output the package produces is a table of
    float64, and a float64 written as text does not come back as the
    number that was written: the decimal form is rounded, and a round
    trip moves the value by up to about 1e-12. That is invisible in a
    printed column and fatal to anything that compares two runs, which
    is most of what this package is used for. Parquet stores the bits.

    The rule was previously applied file by file, which is how it came
    to hold for the training outputs and not for the representations,
    then for the representations and not for the differential
    expression, and so on. Selective application is the failure mode, so
    the decision lives here and every writer in the package calls it.

    THE FORMAT FOLLOWS THE PATH'S EXTENSION, so a configuration that
    names a '.parquet' output gets Parquet and one that names a '.csv'
    output still gets text. Nothing is silently rewritten under a caller
    that asked for something else, and a caller that wants the lossless
    default simply names it."""


#######################################################################


# Import from third-party libraries.
import pandas as pd


#######################################################################


# The extensions that mean Parquet. Kept here so that the two private
# copies that used to live in 'repio' and 'decoutio' cannot drift apart.
PARQUET_EXTENSIONS = (".parquet", ".pq")

# What a new output is called when the caller does not say. Parquet,
# for the reason in the module docstring.
DEFAULT_TABLE_EXT = ".parquet"

# The order a reader tries extensions in when it is looking for a table
# whose format it does not know. Parquet first because it is what is
# written now; '.csv' last because it is what was written before, and
# the data already on disk has to keep loading.
READ_EXTENSIONS = PARQUET_EXTENSIONS + (".csv",)


#######################################################################


def is_parquet(file_path):

    """Whether a path names a Parquet file, by its extension."""

    return str(file_path).lower().endswith(PARQUET_EXTENSIONS)


#---------------------------------------------------------------------#


def save_table(df,
               file_path,
               sep = ",",
               index = True,
               header = True):

    """Write a table, as Parquet or as text, by the path's extension.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        The table to write.

    file_path : :class:`str`
        Where to write it. An extension of '.parquet' or '.pq' selects
        Parquet; anything else is written as delimited text.

    sep : :class:`str`, ``","``
        The column separator, for the text case only.

    index : :class:`bool`, ``True``
        Whether to write the index. Parquet keeps it as the frame's
        index, text writes it as the first column.

    header : :class:`bool`, ``True``
        Whether to write the column names, for the text case only.
        Parquet always carries them, since they are part of the schema.
    """

    if is_parquet(file_path):

        # 'header' has no counterpart in Parquet: the column names are
        # in the schema and cannot be omitted. A caller that passed
        # header = False wanted a headerless text file and is getting
        # Parquet instead, so the names come back, which is the more
        # useful outcome and the one that round-trips.
        df.to_parquet(file_path,
                      engine = "pyarrow",
                      compression = "snappy",
                      index = index)

    else:

        df.to_csv(file_path,
                  sep = sep,
                  index = index,
                  header = header)


#---------------------------------------------------------------------#


def load_table(file_path,
               sep = ",",
               index_col = None):

    """Read a table written by :func:`save_table`."""

    if is_parquet(file_path):

        return pd.read_parquet(file_path)

    return pd.read_csv(file_path, sep = sep, index_col = index_col)


#---------------------------------------------------------------------#


def table_name(stem, ext = None):

    """The file name a new table gets, Parquet unless told otherwise."""

    return f"{stem}{DEFAULT_TABLE_EXT if ext is None else ext}"


#---------------------------------------------------------------------#


def resolve_name(stem, available):

    """Which of a stem's possible file names is actually there.

    WHY A READER NEEDS THIS. The package writes Parquet now and wrote
    text before, and both are on disk: hundreds of thousands of
    per-sample differential expression tables were produced as '.csv'
    and are not going to be regenerated to satisfy a file extension. A
    reader that hardcodes either one is wrong for half the data, so it
    asks for the stem and takes whichever exists, newest format first.

    `available` is any container supporting ``in``: a set of archive
    member names, a directory listing, or a callable-free sequence.
    """

    for ext in READ_EXTENSIONS:

        name = f"{stem}{ext}"

        if name in available:

            return name

    return None
