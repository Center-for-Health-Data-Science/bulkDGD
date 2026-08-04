#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    deaio.py
#
#    Read a model's per-sample differential expression whether it is
#    still loose on disk or already packed into the directory's
#    'dea.zip'.
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


# Set the module's description.
__doc__ = \
    """Read a model's per-sample differential expression whether it is
    still loose on disk or already packed into the directory's
    'dea.zip'.

    An ensemble's differential expression is one file per sample per
    model, so a fifteen-model ensemble over a few thousand samples is
    tens of thousands of small files - enough to matter on a shared
    file system with a file quota. Packing a finished directory into
    one archive is therefore the recommended end state, and everything
    that reads those files afterwards goes through here so it does not
    have to know or care which of the two states a directory is in: if
    a 'dea.zip' is present the sample is read from it, otherwise from
    the loose file."""


#######################################################################


# Import from the standard library.
import io
import os
import zipfile

# Import from third-party libraries.
import pandas as pd

# Import from the package.
from .tableio import is_parquet, resolve_name, READ_EXTENSIONS


#######################################################################


# The name of the archive a packed directory uses.
DEA_ZIP_NAME = "dea.zip"

# The prefix the per-sample files are named with.
DEA_PREFIX = "dea_"


#######################################################################


# An open archive is kept per path so that reading many samples out of
# the same archive - as the consensus does, once per sample per model -
# parses its central directory once, and not once a sample.
_ZIP_CACHE = {}

# The member names of each open archive, so that asking whether a
# sample is present does not go back to the archive.
_NAMES_CACHE = {}

# The process the caches were built in.
_CACHE_PID = None


#######################################################################


def _get_zip_path(dea_dir: str) -> str:
    """Get the path to the archive a packed directory would use.

    Parameters
    ----------
    dea_dir : :class:`str`
        The directory containing the differential expression analysis'
        results.

    Returns
    -------
    zip_path : :class:`str`
        The path to the archive.
    """

    # Return the path to the archive.
    return os.path.join(dea_dir, DEA_ZIP_NAME)


#---------------------------------------------------------------------#


def _get_archive(zip_path: str) -> tuple[zipfile.ZipFile, set]:
    """Get the open archive for a path, and its members' names, both
    cached.

    Parameters
    ----------
    zip_path : :class:`str`
        The path to the archive.

    Returns
    -------
    archive : :class:`zipfile.ZipFile`
        The open archive.

    names : :class:`set`
        The names of the archive's members.
    """

    # Get the process the caches were built in.
    global _CACHE_PID

    # Get the current process.
    pid = os.getpid()

    # The caches are dropped whenever the process changes, so that a
    # worker started by 'multiprocessing' never reads through a file
    # descriptor it inherited from its parent - a shared descriptor
    # gives interleaved reads, and a 'BadZipFile' raised from a file
    # that is perfectly fine. Each process opens, and reads through,
    # its own handles.
    if _CACHE_PID != pid:

        # Empty the caches.
        _ZIP_CACHE.clear()
        _NAMES_CACHE.clear()

        # Record the process the caches now belong to.
        _CACHE_PID = pid

    #-----------------------------------------------------------------#

    # Get the archive, if it was already opened.
    archive = _ZIP_CACHE.get(zip_path)

    # If the archive was not opened yet
    if archive is None:

        # Open it.
        archive = zipfile.ZipFile(zip_path)

        # Cache it, together with its members' names.
        _ZIP_CACHE[zip_path] = archive
        _NAMES_CACHE[zip_path] = set(archive.namelist())

    #-----------------------------------------------------------------#

    # Return the archive and its members' names.
    return archive, _NAMES_CACHE[zip_path]


#---------------------------------------------------------------------#


def has_sample(dea_dir: str,
               sample: str,
               prefix: str = DEA_PREFIX) -> bool:
    """Return whether a sample's differential expression is present,
    packed or loose.

    Parameters
    ----------
    dea_dir : :class:`str`
        The directory containing the differential expression analysis'
        results.

    sample : :class:`str`
        The sample's name.

    prefix : :class:`str`, ``"dea_"``
        The prefix the per-sample files are named with.

    Returns
    -------
    has_sample : :class:`bool`
        Whether the sample is present.
    """

    # Get the path to the archive.
    zip_path = _get_zip_path(dea_dir)

    #-----------------------------------------------------------------#

    # If the directory is packed
    if os.path.exists(zip_path):

        # Get the archive's members' names.
        _, names = _get_archive(zip_path)

        # Present in whichever format, since both are on disk.
        return resolve_name(f"{prefix}{sample}", names) is not None

    #-----------------------------------------------------------------#

    # Otherwise, whether a loose file is there in either format.
    listing = os.listdir(dea_dir) if os.path.isdir(dea_dir) else []

    return resolve_name(f"{prefix}{sample}", listing) is not None


#---------------------------------------------------------------------#


def read_dea(dea_dir: str,
             sample: str,
             prefix: str = DEA_PREFIX,
             **read_csv_kwargs) -> pd.DataFrame:
    """Read one sample's differential expression, from the archive if
    the directory is packed and from the loose file otherwise.

    Parameters
    ----------
    dea_dir : :class:`str`
        The directory containing the differential expression analysis'
        results.

    sample : :class:`str`
        The sample's name.

    prefix : :class:`str`, ``"dea_"``
        The prefix the per-sample files are named with.

    **read_csv_kwargs
        The keyword arguments to be passed to
        :func:`pandas.read_csv`.

    Returns
    -------
    df_dea : :class:`pandas.DataFrame` or :obj:`None`
        The sample's statistics, or :obj:`None` if the sample is not
        present.
    """

    # Get the path to the archive.
    zip_path = _get_zip_path(dea_dir)

    #-----------------------------------------------------------------#

    # If the directory is packed
    if os.path.exists(zip_path):

        # Get the archive and its members' names.
        archive, names = _get_archive(zip_path)

        # WHICHEVER FORMAT IS IN THERE. New runs write Parquet, and the
        # archives already on disk hold text, so the member is looked up
        # by its stem and taken in whatever form it was stored in. A
        # reader that hardcoded one extension would be wrong for half
        # the data the moment the writer changed.
        member = resolve_name(f"{prefix}{sample}", names)

        # If the sample is not in the archive
        if member is None:

            # Return nothing.
            return None

        # Read the member straight out of the archive. It is read in
        # full before being parsed because the handle the archive
        # gives is not seekable, and the parser may need to seek.
        with archive.open(member) as handle:

            data = io.BytesIO(handle.read())

            return (pd.read_parquet(data) if is_parquet(member)
                    else pd.read_csv(data, **read_csv_kwargs))

    #-----------------------------------------------------------------#

    # The same question, for a directory that has not been packed.
    listing = os.listdir(dea_dir) if os.path.isdir(dea_dir) else []

    member = resolve_name(f"{prefix}{sample}", listing)

    # If the file is not there
    if member is None:

        # Return nothing.
        return None

    # Get the path to the loose file.
    path = os.path.join(dea_dir, member)

    # Read the file.
    return (pd.read_parquet(path) if is_parquet(path)
            else pd.read_csv(path, **read_csv_kwargs))


#---------------------------------------------------------------------#


def list_samples(dea_dir: str,
                 prefix: str = DEA_PREFIX) -> list:
    """List the samples a directory holds, packed or loose.

    Parameters
    ----------
    dea_dir : :class:`str`
        The directory containing the differential expression analysis'
        results.

    prefix : :class:`str`, ``"dea_"``
        The prefix the per-sample files are named with.

    Returns
    -------
    samples : :class:`list`
        The samples' names, sorted.
    """

    # Get the path to the archive.
    zip_path = _get_zip_path(dea_dir)

    #-----------------------------------------------------------------#

    # If the directory is packed
    if os.path.exists(zip_path):

        # Get the archive's members' names.
        _, names = _get_archive(zip_path)

    # Otherwise, if the directory exists
    elif os.path.isdir(dea_dir):

        # Get the names of the files it contains.
        names = os.listdir(dea_dir)

    # Otherwise
    else:

        # There are no samples.
        return []

    #-----------------------------------------------------------------#

    # Return the part of each name between the prefix and the
    # extension, sorted.
    #
    # THE EXTENSION IS STRIPPED BY LENGTH, NOT BY A CONSTANT. This took
    # the name up to its last four characters, which is '.csv' and is
    # not '.parquet': against a directory written by a current run it
    # would have returned every sample name with 'rque' still attached,
    # and every lookup keyed on those names would have missed. A
    # directory holding both formats is also possible while a cohort is
    # half regenerated, so the names are de-duplicated.
    out = set()

    for name in names:

        if not name.startswith(prefix):

            continue

        for ext in READ_EXTENSIONS:

            if name.endswith(ext):

                out.add(name[len(prefix):-len(ext)])

                break

    return sorted(out)
