#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    samplesio.py
#
#    Copyright (C) 2023 Valentina Sora 
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


# Standard library
import logging as log
import re
# Third-party packages
import pandas as pd
# bulkDGD
from . import defaults

# Get the module's logger
logger = log.getLogger(__name__)


#------------------------- Private functions -------------------------#


def _load_list(list_file):
    """Load a list of newline-separated entities from
    a plain text file.

    Parameters
    ----------
    list_file : ``str``
        The plain text file containing the entities of interest.

    Returns
    -------
    list_entities : ``list``
        The list of entities.
    """

    # Return the list of entities from the file (exclude blank
    # and comment lines)
    return \
        [l.rstrip("\n") for l in open(list_file, "r") \
         if (not l.startswith("#") and not re.match(r"^\s*$", l))]


#------------------------- Public functions --------------------------#


def load_samples(csv_file,
                 sep = ",",
                 keep_samples_names = True,
                 split = True):
    """Load the data frame containing the gene expression data for
    the samples.

    Parameters
    ----------
    csv_file : `str``
        A CSV file containing the samples' data.

        The rows of the data frame should represent the samples,
        while the columns should represent the genes and any
        additional information about the samples.

        Columns containing gene expression data must be named
        after the genes' Ensembl IDs.

    sep : ``str``, ``","``
        The column separator in the input CSV file.

    keep_samples_names : ``bool``, ``True``
        Whether to keep the names/IDs/indexes assigned to the
        samples in the input data frame.

        If ``True``, the names/IDs/indexes are assumed to be in
        the first column of the input data frame.

    split : ``bool``, ``True``
        Whether to split the input data frame into two data frames,
        one with only the columns containing the gene expression
        data and the other containing only the columns with
        additional information about the samples, if any
        were found.

    Returns
    -------
    df_data : ``pandas.DataFrame``
        A data frame containing the gene expression data.

        Here, the rows represent the samples and the columns
        represent the genes.

        If ``split`` is ``False``, this data frame will
        also contain the columns containing additional
        information about the samples, if any were found.

    df_other_data : ``pandas.DataFrame``
        A data frame containing the additional information
        about the samples found in the input data frame.

        If ``split`` is ``False``, only ``df_data`` is returned.
    """


    #---------------------- Load the data frame ----------------------#


    # If we need to keep the samples' original names
    if keep_samples_names:
        
        # Load the data frame assuming the samples' names
        # are in the first column of the data frame
        df = pd.read_csv(csv_file,
                         sep = sep,
                         index_col = 0,
                         header = 0)

    # Otherwise
    else:

        # Load the data frame assuming no column contains the
        # samples' names
        df = pd.read_csv(csv_file,
                         sep = sep,
                         index_col = False,
                         header = 0)

        # Inform the user that numeric indexes will be used
        infostr = \
            "Since 'keep_samples_names = False', the samples " \
            "will be identified using unique integer indexes " \
            "starting from 1."
        logger.info(infostr)

        # Set the indexes to numeric indexes
        df.index = range(len(df))


    #--------------------- Split the data frame ----------------------#


    # If the user requested splitting the data frame
    if split:

        # Get the names of the columns containing gene expression
        # data from the original data frame
        genes_columns = \
            [col for col in df.columns if col.startswith("ENSG")]

        # Get the names of the other columns
        other_columns = \
            [col for col in df.columns if col not in genes_columns]

        # If additional columns were found
        if other_columns:

            # Inform the user of the other columns found
            infostr = \
                f"{len(other_columns)} column(s) containing " \
                "additional information (not gene expression data) " \
                "was (were) found in the input data frame: " \
                f"{', '.join(other_columns)}."
            logger.info(infostr)

        # Create a data frame with only those columns containing gene
        # expression data    
        df_expr_data = df.loc[:,genes_columns]

        # Create a data frame with only those columns containing
        # additional information    
        df_other_data = df.loc[:,other_columns]

        # Return the data frames
        return df_expr_data, df_other_data


    #------------------ Do not split the data frame ------------------#


    # Otherwise
    else:

        # Return the full data frame
        return df


def save_samples(df,
                 csv_file,
                 sep = ","):
    """Save the samples to a CSV file.

    Parameters
    ----------
    df : ``pandas.DataFrame``
        A data frame containing the samples.

    csv_file : ``str``
        The output CSV file.

    sep : ``str``, ``","``
        The column separator in the output CSV file.
    """

    # Save the representations
    df.to_csv(csv_file,
              sep = sep,
              index = True,
              header = True)


def preprocess_samples(df_samples,
                       genes_txt_file = None):
    """Preprocess new samples.

    Parameters
    ----------
    df_samples : ``pandas.DataFrame``
        A data frame containing the samples to be preprocessed.

    genes_txt_file : ``str``, optional
        A plain text file containing the genes included in the
        model.

        If not provided, the default one
        (``bulkDGD/data/model/genes.txt``) will be used.

    Returns
    -------
    df_preproc : ``pandas.DataFrame``
        The data frame with the preprocessed samples.

    genes_excluded : ``list``
        The list of genes found in the input data frame but not
        belonging to the gene set used to train the DGD model.

        These genes are dropped from the ``df_preproc``
        data frame.

    genes_missing : ``list``
        The list of genes present in the gene set used to train
        the DGD model but not found in the input data frame.

        These genes are added with a count of 0 for all samples
        in the ``df_preproc`` data frame.
    """


    #------------------------- Other columns -------------------------#


    # Get the names of the columns containing gene expression data
    # from the original data frame
    genes_columns_old = \
        [col for col in df_samples.columns if col.startswith("ENSG")]

    # Create an empty list to store the new columns' names
    genes_columns = []

    # For each of the old columns containing genes' names
    for col in genes_columns_old:

        # Get the gene name and the version, if available
        gene_name = col.split(".")

        # If the gene does not have a version associated
        if len(gene_name) == 1:

            # Get only the gene name
            gene = gene_name[0]

        # If the gene has a version
        elif len(gene_name) == 2:

            # Get the gene and the version
            gene, version = gene_name

            # Try to split the version (it may contain the indication of
            # a pseudoautosomal region, like PAR_Y)
            pseudoatom_region = version.split("_")

            # If there is an indication of a pseudoatosomal region
            if len(pseudoatom_region) > 1:

                # Add the information to the unversioned gene name
                gene = gene + "_".join(pseudoatom_region[1:])

        # Add the new column to the list
        genes_columns.append(gene)

    # Rename the columns containing gene expression data
    df_samples = \
        df_samples.rename(\
            mapper = dict(zip(genes_columns_old, genes_columns)),
            axis = 1)

    # Get the names of the other columns
    other_columns = \
        [col for col in df_samples.columns if col not in genes_columns]

    # If additional columns were found
    if other_columns:

        # Inform the user of the other columns found
        infostr = \
            f"{len(other_columns)} column(s) containing additional " \
            "information (not gene expression data) was (were) found " \
            f"in the input data frame: {', '.join(other_columns)}."
        logger.info(infostr)


    #----------------------- Duplicate samples -----------------------#


    # Inform the user that we are about to perform a check on
    # duplicated samples
    infostr = "Now looking for duplicated samples..."
    logger.info(infostr)

    # Get duplicate samples, if any
    duplicated_samples = df_samples.duplicated(keep = "first")

    # If duplicate samples were found
    if duplicated_samples.any():

        # Warn the user that the duplicates will be removed
        warnstr = \
            f"{duplicated_samples.values.sum()} duplicated " \
            "sample(s) was (were() found in the data frame. " \
            "Duplicates will be removed from the data frame, and " \
            "only the first instance of the duplicates row will be " \
            "kept."
        logger.warning(warnstr)

        # Remove the duplicate samples
        df_samples = df_samples[~duplicated_samples]

    # Otherwise
    else:

        # Inform the user that no duplicate samples were found
        infostr = "No duplicated samples were found."
        logger.info(infostr)


    #------------------------ Missing values -------------------------#


    # Inform the user that we are about to perform a check on
    # missing values
    infostr = \
        "Now looking for missing values in the columns containing " \
        "gene expression data..."
    logger.info(infostr)

    # Get samples containing missing values in the columns containing
    # gene counts
    na_samples = df_samples[genes_columns].isnull().any(axis = 1)

    # If there are samples containing missing values
    if na_samples.any():

        # Warn the user of the samples containing missing values 
        warnstr = \
            f"{na_samples.values.sum()} sample(s) with missing " \
            "values in the columns containing gene expression data " \
            "was (were) found. It (they) will be removed from the " \
            "data frame of preprocessed samples."
        logger.warning(warnstr)

        # Remove the samples with missing values
        df_samples = df_samples[~na_samples]

    # Otherwise
    else:

        # Inform the user that no NA values were found in the
        # columns containing gene expression data
        infostr = \
            "No missing values were found in the columns containing " \
            "gene expression data."
        logger.info(infostr)


    #------------------------ Duplicate genes ------------------------#


    # Inform the user that we are looking for duplicate genes
    infostr = "Now looking for duplicated genes..."
    logger.info(infostr)

    # If there are duplicate genes
    if len(genes_columns) > len(set(genes_columns)):
        
        # Get the duplicate genes
        genes_series = pd.Series(genes_columns)
        duplicated_genes = \
            genes_series[genes_series.duplicated()].tolist()

        # Raise an error informing the user of the duplicated
        # genes
        errstr = \
            "Duplicated genes were found in the input data " \
            "frame. The duplicated genes are: " \
            f"{', '.join(duplicated_genes)}."
        raise ValueError(errstr)

    # Inform the user that no duplicated genes were found
    infostr = "No duplicated genes were found."
    logger.info(infostr)


    #------------------------- Final columns -------------------------#


    # If the user did not pass a file with the list of genes
    if genes_txt_file is None:

        # Use the default one
        genes_txt_file = defaults.GENES_FILE

    # Load the list of genes
    genes_list_dgd = _load_list(list_file = genes_txt_file)

    # Warn the user that the genes' columns were rearranged
    infostr = \
        "In the data frame containing the pre-processed samples, " \
        "the columns containing gene expression data will be " \
        "ordered according to the list of genes included in " \
        "the DGD model (taken from " \
        f"'{genes_txt_file}')."
    logger.info(infostr)

    # Warn the user that the other columns were rearranged
    infostr = \
        "In the data frame containing the pre-processed samples, " \
        "the columns found in the input data frame which did not " \
        "contain gene expression data, if any were present, " \
        "will be appended as the last columns of the data frame " \
        "and appear in the same order as they did in the input " \
        "data frame."
    logger.info(infostr)

    # Select only the genes used to train the DGD model
    # from the samples' data frame to obrain the data frame
    # containing the preprocessed samples. Sort the columns
    # (= genes) in the order expected by the DGD model, and,
    # if no data were found for some genes, add a default
    # count of 0
    df_preproc = df_samples.reindex(genes_list_dgd + other_columns,
                                    axis = 1,
                                    fill_value = 0)


    #-------------------- Excluded/missing genes ---------------------#


    # Create a list containing the genes present in the original
    # data frame but not in the list of genes on which the DGD
    # model was trained on. Use lists instead of sets (which
    # would be faster) to preserve the order
    genes_excluded = \
        [gene for gene in genes_columns if gene not in genes_list_dgd]

    # If some genes to be excluded were found
    if genes_excluded:

        # Warn the user
        warnstr = \
            f"{len(genes_excluded)} gene(s) found in the input " \
            "samples is (are) not part of the set of genes used to " \
            "train the DGD model. It (they) will be removed from " \
            "the data frame of preprocessed samples."
        logger.warning(warnstr)

    # Otherwise
    else:

        # Inform the user that no genes to be excluded were found
        infostr = \
            "All genes found in the input samples are part of the " \
            "set of genes used to train the DGD model."
        logger.info(infostr)

    # Create a list containing the genes present in the list of genes
    # used to train the DGD model but not in the original data frame.
    # Use lists instead of sets (which would be faster) to preserve
    # the order
    genes_missing = \
        [gene for gene in genes_list_dgd if gene not in genes_columns]

    # If genes with missing counts were found
    if genes_missing:

        # Warn the user
        warnstr = \
            f"{len(genes_missing)} gene(s) in the set of genes used " \
            "to train the DGD model was (were) not found in the " \
            "input samples. A default count of 0 will be assigned to " \
            "it (them) in all preprocessed samples."
        logger.warning(warnstr)

    # Otherwise
    else:

        # Inform the user that no genes with missing counts were
        # found
        infostr = \
            "All genes used to train the DGD model were found " \
            "in the input samples."
        logger.info(infostr)

    # Return the data frame with the preprocessed samples and the
    # two lists
    return df_preproc, genes_excluded, genes_missing