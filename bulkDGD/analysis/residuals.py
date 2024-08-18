#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    residuals.py
#
#    Utilities to compute vectors of residuals.
#
#    Copyright (C) 2024 Valentina Sora 
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
__doc__ = "Utilities to compute vectors of residuals."


#######################################################################


# Import from the standard library.
import logging as log
# Import from third-party packages.
import pandas as pd
from scipy.stats import nbinom, norm, poisson
import torch


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


def get_residuals(obs_counts,
                  pred_means,
                  r_values,
                  sample_name = None):
    """Calculate the vector of residuals between the observed gene
    expression (counts) and the predicted means of the negative
    binomials modeling the expression of the different genes for a
    single sample.

    Parameters
    ----------
    obs_counts : :class:`pandas.Series`
        The observed gene counts in a single sample.

        This is a series whose index contains either the genes'
        Ensembl IDs or names of fields containing additional
        information about the sample.

    pred_means : :class:`pandas.Series`
        The predicted means of the distributions modelling
        the genes' counts in a single sample.

        This is a series whose index contains either the genes'
        Ensembl IDs or names of fields containing additional
        information about the sample.

        If the genes' counts were modelled using negative binomial
        distributions, the predicted means are scaled by the
        corresponding distributions' r-values.

    r_values : :class:`pandas.Series`, optional
        The predicted r-values of the negative binomial distributions
        modelling the genes' counts in a single sample, if the genes'
        counts were modelled using negative binomial distributions.

        This is a series whose index contains either the genes'
        Ensembl IDs or names of fields containing additional
        information about the sample.

        If ``r_values`` is not provided, it is assumed that the genes'
        counts were modelled using Poisson distributions.

    sample_name : :class:`str`, optional
        The name of the sample under consideration. It is used as
        name for the :class:`pandas.Series` returned.

        If not passed, the series will be unnamed.

    Returns
    -------
    series_residuals : :class:`pandas.Series`
        A series containing the residuals for all genes.
    """

    # Get the names of the cells containing gene expression data from
    # the series containing the observed gene counts.
    genes_obs = \
        [col for col in obs_counts.index if col.startswith("ENSG")]

    #-----------------------------------------------------------------#

    # Get the names of the cells containing gene expression data from
    # the series containing the predicted means.
    genes_pred = \
        [col for col in pred_means.index if col.startswith("ENSG")]

    #-----------------------------------------------------------------#

    # If the lists do not contain the same genes
    if set(genes_obs) != set(genes_pred):

        # Raise an error.
        errstr = \
            "The set of genes in 'obs_counts' and 'pred_means', " \
            "must be the same. It is assumed that the genes are " \
            "specified using their Ensembl IDs."
        raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # If the r-values were passed
    if r_values is not None:

        # Get the names of the cells containing r-values from the
        # series of r-values.
        genes_r_values =  \
            [col for col in r_values.index if col.startswith("ENSG")]

        # If the lists do not contain the same genes
        if set(genes_obs) != set(genes_pred) != set(genes_r_values):

            # Raise an error.
            errstr = \
                "The set of genes in 'obs_counts', 'pred_means', " \
                "and 'r_values' must be the same. It is assumed " \
                "that the genes are specified using their Ensembl IDs."
            raise ValueError(errstr)

        # Create a tensor containing only those columns containing gene
        # expression data for the predicted r-values - the 'loc' syntax
        # should return the columns in the order specified by the
        # selection.
        r_values = \
            torch.Tensor(\
                pd.to_numeric(r_values.loc[genes_r_values]).values)

    #-----------------------------------------------------------------#

    # Create a tensor containing only those columns containing gene
    # expression data for the observed gene counts - the 'loc' syntax
    # should return the columns in the order specified by the
    # selection.
    obs_counts = \
        torch.Tensor(\
            pd.to_numeric(obs_counts.loc[genes_obs]).values)

    #-----------------------------------------------------------------#

    # Create a tensor containing only those columns containing gene
    # expression data for the predicted mean counts - the 'loc' syntax
    # should return the columns in the order specified by the
    # selection.
    pred_means = \
        torch.Tensor(\
            pd.to_numeric(pred_means.loc[genes_obs]).values)

    #-----------------------------------------------------------------#

    # Get the mean gene counts for the sample.
    #
    # The output is a single value.
    obs_counts_mean = torch.mean(obs_counts).unsqueeze(-1)

    #-----------------------------------------------------------------#

    # Rescale the predicted means by the mean gene counts.
    #
    # The output is a 1D tensor containing the rescaled means.
    pred_means = pred_means * obs_counts_mean

    #-----------------------------------------------------------------#

    # Create an empty list to store the residuals.
    residuals = []

    # For each gene's observed count, predicted mean count, and r-value
    for i, (obs_count_gene_i, pred_mean_gene_i) \
        in enumerate(zip(obs_counts, pred_means)):

        #-------------------------------------------------------------#

        # If negative binomial distributions were used to model the
        # genes' counts
        if r_values is not None:

            # Get the r-value for the current gene.
            r_value_gene_i = r_values[i]

            # Calculate the probability of "success" from the r-value
            # (number of successes till the experiment is stopped) from
            # the mean of the negative binomial. This is a single
            # value, and is calculated from the mean 'm' of the
            # negative binomial as:
            #
            # m = r(1-p) / p
            # mp = r - rp
            # mp + rp = r
            # p(m+r) = r
            # p = r / (m + r)
            p_i = pred_mean_gene_i.item() / \
                  (pred_mean_gene_i.item() + r_value_gene_i.item())

            #---------------------------------------------------------#

            # Get the value of the cumulative negative binomial
            # distribution.
            #
            # Since SciPy's negative binomial function is implemented
            # as function of the number of failures, their 'p' is
            # equivalent to our '1-p' and their 'n' is our 'r'
            cdf_nb_value = \
                nbinom.cdf(k = obs_count_gene_i.item(),
                           n = r_value_gene_i.item(),
                           p = 1 - p_i).item()

        #-------------------------------------------------------------#
        
        # If Poisson distributions were used to model the genes' counts
        else:

            # Get the value of the cumulative Poisson distribution.
            cdf_nb_value = \
                poisson.cdf(k = obs_count_gene_i.item(),
                            mu = pred_mean_gene_i.item()).item()

        #-------------------------------------------------------------#
        
        # Get the value of the inverse of the cumulative normal
        # distribution (= the valued of the percentile function) at the
        # point corresponding to the value of the neg. binom. CDF at
        # the observed gene count value.
        residual_gene_i = norm.ppf(q = cdf_nb_value)

        #-------------------------------------------------------------#

        # Save the residual for the current gene.
        residuals.append(residual_gene_i)

    #-----------------------------------------------------------------#

    # Create a series to store the residuals.
    series_residuals = pd.Series(residuals)

    # Set the index of the series equal to the genes' names.
    series_residuals.index = genes_obs

    # If a name was passed for the sample
    if sample_name:

        # Set the name of the series equal to the sample's name.
        series_residuals.name = sample_name

    #-----------------------------------------------------------------#

    # Return the series with the residuals.
    return series_residuals


def get_genes_by_residual_threshold(df_res,
                                    le_than = -1,
                                    ge_than = 1,
                                    sort_genes = False,
                                    ascending = False):
    """Get the genes whose residuals fall in specified intervals and
    are common to a certain number of samples.

    Parameters
    ----------
    df_res : ``pandas.DataFrame``
        A data frame containing the residual vectors for a set of
        samples.

    le_than : ``int`` or ``float``, ``-1``
        Consider only genes whose residual value in all samples is
        lower than or equal to the provided value.

    ge_than : ``int`` or ``float``, ``1``
        Consider only genes whose residual value in all samples is
        greater than or equal to the provided value.

    Returns
    -------
    df_samples_count_le : :class:`pandas.DataFrame`
        A data frame containing each gene with the count of samples
        where the gene's residual value is lower than or equal to
        ``le_than``. 

        The data frame's rows are identified by each gene's ID (as
        provided in the input data frame) and the data frame includes
        one column:

        - 'n_samples': the number of samples where the gene's residual
          meets the condition.

    df_genes_distribution_le : :class:`pandas.DataFrame`
        A data frame displaying the distribution of genes having a
        residual value lower than or equal to ``le_than``  across
        different sample counts.

        Each row represents the number of samples, and the columns are:

        - 'n_genes': the number of genes that meet the condition in
          exactly that many samples.
        - 'genes': a period-separated string listing the genes that
          meet the condition in exactly that many samples.

    df_samples_count_ge : :class:`pandas.DataFrame`
        A data frame containing each gene with the count of samples
        where the gene's residual value is greater than or equal to
        ``ge_than``. 

        The data frame's rows are identified by each gene's ID (as
        provided in the input data frame) and the data frame includes
        one column:

        - 'n_samples': the number of samples where the gene's residual
          meets the condition.

    df_genes_distribution_ge : :class:`pandas.DataFrame`
        A data frame displaying the distribution of genes having a
        residual value greater than or equal to ``le_than``  across
        different sample counts.

        Each row represents the number of samples, and the columns are:
        
        - 'n_genes': the number of genes that meet the condition in
          exactly that many samples.
        - 'genes': a period-separated string listing the genes that
          meet the condition in exactly that many samples..
    """

    # Get the names of the cells containing residual values.
    df_res_data = \
        df_res.loc[:, [col for col in df_res.columns \
                       if col.startswith("ENSG")]]

    #-----------------------------------------------------------------#

    # Set the condition for which the residuals must be lower than or
    # equal to the specified value.
    condition_le = df_res_data <= le_than

    # Set the condition for which the residuals must be greater than
    # or equal to the specified value.
    condition_ge = df_res_data >= ge_than

    #-----------------------------------------------------------------#

    # Count the number of samples meeting the first condition.
    samples_count_le = condition_le.sum(axis = 0)

    # Count the number of samples meeting the second condition.
    samples_count_ge = condition_ge.sum(axis = 0)

    #-----------------------------------------------------------------#

    # Create a data frame containing genes and their count of samples
    # meeting the first condition.
    df_samples_count_le = \
        pd.DataFrame({"gene": df_res_data.columns,
                      "n_samples": samples_count_le})

    # Create a data frame containing genes and their count of samples
    # meeting the second condition.
    df_samples_count_ge = \
        pd.DataFrame({"gene": df_res_data.columns,
                      "n_samples": samples_count_ge})

    #-----------------------------------------------------------------#

    # If the genes must be sorted by the number of samples.
    if sort_genes:

        # Sort the first data frame by the number of samples.
        df_samples_count_le = \
            df_samples_count_le.sort_values(\
                by = "n_samples",
                ascending = ascending)

        # Sort the second data frame by the number of samples.
        df_samples_count_ge = \
            df_samples_count_ge.sort_values(\
                    by = "n_samples",
                    ascending = ascending)

    #-----------------------------------------------------------------#

    # Create another data frame showing how many genes meet the
    # condition in how many samples.

    # Get the total number of samples.
    n_all_samples = df_res_data.shape[0]

    # Create an empty dictionary to store the data about how the genes
    # are distributed according to in how many samples they meet the
    # first condition.
    distribution_data_le = {"n_genes": [], "genes": []}

    # Create an empty dictionary to store the data about how the genes
    # are distributed according to in how many samples they meet the
    # second condition.
    distribution_data_ge = {"n_genes": [], "genes": []}

    # For each possible number of samples in which the genes meet the
    # condition.
    for i in range(n_all_samples + 1):

        # Find the genes that meet the first condition in the current
        # number of samples.
        genes_meeting_i_samples_le = \
            df_samples_count_le[\
                df_samples_count_le["n_samples"] == i]["gene"]
        
        # Find the genes that meet the second condition in the current
        # number of samples.
        genes_meeting_i_samples_ge = \
            df_samples_count_ge[\
                df_samples_count_ge["n_samples"] == i]["gene"]
        
        # Add the number of genes to the first dictionary.
        distribution_data_le["n_genes"].append(\
            len(genes_meeting_i_samples_le))
        
        # Add the number of genes to the second dictionary.
        distribution_data_ge["n_genes"].append(\
            len(genes_meeting_i_samples_ge))

        # Add the list of genes to the first dictionary.
        distribution_data_le["genes"].append(\
            ".".join(genes_meeting_i_samples_le))

        # Add the list of genes to the second dictionary.
        distribution_data_ge["genes"].append(\
            ".".join(genes_meeting_i_samples_ge))
    
    #-----------------------------------------------------------------#

    # Convert the first dictionary into a data frame.
    df_genes_distribution_le = \
        pd.DataFrame(distribution_data_le, 
                     index = range(0, n_all_samples + 1))

    # Convert the second dictionary into a data frame.
    df_genes_distribution_ge = \
        pd.DataFrame(distribution_data_ge, 
                     index = range(0, n_all_samples + 1))

    #-----------------------------------------------------------------#

    # Update the index of the first data frame.
    df_samples_count_le = df_samples_count_le.set_index("gene")

    # Update the index of the second data frame.
    df_samples_count_ge = df_samples_count_ge.set_index("gene")

    #-----------------------------------------------------------------#

    # If the genes must be sorted by the number of samples.
    if sort_genes:

        # Sort the first data frame by the number of samples.
        df_genes_distribution_le = \
            df_genes_distribution_le.sort_index(ascending = ascending)
        
        # Sort the second data frame by the number of samples.
        df_genes_distribution_ge = \
            df_genes_distribution_ge.sort_index(ascending = ascending)

    #-----------------------------------------------------------------#

    # Return the data frames.
    return df_samples_count_le, df_genes_distribution_le, \
           df_samples_count_ge, df_genes_distribution_ge
