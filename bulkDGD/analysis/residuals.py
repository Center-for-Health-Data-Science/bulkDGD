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
                                    ge_than = 1):
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
    df_samples_count : ``pandas.DataFrame``
        A data frame containing each gene with the count of samples
        meeting either the lower than or equal (``le_than``) or
        greater than or equal (``ge_than``) conditions. 

        The data frame's rows are identified by each gene's ID (as
        provided in the input data frame) and the data frame includes
        one column:
        - 'n_samples': the number of samples where the gene's residual
          meets the conditions.

    df_genes_distribution : ``pandas.DataFrame``
        A data frame displaying the distribution of genes meeting the
        condition of having a residual value either lower than or equal
        to ``le_than`` or higher than or equal to ``ge_than`` across
        different sample counts.

        Each row represents the number of samples, and the columns are:
        - 'n_genes': the number of genes that meet the condition in
          exactly that many samples.
        - 'genes': a period-separated string listing the genes that
          meet the condition in exactly that many samples.

    Examples
    --------
    .. code-block:: python
        
        # Assuming 'df_res' is a data frame with residuals.
        >>> df_samples_count, df_genes_distribution = \
                get_genes_by_residual_threshold(df_res, 
                                                le_than = -1.5,
                                                ge_than = 2.0)

        # Visualize the first rows of the first data frame.
        >>> df_samples_count.head()
                   gene  n_samples
        ENSG00000187634         18
        ENSG00000188976          6
        ENSG00000187961         21
        ENSG00000187583         17
        ENSG00000187642         15

        # Visualize the first rows of the second data frame.
        >>> df_genes_distribution.head()
             n_genes                                              genes
                 300  ENSG00000183726.ENSG00000117713.ENSG0000009027...
                 760  ENSG00000160075.ENSG00000248333.ENSG0000015791...
                1020  ENSG00000127054.ENSG00000175756.ENSG0000007836...
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

    # Combine the two conditions into one using the OR connector.
    combined_condition = condition_le | condition_ge

    #-----------------------------------------------------------------#

    # Count the number of samples meeting the full condition for each
    # gene.
    samples_count = combined_condition.sum(axis = 0)

    #-----------------------------------------------------------------#

    # Create a data frame containing genes and their count of samples
    # meeting the condition.
    df_samples_count = \
        pd.DataFrame({"gene": df_res_data.columns,
                      "n_samples": samples_count})

    #-----------------------------------------------------------------#

    # Create another data frame showing how many genes meet the
    # condition in how many samples.

    # Get the total number of samples.
    n_all_samples = df_res_data.shape[0]

    # Create an empty dictionary to store the data about how the genes
    # are distributed according to in how many samples they meet the
    # given condition.
    distribution_data = {"n_genes": [], "genes": []}

    # Add a row for genes not meeting the condition in any samples.

    # Add the number of genes not meeting the condition in any samples.
    distribution_data["n_genes"].append(\
        df_samples_count[\
            df_samples_count["n_samples"] == 0].shape[1])

    # Add the list of genes not meeting the condition in any samples.
    distribution_data["genes"].append(\
        ".".join(df_samples_count[\
            df_samples_count["n_samples"] == 0]["gene"]))

    # For each possible number of samples in which the genes meet the
    # condition.
    for i in range(1, n_all_samples + 1):

        # Find the genes that meet the conditions in the current number
        # of samples.
        genes_meeting_i_samples = \
            df_samples_count[\
                df_samples_count["n_samples"] == i]["gene"]
        
        # Add the number of genes to the dictionary.
        distribution_data["n_genes"].append(\
            len(genes_meeting_i_samples))

        # Add the list of genes to the dictionary.
        distribution_data["genes"].append(\
            ".".join(genes_meeting_i_samples))

    # Convert the dictionary into a data frame.
    df_genes_distribution = \
        pd.DataFrame(distribution_data, 
                     index = range(0, n_all_samples + 1))

    #-----------------------------------------------------------------#

    # Update the index of the first data frame.
    df_samples_count = df_samples_count.set_index("gene")

    #-----------------------------------------------------------------#

    # Return the two data frames.
    return df_samples_count, df_genes_distribution
