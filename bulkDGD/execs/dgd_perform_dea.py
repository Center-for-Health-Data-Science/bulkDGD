#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    dgd_perform_dea.py
#
#    Perform differential expression analysis comparing experimental
#    samples to their "closest normal" sample found in latent space
#    by the DGD model.
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


# Description of the module
__doc__ = \
    "Perform differential expression analysis comparing " \
    "experimental samples to their 'closest normal' sample " \
    "found in latent space by the DGD model."


# Standard library
import argparse
from distributed import (
    as_completed,
    Client,
    LocalCluster)
import logging as log
import os
import sys
# Third-party packages
import pandas as pd
import torch
# bulkDGD
from bulkDGD.core import decoder
from bulkDGD.utils import dgd, misc


def main():


    #-------------------- Command-line arguments ---------------------#


    # Create the argument parser
    parser = argparse.ArgumentParser(description = __doc__)

    # Add the arguments
    is_help = \
        "The input CSV file containing a data frame with " \
        "the gene expression data for the samples. " \
        "The rows must represent the samples. The first " \
        "column must contain the samples' unique names or indexes, " \
        "while the other columns must represent the genes " \
        "(Ensembl IDs)."
    parser.add_argument("-is", "--input-csv-samples",
                        type = str,
                        required = True,
                        help = is_help)

    id_help = \
        "The input CSV file containing the data frame " \
        "with the decoder output for each samples' best " \
        "representation found with 'dgd_get_representations'. " \
        "The rows must represent the samples. The first " \
        "column must contain the samples' unique names or indexes, " \
        "while the other columns must represent the genes " \
        "(Ensembl IDs)."
    parser.add_argument("-id", "--input-csv-dec",
                        type = str,
                        required = True,
                        help = id_help)

    op_default = "dea_"
    op_help = \
        "The prefix of the output CSV file(s) that will contain " \
        "the results of the differential expression analysys. " \
        "Since the analysis will be performed for each sample, " \
        "one file per sample will be created. The files' names " \
        "will have the form {output_csv_prefix}{sample_name}.csv. " \
        f"The default prefix is '{op_default}'."
    parser.add_argument("-op", "--output-csv-prefix",
                        type = str,
                        default = op_default,
                        help = op_help)

    cm_help = \
        "The YAML configuration file specifying the " \
        "DGD model parameters and files containing " \
        "the trained model. If it is a name without " \
        "extension, it is assumed to be the name of a " \
        f"configuration file in '{dgd.CONFIG_MODEL_DIR}'."
    parser.add_argument("-cm", "--config-file-model",
                        type = str,
                        required = True,
                        help = cm_help)

    pr_default = 1
    pr_help = \
        "The resolution at which to sum over the probability " \
        "mass function to compute the p-values. The lower the " \
        "resolution, the more accurate the calculation. A " \
        "resolution of 1 corresponds to an exact calculation " \
        f"of the p-values. The default is {pr_default}."
    parser.add_argument("-pr", "--p-values-resolution",
                        type = lambda x: int(float(x)),
                        default = pr_default,
                        help = pr_help)

    qa_default = 0.05
    qa_help = \
        "The alpha value used to calculate the q-values (adjusted " \
        f"p-values). The default is {qa_default}."
    parser.add_argument("-qa", "--q-values-alpha",
                        type = float,
                        default = qa_default,
                        help = qa_help)

    qm_default = "fdr_bh"
    qm_help = \
        "The method used to calculate the q-values (i.e., to " \
        f"adjust the p-values). The default is '{qm_default}'. " \
        "The available methods can be found in the documentation " \
        "of 'statsmodels.stats.multitest.multipletests', " \
        "which is used to perform the calculation."
    parser.add_argument("-qm", "--q-values-method",
                        type = str,
                        default = qm_default,
                        help = qm_help)

    d_help = \
        "The working directory. The default is the current " \
        "working directory."
    parser.add_argument("-d", "--work-dir",
                        type = str,
                        default = os.getcwd(),
                        help = d_help)

    n_default = 1
    n_help = \
        "The number of processes to start (the higher " \
        "the number, the faster the execution of the program). " \
        "The default number of processes started is " \
        f"{n_default}."
    parser.add_argument("-n", "--n-proc",
                        type = int,
                        default = n_default,
                        help = n_help)


    v_help = "Verbose logging (INFO level)."
    parser.add_argument("-v", "--logging-verbose",
                        action = "store_true",
                        help = v_help)

    vv_help = \
        "Maximally verbose logging for debugging " \
        "purposes (DEBUG level)."
    parser.add_argument("-vv", "--logging-debug",
                        action = "store_true",
                        help = vv_help)

    # Parse the arguments
    args = parser.parse_args()
    input_csv_samples = args.input_csv_samples
    input_csv_dec = args.input_csv_dec
    output_csv_prefix = args.output_csv_prefix
    config_file_model = args.config_file_model
    p_values_resolution = args.p_values_resolution
    q_values_alpha = args.q_values_alpha
    q_values_method = args.q_values_method
    wd = args.work_dir
    n_proc = args.n_proc
    v = args.logging_verbose
    vv = args.logging_debug


    #---------------------------- Logging ----------------------------#


    # Get the module's logger
    logger = log.getLogger(__name__)

    # Set WARNING logging level by default
    level = log.WARNING

    # If the user requested verbose logging
    if v:

        # The minimal logging level will be INFO
        level = log.INFO

    # If the user requested logging for debug purposes
    # (-vv overrides -v if both are provided)
    if vv:

        # The minimal logging level will be DEBUG
        level = log.DEBUG

    # Set the logging level for this module
    log.basicConfig(level = level)

    # Suppress all Dask logs below WARNING
    log.getLogger("distributed").setLevel(log.WARNING)


    #------------------------- Configuration -------------------------#


    # Try to load the configuration
    try:

        config_model = misc.get_config_model(config_file_model)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to load the configuration from " \
            f"'{config_file_model}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the configuration was successfully loaded
    infostr = \
        "The configuration was successfully loaded from " \
        f"'{config_file_model}'."
    logger.info(infostr)


    #----------------------- Input - samples -------------------------#


    # Try to load the samples' data
    try:

        # Get the observed counts
        obs_counts = \
            dgd.load_samples_data(\
                csv_file = input_csv_samples,
                keep_samples_names = True)[0].df

        # Get the samples' names
        obs_counts_names = obs_counts.index.tolist()

        # Get the genes' names
        obs_counts_genes = obs_counts.columns.tolist()

        # Get the observed counts' values
        obs_counts_values = obs_counts.values

        # Convert the observed counts into a tensor
        obs_counts = \
            [torch.Tensor(obs_counts_values[i,:]) for i in \
             range(obs_counts_values.shape[0])]

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to load the samples from " \
            f"'{input_csv_samples}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)


    #-------------------- Input - decoder outputs --------------------#


    # Try to load the decoder outputs
    try:

        # Get the predicted means
        pred_means = \
            pd.read_csv(input_csv_dec,
                        sep = ",",
                        header = 0,
                        index_col = 0)

        # Get the samples' names
        pred_means_names = pred_means.index.tolist()

        # Get the genes' names
        pred_means_genes = pred_means.columns.tolist()

        # Get the predicted means' values
        pred_means_values = pred_means.values

        # Convert the predicted mean into a tensor
        pred_means = \
            [torch.Tensor(pred_means_values[i,:]) for i in \
             range(pred_means_values.shape[0])]

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to load the decoder outputs from " \
            f"'{input_csv_dec}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)


    #------------------- Check the samples' names --------------------#


    # If the samples' names differ between observed counts and
    # predicted means
    if obs_counts_names != pred_means_names:

        # Warn the user and exit
        errstr = \
            "The names of the samples in the observed counts " \
            "(rows' names) must correspond to the names of the " \
            "samples in the decoder outputs (rows' names)."
        logger.exception(errstr)
        sys.exit(errstr)


    #-------------------- Check the genes' names ---------------------#


    # If the genes' names found in the observed counts differ from
    # those found in the predicted means
    if obs_counts_genes != pred_means_genes:

        # Warn the user and exit
        errstr = \
            "The names of the genes in the observed counts " \
            "(columns' names) must correspond to the names of " \
            "the genes in the decoder outputs (columns' names)."
        logger.exception(errstr)
        sys.exit(errstr)


    #----------------------- Get the r-values ------------------------#


    # Get the dimensionality of the latent space
    dim_latent = config_model["dim_latent"]

    # Get the configuration for the decoder
    config_dec = config_model["dec"]

    # Try to get the decoder
    try:

        # Initialize and set the decoder
        dec = dgd.get_decoder(dim = dim_latent,
                              config = config_dec)

        # Get the r values of the negative binomials from the log-r
        # values stored in the negative binomial layer of the decoder.
        # This is a 1D tensor with:
        #
        # - 1st dimension: the dimensionality of the gene space
        r_values = torch.exp(dec.nb.log_r).squeeze().detach()

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to retrieve the r-values from the " \
            f"decoder's paramenters. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the r-values were successfully retrieved
    infostr = \
        "The r-values were successfully retrieved from the " \
        "decoder's paramenters."
    logger.info(infostr)


    #--------------- Set the loocal cluster and client ---------------#


    # Create the local cluster
    cluster = LocalCluster(# Number of workers
                           n_workers = n_proc,
                           # Below which level log messages will
                           # be silenced
                           silence_logs = "WARNING",
                           # Whether to use processes, single-core
                           # or threads
                           processes = True,
                           # How many threads for each worker should
                           # be used
                           threads_per_worker = 1)
    
    # Open the client from the cluster
    client = Client(cluster)


    #---------------------- Get the statistics -----------------------#


    # Set the statistics to be calculated
    statistics = ["p_values", "q_values", "log2_fold_changes"]

    # Create a list to store the futures
    futures = []
    
    # For each samples' observed counts, predicted means, 
    # and name
    for obs_counts_sample, pred_means_sample, sample_name \
    in zip(obs_counts, pred_means, obs_counts_names):
            
        # Submit the calculation to the cluster
        futures.append(\
            client.submit(dgd.perform_dea,
                          obs_counts_sample = obs_counts_sample,
                          pred_means_sample = pred_means_sample,
                          sample_name = sample_name,
                          statistics = statistics,
                          genes_names = obs_counts_genes,
                          r_values = r_values,
                          resolution = p_values_resolution,
                          alpha = q_values_alpha,
                          method = q_values_method))


    #-------------------- Write the output files ---------------------#


    # For each future (which is retrieved as it completes)
    for future in as_completed(futures):
        
        # Get the data frame containing the DEA results for the
        # current sample and the ame of the sample
        df_stats, sample_name = future.result()

        # Set the path to the output file
        output_csv_path = \
            os.path.join(wd,
                         f"{output_csv_prefix}{sample_name}.csv")

        # Try to write the data frame to the output file
        try:

            df_stats.to_csv(output_csv_path,
                            sep = ",",
                            index = True,
                            header = True)

        # If something went wrong
        except Exception as e:

            # Warn the user and exit
            errstr = \
                "It was not possible to write the DEA results " \
                f"for sample '{sample_name}' to " \
                f"'{output_csv_path}'. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the file was successfully written
        infostr = \
            f"The DEA results for sample '{sample_name}' were " \
            f"successfully written to '{output_csv_path}'."
        logger.info(infostr)