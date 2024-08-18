#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    residuals.py
#
#    Find the residual values for a set of samples.
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
__doc__ = "Find the residual values for a set of samples."


#######################################################################


# Import from the standard library.
import logging as log
import os
import sys
# Import from third-party packages.
import pandas as pd
# Import from 'bulkDGD'.
from bulkDGD.analysis import residuals
from bulkDGD import ioutil
from . import util


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


# Define a function to set up the parser.
def set_sub_parser(sub_parsers):

    # Create the argument parser.
    parser = \
        sub_parsers.add_parser(\
            name = "residuals",
            description = __doc__,
            help = __doc__,
            formatter_class = util.CustomHelpFormatter)

    #-----------------------------------------------------------------#

    # Create a group of arguments for the input files.
    input_group = \
        parser.add_argument_group(title = "Input files")

    # Create a group of arguments for the output files.
    output_group = \
        parser.add_argument_group(title = "Output files")

    #-----------------------------------------------------------------#

    # Set a help message.
    is_help = \
        "The input CSV file containing a data frame with " \
        "the gene expression data for the samples."

    # Add the argument to the group.
    input_group.add_argument("-is", "--input-samples",
                             type = str,
                             required = True,
                             help = is_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    im_help = \
        "The input CSV file containing the data frame with the " \
        "predicted means of the distributions used to model the " \
        "genes' counts for each in silico control sample."

    # Add the argument to the group.
    input_group.add_argument("-im", "--input-means",
                             type = str,
                             required = True,
                             help = im_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    iv_help = \
        "The input CSV file containing the data frame with the " \
        "predicted r-values of the negative binomial distributions " \
        "for each in silico control sample, if negative binomial " \
        "distributions were used to model the genes' counts."

    # Add the argument to the group.
    input_group.add_argument("-iv", "--input-rvalues",
                             type = str,
                             default = None,
                             help = iv_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    or_default = "residuals.csv"

    # Set a help message.
    or_help = \
        "The name of the output CSV file where the residuals " \
        "values will be saved. The default file name is " \
        f"'{or_default}'."
    
    # Add the argument to the group.
    output_group.add_argument("-or", "--output-residuals",
                              type = str,
                              default = or_default,
                              help = or_help)

    #-----------------------------------------------------------------#

    # Return the parser.
    return parser


#######################################################################


# Define the 'main' function.
def main(args):

    # Get the argument corresponding to the working directory.
    wd = args.work_dir

    # Get the arguments corresponding to the input files.
    input_samples = args.input_samples
    input_means = args.input_means
    input_rvalues = args.input_rvalues

    # Get the argument corresponding to the output file.
    output_residuals = os.path.join(wd, args.output_residuals)

    #-----------------------------------------------------------------#

    # Try to load the samples.
    try:

        # Get the samples (= observed gene counts).
        df_samples, df_extra = \
            ioutil.load_samples(\
                csv_file = input_samples,
                sep = ",",
                keep_samples_names = True,
                split = True)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the samples from " \
            f"'{input_samples}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the samples were successfully loaded.
    infostr = \
        "The samples were successfully loaded from " \
        f"'{input_samples}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to load the predicted means.
    try:

        # Get the predicted means.
        df_pred_means = \
            ioutil.load_decoder_outputs(csv_file = input_means,
                                        sep = ",",
                                        split = False)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the predicted means from " \
            f"'{input_means}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the predicted means were successfully
    # loaded.
    infostr = \
        "The predicted means were successfully loaded from " \
        f"'{input_means}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # If r-values were passed
    if input_rvalues is not None:

        # Try to load the predicted r-values.
        try:

            # Get the predicted r-values.
            df_r_values = \
                ioutil.load_decoder_outputs(\
                    csv_file = input_rvalues,
                    sep = ",",
                    split = False)

        # If something went wrong
        except Exception as e:

            # Warn the user and exit.
            errstr = \
                "It was not possible to load the predicted r-values " \
                f"from '{input_rvalues}'. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the predicted r-values were successfully
        # loaded.
        infostr = \
            "The predicted r-values were successfully loaded from " \
            f"'{input_rvalues}'."
        logger.info(infostr)

    # Otherwise
    else:

        # The r-values will be None.
        df_r_values = None
    
    #-----------------------------------------------------------------#

    # Initialize an empty list to store the residuals for all
    # samples.
    all_res = []

    # Try to find the residuals.
    try:

        # For each sample
        for i in range(df_samples.shape[0]):
            
            # Get the current sample.
            sample = df_samples.iloc[i]

            # Get the name of the current sample.
            sample_name = df_samples.index[i]

            # Get the predicted means for the current sample.
            means_sample = df_pred_means.iloc[i]
            
            # If the r-values were passed
            if df_r_values is not None:

                # Get the r-values for the current sample.
                r_values_sample = df_r_values.iloc[i]
            
            # Otherwise
            else:

                # The r-values will be None.
                r_values_sample = None

            # Get the residual vector for the current sample.
            res = residuals.get_residuals(obs_counts = sample,
                                          pred_means = means_sample,
                                          r_values = r_values_sample,
                                          sample_name = sample_name)

            # Save the current residual vector in the list of
            # residuals.
            all_res.append(res)
    
    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to find the residuals. " \
            f"Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the residuals were successfully found.
    logger.info(\
        "The residual values for the samples were successfully found.")

    #-----------------------------------------------------------------#

    # Concatenate all residual values into one data frame and
    # transpose it.
    df_res = pd.concat(all_res,
                       axis = 1).T

    # Add the extra columns.
    df_res = pd.concat([df_res, df_extra],
                        axis = 1)

    #-----------------------------------------------------------------#

    # Try to save the residuals to the output CSV file.
    try:

        df_res.to_csv(output_residuals,
                      sep = ",",
                      index = True,
                      header = True)
    
    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to save the residual vectors " \
            f"in '{output_residuals}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the residual vectors were saved.
    logger.info(\
        f"The residual vectors successfully saved in " \
        f"'{output_residuals}'.")
