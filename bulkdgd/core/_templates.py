#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    _templates.py
#
#    Templates for the different configurations.
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
__doc__ = "Templates for the different configurations."


#######################################################################


# Import from the standard library.
import copy

# Import from bulkdgd.
from bulkdgd import _internals
from bulkdgd import core
from . import metrics


#######################################################################


def _override_items(d, paths2values):
    """Return a copy of ``d`` with the value at the end of each "key
    path" REPLACED by the given value.

    Parameters
    ----------
    d : :class:`dict`
        The template, or template section, to copy.

    paths2values : :class:`dict`
        A mapping from "key path" - a tuple of keys leading to the key
        to be set - to the value that key should take.

    Returns
    -------
    :class:`dict`
        The copy, with the values replaced.
    """

    # THIS IS NOT 'bulkdgd._internals.recursive_add_items'. That one
    # only ADDS a key that is missing: it walks to the end of the path
    # and, finding the key already there, leaves the value alone. It is
    # the right thing for filling a template out, and it is the wrong
    # thing here, because every default this module wants to specialize
    # - a learning rate for one section that differs from the learning
    # rate for another, the number of epochs one optimization round
    # runs for - is a default that the shared template already carries
    # a value for, and so is precisely the case that function declines
    # to touch.
    #
    # The copy is deep, so that specializing a shared section - and the
    # optimizer, the scheduler and the per-round options are all shared
    # between several templates - cannot reach back into the section it
    # was specialized from.
    new_d = copy.deepcopy(d)

    # For each key path and the value the key should take
    for key_path, value in paths2values.items():

        # Start at the top of the copy.
        current = new_d

        # Walk down to the dictionary holding the last key, creating
        # the intermediate levels if the template does not have them.
        for key in key_path[:-1]:

            current = current.setdefault(key, {})

        # Set the value, whether or not the key was already there.
        current[key_path[-1]] = value

    # Return the copy.
    return new_d


#######################################################################


# Set the template for the options of the 'lgmm' latent type in the
# model configuration.
_MODEL_LGMM_OPTIONS = {

    # The number of components in the Gaussian mixture model.     
    "n_components" : {
        
        "type" : (int,),
        "condition" : lambda v: v > 0,
        "message" : "must be a positive integer",
        "default" : 45,
        },

    # The type of covariance to use in the Gaussian mixture model.       
    "covariance_type" : {
        "type" : (str,),
        "choices" :  ["fixed",  "isotropic",  "diagonal"],
        "default" : "diagonal",
        },

    # The type of prior distribution to use for the means of the
    # Gaussian components in the Gaussian mixture model.
    "means_prior_type" : {
        "type" : (str,),
        "choices" :  ["softball"],
        "default" : "softball",
        },
    
    # The options for the prior distribution of the means of the
    # Gaussian components in the Gaussian mixture model.
    "means_prior_options" : {
        
        "switch" : {
            "option" : "means_prior_type",
            "cases" : {
                "softball" : {
                    "radius" : {
                        "type" : (float, int),
                        "condition" : lambda v: v > 0,
                        "message" : "must be a positive number",
                        },
                    "sharpness" : {
                        "type" : (float, int),
                        "condition" : lambda v: v > 0,
                        "message" : "must be a positive number"},
                        },
                    },
                },
            },

    # The type of prior distribution to use for the weights of the
    # Gaussian components in the Gaussian mixture model.       
    "weights_prior_type" : {
        "type" : (str,),
        "choices" : ["dirichlet"],
        "default" : "dirichlet",
        },

    # The options for the prior distribution of the weights of the
    # Gaussian components in the Gaussian mixture model.
    "weights_prior_options" : {
        "switch" : {
            "option" :
                "weights_prior_type",
            "cases" : {
                "dirichlet" : {
                    "alpha" : {
                        "type" : (float, int),
                        "condition" : lambda v: v > 0,
                        "message" : "must be a positive number",
                        },
                    },
                },
            },
        },

    # The type of prior distribution to use for the log variances of
    # the Gaussian components in the Gaussian mixture model.
    "log_var_prior_type" : {
        "type" : (str,),
        "choices" : ["gaussian"],
        "default" : "gaussian",
        },

    # The options for the prior distribution of the log variances of
    # the Gaussian components in the Gaussian mixture model.
    "log_var_prior_options" : {
        "switch" : {
            "option" : "log_var_prior_type",
            "cases" : {
                "gaussian" : {
                    "mean" : {
                        "type" : (float, int),
                        },
                    "stddev" : {
                        "type" : (float, int),
                        "condition" : lambda v: v > 0,
                        "message" : "must be a positive number",
                        },
                    },
                },
            },
        },
    }


#---------------------------------------------------------------------#


# Set the template for the options of the 'tgmm' latent type in the
# model configuration.
_MODEL_TGMM_OPTIONS = {
    
    # The number of components in the Gaussian mixture model.
    "n_components" : {
        "type" : (int,),
        "condition" : lambda v: v > 0,
        "message" :"must be a positive integer",
        "default" : 35,
        },

    # The type of covariance to use in the Gaussian mixture model.  
    "covariance_type" : {
        "type" : (str,),
        "choices" : core.latents.GaussianMixtureModelTGMM.COVARIANCE_TYPES,
        "default" : "spherical",
        },

    # The rank of the covariance, for 'low_rank' only.
    #
    # Each component's covariance is then W W' + diag(psi) with W of
    # this rank, which is 'rank * dim + dim' numbers per component
    # instead of the 'dim * (dim + 1) / 2' a full covariance needs.
    #
    # Four is the default because that is where BIC put the optimum on
    # both the thirty-two and the sixty-four dimensional models, and it
    # is the order the ~9-dimensional latent manifold implies a single
    # component should need. It is ignored by every other covariance
    # type.
    "rank" : {
        "type" : (int,),
        "condition" : lambda v: v > 0,
        "message" : "must be a positive integer",
        "default" : 4,
        },
    
    # The method to use for initializing the means of the Gaussian
    # components in the Gaussian mixture model.
    "init_means" : {
        "type" : (str,),
        "choices" : \
            core.latents.GaussianMixtureModelTGMM.INIT_MEANS_METHODS,
        "default" : "maxdist",
        },
    
    # The method to use for initializing the weights of the Gaussian
    # components in the Gaussian mixture model.
    "init_weights" : {
        "type" : (str,),
        "choices" : \
            core.latents.GaussianMixtureModelTGMM.INIT_WEIGHTS_METHODS,
        "default" : "uniform",
        },
    
    # The method to use for initializing the covariances of the
    # Gaussian components in the Gaussian mixture model.
    "init_covariances" : {
        "type" : (str,),
        "choices" : 
            core.latents.GaussianMixtureModelTGMM.INIT_COVARIANCES_METHODS,
        "default" : "empirical",
        },
    
    # The tolerance for convergence in the Gaussian mixture model.
    "tol" : {
        "type" : (float, int),
        "condition" : lambda v: v > 0,
        "message" : "must be a positive number",
        "default" : 1e-4,
        },
    
    # The regularization term.
    "reg_covar" : {
        "type" : (float, int),
        "condition" : lambda v: v >= 0,
        "message" : "must be a non-negative number",
        "default" : 1e-6,
        },
    
    # The number of initializations to perform.
    "n_init" : {
        "type" : (int,),
        "condition" : lambda v: v > 0,
        "message" : "must be a positive integer",
        "default" : 1},
    
    # The random state to use.
    "random_state" : {
        "type" : (int,),
        "default" : None,
        },
    
    # Whether to use the CEM algorithm for fitting the Gaussian mixture
    # model.
    "cem" : {
        "type" : (bool,),
        "default" : False,
        },
    }


#---------------------------------------------------------------------#


# Set the template for the options of the Gaussian mixture model
# fitted to the representations after training - what the latent space
# IS, as opposed to the prior that produced it.
_MODEL_GMM_FINAL_OPTIONS = {

    # A model that does not ask for a final mixture is trained exactly
    # as before and writes no 'gmm_final.pth'. Without this, an absent
    # section would be filled in with its own defaults, which is a
    # request for a final mixture rather than the absence of one.
    "__optional__" : True,

    # The type of covariance the final Gaussian mixture model should
    # have. There is no default: a model that asks for a final mixture
    # is asking for a covariance the prior did not have, and which one
    # is the whole of the request.
    "covariance_type" : {
        "type" : (str,),
        "choices" : core.latents.GaussianMixtureModelTGMM.COVARIANCE_TYPES,
        },

    # How far each per-component covariance is pulled back towards the
    # one shared by all of them.
    #
    # It does nothing for the tied covariance types, which are already
    # the thing being shrunk towards, and it is required for 'full':
    # a per-component full covariance has more numbers in it than most
    # components have samples to fit them from, and comes back with
    # negative variances.
    "shrinkage" : {
        "type" : (float, int),
        "condition" : lambda v: 0.0 <= v <= 1.0,
        "message" : "must be between 0.0 and 1.0",
        "default" : 0.0,
        },

    # The value added to the diagonal of the covariance. If not given,
    # the trained mixture's own value is used. It is nullable, and has
    # to be declared as such: the option defaults to None, so a
    # configuration that was loaded (and therefore had the default
    # filled in) could not be loaded again - the check for the
    # condition was handed a None and raised.
    "reg_covar" : {
        "type" : (float, int, type(None)),
        "condition" : lambda v: v >= 0,
        "message" : "must be a non-negative number",
        "default" : None,
        },

    }


#---------------------------------------------------------------------#


# Set the template for the decoder's options in the model
# configuration.
_MODEL_DECODER_OPTIONS = {

    # The number of units in the hidden layers of the decoder.
    "n_units_hidden_layers" : {
        "type" : (list,),
        "condition" :
            lambda v: len(v) > 0 and all(isinstance(n, int) \
                      and n > 0 for n in v),
        "message" : "must be a non-empty list of positive integers",
        },
    
    # The activation functions to use in the hidden layers of the
    # decoder.
    "activations" : {
        "type" : (list,),
        "choices" : core.decoders.Decoder.ACTIVATIONS,
        },
    
    # The type of normalization to use in the hidden layers of the
    # decoder.
    "dropout" : {
        "type" : (float, int),
        "condition" : lambda v: 0 <= v <= 1,
        "message" : "must be a float between 0 and 1",
        "default" : 0,
        },
    
    # The type of output module to use.
    "output_module_name" : {
        "type" : (str,),
        "choices" : list(core.outputmodules.OUTPUT_MODULES.keys()),
        },
    
    # The options for the output module.
    "output_module_options" : {
        "switch" : {
            "option": "decoder_options.output_module_name",
            "cases" : {
                "poisson" : {
                    "activation" : {
                        "type" : (str,),
                        "choices" : 
                            core.outputmodules.
                                 OutputModulePoisson.
                                    ACTIVATION_FUNCTIONS,
                        },
                    },
            
                "nb_feature_dispersion" : {
                    "activation" : {
                        "type" : (str,),
                        "choices" : 
                            core.outputmodules.
                                 OutputModuleNBFeatureDispersion.
                                    ACTIVATION_FUNCTIONS,
                        },
                    "r_init" : {
                        "type" : (float, int),
                        "condition" : lambda v: v > 0,
                        "message" : "must be a positive number",
                        "default" : 2,
                        },
                    },
            
                "nb_full_dispersion" : {

                    "activation" : {
                        "type" : (str,),
                        "choices" :
                            core.outputmodules.
                                OutputModuleNBFullDispersion.
                                    ACTIVATION_FUNCTIONS,
                        },
                    },

                # Full dispersion, tied to the mean.
                "nb_full_dispersion_tied" : {
                    "activation" : {
                        "type" : (str,),
                        "choices" :
                            core.outputmodules.
                                OutputModuleNBFullDispersion.
                                    ACTIVATION_FUNCTIONS,
                        },
                    "r_init" : {
                        "type" : (float, int),
                        "condition" : lambda v: v > 0,
                        "message" : "must be a positive number",
                        "default" : 2,
                        },
                    },

                # Full dispersion, with a per-gene hierarchical prior
                # on the per-sample deviation whose width is LEARNED.
                #
                # There is no 'shrinkage_lambda' here, and its absence
                # is the point: the strength of the pull is what this
                # module estimates instead of being told. What is set
                # is only where the estimate starts and how far down it
                # may go.
                "nb_full_dispersion_hierarchical" : {
                    "activation" : {
                        "type" : (str,),
                        "choices" :
                            core.outputmodules.
                                OutputModuleNBFullDispersion.
                                    ACTIVATION_FUNCTIONS,
                        },
                    # Narrow to begin with, so training starts near the
                    # stable per-gene dispersion and widens only where
                    # the data ask for it.
                    "sigma_init" : {
                        "type" : (float, int),
                        "condition" : lambda v: v > 0,
                        "message" : "must be a positive number",
                        "default" : 0.1,
                        },
                    # The 'log sigma' term diverges at zero, so the
                    # width needs a floor for a gene whose deviations
                    # all vanish.
                    "sigma_min" : {
                        "type" : (float, int),
                        "condition" : lambda v: v > 0,
                        "message" : "must be a positive number",
                        "default" : 0.01,
                        },
                    # The prior on the widths themselves. Without it
                    # the objective is unbounded - a gene whose
                    # deviations reach zero sends its own width after
                    # them and 'log sigma' to minus infinity.
                    "sigma_prior" : {
                        "type" : (float, int),
                        "condition" : lambda v: v > 0,
                        "message" : "must be a positive number",
                        "default" : 0.1,
                        },
                    "sigma_prior_tau" : {
                        "type" : (float, int),
                        "condition" : lambda v: v > 0,
                        "message" : "must be a positive number",
                        "default" : 1.0,
                        },
                    "r_init" : {
                        "type" : (float, int),
                        "condition" : lambda v: v > 0,
                        "message" : "must be a positive number",
                        "default" : 2,
                        },
                    },
                },
            },
        },
    }


#---------------------------------------------------------------------#


# Set the template for the data loader's options in the training
# configuration.
_TRAIN_DATA_LOADER = {
    
    # The options for the data loader for training data.
    "train" : {
        "batch_size" : {
            "type": (int,),
            "condition": lambda v: v > 0,
            "message": "must be a positive integer",
            "default": 64,
            },
        "shuffle" : {
            "type": (bool,),
            "default": True,
            },
        },
    
    # The options for the data loader for test data.
    "test" : {
        "batch_size" : {
            "type": (int,),
            "condition": lambda v: v > 0,
            "message": "must be a positive integer",
            "default": 64,
            },
        "shuffle" : {
            "type": (bool,),
            "default": False,
            },
        },
    }


#---------------------------------------------------------------------#


# Set the template for the Adam optimizer's options.
_OPTIMIZER_ADAM = {
    
    # The learning rate for the optimizer.
    "lr" : {
        "type": (float, int),
        "condition": lambda v: v > 0,
        "message": "must be a positive number",
        "default": 0.001,
        },
    
    # The weight decay for the optimizer.
    "weight_decay" : {
        "type": (float, int),
        "condition": lambda v: v >= 0,
        "message": "must be non-negative",
        "default": 0.0,
        },
    
    # The beta parameters for the optimizer.
    #
    # A LIST, and not the tuple this used to be. A configuration is
    # validated more than once on the way to a trained model - when it
    # is loaded, and again inside the method that consumes it - and the
    # second pass sees the value the first pass filled in. The check
    # for a list option asks for an actual 'list', so a tuple default
    # made every configuration that says nothing about the betas load
    # cleanly and then be rejected.
    "betas" : {
        "type": (list,),
        "condition": lambda v: len(v) == 2 and \
            all(isinstance(x, (float, int)) for x in v),
        "message": "must be a list of two floats",
        "default": [0.9, 0.999],
        },
    }


#---------------------------------------------------------------------#


# Set the template for the AdamW optimizer's options.
_OPTIMIZER_ADAMW = {
    
    # The learning rate for the optimizer.
    "lr" : {
        "type": (float, int),
        "condition": lambda v: v > 0,
        "message": "must be a positive number",
        "default": 0.001,
        },
    
    # The weight decay for the optimizer.
    "weight_decay" : {
        "type": (float, int),
        "condition": lambda v: v >= 0,
        "message": "must be non-negative",
        "default": 0.0,
        },
    
    # The beta parameters for the optimizer.
    #
    # A LIST, and not the tuple this used to be. A configuration is
    # validated more than once on the way to a trained model - when it
    # is loaded, and again inside the method that consumes it - and the
    # second pass sees the value the first pass filled in. The check
    # for a list option asks for an actual 'list', so a tuple default
    # made every configuration that says nothing about the betas load
    # cleanly and then be rejected.
    "betas" : {
        "type": (list,),
        "condition": lambda v: len(v) == 2 and \
            all(isinstance(x, (float, int)) for x in v),
        "message": "must be a list of two floats",
        "default": [0.9, 0.999],
        },
    }


#---------------------------------------------------------------------#


# Set the template for the options of the L-BFGS optimizer.
_OPTIMIZER_LBFGS = {

    # The learning rate for the optimizer.
    #
    # One, and not the small step a first-order optimizer wants: the
    # line search decides how far to go along the direction the
    # curvature estimate picked, so this only scales its starting
    # guess.
    "lr" : {
        "type": (float, int),
        "condition": lambda v: v > 0,
        "message": "must be a positive number",
        "default": 1.0,
        },

    # How many iterations the optimizer takes per step.
    #
    # An L-BFGS 'step' is a whole optimization, not one move, so this
    # is what the epoch count is for a first-order optimizer - and the
    # epoch count then says how many such optimizations to run.
    "max_iter" : {
        "type": (int,),
        "condition": lambda v: v > 0,
        "message": "must be a positive integer",
        "default": 100,
        },

    # How many past updates are kept to approximate the curvature.
    "history_size" : {
        "type": (int,),
        "condition": lambda v: v > 0,
        "message": "must be a positive integer",
        "default": 20,
        },

    # The line search to use, or null for a fixed step.
    #
    # The strong Wolfe conditions are what make the method worth
    # having: without a line search the step length is guessed and the
    # curvature estimate can be fed a bad pair, which is how L-BFGS
    # diverges.
    "line_search_fn" : {
        "type": (str, type(None)),
        "choices": ["strong_wolfe", None],
        "default": "strong_wolfe",
        },

    # The gradient below which the optimizer stops.
    "tolerance_grad" : {
        "type": (float, int),
        "condition": lambda v: v > 0,
        "message": "must be a positive number",
        "default": 1e-9,
        },

    # The change in the parameters below which it stops.
    "tolerance_change" : {
        "type": (float, int),
        "condition": lambda v: v > 0,
        "message": "must be a positive number",
        "default": 1e-12,
        },
    }


#---------------------------------------------------------------------#


# Set the template for the optimizer's options.
_OPTIMIZER =  {

    # The type of optimizer to use.
    "optimizer_type" : {
        "type": (str,),
        "choices": ["adam", "adamw", "lbfgs"],
        "default": "adamw"},

    # The options for the optimizer.
    "optimizer_options" : {
        "switch" : {
            "option" : "optimizer_type",
            "cases" : {
                "adam" : _OPTIMIZER_ADAM,
                "adamw" : _OPTIMIZER_ADAMW,
                "lbfgs" : _OPTIMIZER_LBFGS,
                },
            },
        },

    # The norm to which the gradients are clipped before the optimizer
    # takes its step.
    #
    # If not set, the gradients are not clipped, which is what happened
    # before this option existed.
    "grad_clipping_max_norm" : {
        "type": (float, int, type(None)),
        "condition": lambda v: v is None or v > 0,
        "message": "must be a positive number, or null to not clip",
        "default": None,
        },
    }


#---------------------------------------------------------------------#


# Set the template for the learning rate scheduler's options for the
# OneCycleLR scheduler.
_LR_SCHEDULER_ONE_CYCLE = {
    
    # The maximum learning rate for the learning rate scheduler.
    "max_lr" : {
        "type": (float, int),
        "condition": lambda v: v > 0,
        "message": "must be a positive number",
        "default": 0.01,
        },
    
    # The percentage of the cycle to use for increasing the learning
    # rate.
    "pct_start" : {
        "type": (float, int),
        "condition": lambda v: 0 <= v <= 1,
        "message": "must be a number between 0 and 1",
        "default": 0.25,
        },
    
    # The annealing strategy to use for the learning rate scheduler.
    "anneal_strategy" : {
        "type": (str,),
        "choices": ["cos", "linear"],
        "default": "cos",
        },
    
    # Whether to use momentum cycling in the learning rate scheduler. 
    "cycle_momentum" : {
        "type": (bool,),
        "default": True,
        },
    
    # The base momentum for the learning rate scheduler.
    "base_momentum" : {
        "type": (float, int),
        "condition": lambda v: 0 <= v <= 1,
        "message": "must be a number between 0 and 1",
        "default": 0.85,
        },
    
    # The maximum momentum for the learning rate scheduler.
    "max_momentum" : {
        "type": (float, int),
        "condition": lambda v: 0 <= v <= 1,
        "message": "must be a number  between 0 and 1",
        "default": 0.9,
        },
    
    # The division factor for the learning rate scheduler.
    "div_factor" : {
        "type": (float, int),
        "condition": lambda v: v > 0,
        "message": "must be a positive number",
        "default": 25.0,
        },
    
    # The final division factor for the learning rate scheduler.
    "final_div_factor" : {
        "type": (float, int),
        "condition": lambda v: v > 0,
        "message": "must be a positive number",
        "default": 1000.0,
        },
    
    # Whether to use the three-phase version of the learning rate
    # scheduler.
    "three_phase" : {
        "type": (bool,),
        "default": False,
        },
    }


#---------------------------------------------------------------------#


# Set the template for the learning rate scheduler's options for the
# CosineAnnealingLR scheduler. The peak learning rate is the optimizer's
# own 'lr'; the schedule anneals it down to 'eta_min' over the whole run
# (one half-cosine, no restarts). 'T_max' is not an option here: it is
# fixed to the number of steps in the run (batches for the decoder,
# epochs for the representations) so the anneal always finishes exactly
# at the end of training, whatever the number of epochs.
_LR_SCHEDULER_COSINE = {

    # The minimum learning rate the schedule anneals down to.
    "eta_min" : {
        "type": (float, int),
        "condition": lambda v: v >= 0,
        "message": "must be a non-negative number",
        "default": 0.0,
        },
    }


#---------------------------------------------------------------------#


# Set the template for learning rate scheduler's options.
_LR_SCHEDULER = {

    # The type of learning rate scheduler to use.
    "lr_scheduler_type" : {
        "type": (str, type(None)),
        "choices": ["one_cycle", "cosine"],
        "default": None,
        },

    # The options for the learning rate scheduler.
    "lr_scheduler_options" : {
        "switch" : {
            "option" : "lr_scheduler_type",
            "cases" : {
                "one_cycle": _LR_SCHEDULER_ONE_CYCLE,
                "cosine": _LR_SCHEDULER_COSINE,
                },
            },
        },
    }


#---------------------------------------------------------------------#


# Set the template for the options for removing collapsed components
# in the training configuration.
_COMPONENTS_REMOVAL = {
    
    # The type of removal to use for the collapsed components.
    "components_removal_type" : {
        "type": (str, type(None)),
        "choices": ["weight_threshold"],
        "default": None,
        },
    
    # The options for the removal of the collapsed components.
    "components_removal_options" : {
        "switch" : {
            "option" : "components_removal_type",
            "cases" : {
                
                "weight_threshold" : {
                    "threshold" : {
                        "type": (float, int),
                        "condition": lambda v: v >= 0,
                        "message": "must be a non-negative number",
                        "default": 1e-8,
                        },
                    },
                },
            },
        },
    }


#---------------------------------------------------------------------#


# Set the template for the options of the 'tgmm' latent type in the
# training configuration.
_TRAIN_TGMM = {

    # The options for the calculation of the loss.
    "loss_calculation" : {
        "lambda" : {
            "type": (float, int),
            "condition": lambda v: v >= 0,
            "message": "must be a non-negative number",
            "default": 1.0,
            },
        },

    # The type of model selection to use for selecting the best model
    # during training.
    "model_selection_type" : {
        "type": (str, type(None)),
        "choices": ["metric"],
        "default": None,
        },
    
    # The options for the model selection to use for selecting the best
    # model during training.
    #
    # This used to say that the options were a single string - the name
    # of the metric - while the code that reads them asks them for a
    # 'metric' key, as a dictionary. A configuration written the way the
    # template described it therefore reached
    # '.get("metric", "bic")' as a string and raised an
    # 'AttributeError'. They are a dictionary, and they are described as
    # one.
    "model_selection_options" : {
        "switch" : {
            "option" : "model_selection_type",
            "cases" : {

                "metric" : {

                    "metric" : {
                        "type": (str,),
                        "choices": [
                            "bic",
                            "silhouette_score",
                            "calinski_harabasz_score",
                            "davies_bouldin_score"],
                        "default": "bic",
                        },

                    # How far from the current number of components to
                    # look.
                    #
                    # The search used to be over the current number of
                    # components, one more, and one fewer, and nothing
                    # else. A model that starts with sixty-four
                    # components and needs thirty can only walk there
                    # one component per refit, and it refits every few
                    # epochs, so it may not arrive at all.
                    "step" : {
                        "type": (int,),
                        "condition": lambda v: v >= 1,
                        "message": "must be a positive integer",
                        "default": 1,
                        },
                    },
                },
            },
        },

    # The epoch at which to start fitting the Gaussian mixture model
    # during training.
    "fitting" : {

        "first_epoch" : {
            "type": (int,),
            "condition": lambda v: v >= 0,
            "message": "must be a non-negative integer",
            "default": 25,
            },
        
        # Whether to refit the Gaussian mixture model at the end of
        # the training period.
        "refit_final" : {
            "type": (bool,),
            "default": True,
            },
        
        # The interval (in epochs) at which to refit the Gaussian
        # mixture model during training.
        "refit_interval" : {
            "type": (int,),
            "condition": lambda v: v >= 0,
            "message": "must be a non-negative integer",
            "default": 0,
            },
        
        # The maximum number of iterations for fitting the Gaussian
        # mixture model during the first epoch of fitting.
        "max_iter_first_epoch" : {
            "type": (int,),
            "condition": lambda v: v > 0,
            "message": "must be a positive integer",
            "default": 1000,
            },
        
        # The maximum number of iterations for fitting the Gaussian
        # mixture model during the epochs of refitting.
        "max_iter_full_refit" : {
            "type": (int,),
            "condition": lambda v: v > 0,
            "message": "must be a positive integer",
            "default": 100,
            },
        
        # The maximum number of iterations for fitting the Gaussian
        # mixture model during the epochs of refitting with warm
        # initialization.
        "max_iter_warm_refit" : {
            "type": (int,),
            "condition": lambda v: v > 0,
            "message": "must be a positive integer",
            "default": 100,
            },
        
        # The maximum number of iterations for fitting the Gaussian
        # mixture model during the final refitting.
        "max_iter_final_refit" : {
            "type": (int,),
            "condition": lambda v: v > 0,
            "message": "must be a positive integer",
            "default": 1000,
            },
        },

    # The options for removing collapsed components.
    **_COMPONENTS_REMOVAL,
    
    }


#---------------------------------------------------------------------#


# Set the template for the options of the 'lgmm' latent type in the
# training configuration.
_TRAIN_LGMM = {

    # The options for the optimizer used to train the latent space.
    **_internals.recursive_add_items(
        d = _OPTIMIZER,
        paths2values = {
            ("optimizer_options",
             "switch",
             "cases",
             "adam",
             "lr",
             "default") : 0.01,
            ("optimizer_options",
             "switch",
             "cases",
             "adamw",
             "lr",
             "default") : 0.01,
            }),

    # The options for the learning rate scheduler used to train the
    # latent space.
    **_LR_SCHEDULER,

    # The options for removing collapsed components.
    **_COMPONENTS_REMOVAL,

    }


#---------------------------------------------------------------------#


# Set the template for the options for fitting the Gaussian mixture
# model that describes the latent space after training.
# The options for the per-sample training diagnostics.
#
# Optional, and off unless present: the section answers whether some
# training samples drive the decoder more than others, which is a
# question about the gradient rather than the loss, and the hooks that
# measure it are cheap but not free.
_TRAIN_DIAGNOSTICS = {

    "__optional__" : True,

    # Whether to record each sample's contribution to the decoder's
    # gradient norm, once per epoch.
    "per_sample_grad_norm" : {
        "type": (bool,),
        "default": False,
        },

    # Save the decoder's state every this many epochs, so that a
    # TracIn pass can be run offline afterwards. Zero disables it.
    "checkpoint_every" : {
        "type": (int,),
        "default": 0,
        },

    # Where the per-epoch records and the checkpoints are written,
    # relative to the model's working directory.
    "output_dir" : {
        "type": (str,),
        "default": "diagnostics",
        },
    }


_TRAIN_GMM_FINAL = {

    # A training configuration that says nothing about the final
    # mixture is not asking for one.
    "__optional__" : True,

    # What the fit is allowed to move.
    #
    # 'covariance_only' freezes the means and the weights where
    # training left them and refits only the covariance, in one
    # closed-form M-step against the trained model's own
    # responsibilities. Nothing iterates, so no component can drift
    # onto a different part of the latent space, and every mapping
    # from a component to a label - a tissue, a cancer type -
    # established on the trained mixture stays valid.
    #
    # 'full_em' re-estimates the means and the weights as well. It
    # fits the representations better and it is a different set of
    # components: anything that labelled the old ones does not carry
    # over.
    "fit" : {
        "type" : (str,),
        "choices" : ["covariance_only", "full_em"],
        "default" : "covariance_only",
        },

    # The maximum number of iterations, used only by 'full_em'.
    "max_iter" : {
        "type" : (int,),
        "condition" : lambda v: v > 0,
        "message" : "must be a positive integer",
        "default" : 1000,
        },

    }


#---------------------------------------------------------------------#


# Set the template for the decoder training options.
#
# The defaults here are the ones the published ensemble was trained
# with, so that a configuration which leaves the decoder's training
# unspecified trains it the way the shipped model's decoder was
# trained. See 'configs/training/training.yaml', which sets all of them
# explicitly.
_TRAIN_DECODER = {

    # The options for the optimizer used to train the decoder.
    #
    # THE WEIGHT DECAY IS THE DECODER'S ALONE. It is the part of the
    # model with the parameters - one weight per latent dimension per
    # gene - and it is the only part that is regularized this way. The
    # representations are given no weight decay at all, because
    # shrinking a representation towards the origin is shrinking it
    # towards the middle of the latent space, which is a statement
    # about the sample rather than a regularizer; that is why the two
    # sections specialize the same optimizer template differently
    # instead of sharing one.
    **_override_items(
        d = _OPTIMIZER,
        paths2values = {
            ("optimizer_options",
             "switch",
             "cases",
             "adam",
             "lr",
             "default") : 0.01,
            ("optimizer_options",
             "switch",
             "cases",
             "adamw",
             "lr",
             "default") : 0.01,
            ("optimizer_options",
             "switch",
             "cases",
             "adamw",
             "weight_decay",
             "default") : 0.1,
            }),

    # The options for the learning rate scheduler used to train the
    # decoder.
    #
    # The one-cycle schedule is on by default, because the learning
    # rate above is the PEAK of that schedule and not a rate to be used
    # flat: run unscheduled, it is a large constant rate rather than
    # the rate the model was trained at.
    **_override_items(
        d = _LR_SCHEDULER,
        paths2values = {
            ("lr_scheduler_type",
             "default") : "one_cycle",
            }),

    }


#---------------------------------------------------------------------#


# Set the template for the representations training options.
#
# The defaults here are the ones the published ensemble was trained
# with, for the same reason the decoder's are. See
# 'configs/training/training.yaml', which sets all of them explicitly.
_TRAIN_REPRESENTATIONS = {

    # The type of noise to add to the representations during training.
    #
    # ON by default, which is what training does and what finding a
    # representation does not. The representations are free parameters,
    # one per training sample, and nothing stops a decoder from
    # memorizing a representation only that one sample can reach;
    # perturbing them while they are learned forces the decoder to
    # produce the sample from a NEIGHBOURHOOD of the representation
    # rather than from a point, which is the property the search for a
    # new sample's representation later relies on.
    #
    # 'none' is a CHOICE and not only a value the default used to take.
    # A configuration is validated more than once on the way to a
    # trained model - when it is loaded, and again inside 'train' - and
    # the second pass sees the key the first pass filled in, so a
    # 'none' that was not also a legal value made every configuration
    # saying nothing about the noise load and then be rejected.
    "train_noise_type" : {
        "type": (str, type(None)),
        "choices": ["gaussian", "none"],
        "default": "gaussian",
        },

    # The options for the noise to add to the representations during
    # training.
    "train_noise_options" : {
        "switch" : {
            "option" : "train_noise_type",
            "cases" : {
                "gaussian" : {

                    "scale" : {
                        "type": (float, int),
                        "condition": lambda v: v >= 0,
                        "message": "must be a non-negative number",
                        "default": 0.1,
                        },

                    "start" : {
                        "type": (float, int),
                        "condition": lambda v: v >= 0,
                        "message": "must be a non-negative number",
                        "default": 1.0,
                        },
                    
                    "end" : {
                        "type": (float, int),
                        "condition": lambda v: v >= 0,
                        "message": "must be a non-negative number",
                        "default": 0.01,
                        },
                    
                    "within_radius_prob" : {
                        "type": (float, int),
                        "condition": lambda v: 0 <= v <= 1,
                        "message": "must be a number between 0 and 1",
                        "default": 0.95,
                        },
                    
                    # The final multiplier on the noise.
                    #
                    # Four, which is what the shipped model was trained
                    # with. It is not a free parameter for anything
                    # meaning to reproduce that model: a different gain
                    # perturbs the representations by a different
                    # amount than the ones this decoder was fitted
                    # against were perturbed by.
                    "gain" : {
                        "type": (float, int),
                        "condition": lambda v: v >= 0,
                        "message": "must be a non-negative number",
                        "default": 4.0,
                        },
                    },
                },
            },
        },

    # The options for the optimizer used to train the representations.
    #
    # NO WEIGHT DECAY, unlike the decoder's - see there for why.
    **_override_items(
        d = _OPTIMIZER,
        paths2values = {
            ("optimizer_options",
             "switch",
             "cases",
             "adam",
             "lr",
             "default") : 0.01,
            ("optimizer_options",
             "switch",
             "cases",
             "adamw",
             "lr",
             "default") : 0.01,
            }),

    # The options for the learning rate scheduler used to train the
    # representations.
    #
    # The representations are scheduled on epochs and the decoder on
    # batches, but the two are given the same shape on purpose: a
    # decoder annealing towards its final weights while the
    # representations it is fitted against are still moving at their
    # starting rate is a decoder fitted to a moving target.
    **_override_items(
        d = _LR_SCHEDULER,
        paths2values = {
            ("lr_scheduler_type",
             "default") : "one_cycle",
            }),

    }


#---------------------------------------------------------------------#


# Set the template for the loss options.
#
# THE NORMALIZATION IS PER SAMPLE BY DEFAULT, which is what the
# published runs used. The cohorts a representation is found for differ
# in size by orders of magnitude - a few dozen samples for one disease,
# thousands for TCGA - so an unnormalized loss says more about how many
# samples were in the file than about how well the model reached them,
# and the number is looked at precisely to compare one cohort with
# another.
_LOSS_OPTIONS = {

    # The type of reduction to use for the loss.
    "reduction_type" : {
        "type": (str,),
        "choices": ["mean", "sum"],
        "default": "sum",
        },

    # The options for the normalization of the loss for the latent
    # space.
    "latent" : {
        "norm_type" : {
            "type": (str,),
            "choices": \
                ["none", "n_samples", "n_samples * latent_dim"],
            "default": "n_samples"},
        "lambda" : {
            "type": (float, int),
            "condition": lambda v: v >= 0,
            "message": "must be a non-negative number",
            "default": 1.0,
            },
        },

    # The options for the normalization of the loss for the decoder.
    "decoder" : {
        "norm_type" : {
            "type": (str,),
            "choices": \
                ["none", "n_samples", "n_samples * n_genes"],
            "default": "n_samples",
            },
        },

    # The options for the normalization of the loss for the total loss.
    "total" : {
        "norm_type" : {
            "type": (str,),
            "choices": \
                ["none", "n_samples", "n_samples * n_genes"],
            "default": "n_samples",
            },
        },
    }


#---------------------------------------------------------------------#


# Set the template for the reporting options.
_REPORTING_OPTIONS = {

    # The options for the loss.
    #
    # THE NORMALIZATION IS PER SAMPLE BY DEFAULT, which is what the
    # published runs used. The training and test splits do not hold the
    # same number of samples, so an unnormalized loss cannot be
    # compared between them - and that comparison is what the loss is
    # written out for.
    "loss" : {

        # The options for the normalization of the loss for the latent
        # space.
        "latent" : {
            "norm_type" : {
                "type": (str,),
                "choices": \
                    ["none", "n_samples", "n_samples * latent_dim"],
                "default": "n_samples",
                },
            },

        # The options for the normalization of the loss for the
        # decoder.
        "decoder" : {
            "norm_type" : {
                "type": (str,),
                "choices": \
                    ["none", "n_samples", "n_samples * n_genes"],
                "default": "n_samples",
                },
            },

        # The options for the normalization of the loss for the total
        # loss.
        "total" : {
            "norm_type" : {
                "type": (str,),
                "choices": \
                    ["none", "n_samples", "n_samples * n_genes"],
                "default": "n_samples",
                },
            },
        },

    # The options for the metrics to calculate during training.
    "metrics" : {

        # The options for the metrics to calculate for the latent
        # space.
        #
        # The six the published ensemble was trained with. The first
        # four are unsupervised and are always available; the last two
        # compare the mixture's components against ground-truth labels,
        # so they are calculated only if labels were passed to 'train'
        # and are dropped with a warning if they were not. Asking for
        # them by default therefore costs nothing to a caller who has
        # no labels, and saves the caller who does from having to know
        # to ask.
        "latent" : {
            "type" : (list,),
            "choices" : [*list(metrics.UNSUPERVISED_METRICS.keys()),
                         *list(metrics.SUPERVISED_METRICS.keys())],
            "default" : ["bic",
                         "silhouette_score",
                         "davies_bouldin_score",
                         "calinski_harabasz_score",
                         "adjusted_rand_index_score",
                         "adjusted_mutual_info_score"],
            },
        },
    
    # The options for the optional outputs.
    "optional_outputs" : {

        # The options for the model to output at the end of each epoch
        # during training.
        #
        # Training writes the model out only when it is over, so a run
        # that dies at the last epoch - or is killed, or runs out of
        # time - leaves nothing behind. Enabling this saves the
        # decoder's weights and the latent space's parameters as the
        # run goes, so it can be picked up from where it got to.
        "model_epoch" : {
            "enabled" : {
                "type": (bool,),
                "default": False,
                },
            "stride" : {
                "type": (int,),
                "condition": lambda v: v > 0,
                "message": "must be a positive integer",
                "default": 1,
                },
            "dir" : {
                "type": (str, type(None)),
                "default": None,
                },
            },

        # The options for the representations to output at the end of
        # each epoch during training.
        "representations_epoch" : {
            "enabled" : {
                "type": (bool,),
                "default": False,
                },
            "stride" : {
                "type": (int,),
                "condition": lambda v: v > 0,
                "message": "must be a positive integer",
                "default": 1,
                },
            "dir" : {
                "type": (str, type(None)),
                "default": None,
                },
            },
        
        # The options for the latent probabilities to output at the
        # end of each epoch during training.
        "latent_probs_epoch" : {
            "enabled" : {
                "type": (bool,),
                "default": False,
                },
            "stride" : {
                "type": (int,),
                "condition": lambda v: v > 0,
                "message": "must be a positive integer",
                "default": 1,
                },
            "dir" : {
                "type": (str, type(None)),
                "default": None,
                },
            },
        
        # The options for the latent means to output at the end of each
        # epoch during training.
        "latent_means_epoch" : {
            "enabled" : {
                "type": (bool,),
                "default": False,
                },
            "stride" : {
                "type": (int,),
                "condition": lambda v: v > 0,
                "message": "must be a positive integer",
                "default": 1,
                },
            "dir" : {
                "type": (str, type(None)),
                "default": None,
                },
            },
        
        # The options for the gene-level saliency maps to output at the
        # end of each epoch during training.
        "genes_saliency_maps_epoch" : {
            "enabled" : {
                "type": (bool,),
                "default": False,
                },
            "stride" : {
                "type": (int,),
                "condition": lambda v: v > 0,
                "message": "must be a positive integer",
                "default": 1,
                },
            "dir" : {
                "type": (str, type(None)),
                "default": None,
                },
            },
        
        # The options for the pathway-level saliency maps to output at
        # the end of each epoch during training.
        "pathways_saliency_maps_epoch" : {
            "enabled" : {
                "type": (bool,),
                "default": False,
                },
            "stride" : {
                "type": (int,),
                "condition": lambda v: v > 0,
                "message": "must be a positive integer",
                "default": 1,
                },
            "dir" : {
                "type": (str, type(None)),
                "default": None,
                },
            },
        },
    }


#---------------------------------------------------------------------#


# Set the template for the options for the optimizations.
_REP_OPTIMIZATION = {
    
    # The number of epochs for the optimization.
    "epochs" : {
        "type": (int,),
        "condition": lambda v: v > 0,
        "message": "must be a positive integer",
        "default": 50,
        },
    
    # Whether to use automatic learning rate for the optimization.
    "auto_lr" : {
        "type": (bool,),
        "default": False,
        },

    # The type of noise to add to the representations while they are
    # being optimized - the same perturbation training applies to its
    # own representations, under the same options.
    #
    # It defaults to OFF, unlike training's, and deliberately: the
    # decoder is fixed here and the representation is an inference
    # about a sample, so a configuration that says nothing about noise
    # must keep finding the representation it found before this option
    # existed.
    # 'none' is a CHOICE and not only the default. Validation happens
    # more than once on the way to a representation - the config is
    # parsed when it is loaded and parsed again inside
    # 'get_representations' - and the second pass sees the key the
    # first pass filled in. If the default were not also a legal value,
    # every config that says nothing about noise would load and then be
    # rejected, which is the whole installed base of them.
    "noise_type" : {
        "type": (str, type(None)),
        "choices": ["gaussian", "none"],
        "default": "none",
        },

    # The options for that noise, with the same meanings they have in
    # '_TRAIN_REPRESENTATIONS'. The scale is annealed, cosine, from
    # 'start' to 'end' across THIS optimization's epochs, and the noise
    # is divided by the radius of the hypersphere holding
    # 'within_radius_prob' of the mass so that a scale means the same
    # thing whatever the latent dimensionality.
    "noise_options" : {
        "switch" : {
            "option" : "noise_type",
            "cases" : {
                "gaussian" : {

                    "scale" : {
                        "type": (float, int),
                        "condition": lambda v: v >= 0,
                        "message": "must be a non-negative number",
                        "default": 0.0,
                        },

                    "start" : {
                        "type": (float, int),
                        "condition": lambda v: v >= 0,
                        "message": "must be a non-negative number",
                        "default": 1.0,
                        },

                    "end" : {
                        "type": (float, int),
                        "condition": lambda v: v >= 0,
                        "message": "must be a non-negative number",
                        "default": 0.01,
                        },

                    "within_radius_prob" : {
                        "type": (float, int),
                        "condition": lambda v: 0 <= v <= 1,
                        "message": "must be a number between 0 and 1",
                        "default": 0.95,
                        },

                    "gain" : {
                        "type": (float, int),
                        "condition": lambda v: v >= 0,
                        "message": "must be a non-negative number",
                        "default": 1.0,
                        },
                    },
                },
            },
        },

    # The options for the optimizer (spread flat, matching how
    # '_get_representations_two_opt' actually reads 'optimizer_type'/
    # 'optimizer_options', and matching '_TRAIN_DECODER''s pattern --
    # not nested under an 'optimizer' key).
    #
    # The learning rate is the one every published set of
    # representations was found at, and is ten times the optimizer's
    # own default: a representation is being optimized from scratch
    # against a fixed decoder, not nudged from a good starting point,
    # and there are only a few hundred epochs to get there in.
    **_override_items(
        d = _OPTIMIZER,
        paths2values = \
            {("optimizer_options",
                "switch",
                "cases",
                "adam",
                "lr",
                "default") : 0.01,
                ("optimizer_options",
                "switch",
                "cases",
                "adamw",
                "lr",
                "default") : 0.01,
            }),
    }


#---------------------------------------------------------------------#


# Set the template for the options of the optimizers in the 
# representations configuration for the 'two_opt' scheme when
# the latent space is the legacy Gaussian mixture model.
_REP_TWO_OPT_LGMM = {
    # How much of a sample the model is allowed to give up on when
    # FINDING A REPRESENTATION. Zero is the plain negative binomial and
    # is what training uses; a small value bounds what a gene the model
    # cannot reach may do to the representation, which matters for a
    # tumour because those genes are the signal. See
    # 'OutputModuleNBFullDispersion.loss'.
    "contamination" : {
        "type": (float, int),
        "condition": lambda v: 0.0 <= v < 1.0,
        "message": "must be in [0, 1)",
        "default": 0.0,
        },

    # The r-value of the outlier component of that mixture.
    "contamination_r" : {
        "type": (float, int),
        "condition": lambda v: v > 0,
        "message": "must be a positive number",
        "default": 0.05,
        },


    # The reduction method to use for the loss.
    "loss_reduction_type" : {
        "type": (str,),
        "choices": ["mean", "sum"],
        "default": "sum",
        },
    
    # A data-driven starting point for the search, replacing one of
    # the mixture draws. Optional, and absent by default.
    #
    # See 'core/warmstart.py': the prediction takes the slot of a
    # single candidate rather than being added to them, so every count
    # the scheme assumes stays the same. The section sits on BOTH
    # two-optimization schemes because the integration is in the shared
    # '_get_representations_two_opt'.
    "warm_start" : {

        "__optional__" : True,

        # The fitted predictor, written by
        # 'warmstart.fit_from_model_dir'.
        "pth_file" : {
            "type": (str, type(None)),
            "default": None,
            },
        },

    # The options for the first optimization of the representations.
    "optimization_1" : \
        _internals.recursive_add_items(
            d = _REP_OPTIMIZATION,
            paths2values = \
                {("epochs",
                  "default") : 10}),

    # The options for the second optimization of the representations.
    "optimization_2" : _REP_OPTIMIZATION,
    
    }

#---------------------------------------------------------------------#


# Set the template for the options of the optimizers in the
# representations configuration for the 'two_opt' scheme when
# the latent space is the TorchGMM wrapper.
_REP_TWO_OPT_TGMM = {
    # How much of a sample the model is allowed to give up on when
    # FINDING A REPRESENTATION. Zero is the plain negative binomial and
    # is what training uses; a small value bounds what a gene the model
    # cannot reach may do to the representation, which matters for a
    # tumour because those genes are the signal. See
    # 'OutputModuleNBFullDispersion.loss'.
    "contamination" : {
        "type": (float, int),
        "condition": lambda v: 0.0 <= v < 1.0,
        "message": "must be in [0, 1)",
        "default": 0.0,
        },

    # The r-value of the outlier component of that mixture.
    "contamination_r" : {
        "type": (float, int),
        "condition": lambda v: v > 0,
        "message": "must be a positive number",
        "default": 0.05,
        },


    # The reduction method to use for the loss.
    "loss_reduction_type" : {
        "type": (str,),
        "choices": ["mean", "sum"],
        "default": "sum",
        },

    # The options for calculating the loss of the latent space.
    "latent_loss_calculation" : {
        "lambda" : {
            "type": (float, int),
            "condition": lambda v: v >= 0,
            "message": "must be a non-negative number",
            "default": 1.0,
            },
        },

    # A data-driven starting point for the search, replacing one of
    # the mixture draws. Optional, and absent by default.
    #
    # See 'core/warmstart.py': the prediction takes the slot of a
    # single candidate rather than being added to them, so every count
    # the scheme assumes stays the same. The section sits on BOTH
    # two-optimization schemes because the integration is in the shared
    # '_get_representations_two_opt'.
    "warm_start" : {

        "__optional__" : True,

        # The fitted predictor, written by
        # 'warmstart.fit_from_model_dir'.
        "pth_file" : {
            "type": (str, type(None)),
            "default": None,
            },
        },

    # The options for the first optimization of the representations.
    #
    # THE EPOCH COUNTS ARE THE PUBLISHED ONES. This round optimizes
    # every candidate of every sample, and its only job is to make the
    # selection between them meaningful: a candidate that is behind
    # because it started further away has to be given enough steps to
    # catch up, or the selection picks the luckiest initialization
    # rather than the best component.
    "optimization_1" : \
        _override_items(
            d = _REP_OPTIMIZATION,
            paths2values = \
                {("epochs",
                  "default") : 300}),

    # The options for the second optimization of the representations.
    #
    # Longer than the first, because this is the round whose answer is
    # kept. The first only has to rank the candidates; this one has to
    # converge the winner, and it is optimizing one representation per
    # sample rather than one per component per sample, so the extra
    # epochs cost a fraction of what they would have cost above.
    "optimization_2" : \
        _override_items(
            d = _REP_OPTIMIZATION,
            paths2values = \
                {("epochs",
                  "default") : 500}),

    }


#######################################################################
 

# Set the template for the model's configuration.
CONFIG_MODEL = {
    
    # The path to the file containing the genes to use for the model.
    "genes_txt_file" : {
        "type" : (str,),
        },
    
    # The dimension of the latent space.
    "latent_dim" : {
        "type" : (int,),
        "condition" : lambda v: v > 0,
        "message" : "must be a positive integer",
        "default" : 64,
        },
    
    # The type of latent space to use in the model.
    "latent_type" : {
        "type" : (str,),
        "choices" : ["lgmm", "tgmm"],
        "default" : "tgmm",
        },
    
    # The options for the latent space in the model.
    "latent_options" : {
        "switch" : {
            "option" : "latent_type",
            "cases" : {
                "lgmm" : _MODEL_LGMM_OPTIONS,
                "tgmm" : _MODEL_TGMM_OPTIONS,
                },
            },
        },
    
    # The options for the Gaussian mixture model fitted to the
    # representations after training.
    #
    # This section is optional, and a model without it is trained
    # exactly as before and writes no 'gmm_final.pth'. It is a
    # separate mixture from the one in 'latent_options': that one is
    # the prior, it is what finding a representation for a new sample
    # goes through, and it is not replaced.
    "gmm_final" : _MODEL_GMM_FINAL_OPTIONS,

    # The options for the decoder in the model.
    "decoder_options" : _MODEL_DECODER_OPTIONS,

    # How the scaling factor of a sample is computed - the number the
    # decoder's predicted means are multiplied by to put them on the
    # scale of the sample's own counts.
    #
    # It belongs to the model and not to a run of it: the decoder is
    # fitted against it, the median of a sample's counts is about a
    # third of its mean, and a model trained with one and used with the
    # other has every predicted mean wrong by that ratio - without
    # failing.
    #
    # The default is the mean, which is what every model built before
    # there was anything else to choose was trained with.
    "scaling_factor" : {
        "type" : (str,),
        "choices" : ["mean", "median"],
        "default" : "mean",
        },

    # The precision the model's parameters are built in.
    #
    # It belongs to the model for the same reason the scaling factor
    # does, and more plainly: it decides what the parameters are made
    # of, and a checkpoint is read back into the parameters that are
    # already there. A float64 model read in float32 comes back a
    # float32 model, and the only thing that objects is the mixture,
    # which keeps its own tensors and then refuses to multiply with the
    # decoder.
    #
    # The default is single precision, which is torch's own and what
    # every model built before there was anything to choose was built
    # in.
    "dtype" : {
        "type" : (str,),
        "choices" : ["float32", "float64"],
        "default" : "float32",
        },

    }


# Set the template for the training configuration.
CONFIG_TRAIN = {
    
    # The number of epochs for training the model.
    "n_epochs" : {
        "type": (int,),
        "condition": lambda v: v > 0,
        "message": "must be a positive integer",
        "default": 200,
        },
    
    # The reduction method to use for the loss.
    "loss_reduction_type" : {
        "type": (str,),
        "choices": ["mean", "sum"],
        "default": "sum",
        },

    # The options for the data loaders for training and test data.
    "data_loader_options" : _TRAIN_DATA_LOADER,

    # The options for reporting during training.
    "reporting_options" : _REPORTING_OPTIONS,
    
    # The type of latent space used in the model.
    #
    # 'tgmm' is the latent space the published model has, and the one
    # every model trained with this package since it existed has. The
    # legacy implementation has to be asked for by name, because a
    # training configuration that does not say which latent space it is
    # for is not a configuration for the legacy one.
    "latent_type" : {
        "type": (str,),
        "choices": ["lgmm", "tgmm"],
        "default": "tgmm",
        },

    # The options for the latent space in the training configuration.
    "latent_training_options" : {
        "switch" : {
            "option" : "latent_type",
            "cases" : {
                "lgmm" : _TRAIN_LGMM,
                "tgmm" : _TRAIN_TGMM,
                },
            },
        },
    
    # The options for fitting the Gaussian mixture model that is
    # fitted to the representations after training.
    #
    # What that mixture IS lives in the model's configuration, under
    # 'gmm_final'; how it is obtained lives here. A model without a
    # 'gmm_final' section ignores this one.
    "gmm_final_training_options" : _TRAIN_GMM_FINAL,

    # The options for the per-sample training diagnostics. Optional.
    "training_diagnostics" : _TRAIN_DIAGNOSTICS,

    # Where to write the per-epoch learning rates, indexed by epoch.
    #
    # Optional and absent by default. 'loss.csv' has no learning-rate
    # column, so a schedule is otherwise recoverable only by parsing
    # the log text - which anything needing it after the fact will
    # quietly decline to do.
    "output_lr_file" : {
        "type": (str, type(None)),
        "default": None,
        },

    # The options for the decoder in the training configuration.
    "decoder_training_options" : _TRAIN_DECODER,

    # The options for the representations in the training
    # configuration.
    "representations_training_options" : _TRAIN_REPRESENTATIONS,

    # The type of early stopping to use during training.
    "early_stopping_type" : {
        "type": (str, type(None)),
        "choices": ["loss"],
        "default": None,
        },
    
    # The options for early stopping during training.
    "early_stopping_options" : {
        "patience" : {
            "type": (int,),
            "condition": lambda v: v > 0,
            "message": "must be a positive integer",
            "default": 10},
        },
    }


# Set the template for the configuration to find the representations
# for a new set of samples.
CONFIG_REP = {
    
    # The type of scheme to use for finding the representations for a
    # new set of samples.
    #
    # 'two_opt' is the only scheme. There was a 'one_opt' - a single
    # optimization over candidates drawn from every component, with no
    # selection step and no second descent - and it was retired: every
    # result in this project was produced with 'two_opt', so 'one_opt'
    # was an untested path that still had to be kept working.
    #
    # The switch below is left in place with one case rather than
    # collapsed away, because the point of it is that a scheme can be
    # added, and adding one should be writing a case and not rebuilding
    # the dispatch.
    # 'two_opt' is the default because it is the scheme every published
    # set of representations was found with, and because it is the one
    # that returns a single representation per sample; the multi-seed
    # scheme returns one per seed, which is a different output and has
    # to be asked for.
    "scheme_type" : {
        "type": (str,),
        "choices": ["two_opt", "two_opt_multiseed"],
        "default": "two_opt",
        },

    # The type of latent space used in the model.
    #
    # It must be the latent space the model being used actually has.
    # 'tgmm' is the shipped model's, and the legacy implementation has
    # to be asked for by name.
    "latent_type" : {
        "type": (str,),
        "choices": ["lgmm", "tgmm"],
        "default": "tgmm",
        },

    # The number of initial representations to sample per component of
    # the latent space.
    "n_rep_per_comp" : {
        "type": (int,),
        "condition": lambda v: v > 0,
        "message": "must be a positive integer",
        "default": 1,
        },

    # The options for the data loader for the new set of samples.
    #
    # A BATCH HERE IS NOT A BATCH OF SAMPLES. Every sample carries one
    # candidate representation per component of the mixture through the
    # first optimization, so with a few dozen components a batch of
    # sixteen samples is already several hundred representations being
    # optimized at once. That is what the sixteen has to be read
    # against, and it is why it is so much smaller than a training
    # batch.
    "data_loader_options" : {
        "batch_size" : {
            "type": (int,),
            "condition": lambda v: v > 0,
            "message": "must be a positive integer",
            "default": 16,
            },
        "shuffle" : {
            "type": (bool,),
            "default": False,
            },
        },

    # The options for reporting during the optimization(s).
    "reporting_options" : {
        "loss" : _LOSS_OPTIONS,
        },
    
    # The options for the specific optimization scheme.
    "scheme_options" : {
        "switch" : {
            "option" : "scheme_type",
            "cases" : {
                "two_opt" : {
                    "switch" : {
                        "option" : "latent_type",
                        "cases" : {
                            "lgmm" : _REP_TWO_OPT_LGMM,
                            "tgmm" : _REP_TWO_OPT_TGMM,
                            },
                        },
                    },

                # The multi-seed scheme takes the same options as the
                # scheme it runs, because it IS that scheme run once
                # per seed. The only difference is in
                # 'initialization', which carries 'seeds' rather than
                # 'seed'; that block is read with '.get' and is not
                # validated here, for either scheme.
                "two_opt_multiseed" : {
                    "switch" : {
                        "option" : "latent_type",
                        "cases" : {
                            "lgmm" : _REP_TWO_OPT_LGMM,
                            "tgmm" : _REP_TWO_OPT_TGMM,
                            },
                        },
                    },
                },
            },
        },
    }