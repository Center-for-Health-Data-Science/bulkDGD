#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    ensemble.py
#
#    An ensemble of bulkDGD models differing only in the seed they were
#    trained with, and the tiered consensus drawn from them.
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
    """An ensemble of bulkDGD models differing only in the seed they
    were trained with, and the tiered consensus drawn from them.

    A single model's differentially expressed genes depend on the seed
    it was trained with. Which genes are real is therefore not a
    question a single model can answer - the answer is how many of
    several identically trained models agree, and that is what an
    ensemble is for. An ensemble here is a set of models differing
    ONLY in their training seed: same architecture, same data, same
    split, same training options. Models that differ in anything else
    are not repeated draws of the same experiment, and counting their
    agreement means nothing, so the class is built so that a
    heterogeneous ensemble cannot be assembled by accident - there is
    one model configuration, shared, and a per-model configuration
    that carries only the seed and where that model's files live."""


#######################################################################


# Import from the standard library.
import copy
import logging as log
import multiprocessing as mp
import os
import time
import traceback
from typing import Optional
import zipfile

# Import from third-party libraries.
import pandas as pd

# Import from the package.
import torch
import yaml

# Import from 'bulkdgd'.
import bulkdgd
from bulkdgd import defaults
from bulkdgd.analysis import dea as analysis_dea
from bulkdgd.core.model import BulkDGD
from bulkdgd.ioutil import deaio
from bulkdgd.ioutil.tableio import save_table
from bulkdgd.reproducibility import set_seeds


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


def _get_genes_called(item: tuple) -> tuple:
    """Get, for one sample, the genes each model of the ensemble calls
    for it.

    This runs in a worker process, so everything it needs travels in
    its argument.

    Parameters
    ----------
    item : :class:`tuple`
        A tuple containing the sample's name, the group the sample
        belongs to, the directories containing the models' results,
        the prefix the per-sample files are named with, the columns to
        read, and the thresholds a gene must pass to be called.

    Returns
    -------
    result : :class:`tuple` or :obj:`None`
        A tuple containing the group the sample belongs to and the
        list of genes each model calls for it, or :obj:`None` if any
        of the models has no results for the sample.
    """

    # Unpack the item.
    sample, group, dea_dirs, prefix, usecols, q_val, log2_fold_change \
        = item

    #-----------------------------------------------------------------#

    # Initialize the list to store the genes each model calls.
    genes_called = []

    # For each model's results
    for dea_dir in dea_dirs:

        # Read the sample's statistics, from the archive if the
        # directory is packed and from the loose file otherwise.
        df_stats = deaio.read_dea(dea_dir,
                                   sample,
                                   prefix = prefix,
                                   index_col = 0,
                                   header = 0,
                                   usecols = usecols)

        # If the model has no results for the sample, the sample is
        # dropped for the whole ensemble rather than scored against
        # fewer models than the others - a gene's tier is how many
        # models agree on it, and it is only comparable between genes
        # if every gene was offered the same models.
        if df_stats is None:

            # Return nothing.
            return None

        # Name the columns. They were selected by position, so they
        # come in the order they have in the file.
        df_stats.columns = ["q_value", "log2_fold_change"]

        # Get the genes the model calls for the sample.
        genes = \
            df_stats.index[
                (df_stats["q_value"] < q_val) \
                & (df_stats["log2_fold_change"].abs() \
                    > log2_fold_change)]

        # Add them to the list.
        genes_called.append(list(genes))

    #-----------------------------------------------------------------#

    # Return the group the sample belongs to and the genes called.
    return group, genes_called


#######################################################################


class BulkDGDEnsemble:

    """A class implementing an ensemble of bulkDGD models differing
    only in the seed they were trained with.
    """

    ######################## CLASS ATTRIBUTES #########################


    # The keys each member of the ensemble must have.
    REQUIRED_KEYS = ("seed", "model_dir", "results_dir")

    #-----------------------------------------------------------------#

    # The file storing a trained latent space's parameters.
    GMM_PTH_FILE = "gmm.pth"

    # The file storing a trained decoder's parameters.
    DEC_PTH_FILE = "dec.pth"

    # The file storing the final Gaussian mixture model's parameters.
    GMM_FINAL_PTH_FILE = "gmm_final.pth"

    # The file recording what a member was seeded with. A seed that is
    # set and not recorded is a seed nobody has, and the ensemble's
    # whole claim is that its members differ only in it.
    SEEDS_FILE = "seeds.yaml"

    #-----------------------------------------------------------------#

    # The files a member's training writes beside its parameters.
    LOSS_FILE = "loss.csv"

    TIME_FILE = "time.csv"

    #-----------------------------------------------------------------#

    # The files a member's representations are written to.
    REP_FILE = "representations.csv"

    PRED_MEANS_FILE = "pred_means.csv"

    PRED_R_VALUES_FILE = "pred_r_values.csv"

    #-----------------------------------------------------------------#

    # The file a member's enrichment scores are written to.
    E_SCORES_FILE = "e_scores.csv"

    #-----------------------------------------------------------------#

    # The default thresholds a gene must pass to be called for a
    # sample, and the default share of a group's samples a model must
    # call it in for the model to consider it. They are the values the
    # bulkDGD paper's consensus was drawn with.
    DEFAULT_Q_VAL = 0.05

    DEFAULT_LOG2_FOLD_CHANGE = 1.0

    DEFAULT_RECURRENCE = 0.20

    DEFAULT_MIN_TIER = 2


    ######################### INITIALIZATION ##########################


    def __init__(self,
                 config_model: Optional[dict[str, object]] = None,
                 config_ensemble: \
                     Optional[dict[str, dict[str, object]]] = None,
                 device: str = "cpu") -> None:
        """Initialize an instance of the class.

        Parameters
        ----------
        config_model : :class:`dict`
            The configuration of the model the ensemble is made of.

            It is the configuration a single
            :class:`bulkdgd.core.model.BulkDGD` takes, and it is
            SHARED: every member of the ensemble is built from it, and
            differs from the others only in the seed it is trained
            with.

            For the available options, refer to the
            :ref:`model_config_options` page.

        config_ensemble : :class:`dict`
            The configuration of the ensemble - which members it has,
            and where each member's files live.

            It is a dictionary mapping each member's name, which is
            yours to choose, to a dictionary with these keys:

            * ``"seed"`` (:class:`int`) - the seed the member is
              trained with. It is what makes the member different from
              the others, so no two members may share it.

            * ``"model_dir"`` (:class:`str`) - the directory where the
              member's trained parameters live.

            * ``"results_dir"`` (:class:`str`) - the directory where
              the analyses run with the member live.

        device : :class:`str`, ``"cpu"``
            The device the members are placed on when they are built.
        """

        # A BARE 'BulkDGDEnsemble()' IS THE SHIPPED ENSEMBLE.
        #
        # Its members differ only in their seed, so describing them by
        # hand means writing the same architecture fifteen times and
        # fifteen paths that have to agree with it. Given neither
        # configuration, both are built from what the package ships.
        #
        # Members are still built one at a time, by 'get_model', so
        # nothing is loaded until it is asked for: the decoders are
        # 1.79 GiB each and fetched on first use.
        if config_model is None and config_ensemble is None:

            config_model, config_ensemble = self.shipped_config()

        elif config_model is None or config_ensemble is None:

            errstr = \
                "'config_model' and 'config_ensemble' describe the " \
                "ensemble together and must be given together. Give " \
                "neither to use the ensemble that ships with the " \
                "package."
            raise ValueError(errstr)

        #-------------------------------------------------------------#

        # Save the model's configuration. It is copied because it is
        # the ensemble's defining property - a caller that edits the
        # dictionary afterwards would otherwise be editing what the
        # members already built were built from.
        self._config_model = copy.deepcopy(config_model)

        #-------------------------------------------------------------#

        # Check and save the ensemble's configuration.
        self._config_ensemble = \
            self._check_config_ensemble(
                config_ensemble = config_ensemble)

        #-------------------------------------------------------------#

        # Save the device the members are placed on.
        self._device = device

        #-------------------------------------------------------------#

        # Inform the user about the ensemble that was set.
        logger.info(
            f"The ensemble was successfully set ({self.n_models} "
            f"models, seeds: "
            f"{', '.join(str(s) for s in self.seeds)}).")

        #-------------------------------------------------------------#

        # The model's configuration is shared, so anything random in
        # it is shared too. The mixture's own seed is the one such
        # thing, and it is set apart from the seed that makes the
        # members different: two members that agree on it have that
        # much less to disagree about, and it is the disagreement that
        # the consensus' tiers are drawn from.
        if self._config_model.get(
                "latent_options", {}).get("random_state") is not None:

            logger.warning(
                "The model's configuration sets "
                "'latent_options.random_state', which every member of "
                "the ensemble is built with. The members will be less "
                "different from each other than their seeds suggest. "
                "Leave it unset unless you mean it.")



    @staticmethod
    def shipped_config() -> tuple:

        """The configuration of the ensemble that ships with the
        package: the shared model configuration, and one entry per
        member.

        The members differ in nothing but the seed, so the model
        configuration is read once, from the base member, and every
        entry points at its own directory of fitted parameters.
        """

        from bulkdgd.core import _util

        # Read the shared architecture from the base member. The
        # per-member paths are filled in by 'get_model', so the
        # parameter files this returns are deliberately left out.
        config_model = _util.load_shipped_model(
            seed = defaults.BASE_SEED)

        for key in ("latent_options", "decoder_options"):
            config_model[key] = dict(config_model[key])

        config_model["latent_options"].pop("latent_pth_file", None)
        config_model["decoder_options"].pop("decoder_pth_file", None)

        # Results land under the working directory: there is nowhere
        # inside an installed package that a user's output belongs,
        # and writing there would fail on a system-wide install.
        results_root = os.path.join(os.getcwd(),
                                    "bulkdgd_ensemble_results")

        config_ensemble = {
            seed : {"seed" : int(seed.removeprefix("seed")),
                    "model_dir" : defaults.model_dir(seed),
                    "results_dir" : os.path.join(results_root, seed)}
            for seed in defaults.ENSEMBLE_SEEDS}

        return config_model, config_ensemble


    def _check_config_ensemble(
            self,
            config_ensemble: dict[str, dict[str, object]]) -> \
                dict[str, dict[str, object]]:
        """Check the ensemble's configuration.

        Parameters
        ----------
        config_ensemble : :class:`dict`
            The ensemble's configuration.

        Returns
        -------
        config_ensemble : :class:`dict`
            The ensemble's configuration, checked.
        """

        # If the configuration is empty.
        if not config_ensemble:

            # Raise an error.
            raise ValueError(
                "The ensemble's configuration is empty. It must "
                "contain at least one member.")

        #-------------------------------------------------------------#

        # Initialize an empty dictionary to store the seeds found so
        # far, and which member had them.
        seeds_found = {}

        # For each member of the ensemble
        for name, options in config_ensemble.items():

            # If the member's options are not a dictionary
            if not isinstance(options, dict):

                # Raise an error.
                raise TypeError(
                    f"The options for the member '{name}' must be a "
                    "dictionary.")

            #---------------------------------------------------------#

            # Get the keys the member is missing.
            keys_missing = \
                [key for key in self.REQUIRED_KEYS
                 if key not in options]

            # If the member is missing any key
            if keys_missing:

                # Raise an error.
                raise KeyError(
                    f"The member '{name}' is missing these required "
                    f"keys: {', '.join(keys_missing)}.")

            #---------------------------------------------------------#

            # Get the member's seed.
            seed = options["seed"]

            # If the seed is not an integer
            if not isinstance(seed, int) or isinstance(seed, bool):

                # Raise an error.
                raise TypeError(
                    f"The seed of the member '{name}' must be an "
                    "integer.")

            # If the seed was already used by another member. Two
            # members sharing a seed are the same model twice, and
            # counting them twice would inflate the agreement the
            # consensus reports.
            if seed in seeds_found:

                # Raise an error.
                raise ValueError(
                    f"The members '{seeds_found[seed]}' and "
                    f"'{name}' have the same seed ({seed}). The "
                    "members of an ensemble differ in the seed they "
                    "are trained with, so each seed may appear only "
                    "once.")

            # Record which member had the seed.
            seeds_found[seed] = name

        #-------------------------------------------------------------#

        # Return a copy of the configuration, for the same reason the
        # model's configuration is copied.
        return copy.deepcopy(config_ensemble)


    @classmethod
    def from_existing(
            cls,
            config_model: dict[str, object],
            config_ensemble: dict[str, dict[str, object]],
            device: str = "cpu") -> "BulkDGDEnsemble":
        """Build an ensemble from models that were already trained,
        without training anything.

        The models are not loaded - only checked, so that an ensemble
        whose members are not all there fails now rather than in the
        middle of the first analysis run with it.

        Parameters
        ----------
        config_model : :class:`dict`
            The configuration of the model the ensemble is made of.

        config_ensemble : :class:`dict`
            The configuration of the ensemble.

        device : :class:`str`, ``"cpu"``
            The device the members are placed on when they are built.

        Returns
        -------
        ensemble : :class:`BulkDGDEnsemble`
            The ensemble.
        """

        # Build the ensemble.
        ensemble = cls(config_model = config_model,
                       config_ensemble = config_ensemble,
                       device = device)

        #-------------------------------------------------------------#

        # Initialize an empty list to store the members whose trained
        # parameters are missing.
        members_missing = []

        # For each member of the ensemble
        for name, options in ensemble.config_ensemble.items():

            # For each file storing the trained parameters
            for pth_file in (cls.GMM_PTH_FILE, cls.DEC_PTH_FILE):

                # Get the path to the file.
                path = os.path.join(options["model_dir"], pth_file)

                # If the file is not there
                if not os.path.isfile(path):

                    # Add it to the list.
                    members_missing.append(f"'{name}' ({path})")

        #-------------------------------------------------------------#

        # If any member's parameters are missing
        if members_missing:

            # Raise an error.
            raise FileNotFoundError(
                "These members have no trained parameters: "
                f"{'; '.join(members_missing)}. Train them, or point "
                "the ensemble's configuration at the directories "
                "where they live.")

        #-------------------------------------------------------------#

        # Inform the user that the models were found.
        logger.info(
            f"The trained parameters of all {ensemble.n_models} "
            "members were found.")

        #-------------------------------------------------------------#

        # Return the ensemble.
        return ensemble


    ############################ PROPERTIES ###########################


    @property
    def config_model(self) -> dict[str, object]:
        """The configuration of the model the ensemble is made of.
        """

        return self._config_model


    @config_model.setter
    def config_model(self,
                     value) -> None:
        """Raise an exception if the user tries to modify the
        configuration of the model the ensemble is made of.
        """

        raise ValueError(
            "The configuration of the model the ensemble is made of "
            "cannot be changed after the ensemble is initialized.")


    #-----------------------------------------------------------------#


    @property
    def config_ensemble(self) -> dict[str, dict[str, object]]:
        """The configuration of the ensemble.
        """

        return self._config_ensemble


    @config_ensemble.setter
    def config_ensemble(self,
                        value) -> None:
        """Raise an exception if the user tries to modify the
        configuration of the ensemble.
        """

        raise ValueError(
            "The configuration of the ensemble cannot be changed "
            "after the ensemble is initialized. Pass a different "
            "configuration to the method you are calling to write "
            "its results elsewhere.")


    #-----------------------------------------------------------------#


    @property
    def names(self) -> list:
        """The names of the ensemble's members.
        """

        return list(self._config_ensemble.keys())


    #-----------------------------------------------------------------#


    @property
    def seeds(self) -> list:
        """The seeds the ensemble's members are trained with.
        """

        return [options["seed"]
                for options in self._config_ensemble.values()]


    #-----------------------------------------------------------------#


    @property
    def n_models(self) -> int:
        """The number of models the ensemble is made of.
        """

        return len(self._config_ensemble)


    #-----------------------------------------------------------------#


    @property
    def device(self) -> str:
        """The device the ensemble's members are placed on.
        """

        return self._device


    ######################### PRIVATE METHODS #########################


    def _get_config_ensemble(
            self,
            config_ensemble: dict[str, dict[str, object]] = None) -> \
                dict[str, dict[str, object]]:
        """Get the ensemble's configuration to be used for an
        operation - the one passed, if any, and the ensemble's own
        otherwise.

        Parameters
        ----------
        config_ensemble : :class:`dict`, optional
            The configuration to be used.

        Returns
        -------
        config_ensemble : :class:`dict`
            The configuration to be used.
        """

        # If no configuration was passed
        if config_ensemble is None:

            # Use the ensemble's own.
            return self._config_ensemble

        #-------------------------------------------------------------#

        # Otherwise, check the one passed. It is the ensemble's
        # members that it points elsewhere, so it must describe the
        # same members.
        config_ensemble = \
            self._check_config_ensemble(
                config_ensemble = config_ensemble)

        # If it does not describe the same members
        if set(config_ensemble.keys()) != set(self.names):

            # Raise an error.
            raise ValueError(
                "The configuration passed describes different "
                "members than the ensemble's own "
                f"({', '.join(sorted(config_ensemble.keys()))} "
                f"against {', '.join(sorted(self.names))}). It may "
                "point the ensemble's members at different "
                "directories, but it may not change which members "
                "there are.")

        #-------------------------------------------------------------#

        # Return the configuration.
        return config_ensemble


    def _get_dea_dir(self,
                     options: dict[str, object],
                     dea_dir: str) -> str:
        """Get the directory containing a member's differential
        expression analysis' results.

        Parameters
        ----------
        options : :class:`dict`
            The member's options.

        dea_dir : :class:`str`
            The directory containing the results. If it is a relative
            path, it is taken relative to the member's results'
            directory.

        Returns
        -------
        dea_dir : :class:`str`
            The directory containing the results.
        """

        # If the directory is an absolute path
        if os.path.isabs(dea_dir):

            # Return it as it is.
            return dea_dir

        #-------------------------------------------------------------#

        # Otherwise, take it relative to the member's results.
        return os.path.join(options["results_dir"], dea_dir)


    ######################### PUBLIC METHODS ##########################


    def get_model(self,
                  name: str,
                  config_ensemble: dict[str, dict[str, object]] = \
                    None) -> BulkDGD:
        """Get one member of the ensemble, with its trained parameters
        loaded.

        The members are built one at a time, and only when they are
        needed: a decoder is large, and an ensemble's worth of them at
        once is more memory than the machine running the analysis is
        likely to have.

        Parameters
        ----------
        name : :class:`str`
            The member's name.

        config_ensemble : :class:`dict`, optional
            The configuration to be used. If not passed, the
            ensemble's own is used.

        Returns
        -------
        model : :class:`bulkdgd.core.model.BulkDGD`
            The member.
        """

        # Get the configuration to be used.
        config_ensemble = \
            self._get_config_ensemble(
                config_ensemble = config_ensemble)

        #-------------------------------------------------------------#

        # If the member is not in the ensemble
        if name not in config_ensemble:

            # Raise an error.
            raise KeyError(
                f"'{name}' is not a member of the ensemble. The "
                f"members are: {', '.join(self.names)}.")

        #-------------------------------------------------------------#

        # Get the member's options.
        options = config_ensemble[name]

        # Get the model's configuration, and point it at the member's
        # trained parameters.
        config_model = copy.deepcopy(self._config_model)

        config_model["latent_options"]["latent_pth_file"] = \
            os.path.join(options["model_dir"], self.GMM_PTH_FILE)

        decoder_pth_file = \
            os.path.join(options["model_dir"], self.DEC_PTH_FILE)

        # THE DECODER IS FETCHED HERE, NOT SHIPPED. Each is 1.79 GiB,
        # so they live on the release rather than in the package, and
        # a member's is downloaded the first time that member is
        # built. Only members the package ships can be fetched; for
        # any other directory the missing file is the caller's to
        # provide, and 'BulkDGD' will say so.
        if not os.path.isfile(decoder_pth_file) \
                and os.path.basename(options["model_dir"]) \
                    in defaults.ENSEMBLE_SEEDS:

            bulkdgd._internals.util.download_decoder_pth(
                dest_path = decoder_pth_file)

        config_model["decoder_options"]["decoder_pth_file"] = \
            decoder_pth_file

        #-------------------------------------------------------------#

        # Return the model.
        return BulkDGD(device = self._device, **config_model)


    def status(self,
               config_ensemble: dict[str, dict[str, object]] = None,
               dea_dir: str = "dea") -> pd.DataFrame:
        """Report what each member of the ensemble already has on
        disk.

        Parameters
        ----------
        config_ensemble : :class:`dict`, optional
            The configuration to be used. If not passed, the
            ensemble's own is used.

        dea_dir : :class:`str`, ``"dea"``
            The directory containing a member's differential
            expression analysis' results. If it is a relative path, it
            is taken relative to the member's results' directory.

        Returns
        -------
        df_status : :class:`pandas.DataFrame`
            A data frame with one row per member, reporting the seed
            the member is trained with, whether its trained parameters
            are there, and how many samples it has results for.
        """

        # Get the configuration to be used.
        config_ensemble = \
            self._get_config_ensemble(
                config_ensemble = config_ensemble)

        #-------------------------------------------------------------#

        # Initialize an empty list to store the members' status.
        rows = []

        # For each member of the ensemble
        for name, options in config_ensemble.items():

            # Get the directory containing the member's differential
            # expression analysis' results.
            dea_dir_member = \
                self._get_dea_dir(options = options,
                                  dea_dir = dea_dir)

            # Get the samples the member has results for.
            samples = deaio.list_samples(dea_dir_member)

            #---------------------------------------------------------#

            # Add the member's status to the list.
            rows.append(
                {"name" : name,
                 "seed" : options["seed"],
                 "trained" : \
                    all(os.path.isfile(
                            os.path.join(options["model_dir"],
                                         pth_file))
                        for pth_file in (self.GMM_PTH_FILE,
                                         self.DEC_PTH_FILE)),
                 "n_samples_dea" : len(samples),
                 "dea_packed" : \
                    os.path.isfile(
                        os.path.join(dea_dir_member,
                                     deaio.DEA_ZIP_NAME))})

        #-------------------------------------------------------------#

        # Return a data frame with the members' status.
        return pd.DataFrame(rows).set_index("name")


    def _get_model_untrained(self) -> BulkDGD:
        """Build a member of the ensemble as it is before it is
        trained.

        Returns
        -------
        model : :class:`bulkdgd.core.model.BulkDGD`
            The untrained member.
        """

        # Get the model's configuration.
        config_model = copy.deepcopy(self._config_model)

        # Drop the trained parameters, if the configuration points at
        # any - a member that is about to be trained starts from a
        # model that was not.
        config_model.get("latent_options", {}).pop(
            "latent_pth_file", None)

        config_model.get("decoder_options", {}).pop(
            "decoder_pth_file", None)

        #-------------------------------------------------------------#

        # Return the model.
        return BulkDGD(device = self._device, **config_model)


    def _write_seeds(self,
                     model_dir: str,
                     seeds: dict) -> str:
        """Write down what a member was seeded with, beside the member
        it produced.

        Parameters
        ----------
        model_dir : :class:`str`
            The directory where the member's parameters live.

        seeds : :class:`dict`
            What the member was seeded with.

        Returns
        -------
        seeds_file : :class:`str`
            The file the seeds were written to.
        """

        # Get the path to the file.
        seeds_file = os.path.join(model_dir, self.SEEDS_FILE)

        #-------------------------------------------------------------#

        # Write the seeds down.
        with open(seeds_file, "w") as f:

            yaml.safe_dump(
                {"seeds" : seeds,
                 "dtype" : self._config_model.get("dtype"),
                 # Both versions are made strings rather than written
                 # as they come: 'torch.__version__' is a
                 # 'TorchVersion', which is a subclass of 'str' that
                 # 'yaml.safe_dump' refuses to represent, and it
                 # refuses by raising in the middle of a run that has
                 # already trained its model.
                 "bulkdgd_version" : str(bulkdgd.__version__),
                 "torch_version" : str(torch.__version__),
                 # The Gaussian mixture model's own seed, which is
                 # shared by every member because the model's
                 # configuration is.
                 "latent_random_state" : \
                    self._config_model.get(
                        "latent_options", {}).get("random_state"),
                 "device" : self._device},
                f,
                default_flow_style = False,
                sort_keys = False)

        #-------------------------------------------------------------#

        # Return the file.
        return seeds_file


    def _run_members(self,
                     config_ensemble: dict,
                     stage: str,
                     get_outputs_done: object,
                     run_member: object,
                     resume: bool,
                     return_data: bool) -> dict[str, dict]:
        """Run one stage over every member of the ensemble, one member
        at a time.

        A member that fails is recorded and the ones after it are run
        anyway: fourteen good models are not worth throwing away
        because the fifteenth could not be read.

        Parameters
        ----------
        config_ensemble : :class:`dict`
            The configuration to be used.

        stage : :class:`str`
            The name of the stage, for the messages.

        get_outputs_done : callable
            Given a member's name and options, the outputs it already
            has, or :obj:`None` if it does not have them.

        run_member : callable
            Given a member's name and options, run the stage for it,
            and return its outputs and its data.

        resume : :class:`bool`
            Whether to skip the members that are already done.

        return_data : :class:`bool`
            Whether to keep each member's data in what is returned.

        Returns
        -------
        results : :class:`dict`
            What happened to each member.
        """

        # Initialize an empty dictionary to store what happened to
        # each member.
        results = {}

        # For each member of the ensemble
        for name, options in config_ensemble.items():

            # Start the member's record.
            result = {"name" : name,
                      "seed" : options["seed"],
                      "status" : "pending",
                      "outputs" : {},
                      "error" : None,
                      "elapsed" : 0.0}

            #---------------------------------------------------------#

            # Get the outputs the member already has, if it is to be
            # skipped when it has them.
            outputs_done = \
                get_outputs_done(name, options) if resume else None

            # If the member is already done
            if outputs_done is not None:

                # Record it, and move on to the next member.
                result["status"] = "skipped"
                result["outputs"] = outputs_done

                results[name] = result

                logger.info(
                    f"[{stage}] '{name}' is already done - skipping "
                    "it. Pass 'resume = False' to run it again.")

                continue

            #---------------------------------------------------------#

            # Take the time the member started at.
            time_start = time.time()

            # Inform the user that the member is starting.
            logger.info(f"[{stage}] '{name}' (seed "
                        f"{options['seed']}) started.")

            # Try to run the stage for the member
            try:

                # Run it.
                outputs, data = run_member(name, options)

                # Record that it is done.
                result["status"] = "done"
                result["outputs"] = outputs

                # Keep the data, if they were asked for. They are left
                # out by default because an ensemble's worth of
                # predicted means is more memory than the machine
                # drawing the consensus is likely to have - everything
                # is on disk either way.
                if return_data:
                    result["data"] = data

            # If anything went wrong
            except Exception:

                # Record the failure, with what went wrong, and carry
                # on with the other members.
                result["status"] = "failed"
                result["error"] = traceback.format_exc()

                logger.error(
                    f"[{stage}] '{name}' failed:\n"
                    f"{result['error']}")

            #---------------------------------------------------------#

            # Take the time the member took.
            result["elapsed"] = time.time() - time_start

            # Store the member's record.
            results[name] = result

        #-------------------------------------------------------------#

        # Get how many members ended in each state.
        n_done = sum(1 for r in results.values()
                     if r["status"] == "done")

        n_skipped = sum(1 for r in results.values()
                        if r["status"] == "skipped")

        n_failed = sum(1 for r in results.values()
                       if r["status"] == "failed")

        # Inform the user about how the stage went.
        logger.info(
            f"[{stage}] {n_done} members ran, {n_skipped} were "
            f"already done, and {n_failed} failed.")

        # If any member failed, say so where it cannot be missed - a
        # stage that is short of a member gives a consensus whose
        # tiers are drawn from fewer models than the user thinks.
        if n_failed:

            # Get the members that failed.
            names_failed = \
                [name for name, result in results.items()
                 if result["status"] == "failed"]

            logger.warning(
                f"[{stage}] These members failed: "
                f"{', '.join(names_failed)}. Look at their 'error' "
                "before going on.")

        #-------------------------------------------------------------#

        # Return what happened to each member.
        return results


    def train(self,
              df_samples: pd.DataFrame,
              names_train: list,
              names_test: list,
              config_train: dict[str, object],
              config_ensemble: dict[str, dict[str, object]] = None,
              gmm_pth_file: str = None,
              dec_pth_file: str = None,
              gmm_final_pth_file: str = None,
              pathways: pd.DataFrame = None,
              labels_train: object = None,
              labels_test: object = None,
              resume: bool = True,
              return_data: bool = False) -> dict[str, dict]:
        """Train every member of the ensemble.

        Each member is seeded with its own seed BEFORE it is built,
        because building a model is already random - the decoder's
        weights are drawn, the mixture's components are placed, and
        the representations are initialized - and seeding a model that
        already exists seeds none of that. What each member was seeded
        with is written beside it.

        Parameters
        ----------
        df_samples : :class:`pandas.DataFrame`
            The samples to train on.

        names_train : :class:`list`
            The names of the samples to train on.

        names_test : :class:`list`
            The names of the samples to test on.

        config_train : :class:`dict`
            The configuration for the training. It is the one a single
            :class:`bulkdgd.core.model.BulkDGD` takes, and it is
            shared: the members differ only in their seed.

        config_ensemble : :class:`dict`, optional
            The configuration to be used. If not passed, the
            ensemble's own is used.

        gmm_pth_file : :class:`str`, optional
            The name of the file where a member's trained latent
            space's parameters are written, inside the member's own
            directory.

        dec_pth_file : :class:`str`, optional
            The name of the file where a member's trained decoder's
            parameters are written.

        gmm_final_pth_file : :class:`str`, optional
            The name of the file where a member's final Gaussian
            mixture model's parameters are written.

        pathways : :class:`pandas.DataFrame`, optional
            The pathways, if the saliency maps are to be computed.

        labels_train : optional
            The labels of the samples to train on.

        labels_test : optional
            The labels of the samples to test on.

        resume : :class:`bool`, ``True``
            Whether to skip the members that are already trained.

        return_data : :class:`bool`, ``False``
            Whether to keep what the training returned for each
            member. Everything is written to disk either way.

        Returns
        -------
        results : :class:`dict`
            What happened to each member - whether it ran, was
            skipped, or failed, what it wrote, and how long it took.
        """

        # Get the configuration to be used.
        config_ensemble = \
            self._get_config_ensemble(
                config_ensemble = config_ensemble)

        # Get the names of the files the parameters are written to.
        gmm_pth_file = gmm_pth_file or self.GMM_PTH_FILE
        dec_pth_file = dec_pth_file or self.DEC_PTH_FILE
        gmm_final_pth_file = \
            gmm_final_pth_file or self.GMM_FINAL_PTH_FILE

        #-------------------------------------------------------------#

        # Define what it means for a member to be already trained.
        def get_outputs_done(name, options):

            # Get where the member's parameters would be.
            paths = \
                {"gmm" : os.path.join(options["model_dir"],
                                      gmm_pth_file),
                 "decoder" : os.path.join(options["model_dir"],
                                          dec_pth_file)}

            # The member is done if they are all there.
            if all(os.path.isfile(path) for path in paths.values()):
                return paths

            # Otherwise, it is not.
            return None

        #-------------------------------------------------------------#

        # Define how a member is trained.
        def run_member(name, options):

            # Make the directory the member lives in.
            os.makedirs(options["model_dir"], exist_ok = True)

            # Seed everything with the member's seed, before the model
            # is built.
            seeds = \
                set_seeds(
                    seed = options["seed"],
                    deterministic = \
                        config_train.get("deterministic", False))

            # Write down what the member was seeded with.
            seeds_file = \
                self._write_seeds(model_dir = options["model_dir"],
                                  seeds = seeds)

            #---------------------------------------------------------#

            # Build the member, untrained.
            model = self._get_model_untrained()

            # Train it.
            data = \
                model.train(
                    df_samples = df_samples,
                    names_train = names_train,
                    names_test = names_test,
                    config_train = config_train,
                    gmm_pth_file = \
                        os.path.join(options["model_dir"],
                                     gmm_pth_file),
                    dec_pth_file = \
                        os.path.join(options["model_dir"],
                                     dec_pth_file),
                    gmm_final_pth_file = \
                        os.path.join(options["model_dir"],
                                     gmm_final_pth_file),
                    pathways = pathways,
                    labels_train = labels_train,
                    labels_test = labels_test)

            #---------------------------------------------------------#

            # Write down what the training produced.
            outputs = \
                self._write_train_outputs(
                    model_dir = options["model_dir"],
                    data = data)

            # Record the parameters and the seeds among the outputs.
            outputs["gmm"] = os.path.join(options["model_dir"],
                                          gmm_pth_file)

            outputs["decoder"] = os.path.join(options["model_dir"],
                                              dec_pth_file)

            outputs["seeds"] = seeds_file

            #---------------------------------------------------------#

            # Return the outputs and the data.
            return outputs, data

        #-------------------------------------------------------------#

        # Train the members.
        return self._run_members(
                    config_ensemble = config_ensemble,
                    stage = "train",
                    get_outputs_done = get_outputs_done,
                    run_member = run_member,
                    resume = resume,
                    return_data = return_data)


    def _write_train_outputs(self,
                             model_dir: str,
                             data: tuple) -> dict[str, str]:
        """Write down what a member's training produced.

        Parameters
        ----------
        model_dir : :class:`str`
            The directory where the member's parameters live.

        data : :class:`tuple`
            What the training returned.

        Returns
        -------
        outputs : :class:`dict`
            The files that were written.
        """

        # Unpack what the training returned.
        dfs_rep, dfs_pred_means, dfs_pred_r_values, df_loss, \
            dfs_metrics, df_time = data

        # Initialize an empty dictionary to store the files written.
        outputs = {}

        #-------------------------------------------------------------#

        # Define how one data frame is written.
        def write(df, name):

            # Get the path to the file.
            path = os.path.join(model_dir, name)

            # Write the data frame.
            save_table(df, path, sep = ",", index = True)

            # Record the file.
            outputs[name.rsplit(".", 1)[0]] = path

        #-------------------------------------------------------------#

        # Write the losses and the training times.
        write(df_loss, self.LOSS_FILE)
        write(df_time, self.TIME_FILE)

        #-------------------------------------------------------------#

        # Write the representations and the predicted means, which
        # come as one data frame for the training samples and one for
        # the test samples.
        for dfs, stem in ((dfs_rep, "representations"),
                          (dfs_pred_means, "pred_means")):

            write(dfs[0], f"{stem}_train.parquet")
            write(dfs[1], f"{stem}_test.parquet")

        #-------------------------------------------------------------#

        # Write the predicted r-values, which are missing for a
        # Poisson output module, one data frame for a per-gene
        # dispersion, and one per split for a full dispersion.
        if dfs_pred_r_values is not None:

            if isinstance(dfs_pred_r_values, tuple):

                write(dfs_pred_r_values[0], "pred_r_values_train.csv")
                write(dfs_pred_r_values[1], "pred_r_values_test.csv")

            else:

                write(dfs_pred_r_values, self.PRED_R_VALUES_FILE)

        #-------------------------------------------------------------#

        # Write the metrics, if any were computed.
        if dfs_metrics is not None:

            write(dfs_metrics[0], "metrics_train.csv")
            write(dfs_metrics[1], "metrics_test.csv")

        #-------------------------------------------------------------#

        # Return the files that were written.
        return outputs


    def find_representations(
            self,
            df_samples: pd.DataFrame,
            config_rep: dict[str, object],
            config_ensemble: dict[str, dict[str, object]] = None,
            get_saliency_map: bool = False,
            genes_mask: torch.Tensor = None,
            resume: bool = True,
            return_data: bool = False) -> dict[str, dict]:
        """Find the representations of a set of samples with every
        member of the ensemble.

        Parameters
        ----------
        df_samples : :class:`pandas.DataFrame`
            The samples to find the representations for.

        config_rep : :class:`dict`
            The configuration for the search. It is the one a single
            :class:`bulkdgd.core.model.BulkDGD` takes.

        config_ensemble : :class:`dict`, optional
            The configuration to be used. If not passed, the
            ensemble's own is used. Pass a different one to run a
            second cohort through the same members without their
            results landing on top of the first cohort's.

        get_saliency_map : :class:`bool`, ``False``
            Whether to compute the saliency maps.

        genes_mask : :class:`torch.Tensor`, optional
            The mask for the genes to be considered.

        resume : :class:`bool`, ``True``
            Whether to skip the members that already have the
            representations.

        return_data : :class:`bool`, ``False``
            Whether to keep what the search returned for each member.
            Everything is written to disk either way.

        Returns
        -------
        results : :class:`dict`
            What happened to each member.
        """

        # Get the configuration to be used.
        config_ensemble = \
            self._get_config_ensemble(
                config_ensemble = config_ensemble)

        #-------------------------------------------------------------#

        # Define what it means for a member to already have the
        # representations.
        def get_outputs_done(name, options):

            # Get where they would be.
            paths = \
                {"representations" : \
                    os.path.join(options["results_dir"],
                                 self.REP_FILE),
                 "pred_means" : \
                    os.path.join(options["results_dir"],
                                 self.PRED_MEANS_FILE)}

            # The member is done if they are all there.
            if all(os.path.isfile(path) for path in paths.values()):
                return paths

            # Otherwise, it is not.
            return None

        #-------------------------------------------------------------#

        # Define how a member's representations are found.
        def run_member(name, options):

            # Make the directory the results live in.
            os.makedirs(options["results_dir"], exist_ok = True)

            # Get the member, with its trained parameters.
            model = self.get_model(name = name,
                                   config_ensemble = config_ensemble)

            #---------------------------------------------------------#

            # Find the representations.
            df_rep, df_pred_means, df_pred_r_values, df_time = \
                model.get_representations(
                    df_samples = df_samples,
                    config_rep = config_rep,
                    get_saliency_map = get_saliency_map,
                    genes_mask = genes_mask)

            #---------------------------------------------------------#

            # Initialize an empty dictionary to store the files.
            outputs = {}

            # Write the representations, the predicted means, and the
            # times.
            for df, name_file, key in (
                    (df_rep, self.REP_FILE, "representations"),
                    (df_pred_means, self.PRED_MEANS_FILE,
                     "pred_means"),
                    (df_time, self.TIME_FILE, "time")):

                path = os.path.join(options["results_dir"], name_file)

                save_table(df, path, sep = ",", index = True)

                outputs[key] = path

            #---------------------------------------------------------#

            # Write the predicted r-values, if the output module has
            # any.
            if df_pred_r_values is not None:

                path = os.path.join(options["results_dir"],
                                    self.PRED_R_VALUES_FILE)

                save_table(df_pred_r_values, path, sep = ",", index = True)

                outputs["pred_r_values"] = path

            #---------------------------------------------------------#

            # Return the outputs and the data.
            return outputs, (df_rep, df_pred_means, df_pred_r_values,
                             df_time)

        #-------------------------------------------------------------#

        # Find the representations with each member.
        return self._run_members(
                    config_ensemble = config_ensemble,
                    stage = "find_representations",
                    get_outputs_done = get_outputs_done,
                    run_member = run_member,
                    resume = resume,
                    return_data = return_data)


    def dea(self,
            df_samples: pd.DataFrame,
            config_dea: dict[str, object],
            config_ensemble: dict[str, dict[str, object]] = None,
            resume: bool = True,
            return_data: bool = False) -> dict[str, dict]:
        """Perform differential expression analysis with every member
        of the ensemble.

        The results are written one file per sample, which is what
        makes the analysis resumable where it matters: a run that dies
        halfway picks up at the first sample that has no file, and not
        at the first member.

        Parameters
        ----------
        df_samples : :class:`pandas.DataFrame`
            The samples' observed counts.

        config_dea : :class:`dict`
            The configuration for the analysis. These keys belong to
            the ensemble:

            * ``"dea_dir"`` (:class:`str`, optional) - where a
              member's results are written. If it is a relative path,
              it is taken relative to the member's results'
              directory. It defaults to ``"dea"``.

            * ``"zip_results"`` (:class:`bool`, optional) - whether to
              pack a member's results into one archive once they are
              all there, and remove the loose files. It defaults to
              :obj:`False`, and is worth turning on for an ensemble of
              any size: one file per sample per model is tens of
              thousands of small files. The archive is verified before
              anything is removed, and everything that reads these
              results afterwards reads it either way.

            * ``"pred_means_file"``, ``"pred_r_values_file"``
              (:class:`str`, optional) - the files the predicted means
              and r-values are read from, inside a member's results'
              directory.

            Every other key is passed on to
            :func:`bulkdgd.analysis.dea.get_statistics`.

        config_ensemble : :class:`dict`, optional
            The configuration to be used. If not passed, the
            ensemble's own is used.

        resume : :class:`bool`, ``True``
            Whether to skip the samples that already have results.

        return_data : :class:`bool`, ``False``
            Whether to keep each sample's statistics. They are written
            to disk either way, and an ensemble's worth of them is a
            great deal of memory.

        Returns
        -------
        results : :class:`dict`
            What happened to each member.
        """

        # Get the configuration to be used.
        config_ensemble = \
            self._get_config_ensemble(
                config_ensemble = config_ensemble)

        #-------------------------------------------------------------#

        # Take the options that belong to the ensemble out of the
        # configuration - what is left is what the statistics take.
        config_dea = copy.deepcopy(config_dea)

        dea_dir = config_dea.pop("dea_dir", "dea")

        zip_results = config_dea.pop("zip_results", False)

        prefix = config_dea.pop("prefix", deaio.DEA_PREFIX)

        pred_means_file = \
            config_dea.pop("pred_means_file", self.PRED_MEANS_FILE)

        pred_r_values_file = \
            config_dea.pop("pred_r_values_file",
                           self.PRED_R_VALUES_FILE)

        # The statistics are computed against a scaled predicted mean,
        # and which scaling is a property of the model - a model
        # trained on the median of a sample's counts and analyzed as
        # if it were trained on the mean gives fold changes that are
        # wrong rather than absent. The statistics cannot ask the
        # model, but the ensemble has its configuration, so it is
        # taken from there unless the caller says otherwise.
        config_dea.setdefault(
            "scaling_factor",
            self._config_model.get("scaling_factor", "mean"))

        #-------------------------------------------------------------#

        # Define how a member's differential expression is computed.
        # There is no check for a member being done as a whole: the
        # samples are checked one by one inside, which is finer.
        def run_member(name, options):

            # Get where the member's results go, and make it.
            dea_dir_member = \
                self._get_dea_dir(options = options,
                                  dea_dir = dea_dir)

            os.makedirs(dea_dir_member, exist_ok = True)

            #---------------------------------------------------------#

            # Get the member's predicted means.
            df_pred_means = \
                pd.read_csv(os.path.join(options["results_dir"],
                                         pred_means_file),
                            index_col = 0)

            # Get the member's predicted r-values, if it has any.
            path_r_values = \
                os.path.join(options["results_dir"],
                             pred_r_values_file)

            df_pred_r_values = \
                pd.read_csv(path_r_values, index_col = 0) \
                if os.path.isfile(path_r_values) else None

            # The genes are the ones the model predicts for.
            genes = list(df_pred_means.columns)

            # Get the genes the samples have no counts for. They are
            # checked here because the alternative is a sample-by-
            # sample lookup failing with a list of fourteen thousand
            # gene names and no hint of why - and the usual why is
            # that the counts still carry the genes' versions
            # ('ENSG00000000003.15') while the model's genes do not.
            genes_missing = \
                [gene for gene in genes
                 if gene not in df_samples.columns]

            # If the samples are missing any gene
            if genes_missing:

                # Raise an error saying so.
                raise KeyError(
                    f"The samples have no counts for "
                    f"{len(genes_missing)} of the "
                    f"{len(genes)} genes the model predicts for "
                    f"(the first are: "
                    f"{', '.join(genes_missing[:3])}). Check that "
                    "the counts are the ones the model was run on, "
                    "and that their genes are named the same way.")

            #---------------------------------------------------------#

            # Initialize the statistics kept, and the number of
            # samples written.
            dfs_stats = {}

            n_written = 0

            n_skipped = 0

            #---------------------------------------------------------#

            # For each sample
            for sample in df_pred_means.index:

                # If the sample already has results, leave it alone.
                if resume \
                and deaio.has_sample(dea_dir_member,
                                     sample,
                                     prefix = prefix):

                    n_skipped += 1

                    continue

                #-----------------------------------------------------#

                # Get the sample's observed counts and predicted
                # means.
                obs_counts = df_samples.loc[sample, genes]

                pred_means = df_pred_means.loc[sample, genes]

                # Get the sample's predicted r-values. They are one
                # row per sample for a full dispersion, and a single
                # row for a per-gene one.
                r_values = \
                    self._get_r_values(
                        df_pred_r_values = df_pred_r_values,
                        sample = sample,
                        genes = genes)

                #-----------------------------------------------------#

                # Compute the statistics.
                df_stats, _ = \
                    analysis_dea.get_statistics(
                        obs_counts = obs_counts,
                        pred_means = pred_means,
                        r_values = r_values,
                        sample_name = sample,
                        **config_dea)

                # Keep what the model predicted beside the statistics
                # drawn from it, so that a sample's file says what it
                # was computed from.
                df_stats["dgd_mean"] = pred_means

                if r_values is not None:
                    df_stats["dgd_r"] = r_values

                #-----------------------------------------------------#

                # Write the sample's statistics.
                save_table(df_stats, 
                    os.path.join(dea_dir_member,
                                 f"{prefix}{sample}.parquet"),
                    sep = ",",
                    index = True)

                n_written += 1

                # Keep them, if they were asked for.
                if return_data:
                    dfs_stats[sample] = df_stats

            #---------------------------------------------------------#

            # Inform the user about what was done.
            logger.info(
                f"[dea] '{name}': {n_written} samples were analyzed, "
                f"and {n_skipped} already had results.")

            #---------------------------------------------------------#

            # Start the outputs with where the results are.
            outputs = {"dea_dir" : dea_dir_member,
                       "n_samples_written" : n_written,
                       "n_samples_skipped" : n_skipped}

            # Pack the results, if they are to be packed.
            if zip_results:

                outputs["dea_zip"] = \
                    self._zip_dea(dea_dir = dea_dir_member,
                                  prefix = prefix)

            #---------------------------------------------------------#

            # Return the outputs and the data.
            return outputs, dfs_stats

        #-------------------------------------------------------------#

        # Analyze the samples with each member.
        return self._run_members(
                    config_ensemble = config_ensemble,
                    stage = "dea",
                    get_outputs_done = lambda name, options: None,
                    run_member = run_member,
                    resume = resume,
                    return_data = return_data)


    def _get_r_values(self,
                      df_pred_r_values: pd.DataFrame,
                      sample: str,
                      genes: list) -> pd.Series:
        """Get a sample's predicted r-values.

        Parameters
        ----------
        df_pred_r_values : :class:`pandas.DataFrame` or :obj:`None`
            The predicted r-values.

        sample : :class:`str`
            The sample's name.

        genes : :class:`list`
            The genes.

        Returns
        -------
        r_values : :class:`pandas.Series` or :obj:`None`
            The sample's predicted r-values.
        """

        # If the output module has no r-values.
        if df_pred_r_values is None:

            # There are none.
            return None

        #-------------------------------------------------------------#

        # If there is one row per sample - a full dispersion.
        if sample in df_pred_r_values.index:

            return df_pred_r_values.loc[sample, genes]

        #-------------------------------------------------------------#

        # If there is a single row for every sample - a per-gene
        # dispersion.
        if len(df_pred_r_values) == 1:

            return df_pred_r_values.iloc[0][genes]

        #-------------------------------------------------------------#

        # Otherwise, the r-values do not cover the sample.
        raise KeyError(
            f"No predicted r-values were found for the sample "
            f"'{sample}'.")


    def _zip_dea(self,
                 dea_dir: str,
                 prefix: str = None) -> str:
        """Pack a member's differential expression into one archive,
        and remove the loose files.

        The archive is written, closed, and read back before anything
        is removed: an archive that was interrupted while it was being
        written is worse than the files it was meant to replace, since
        it also stops everything downstream from reading them.

        Parameters
        ----------
        dea_dir : :class:`str`
            The directory containing the results.

        prefix : :class:`str`, optional
            The prefix the per-sample files are named with.

        Returns
        -------
        zip_path : :class:`str`
            The archive.
        """

        # Get the prefix.
        prefix = prefix or deaio.DEA_PREFIX

        # Get where the archive goes, and where it is built.
        zip_path = os.path.join(dea_dir, deaio.DEA_ZIP_NAME)

        zip_path_partial = f"{zip_path}.partial"

        #-------------------------------------------------------------#

        # If the results are already packed
        if os.path.isfile(zip_path):

            # Leave them alone.
            logger.info(
                f"The results in '{dea_dir}' are already packed.")

            return zip_path

        #-------------------------------------------------------------#

        # Get the files to be packed.
        names = sorted(name for name in os.listdir(dea_dir)
                       if name.startswith(prefix)
                       and name.endswith(".csv"))

        # If there is nothing to pack
        if not names:

            # Raise an error, rather than leaving an empty archive
            # where the results should be.
            raise FileNotFoundError(
                f"There are no results to pack in '{dea_dir}'.")

        #-------------------------------------------------------------#

        # Build the archive under a name of its own, so that a run
        # that dies while writing it does not leave something that
        # looks like a finished archive.
        with zipfile.ZipFile(zip_path_partial,
                             "w",
                             compression = zipfile.ZIP_DEFLATED) \
                as archive:

            for name in names:

                archive.write(os.path.join(dea_dir, name),
                              arcname = name)

        #-------------------------------------------------------------#

        # Read the archive back, and check that every file made it in.
        with zipfile.ZipFile(zip_path_partial) as archive:

            if archive.testzip() is not None:

                raise RuntimeError(
                    f"The archive built for '{dea_dir}' is corrupted. "
                    "The loose files were left alone.")

            names_packed = set(archive.namelist())

        # Get the files that did not make it in.
        names_missing = [name for name in names
                         if name not in names_packed]

        # If any file did not make it in
        if names_missing:

            # Raise an error, leaving the loose files alone.
            raise RuntimeError(
                f"{len(names_missing)} of the {len(names)} files in "
                f"'{dea_dir}' are missing from the archive built for "
                "it. The loose files were left alone.")

        #-------------------------------------------------------------#

        # Put the archive in place now that it is known to be good.
        os.replace(zip_path_partial, zip_path)

        # Remove the loose files.
        for name in names:

            os.remove(os.path.join(dea_dir, name))

        #-------------------------------------------------------------#

        # Inform the user about what was packed.
        logger.info(
            f"{len(names)} files in '{dea_dir}' were packed into "
            f"'{deaio.DEA_ZIP_NAME}' and removed.")

        #-------------------------------------------------------------#

        # Return the archive.
        return zip_path


    def gsea(self,
             genes_sets: dict[str, list],
             config_gsea: dict[str, object],
             config_ensemble: dict[str, dict[str, object]] = None,
             resume: bool = True,
             return_data: bool = False) -> dict[str, dict]:
        """Compute the enrichment scores of the genes every member of
        the ensemble calls.

        The results of the differential expression analysis are read
        whether they are still loose on disk or already packed.

        Parameters
        ----------
        genes_sets : :class:`dict`
            The sets of genes of interest.

        config_gsea : :class:`dict`
            The configuration for the analysis. These keys belong to
            the ensemble:

            * ``"dea_dir"`` (:class:`str`, optional) - where a
              member's differential expression is read from. It
              defaults to ``"dea"``.

            * ``"gsea_dir"`` (:class:`str`, optional) - where a
              member's enrichment scores are written. It defaults to
              ``"gsea"``.

            * ``"genes_all"`` (:class:`list`, optional) - the genes
              the analysis was run on. It defaults to the genes the
              differential expression covers.

            Every other key is passed on to
            :func:`bulkdgd.analysis.dea.get_significant_genes`.

        config_ensemble : :class:`dict`, optional
            The configuration to be used. If not passed, the
            ensemble's own is used.

        resume : :class:`bool`, ``True``
            Whether to skip the members that already have the
            enrichment scores.

        return_data : :class:`bool`, ``False``
            Whether to keep each member's enrichment scores. They are
            written to disk either way.

        Returns
        -------
        results : :class:`dict`
            What happened to each member.
        """

        # Get the configuration to be used.
        config_ensemble = \
            self._get_config_ensemble(
                config_ensemble = config_ensemble)

        #-------------------------------------------------------------#

        # Take the options that belong to the ensemble out of the
        # configuration - what is left is what the significant genes
        # take.
        config_gsea = copy.deepcopy(config_gsea)

        dea_dir = config_gsea.pop("dea_dir", "dea")

        gsea_dir = config_gsea.pop("gsea_dir", "gsea")

        genes_all = config_gsea.pop("genes_all", None)

        prefix = config_gsea.pop("prefix", deaio.DEA_PREFIX)

        #-------------------------------------------------------------#

        # Define what it means for a member to already have the
        # enrichment scores.
        def get_outputs_done(name, options):

            # Get where they would be.
            path = os.path.join(options["results_dir"],
                                gsea_dir,
                                self.E_SCORES_FILE)

            # The member is done if they are there.
            if os.path.isfile(path):
                return {"e_scores" : path}

            # Otherwise, it is not.
            return None

        #-------------------------------------------------------------#

        # Define how a member's enrichment scores are computed.
        def run_member(name, options):

            # Get where the member's differential expression is.
            dea_dir_member = \
                self._get_dea_dir(options = options,
                                  dea_dir = dea_dir)

            # Get where the enrichment scores go, and make it.
            gsea_dir_member = \
                os.path.join(options["results_dir"], gsea_dir) \
                if not os.path.isabs(gsea_dir) else gsea_dir

            os.makedirs(gsea_dir_member, exist_ok = True)

            #---------------------------------------------------------#

            # Get the samples the member has results for.
            samples = \
                deaio.list_samples(dea_dir_member, prefix = prefix)

            # If it has none
            if not samples:

                # Raise an error.
                raise FileNotFoundError(
                    "There is no differential expression to draw the "
                    f"enrichment scores from in '{dea_dir_member}'.")

            #---------------------------------------------------------#

            # Initialize an empty list to store each sample's scores.
            dfs_e_scores = []

            # Get the genes the analysis was run on, if they were not
            # given.
            genes_all_member = genes_all

            #---------------------------------------------------------#

            # For each sample
            for sample in samples:

                # Get the sample's statistics.
                df_stats = \
                    deaio.read_dea(dea_dir_member,
                                   sample,
                                   prefix = prefix,
                                   index_col = 0)

                # The genes the analysis was run on are the ones it
                # has statistics for.
                if genes_all_member is None:
                    genes_all_member = df_stats.index.tolist()

                #-----------------------------------------------------#

                # Get the genes the member calls for the sample.
                df_significant_genes = \
                    analysis_dea.get_significant_genes(
                        df_stats = df_stats,
                        **config_gsea)

                # Compute their enrichment scores.
                df_e_scores = \
                    analysis_dea.get_enrichment_scores(
                        df_significant_genes = df_significant_genes,
                        genes_sets = genes_sets,
                        genes_all = genes_all_member)

                # Say which sample they are for - they are all written
                # to one file.
                df_e_scores.insert(0, "sample", sample)

                dfs_e_scores.append(df_e_scores)

            #---------------------------------------------------------#

            # Put every sample's scores together.
            df_e_scores_all = \
                pd.concat(dfs_e_scores, ignore_index = True)

            # Write them.
            path = os.path.join(gsea_dir_member, self.E_SCORES_FILE)

            save_table(df_e_scores_all, path, sep = ",", index = False)

            #---------------------------------------------------------#

            # Inform the user about what was done.
            logger.info(
                f"[gsea] '{name}': the enrichment scores of "
                f"{len(samples)} samples over {len(genes_sets)} sets "
                "of genes were computed.")

            #---------------------------------------------------------#

            # Return the outputs and the data.
            return {"e_scores" : path}, df_e_scores_all

        #-------------------------------------------------------------#

        # Compute the enrichment scores with each member.
        return self._run_members(
                    config_ensemble = config_ensemble,
                    stage = "gsea",
                    get_outputs_done = get_outputs_done,
                    run_member = run_member,
                    resume = resume,
                    return_data = return_data)


    def consensus(
            self,
            df_metadata: pd.DataFrame,
            config_consensus: dict[str, object],
            config_ensemble: dict[str, dict[str, object]] = \
                None) -> dict[str, pd.DataFrame]:
        """Get the tiered list of genes the ensemble agrees on.

        A model calls a gene for a sample when the gene's q-value and
        log2-fold change pass the given thresholds. A model CONSIDERS
        a gene for a group of samples when it calls it in at least a
        given share of that group's samples. A gene's TIER is how many
        of the ensemble's models consider it - the number of models
        that agree on it - and the consensus recurrence reported for a
        gene of tier K is the K-th largest of its per-model
        recurrences, which is the level at which K models agree.

        Samples that any one model has no results for are dropped for
        the whole ensemble, so that every gene was offered the same
        models and the tiers are comparable.

        Parameters
        ----------
        df_metadata : :class:`pandas.DataFrame`
            The samples' metadata, indexed by the samples' names.

            It must contain the column the samples are grouped by, and
            the column they are filtered on, if any.

        config_consensus : :class:`dict`
            The configuration for the consensus. The keys are:

            * ``"group_column"`` (:class:`str`) - the metadata column
              the samples are grouped by, and within which a gene's
              recurrence is computed. It is the cancer type for a
              cohort of tumours, but it is whatever condition the
              samples are grouped by for any other cohort.

            * ``"group_name"`` (:class:`str`, optional) - the name the
              group is given in the output. It defaults to the name of
              the column the samples are grouped by.

            * ``"filter_column"`` (:class:`str`, optional) - a
              metadata column the samples are filtered on before
              anything else. For a cohort of tumours, this is how the
              primary tumours are kept and the metastatic ones left
              out.

            * ``"filter_value"`` (optional) - the value the samples
              must have in ``"filter_column"`` to be kept.

            * ``"dea_dir"`` (:class:`str`, optional) - the directory
              containing a member's differential expression analysis'
              results. If it is a relative path, it is taken relative
              to the member's results' directory. It defaults to
              ``"dea"``.

            * ``"q_val"`` (:class:`float`, optional) - the q-value
              below which a gene is called for a sample. It defaults
              to ``0.05``.

            * ``"log2_fold_change"`` (:class:`float`, optional) - the
              absolute log2-fold change above which a gene is called
              for a sample. It defaults to ``1.0``.

            * ``"recurrence"`` (:class:`float`, optional) - the share
              of a group's samples a model must call a gene in for the
              model to consider it. It defaults to ``0.20``.

            * ``"min_tier"`` (:class:`int`, optional) - the tier below
              which a gene is left out of the output. It defaults to
              ``2``, since a gene only one model considers is not
              something the ensemble agrees on.

            * ``"genes_symbols"`` (:class:`dict`, optional) - the
              genes' symbols, mapped from the genes' names.

            * ``"n_processes"`` (:class:`int`, optional) - how many
              processes to score the samples with. It defaults to the
              number of available CPUs.

        config_ensemble : :class:`dict`, optional
            The configuration to be used. If not passed, the
            ensemble's own is used.

        Returns
        -------
        dfs_consensus : :class:`dict`
            A dictionary mapping each group of samples to a data frame
            with one row per gene, reporting the gene's tier, its
            consensus recurrence, and every per-model recurrence.
        """

        # Get the configuration to be used.
        config_ensemble = \
            self._get_config_ensemble(
                config_ensemble = config_ensemble)

        #-------------------------------------------------------------#

        # Get the column the samples are grouped by.
        group_column = config_consensus["group_column"]

        # Get the name the group is given in the output.
        group_name = \
            config_consensus.get("group_name", group_column)

        # Get the column the samples are filtered on, and the value
        # they must have in it.
        filter_column = config_consensus.get("filter_column")
        filter_value = config_consensus.get("filter_value")

        # Get the directory containing the results.
        dea_dir = config_consensus.get("dea_dir", "dea")

        # Get the thresholds a gene must pass to be called.
        q_val = \
            config_consensus.get("q_val", self.DEFAULT_Q_VAL)

        log2_fold_change = \
            config_consensus.get("log2_fold_change",
                                 self.DEFAULT_LOG2_FOLD_CHANGE)

        # Get the share of a group's samples a model must call a gene
        # in to consider it, and the tier below which a gene is left
        # out.
        recurrence = \
            config_consensus.get("recurrence",
                                 self.DEFAULT_RECURRENCE)

        min_tier = \
            config_consensus.get("min_tier", self.DEFAULT_MIN_TIER)

        # Get the genes' symbols.
        genes_symbols = config_consensus.get("genes_symbols") or {}

        # Get how many processes to score the samples with.
        n_processes = \
            config_consensus.get("n_processes") or os.cpu_count()

        #-------------------------------------------------------------#

        # Get the directories containing the members' results, in the
        # order the members are given in - the per-model recurrences
        # are reported in that order.
        dea_dirs = \
            [self._get_dea_dir(options = config_ensemble[name],
                               dea_dir = dea_dir)
             for name in self.names]

        # Get the members whose results are missing entirely. A member
        # with no results would drop every sample, and the consensus
        # would silently come back empty.
        dirs_missing = \
            [f"'{name}' ({dea_dir_member})"
             for name, dea_dir_member in zip(self.names, dea_dirs)
             if not deaio.list_samples(dea_dir_member)]

        # If any member's results are missing
        if dirs_missing:

            # Raise an error.
            raise FileNotFoundError(
                "These members have no differential expression "
                f"analysis' results: {'; '.join(dirs_missing)}. A "
                "gene's tier is how many of the ensemble's models "
                "agree on it, so it can only be computed with every "
                "member's results.")

        #-------------------------------------------------------------#

        # Get the samples the first member has results for. Any sample
        # the others are missing is dropped as it is scored.
        samples = deaio.list_samples(dea_dirs[0])

        # If the samples are to be filtered
        if filter_column is not None:

            # Get the samples that pass the filter.
            samples_kept = \
                set(df_metadata.index[
                        df_metadata[filter_column].astype(str) \
                            == str(filter_value)])

            # Keep only those.
            samples = [sample for sample in samples
                       if sample in samples_kept]

        #-------------------------------------------------------------#

        # Get the group each sample belongs to. It is taken as a
        # dictionary because a metadata table with a repeated sample
        # would otherwise give a series where a single group is
        # expected.
        groups = df_metadata[group_column].astype(str).to_dict()

        # Pair each sample with its group, leaving out the samples
        # that have no metadata.
        items_samples = \
            [(sample, groups[sample]) for sample in samples
             if sample in groups]

        # If no samples are left
        if not items_samples:

            # Raise an error.
            raise ValueError(
                "No samples are left to draw the consensus from. "
                "Check that the metadata cover the samples the "
                "members have results for, and that the filter, if "
                "any, is not excluding all of them.")

        #-------------------------------------------------------------#

        # Inform the user about what the consensus will be drawn from.
        logger.info(
            f"The consensus will be drawn from {len(items_samples)} "
            f"samples across "
            f"{len(set(group for _, group in items_samples))} groups "
            f"and {self.n_models} models.")

        #-------------------------------------------------------------#

        # Get the columns to read from the samples' statistics. They
        # are selected by position because that is much cheaper than
        # parsing the whole file, but the positions are found from the
        # file's own header rather than assumed.
        usecols = self._get_usecols(dea_dir = dea_dirs[0],
                                    sample = items_samples[0][0])

        # Build the items to be scored.
        items = \
            [(sample, group, dea_dirs, deaio.DEA_PREFIX, usecols,
              q_val, log2_fold_change)
             for sample, group in items_samples]

        #-------------------------------------------------------------#

        # Initialize, for each model, the number of a group's samples
        # it calls each gene in.
        counts = \
            [{} for _ in range(self.n_models)]

        # Initialize the number of samples scored in each group.
        n_samples = {}

        # Initialize the number of samples dropped because some model
        # had no results for them.
        n_dropped = 0

        #-------------------------------------------------------------#

        # Score the samples in parallel.
        with mp.Pool(n_processes) as pool:

            # For each sample scored
            for i, result in enumerate(
                    pool.imap_unordered(_get_genes_called,
                                        items,
                                        chunksize = 8)):

                # Every so often, report the progress.
                if i % 500 == 0:

                    logger.info(
                        f"{i}/{len(items)} samples were scored.")

                # If some model had no results for the sample
                if result is None:

                    # Count it as dropped, and move on.
                    n_dropped += 1

                    continue

                #-----------------------------------------------------#

                # Unpack the result.
                group, genes_called = result

                # Count the sample in its group.
                n_samples[group] = n_samples.get(group, 0) + 1

                # For each model's called genes
                for i_model, genes in enumerate(genes_called):

                    # Get the counts for the model and the group.
                    counts_group = \
                        counts[i_model].setdefault(group, {})

                    # For each gene the model called
                    for gene in genes:

                        # Count it.
                        counts_group[gene] = \
                            counts_group.get(gene, 0) + 1

        #-------------------------------------------------------------#

        # If any sample was dropped
        if n_dropped:

            # Inform the user, since the tiers were drawn from fewer
            # samples than the user asked for.
            logger.warning(
                f"{n_dropped} samples were dropped because at least "
                "one member had no results for them.")

        #-------------------------------------------------------------#

        # Build the consensus.
        return self._get_dfs_consensus(
                    counts = counts,
                    n_samples = n_samples,
                    recurrence = recurrence,
                    min_tier = min_tier,
                    genes_symbols = genes_symbols,
                    group_name = group_name)


    def _get_usecols(self,
                     dea_dir: str,
                     sample: str) -> list:
        """Get the positions of the columns to be read from a sample's
        statistics.

        Parameters
        ----------
        dea_dir : :class:`str`
            The directory containing the results.

        sample : :class:`str`
            The sample whose statistics are inspected.

        Returns
        -------
        usecols : :class:`list`
            The positions of the genes' names, the q-values, and the
            log2-fold changes.
        """

        # Read only the header of the sample's statistics.
        df_head = deaio.read_dea(dea_dir,
                                  sample,
                                  index_col = 0,
                                  header = 0,
                                  nrows = 0)

        # Get the columns the file has, with the genes' names first.
        columns = [""] + list(df_head.columns)

        #-------------------------------------------------------------#

        # Initialize the list to store the positions.
        usecols = [0]

        # For each column that is needed
        for column in ("q_value", "log2_fold_change"):

            # If the file does not have it
            if column not in columns:

                # Raise an error.
                raise KeyError(
                    f"The statistics found in '{dea_dir}' have no "
                    f"'{column}' column. They have: "
                    f"{', '.join(columns[1:])}.")

            # Add its position to the list.
            usecols.append(columns.index(column))

        #-------------------------------------------------------------#

        # Return the positions.
        return usecols


    def _get_dfs_consensus(self,
                           counts: list,
                           n_samples: dict,
                           recurrence: float,
                           min_tier: int,
                           genes_symbols: dict,
                           group_name: str) -> dict[str, pd.DataFrame]:
        """Turn the counts of how many of a group's samples each model
        calls each gene in into the tiered consensus.

        Parameters
        ----------
        counts : :class:`list`
            For each model, the number of a group's samples it calls
            each gene in.

        n_samples : :class:`dict`
            The number of samples scored in each group.

        recurrence : :class:`float`
            The share of a group's samples a model must call a gene in
            to consider it.

        min_tier : :class:`int`
            The tier below which a gene is left out.

        genes_symbols : :class:`dict`
            The genes' symbols, mapped from the genes' names.

        group_name : :class:`str`
            The name the group is given in the output.

        Returns
        -------
        dfs_consensus : :class:`dict`
            A dictionary mapping each group to its consensus.
        """

        # Initialize an empty dictionary to store the consensus for
        # each group.
        dfs_consensus = {}

        # For each group and the number of samples scored in it
        for group, n_samples_group in n_samples.items():

            # Get every gene any model called in the group.
            genes = set()

            for counts_model in counts:
                genes |= set(counts_model.get(group, {}).keys())

            #---------------------------------------------------------#

            # Initialize an empty list to store the group's genes.
            rows = []

            # For each gene
            for gene in genes:

                # Get the share of the group's samples each model
                # called it in, largest first.
                recurrences = \
                    sorted((counts_model.get(group, {}).get(gene, 0) \
                                / n_samples_group
                            for counts_model in counts),
                           reverse = True)

                # The gene's tier is how many models considered it.
                tier = \
                    sum(1 for r in recurrences if r >= recurrence)

                # If too few models considered it
                if tier < min_tier:

                    # Move on to the next gene.
                    continue

                #-----------------------------------------------------#

                # Add the gene to the list. The consensus recurrence
                # is the recurrence of the last model that considered
                # it - the level at which that many models agree.
                rows.append(
                    {group_name : group,
                     "gene" : gene,
                     "symbol" : genes_symbols.get(gene, gene),
                     "tier" : int(tier),
                     "consensus_recurrence" : \
                        round(recurrences[tier - 1], 4),
                     "rec_sorted" : \
                        "|".join(f"{r:.3f}" for r in recurrences)})

            #---------------------------------------------------------#

            # If no gene made it into the group's consensus
            if not rows:

                # Move on to the next group.
                continue

            # Store the group's consensus, with the genes the most
            # models agree on first.
            dfs_consensus[group] = \
                pd.DataFrame(rows).sort_values(
                    ["tier", "consensus_recurrence"],
                    ascending = [False, False]).reset_index(drop = True)

        #-------------------------------------------------------------#

        # Inform the user about the consensus that was drawn.
        logger.info(
            f"The consensus was drawn for {len(dfs_consensus)} "
            "groups "
            f"({sum(len(df) for df in dfs_consensus.values())} genes "
            "in total).")

        #-------------------------------------------------------------#

        # Return the consensus.
        return dfs_consensus
