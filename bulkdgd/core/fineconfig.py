#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    fineconfig.py
#
#    Validation of fine-tuning configurations.
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
__doc__ = "Validation of fine-tuning configurations."


#######################################################################


# Import from the standard library.
import copy
from typing import Optional

# Import from 'bulkdgd'.
from .finetuning import FINE_TUNING_SCHEMES


#######################################################################


_COMMON_DEFAULTS = \
    {"seed" : 37,
     "deterministic_algorithms" : True,
     "n_epochs" : 50,
     "data_loader_options" :
         {"batch_size" : 16, "shuffle" : True},
     "target_representation_options" : {"config" : None},
     "replay_options" :
         {"replay_ratio" : 1.0, "stratify_by" : None},
     "anchoring_options" :
         {"l2_sp" : 0.0, "functional" : 0.0},
     "checkpoint_options" :
         {"resume" : False, "every_n_epochs" : 1},
     "early_stopping_options" :
         {"patience" : 10, "min_delta" : 0.0},
     "reporting_options" : {"fixed_panels" : True}}


_SCHEME_DEFAULTS = {
    "add_gmm_components" :
        {"component_fit_seed" : 37,
         "n_new_components" : 4,
         "new_component_weight" : 0.10,
         "initialization" : "kpp",
         "max_iter" : 100,
         "tol" : 1.0e-5,
         "reg_covar" : 1.0e-6,
         "outer_iterations" : 1},
    "dispersion_only" :
        {"learning_rate" : 1.0e-4,
         "dispersion_anchor" : 1.0},
    "output_head" :
        {"submode" : "means_only",
         "learning_rate" : 1.0e-5},
    "low_rank_adapter" :
        {"rank" : 8,
         "alpha" : 8.0,
         "dropout" : 0.0,
         "learning_rate" : 1.0e-4},
    "joint_replay" :
        {"target_learning_rate" : 1.0e-3,
         "replay_learning_rate" : 1.0e-3,
         "decoder_learning_rate" : 1.0e-5}}


#######################################################################


def _is_number(value: object) -> bool:
    """Return whether a value is a non-boolean number."""

    return not isinstance(value, bool) and \
        isinstance(value, (int, float))


def _resolve_section(
        config: dict[str, object],
        section: str,
        defaults: dict[str, object],
        errors: list[str]) -> dict[str, object]:
    """Resolve a strict nested configuration section."""

    value = config.get(section, {})

    if not isinstance(value, dict):
        errors.append(f"{section}: must be a dictionary.")
        value = {}

    unknown = sorted(set(value) - set(defaults))

    if unknown:
        errors.append(
            f"{section}: unsupported option(s): " +
            ", ".join(unknown) + ".")

    resolved = copy.deepcopy(defaults)
    resolved.update(value)
    config[section] = resolved

    return resolved


def _validate_common(
        config: dict[str, object],
        errors: list[str]) -> None:
    """Validate options shared by all fine-tuning schemes."""

    if type(config["seed"]) is not int:
        errors.append("seed: must be an integer.")

    if type(config["deterministic_algorithms"]) is not bool:
        errors.append(
            "deterministic_algorithms: must be a boolean.")

    if type(config["n_epochs"]) is not int or \
            config["n_epochs"] <= 0:
        errors.append("n_epochs: must be a positive integer.")

    loader = config["data_loader_options"]

    if type(loader["batch_size"]) is not int or \
            loader["batch_size"] <= 0:
        errors.append(
            "data_loader_options.batch_size: must be a positive "
            "integer.")

    if type(loader["shuffle"]) is not bool:
        errors.append(
            "data_loader_options.shuffle: must be a boolean.")

    target = config["target_representation_options"]["config"]

    if target is not None and not isinstance(target, (dict, str)):
        errors.append(
            "target_representation_options.config: must be a "
            "dictionary, path, configuration name, or null.")

    replay = config["replay_options"]

    if not _is_number(replay["replay_ratio"]) or \
            replay["replay_ratio"] < 0:
        errors.append(
            "replay_options.replay_ratio: must be non-negative.")

    if replay["stratify_by"] is not None and \
            not isinstance(replay["stratify_by"], str):
        errors.append(
            "replay_options.stratify_by: must be a string or null.")

    for key, value in config["anchoring_options"].items():

        if not _is_number(value) or value < 0:
            errors.append(
                f"anchoring_options.{key}: must be non-negative.")

    checkpoint = config["checkpoint_options"]

    if type(checkpoint["resume"]) is not bool:
        errors.append(
            "checkpoint_options.resume: must be a boolean.")

    if type(checkpoint["every_n_epochs"]) is not int or \
            checkpoint["every_n_epochs"] <= 0:
        errors.append(
            "checkpoint_options.every_n_epochs: must be a positive "
            "integer.")

    stopping = config["early_stopping_options"]

    if type(stopping["patience"]) is not int or \
            stopping["patience"] <= 0:
        errors.append(
            "early_stopping_options.patience: must be a positive "
            "integer.")

    if not _is_number(stopping["min_delta"]) or \
            stopping["min_delta"] < 0:
        errors.append(
            "early_stopping_options.min_delta: must be non-negative.")

    if type(config["reporting_options"]["fixed_panels"]) is not bool:
        errors.append(
            "reporting_options.fixed_panels: must be a boolean.")


def _validate_scheme(
        scheme: str,
        options: dict[str, object],
        replay_ratio: float,
        errors: list[str]) -> None:
    """Validate the selected scheme's options."""

    for key, value in options.items():

        if key in {"initialization", "submode"}:
            continue

        if not _is_number(value):
            errors.append(f"scheme_options.{key}: must be numeric.")

    if scheme == "add_gmm_components":

        if type(options["component_fit_seed"]) is not int:
            errors.append(
                "scheme_options.component_fit_seed: must be an "
                "integer.")

        if type(options["n_new_components"]) is not int or \
                options["n_new_components"] <= 0:
            errors.append(
                "scheme_options.n_new_components: must be a positive "
                "integer.")

        weight = options["new_component_weight"]

        if not _is_number(weight) or not 0 < weight < 1:
            errors.append(
                "scheme_options.new_component_weight: must be between "
                "zero and one.")

        if options["initialization"] not in {"kpp", "maxdist"}:
            errors.append(
                "scheme_options.initialization: must be 'kpp' or "
                "'maxdist'.")

        for key in ("max_iter", "outer_iterations"):

            if type(options[key]) is not int or options[key] <= 0:
                errors.append(
                    f"scheme_options.{key}: must be a positive "
                    "integer.")

        for key in ("tol", "reg_covar"):

            if not _is_number(options[key]) or options[key] <= 0:
                errors.append(
                    f"scheme_options.{key}: must be positive.")

    if scheme == "output_head" and \
            options["submode"] not in \
            {"means_only", "full_output"}:
        errors.append(
            "scheme_options.submode: must be 'means_only' or "
            "'full_output'.")

    if scheme == "low_rank_adapter":

        if type(options["rank"]) is not int or options["rank"] <= 0:
            errors.append(
                "scheme_options.rank: must be a positive integer.")

        dropout = options["dropout"]

        if not _is_number(dropout) or not 0 <= dropout < 1:
            errors.append(
                "scheme_options.dropout: must be in [0, 1).")

    if scheme != "add_gmm_components" and replay_ratio <= 0:
        errors.append(
            f"replay_options.replay_ratio: '{scheme}' requires "
            "replay.")


#######################################################################


def parse_config(
        config: Optional[dict[str, object]]) -> \
            tuple[dict[str, object], list[str], list[str]]:
    """Validate a fine-tuning configuration strictly.

    Parameters
    ----------
    config : :class:`dict`, optional
        The fine-tuning configuration.

    Returns
    -------
    config : :class:`dict`
        The validated configuration with defaults filled in.

    errors : :class:`list`
        The validation errors.

    warnings : :class:`list`
        The warnings emitted when defaults are used.
    """

    errors = []
    warnings = []

    if config is None:
        config = {}

    if not isinstance(config, dict):
        return {}, ["The fine-tuning configuration must be a "
                    "dictionary."], warnings

    config = copy.deepcopy(config)
    scheme = config.get("fine_tuning_scheme",
                        "add_gmm_components")
    allowed = set(_COMMON_DEFAULTS) | \
        {"fine_tuning_scheme", "scheme_options"}
    unknown = sorted(set(config) - allowed)

    if unknown:
        errors.append(
            "Unsupported fine-tuning option(s): " +
            ", ".join(unknown) + ".")

    if scheme not in FINE_TUNING_SCHEMES:
        errors.append(
            "fine_tuning_scheme: must be one of: " +
            ", ".join(FINE_TUNING_SCHEMES) + ".")
        selected_scheme = "add_gmm_components"
    else:
        selected_scheme = scheme

    if "fine_tuning_scheme" not in config:
        warnings.append(
            "fine_tuning_scheme: the default value "
            "'add_gmm_components' will be used.")

    resolved = {"fine_tuning_scheme" : scheme}

    for key, default in _COMMON_DEFAULTS.items():

        if isinstance(default, dict):
            resolved[key] = _resolve_section(
                config = config,
                section = key,
                defaults = default,
                errors = errors)
        else:
            resolved[key] = config.get(key, default)

        if key not in config:
            warnings.append(f"{key}: its default value will be used.")

    options = config.get("scheme_options", {})

    if not isinstance(options, dict):
        errors.append("scheme_options: must be a dictionary.")
        options = {}

    defaults = _SCHEME_DEFAULTS[selected_scheme]
    unknown = sorted(set(options) - set(defaults))

    if unknown:
        errors.append(
            "scheme_options: unsupported option(s) for "
            f"'{scheme}': " + ", ".join(unknown) + ".")

    scheme_options = copy.deepcopy(defaults)
    scheme_options.update(options)
    resolved["scheme_options"] = scheme_options

    _validate_common(config = resolved, errors = errors)
    _validate_scheme(
        scheme = selected_scheme,
        options = scheme_options,
        replay_ratio = resolved["replay_options"]["replay_ratio"],
        errors = errors)

    return resolved, errors, warnings


#######################################################################
