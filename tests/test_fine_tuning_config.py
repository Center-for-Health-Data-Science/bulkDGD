#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    test_fine_tuning_config.py
#
#    Tests for the fine-tuning configuration.


#######################################################################


# Import from the standard library.
import copy

# Import from 'bulkdgd'.
from bulkdgd.core import fineconfig
from bulkdgd.core import finetuning


#######################################################################


def test_defaults_select_anchored_components():
    """The default is the least invasive fine-tuning scheme."""

    config, errors, warnings = fineconfig.parse_config(config = None)

    assert not errors
    assert warnings
    assert config["fine_tuning_scheme"] == "add_gmm_components"
    assert config["scheme_options"]["component_fit_seed"] == 37
    assert config["scheme_options"]["n_new_components"] == 4
    assert config["deterministic_algorithms"] is True


def test_every_scheme_has_strict_defaults():
    """Every public scheme resolves to its own options."""

    for scheme in finetuning.FINE_TUNING_SCHEMES:

        config, errors, _ = fineconfig.parse_config(
            config = {"fine_tuning_scheme" : scheme})

        assert not errors
        assert config["fine_tuning_scheme"] == scheme
        assert config["scheme_options"]


def test_unknown_options_are_rejected_without_mutation():
    """A misspelled safeguard cannot silently disappear."""

    original = \
        {"fine_tuning_scheme" : "output_head",
         "replay_options" :
             {"replay_ratio" : 2.0,
              "stratifiy_by" : "tissue"}}
    passed = copy.deepcopy(original)
    _, errors, _ = fineconfig.parse_config(config = passed)

    assert errors
    assert any("stratifiy_by" in error for error in errors)
    assert passed == original


def test_scheme_options_do_not_cross_schemes():
    """An option belonging to another scheme is rejected."""

    _, errors, _ = fineconfig.parse_config(
        config =
            {"fine_tuning_scheme" : "dispersion_only",
             "scheme_options" : {"rank" : 4}})

    assert any("rank" in error for error in errors)


def test_decoder_moving_schemes_require_replay():
    """A decoder-moving scheme cannot disable replay."""

    for scheme in finetuning.FINE_TUNING_SCHEMES[1:]:

        _, errors, _ = fineconfig.parse_config(
            config =
                {"fine_tuning_scheme" : scheme,
                 "replay_options" : {"replay_ratio" : 0.0}})

        assert any("requires replay" in error for error in errors)


def test_invalid_component_and_adapter_options():
    """Component counts, masses, ranks and dropout are bounded."""

    invalid = [
        {"fine_tuning_scheme" : "add_gmm_components",
         "scheme_options" : {"n_new_components" : 0}},
        {"fine_tuning_scheme" : "add_gmm_components",
         "scheme_options" : {"new_component_weight" : 1.0}},
        {"fine_tuning_scheme" : "add_gmm_components",
         "scheme_options" : {"component_fit_seed" : 1.5}},
        {"fine_tuning_scheme" : "low_rank_adapter",
         "scheme_options" : {"rank" : 0}},
        {"fine_tuning_scheme" : "low_rank_adapter",
         "scheme_options" : {"dropout" : 1.0}}]

    for config in invalid:

        _, errors, _ = fineconfig.parse_config(config = config)
        assert errors


#######################################################################
