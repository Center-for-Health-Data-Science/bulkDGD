#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

"""Tests for the public fine-tuning dispatch API."""

import pandas as pd
import pytest

from bulkdgd.core import fineapi
from bulkdgd.core.model import BulkDGD


class _Dispatcher(fineapi.FineTuningMixin):

    """Record which explicit private scheme receives a call."""

    def _fine_tune_add_gmm_components(self, **kwargs):
        return "add_gmm_components", kwargs


def _config(scheme="add_gmm_components"):
    """Return a minimal valid fine-tuning configuration."""

    return {"fine_tuning_scheme" : scheme}


def test_bulkdgd_exposes_explicit_scheme_methods():
    """BulkDGD inherits the public and five private entry points."""

    assert issubclass(BulkDGD, fineapi.FineTuningMixin)
    assert hasattr(BulkDGD, "fine_tune")

    for scheme in (
            "add_gmm_components",
            "dispersion_only",
            "output_head",
            "low_rank_adapter",
            "joint_replay"):
        assert hasattr(BulkDGD, f"_fine_tune_{scheme}")


def test_public_api_dispatches_by_named_scheme(tmp_path):
    """The public method selects one explicit private implementation."""

    model = _Dispatcher()
    samples = pd.DataFrame({"ENSG1" : [1]}, index = ["sample"])
    scheme, arguments = model.fine_tune(
        df_samples = samples,
        names_train = ["sample"],
        names_test = [],
        config_fine_tune = _config(),
        output_dir = str(tmp_path))

    assert scheme == "add_gmm_components"
    assert arguments["config"]["fine_tuning_scheme"] == scheme


def test_public_api_rejects_scheme_mismatch(tmp_path):
    """The named scheme cannot disagree with the configuration."""

    model = _Dispatcher()

    with pytest.raises(ValueError, match="disagree"):
        model.fine_tune(
            df_samples = pd.DataFrame(),
            names_train = [],
            names_test = [],
            fine_tuning_scheme = "output_head",
            config_fine_tune = _config(),
            output_dir = str(tmp_path))


def test_public_api_requires_derived_output_directory():
    """Fine-tuning may not overwrite its parent implicitly."""

    model = _Dispatcher()

    with pytest.raises(ValueError, match="new output_dir"):
        model.fine_tune(
            df_samples = pd.DataFrame(),
            names_train = [],
            names_test = [],
            config_fine_tune = _config())
