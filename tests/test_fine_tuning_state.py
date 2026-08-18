#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    test_fine_tuning_state.py
#
#    Tests for fine-tuning state and result utilities.


#######################################################################


# Import from the standard library.
import random

# Import from third-party libraries.
import numpy as np
import pandas as pd
import torch

# Import from 'bulkdgd'.
from bulkdgd.core import finetuning


#######################################################################


def test_rng_state_round_trip():
    """Restoring a state reproduces every CPU random stream."""

    random.seed(3)
    np.random.seed(5)
    torch.manual_seed(7)

    state = finetuning.capture_rng_state()
    expected = \
        (random.random(),
         float(np.random.random()),
         float(torch.rand(1)))

    finetuning.restore_rng_state(state = state)
    observed = \
        (random.random(),
         float(np.random.random()),
         float(torch.rand(1)))

    assert observed == expected


def test_result_has_named_stable_fields(tmp_path):
    """Fine-tuning returns a named result instead of a tuple."""

    empty = pd.DataFrame()
    result = finetuning.FineTuningResult(
        model = object(),
        representations_train = empty,
        representations_test = empty,
        replay_representations = None,
        predicted_means = None,
        predicted_r_values = None,
        loss = empty,
        metrics = empty,
        output_dir = str(tmp_path),
        manifest_path = str(tmp_path / "manifest.yaml"),
        fine_tuning_scheme = "add_gmm_components",
        stop_epoch = 0)

    assert result.fine_tuning_scheme == "add_gmm_components"
    assert result.replay_representations is None


#######################################################################
