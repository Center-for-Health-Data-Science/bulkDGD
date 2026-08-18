#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    finetuning.py
#
#    Utilities shared by the fine-tuning schemes.
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
__doc__ = "Utilities shared by the fine-tuning schemes."


#######################################################################


# Import from the standard library.
from dataclasses import dataclass
import random
from typing import Any, Optional

# Import from third-party libraries.
import numpy as np
import pandas as pd
import torch


#######################################################################


# Set the fine-tuning schemes exposed by the public API.
FINE_TUNING_SCHEMES = \
    ("add_gmm_components",
     "dispersion_only",
     "output_head",
     "low_rank_adapter",
     "joint_replay")


#######################################################################


@dataclass
class FineTuningResult:

    """Store a derived model and its fine-tuning artifacts.

    Attributes
    ----------
    model : :class:`object`
        The derived :class:`bulkdgd.core.model.BulkDGD` object.

    representations_train : :class:`pandas.DataFrame`
        The optimized target training representations.

    representations_test : :class:`pandas.DataFrame`
        The optimized target test representations.

    replay_representations : :class:`pandas.DataFrame`, optional
        The optimized replay representations, when replay is used.

    predicted_means : :class:`pandas.DataFrame`, optional
        The predicted means for the target test samples.

    predicted_r_values : :class:`pandas.DataFrame`, optional
        The predicted dispersions for the target test samples.

    loss : :class:`pandas.DataFrame`
        The fine-tuning loss history.

    metrics : :class:`pandas.DataFrame`
        The fine-tuning and preservation metrics.

    output_dir : :class:`str`
        The directory containing the derived artifacts.

    manifest_path : :class:`str`
        The path to the provenance manifest.

    fine_tuning_scheme : :class:`str`
        The fine-tuning scheme that produced the result.

    stop_epoch : :class:`int`
        The final completed epoch or outer iteration.

    parent_representations_train : :class:`pandas.DataFrame`, optional
        Parent-inferred training coordinates used to fit the child.

    parent_representations_test : :class:`pandas.DataFrame`, optional
        Parent-inferred test coordinates retained for comparison.
    """

    model: Any
    representations_train: pd.DataFrame
    representations_test: pd.DataFrame
    replay_representations: Optional[pd.DataFrame]
    predicted_means: Optional[pd.DataFrame]
    predicted_r_values: Optional[pd.DataFrame]
    loss: pd.DataFrame
    metrics: pd.DataFrame
    output_dir: str
    manifest_path: str
    fine_tuning_scheme: str
    stop_epoch: int
    parent_representations_train: Optional[pd.DataFrame] = None
    parent_representations_test: Optional[pd.DataFrame] = None


#######################################################################


def capture_rng_state() -> dict[str, object]:
    """Capture all random-number-generator states needed to resume.

    Returns
    -------
    state : :class:`dict`
        Python, NumPy, Torch CPU, and Torch CUDA generator states.
    """

    # Capture the CPU states.
    state = \
        {"python" : random.getstate(),
         "numpy" : np.random.get_state(),
         "torch_cpu" : torch.random.get_rng_state()}

    # Capture every CUDA generator if CUDA is available.
    state["torch_cuda"] = \
        torch.cuda.get_rng_state_all() \
        if torch.cuda.is_available() else None

    # Return the complete state.
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    """Restore random-number-generator states from a checkpoint.

    Parameters
    ----------
    state : :class:`dict`
        A dictionary returned by :func:`capture_rng_state`.
    """

    # Restore the CPU generators.
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])

    # Restore CUDA only when both the checkpoint and runtime have it.
    cuda_state = state.get("torch_cuda")

    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


#######################################################################
