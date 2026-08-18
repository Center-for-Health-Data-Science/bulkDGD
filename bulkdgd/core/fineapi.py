#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    fineapi.py
#
#    Public and private fine-tuning dispatch methods.
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
__doc__ = "Public and private fine-tuning dispatch methods."


#######################################################################


# Import from the standard library.
from typing import Optional

# Import from third-party libraries.
import pandas as pd

# Import from 'bulkdgd'.
from . import _util


#######################################################################


class FineTuningMixin:

    """Add the fine-tuning API to :class:`core.model.BulkDGD`."""

    def fine_tune(
            self,
            df_samples: pd.DataFrame,
            names_train: list,
            names_test: list,
            fine_tuning_scheme: str = "add_gmm_components",
            config_fine_tune: Optional[dict[str, object]] = None,
            df_replay: Optional[pd.DataFrame] = None,
            replay_representations: Optional[pd.DataFrame] = None,
            output_dir: Optional[str] = None):
        """Derive a fine-tuned model without changing its parent.

        Parameters
        ----------
        df_samples : :class:`pandas.DataFrame`
            Target samples with the parent's genes in exact order.

        names_train : :class:`list`
            Target training sample IDs.

        names_test : :class:`list`
            Target test sample IDs.

        fine_tuning_scheme : :class:`str`
            The fine-tuning scheme.

        config_fine_tune : :class:`dict`, optional
            The fine-tuning configuration.  If absent, the shipped
            configuration for ``fine_tuning_scheme`` is used.

        df_replay : :class:`pandas.DataFrame`, optional
            Healthy replay counts for decoder-moving schemes.

        replay_representations : :class:`pandas.DataFrame`, optional
            Parent replay representations.

        output_dir : :class:`str`, optional
            A new directory for the derived model and its provenance.

        Returns
        -------
        result : :class:`bulkdgd.core.finetuning.FineTuningResult`
            The derived model and its named result artifacts.
        """

        # Load the shipped configuration for the requested scheme when
        # no configuration was supplied.
        if config_fine_tune is None:

            from bulkdgd.ioutil import configio

            config_fine_tune = configio.load_config_fine_tune(
                config_file = None,
                fine_tuning_scheme = fine_tuning_scheme)

        # Validate a caller-provided dictionary.
        else:
            config_fine_tune, errors, _ = \
                _util.parse_config_fine_tune(
                    config = config_fine_tune)

            if errors:
                raise ValueError(
                    "Fine-tuning configuration errors: " +
                    " ".join(errors))

        # The argument and configuration must agree explicitly.
        configured = config_fine_tune["fine_tuning_scheme"]

        if configured != fine_tuning_scheme:
            raise ValueError(
                "The fine_tuning_scheme argument and configuration "
                f"disagree: '{fine_tuning_scheme}' versus "
                f"'{configured}'.")

        if output_dir is None:
            raise ValueError(
                "Fine-tuning requires a new output_dir for the derived "
                "model and its provenance.")

        # Select one explicit private method, following the same public
        # dispatch convention as get_representations().
        if fine_tuning_scheme == "add_gmm_components":
            method = self._fine_tune_add_gmm_components
        elif fine_tuning_scheme == "dispersion_only":
            method = self._fine_tune_dispersion_only
        elif fine_tuning_scheme == "output_head":
            method = self._fine_tune_output_head
        elif fine_tuning_scheme == "low_rank_adapter":
            method = self._fine_tune_low_rank_adapter
        elif fine_tuning_scheme == "joint_replay":
            method = self._fine_tune_joint_replay
        else:
            raise ValueError(
                f"Unsupported fine-tuning scheme "
                f"'{fine_tuning_scheme}'.")

        return method(
            df_samples = df_samples,
            names_train = names_train,
            names_test = names_test,
            config = config_fine_tune,
            df_replay = df_replay,
            replay_representations = replay_representations,
            output_dir = output_dir)

    def _fine_tune_add_gmm_components(self, **kwargs):
        """Append anchored target components to the parent TGMM."""

        from . import finetuner

        kwargs.pop("df_replay")
        kwargs.pop("replay_representations")

        return finetuner.run_add_gmm_components(
            model = self, **kwargs)

    def _fine_tune_dispersion_only(self, **kwargs):
        """Dispatch dispersion-only adaptation."""

        from . import finetuner

        return finetuner.run_unimplemented("dispersion_only")

    def _fine_tune_output_head(self, **kwargs):
        """Dispatch output-head adaptation."""

        from . import finetuner

        return finetuner.run_unimplemented("output_head")

    def _fine_tune_low_rank_adapter(self, **kwargs):
        """Dispatch low-rank-adapter adaptation."""

        from . import finetuner

        return finetuner.run_unimplemented("low_rank_adapter")

    def _fine_tune_joint_replay(self, **kwargs):
        """Dispatch joint target/replay adaptation."""

        from . import finetuner

        return finetuner.run_unimplemented("joint_replay")


#######################################################################
