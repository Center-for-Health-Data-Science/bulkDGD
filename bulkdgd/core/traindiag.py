#!/usr/bin/env python

#    traindiag.py
#
#    Per-sample training diagnostics: which samples drive the model.
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
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#    General Public License for more details.
#
#    You should have received a copy of the GNU General Public
#    License along with this program. If not, see
#    <http://www.gnu.org/licenses/>.

"""Per-sample training diagnostics.

The question this answers is whether some training samples drive the
model more than others - whether a handful of samples teach most of
what the decoder knows about healthy tissue, or whether the pull is
spread evenly across the cohort.

**The instrument is the gradient, not the loss.** A sample the model
fits badly sits at a high loss for two hundred epochs without
necessarily changing any parameter that matters; a sample that CHANGES
the model is one whose gradient is large and points somewhere the other
gradients do not. Logging per-sample loss would answer a different and
less interesting question.

Two quantities are recorded, both per sample and per epoch.

``grad_norm_decoder``
    The norm of the sample's own contribution to the DECODER's
    gradient. This is the influence proper: how hard this one sample
    pulls on the shared parameters.

``grad_norm_rep``
    The norm of the gradient with respect to the sample's own
    representation. Free - the representation layer already computes
    it - and a useful contrast: a sample can be pulling its own
    representation hard while asking nothing of the decoder.

**Computing the first one cheaply.** The naive route is one backward
pass per sample, which is a sixty-fourfold slowdown at batch size 64
and would make the instrumentation cost more than the training. It is
avoided by the standard per-sample gradient identity for a linear
layer: if the layer maps ``a`` to ``W a + b`` and the gradient arriving
at its output is ``delta``, then that sample's gradient with respect to
``W`` is the outer product ``delta a'``, whose Frobenius norm is

    ||delta a'||_F = ||a|| * ||delta||

so the per-sample norm needs only the two vector norms, never the
outer product. Summing the squares over layers and taking the root
gives the whole decoder's per-sample gradient norm. The cost is two
norms per layer per batch - a few percent, not a few thousand.

The activations are captured by forward hooks and the output gradients
by TENSOR hooks registered from inside them - a module backward hook
wraps the module's output in a custom autograd Function, which autograd
refuses to combine with the in-place activations this decoder uses. See
:meth:`TrainingDiagnostics._attach`. Nothing in the training loop needs
to know any of this.
"""


import logging as log
import os

import pandas as pd
import torch
import torch.nn as nn


# Get the module's logger.
logger = log.getLogger(__name__)


class TrainingDiagnostics:

    """Record per-sample influence while a model trains.

    The object is attached to a decoder once, told the index of each
    batch's samples before the backward pass, and asked to write what
    it has collected at the end of each epoch.
    """

    def __init__(self,
                 decoder,
                 output_dir,
                 n_samples,
                 checkpoint_every = 0,
                 device = "cpu"):
        """
        Parameters
        ----------
        decoder :
            The decoder whose per-sample gradients are recorded.

        output_dir : :class:`str`
            Where the per-epoch records and any checkpoints are
            written.

        n_samples : :class:`int`
            How many training samples there are. The records are dense
            arrays of this length, indexed as the data loader indexes.

        checkpoint_every : :class:`int`, optional
            Save the decoder's state every this many epochs, for a
            later TracIn pass. Zero disables it.

        device : optional
            The device the accumulators live on.
        """

        self._decoder = decoder
        self._output_dir = output_dir
        self._n_samples = n_samples
        self._checkpoint_every = checkpoint_every
        self._device = device

        # The layers whose per-sample gradients are tracked, and the
        # per-batch activations and output gradients captured for
        # them.
        self._layers = \
            [m for m in decoder.modules() if isinstance(m, nn.Linear)]

        self._acts = {}
        self._grads = {}
        self._handles = []

        # The current batch's sample indices, set by the training loop
        # before the backward pass.
        self._ixs = None

        # The per-epoch accumulators.
        self._reset_epoch()

        os.makedirs(output_dir, exist_ok = True)

        self._attach()

        logger.info(
            f"Per-sample training diagnostics are on. "
            f"{len(self._layers)} linear layer(s) are tracked, and the "
            f"records will be written in '{output_dir}'.")

    #-----------------------------------------------------------------#

    def _reset_epoch(self):
        """Empty the per-epoch accumulators."""

        self._gn_dec = torch.zeros(self._n_samples,
                                   dtype = torch.float64)
        self._gn_rep = torch.zeros(self._n_samples,
                                   dtype = torch.float64)
        self._loss = torch.zeros(self._n_samples, dtype = torch.float64)
        self._seen = torch.zeros(self._n_samples, dtype = torch.float64)

    #-----------------------------------------------------------------#

    def _attach(self):
        """Attach the hooks that capture the two halves of the
        per-sample gradient identity.

        The output gradient is taken with a TENSOR hook registered from
        inside the forward hook, not with a module backward hook.
        ``register_full_backward_hook`` wraps a module's output in a
        custom autograd Function, and this decoder modifies a linear
        layer's output in place (the activations are in-place), which
        autograd refuses to combine with a wrapped output:

            Output 0 of BackwardHookFunctionBackward is a view and is
            being modified inplace [...] This behavior is forbidden.

        A tensor hook attaches to the graph node instead of wrapping
        the module, so it neither changes the forward semantics nor
        objects to what happens to the tensor afterwards.
        """

        for i, layer in enumerate(self._layers):

            def fwd(_mod, inp, out, i = i):

                # The layer's INPUT - the 'a' of the identity.
                self._acts[i] = inp[0].detach()

                # The gradient that will arrive at the layer's OUTPUT -
                # the 'delta'. Registered here so that it is renewed
                # for every forward pass.
                if isinstance(out, torch.Tensor) and out.requires_grad:

                    out.register_hook(
                        lambda g, i = i: self._grads.__setitem__(
                            i, g.detach()))

            self._handles.append(layer.register_forward_hook(fwd))

    def detach(self):
        """Remove every hook. The object is inert afterwards."""

        for h in self._handles:
            h.remove()

        self._handles = []

    #-----------------------------------------------------------------#

    def set_batch(self, ixs):
        """Tell the diagnostics which samples the next backward pass
        belongs to."""

        self._ixs = ixs.detach().cpu() if torch.is_tensor(ixs) \
            else torch.as_tensor(ixs)

    #-----------------------------------------------------------------#

    def record_batch(self,
                     z = None,
                     per_sample_loss = None):
        """Fold the batch that has just been back-propagated into the
        epoch's accumulators.

        Call this AFTER ``loss.backward()`` and BEFORE the optimizers
        step, since the hooks' contents are overwritten by the next
        forward pass and the gradients by the next backward one.

        Parameters
        ----------
        z : :class:`torch.Tensor`, optional
            The batch's representations, whose ``.grad`` gives the
            per-sample representation gradient.

        per_sample_loss : :class:`torch.Tensor`, optional
            The per-sample loss, if the caller has it un-reduced.
        """

        if self._ixs is None:
            return

        ixs = self._ixs

        #-------------------------------------------------------------#

        # The decoder's per-sample gradient norm, from the identity
        # ||delta a'||_F = ||a|| * ||delta||, summed in quadrature over
        # the tracked layers.
        sq = None

        for i in range(len(self._layers)):

            a = self._acts.get(i)
            g = self._grads.get(i)

            if a is None or g is None:
                continue

            # Flatten anything before the feature dimension so that a
            # layer applied to a batch of vectors and one applied to a
            # batch of sequences are handled alike.
            a = a.reshape(a.shape[0], -1).double()
            g = g.reshape(g.shape[0], -1).double()

            if a.shape[0] != len(ixs) or g.shape[0] != len(ixs):
                continue

            # The bias contributes ||delta||^2 on its own.
            contrib = (a.pow(2).sum(-1) + 1.0) * g.pow(2).sum(-1)

            sq = contrib if sq is None else sq + contrib

        if sq is not None:
            self._gn_dec.index_add_(0, ixs, sq.sqrt().cpu())

        #-------------------------------------------------------------#

        # The representation's own gradient norm, which the
        # representation layer has already computed.
        if z is not None and z.grad is not None:

            gr = z.grad.detach().reshape(len(ixs), -1).double()
            self._gn_rep.index_add_(0, ixs, gr.norm(dim = -1).cpu())

        #-------------------------------------------------------------#

        # The per-sample loss, kept for contrast rather than as the
        # influence measure.
        if per_sample_loss is not None:

            pl = per_sample_loss.detach().reshape(len(ixs), -1).sum(-1)
            self._loss.index_add_(0, ixs, pl.double().cpu())

        self._seen.index_add_(0, ixs, torch.ones(len(ixs),
                                                 dtype = torch.float64))

        self._ixs = None

    #-----------------------------------------------------------------#

    def end_epoch(self,
                  epoch,
                  sample_names = None):
        """Write the epoch's record and, if it is due, a checkpoint."""

        seen = self._seen.clamp(min = 1.0)

        df = pd.DataFrame(
            {"grad_norm_decoder": (self._gn_dec / seen).numpy(),
             "grad_norm_rep": (self._gn_rep / seen).numpy(),
             "loss": (self._loss / seen).numpy(),
             "times_seen": self._seen.numpy()})

        if sample_names is not None and len(sample_names) == len(df):
            df.index = sample_names
            df.index.name = "sample"

        df.to_csv(
            os.path.join(self._output_dir,
                         f"per_sample_epoch_{epoch}.csv"))

        # The concentration of the pull, which is the number the whole
        # exercise is about: how much of the total gradient norm the
        # busiest one percent of samples holds.
        g = torch.as_tensor(df["grad_norm_decoder"].to_numpy())
        total = float(g.sum())

        if total > 0:

            top = int(max(1, round(0.01 * len(g))))
            share = float(g.topk(top).values.sum()) / total

            srt = g.sort().values
            n = len(srt)
            gini = float(
                (2.0 * torch.arange(1, n + 1) - n - 1).double()
                @ srt.double()) / (n * total)

            logger.info(
                f"Epoch {epoch} [influence]: the top 1% of "
                f"samples hold "
                f"{100*share:.2f}% of the decoder's gradient norm "
                f"(Gini {gini:.4f}). An even pull would give "
                f"{100*top/len(g):.2f}% and a Gini of 0.")

        if self._checkpoint_every \
            and epoch % self._checkpoint_every == 0:

            torch.save(
                {k: v.detach().cpu()
                 for k, v in self._decoder.state_dict().items()},
                os.path.join(self._output_dir,
                             f"decoder_epoch_{epoch}.pth"))

        self._reset_epoch()
