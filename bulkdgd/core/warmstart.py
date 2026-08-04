#!/usr/bin/env python

#    warmstart.py
#
#    A data-driven starting point for the representation search.
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

"""A ridge predictor from counts to representations, used to seed the
representation search.

The search initialises one candidate in each component of the mixture
and keeps whichever wins. Those starting points are not a prior over
where the answer is - RESULTS Sec.36 found that 83.4% of winners end in
a different component from the one they started in - so the seed
instability is what that lottery looks like when it is run twice.

A cheaper starting point is available for free. Sec.38.4 measured that
a sample's representation is recoverable from its counts by RIDGE
REGRESSION at `R^2 = 0.84`: the map is very nearly linear, not because
the decoder is linear (it is not - a linear map reproduces only 54% of
it) but because the inverse problem is massively overdetermined, with
14,740 genes projecting down to 32 numbers.

**What this is NOT.** It is not an encoder, and the DGD stays
encoder-free by design: nothing here enters the generative model, the
objective, or the training. It only chooses where the search starts.

**And it does not replace the competition.** Sec.40.2 measured a ridge
start against three mixture draws and found the ridge candidate wins
only 46.5% of the time - in most samples a random draw finds a lower
optimum. Using it ALONE costs 7% of the COSMIC enrichment. So it is
used as ONE candidate among the usual ones, taking the slot of a single
mixture draw out of `n_rep_per_comp * n_components`, which leaves every
count in the scheme unchanged and can only improve the winner.

The predictor is fitted on a model's OWN training representations,
which are already on disk and are by construction the answers that
model considers correct.
"""


import logging as log

import numpy as np
import pandas as pd
import torch


# Get the module's logger.
logger = log.getLogger(__name__)


class RidgeWarmStart:

    """Predict a sample's representation from its counts.

    Fitted once from a model's training representations, saved beside
    the model, and applied to any batch of counts on the same genes.
    """

    def __init__(self,
                 weights,
                 mean,
                 scale,
                 lam,
                 genes):
        """The fitted state. Build with :meth:`fit` or
        :meth:`from_file` rather than calling this directly.

        Note that the KERNEL form of the ridge, which is what has to be
        solved when there are more genes than samples, produces dual
        coefficients that must be carried alongside the whole training
        matrix to predict with. That matrix is 9,076 x 14,740 doubles -
        a gigabyte - and would be loaded on every representation run.
        It collapses once and for all into the primal weights

            W = X_train' alpha                (n_genes, n_dim)

        which is four megabytes and turns prediction into a single
        matrix multiplication.
        """

        self.weights = weights
        self.mean = mean
        self.scale = scale
        self.lam = lam
        self.genes = list(genes)

    #-----------------------------------------------------------------#

    @staticmethod
    def _featurise(counts, mean = None, scale = None):
        """Median-scale, log1p, and standardise per gene.

        The same scaling the model itself applies, so that the
        predictor sees what the decoder is asked to reproduce.
        """

        x = torch.as_tensor(counts, dtype = torch.float64)

        scal = x.median(dim = 1, keepdim = True).values.clamp(min = 1.0)
        x = torch.log1p(x / scal)

        if mean is None:
            mean = x.mean(0)

        if scale is None:
            scale = x.std(0).clamp(min = 1.0e-6)

        return (x - mean) / scale, mean, scale

    #-----------------------------------------------------------------#

    @classmethod
    def fit(cls,
            counts,
            representations,
            genes,
            lambdas = (1.0e3, 1.0e4, 3.0e4),
            n_val = 500,
            seed = 0):
        """Fit the predictor, choosing the ridge parameter on a
        held-out split.

        The split is RANDOM. The counts files are ordered by tissue, so
        taking the last rows as validation holds out whole organs and
        asks the ridge to extrapolate to tissues it has never seen -
        which scores `R^2 = -2.3` and is not the question being asked.
        """

        X, mean, scale = cls._featurise(counts)
        Y = torch.as_tensor(representations, dtype = torch.float64)

        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(X), generator = g)

        n_val = min(n_val, len(X) // 4)
        va, tr = perm[:n_val], perm[n_val:]

        best_lam, best_r2, best_alpha = None, -np.inf, None

        for lam in lambdas:

            # Solved in SAMPLE space: there are ~14,700 genes against
            # ~9,000 samples, so the Gram matrix is the smaller one.
            G = X[tr] @ X[tr].T
            A = G + lam * torch.eye(len(tr), dtype = torch.float64)
            alpha = torch.linalg.solve(A, Y[tr])

            pred = (X[va] @ X[tr].T) @ alpha

            ss_res = ((pred - Y[va]) ** 2).sum()
            ss_tot = ((Y[va] - Y[va].mean(0)) ** 2).sum()
            r2 = float(1.0 - ss_res / ss_tot)

            logger.info(
                f"The ridge warm start scored R^2 = {r2:.4f} on the "
                f"held-out samples with lambda = {lam:g}.")

            if r2 > best_r2:
                best_lam, best_r2, best_alpha = lam, r2, alpha

        # Refit on everything with the chosen lambda, then collapse
        # the dual solution into primal weights.
        G = X @ X.T
        A = G + best_lam * torch.eye(len(X), dtype = torch.float64)
        alpha = torch.linalg.solve(A, Y)

        weights = X.T @ alpha

        logger.info(
            f"The ridge warm start was fitted with lambda = "
            f"{best_lam:g}, which explained {100*best_r2:.1f}% of the "
            f"variance of the held-out representations.")

        return cls(weights = weights, mean = mean,
                   scale = scale, lam = best_lam, genes = genes)

    #-----------------------------------------------------------------#

    def predict(self,
                counts,
                device = None):
        """The predicted representation of each row of ``counts``."""

        X, _, _ = self._featurise(counts,
                                  mean = self.mean,
                                  scale = self.scale)

        z = X @ self.weights

        return z.to(device) if device is not None else z

    #-----------------------------------------------------------------#

    def save(self, path):
        """Write the fitted state."""

        torch.save({"weights": self.weights.cpu(),
                    "mean": self.mean.cpu(),
                    "scale": self.scale.cpu(),
                    "lam": self.lam,
                    "genes": self.genes},
                   path)

        logger.info(f"The ridge warm start was saved in '{path}'.")

    @classmethod
    def from_file(cls, path):
        """Load a fitted predictor."""

        d = torch.load(path, map_location = "cpu", weights_only = False)

        return cls(weights = d["weights"], mean = d["mean"],
                   scale = d["scale"], lam = d["lam"],
                   genes = d["genes"])


#---------------------------------------------------------------------#


def fit_from_model_dir(model_dir,
                       counts_file,
                       genes,
                       output_file = None):
    """Fit a warm start from a model's own training representations.

    Parameters
    ----------
    model_dir : :class:`str`
        A trained model's directory, holding
        ``representations_train.csv``.

    counts_file : :class:`str`
        The counts the model was trained on.

    genes : :class:`list`
        The genes, in the order the model expects them.

    output_file : :class:`str`, optional
        Where to save the fitted predictor.
    """

    import os

    # WHICHEVER FORMAT IT IS IN. 'train.py' writes these as Parquet
    # now; a tree built earlier holds the CSV. An exact hit wins, then
    # Parquet, then text.
    _stem = os.path.join(model_dir, "representations_train")

    _path = next((f"{_stem}{e}" for e in (".parquet", ".pq", ".csv")
                  if os.path.isfile(f"{_stem}{e}")), None)

    if _path is None:
        raise FileNotFoundError(
            f"no 'representations_train.{{parquet,csv}}' in "
            f"'{model_dir}'.")

    reps = (pd.read_parquet(_path) if _path.endswith((".parquet", ".pq"))
            else pd.read_csv(_path, index_col = 0))

    reps = reps[[c for c in reps.columns
                 if c.startswith("latent_dim_")]]

    counts = pd.read_csv(counts_file, index_col = 0)

    common = reps.index.intersection(counts.index)

    if len(common) == 0:
        raise ValueError(
            "No sample is shared between the model's training "
            "representations and the counts file, so the warm start "
            "cannot be fitted.")

    logger.info(
        f"The ridge warm start will be fitted on {len(common)} "
        f"sample(s) shared between the model's training "
        f"representations and '{counts_file}'.")

    ws = RidgeWarmStart.fit(
        counts = counts.loc[common, genes].to_numpy(),
        representations = reps.loc[common].to_numpy(),
        genes = genes)

    if output_file is not None:
        ws.save(output_file)

    return ws
