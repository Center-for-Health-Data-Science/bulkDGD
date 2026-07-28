#!/usr/bin/env python

#    lowrank.py
#
#    A mixture of factor analysers for the latent space.
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

"""A low-rank (factor-analyser) covariance for the Gaussian mixture.

RESULTS Sec.38.3 measured that the mixture gets the right AMOUNT of
variance and the wrong SHAPE. The distance from a sample to its
assigned component matches what a genuine draw from that component
would give to within 4%, so the scale is correct - but the
participation ratio of each component's own cloud is 7.4 of 32 and 9.5
of 64, where an isotropic Gaussian would give 32 and 64. The tied
spherical covariance spreads isotropically the variance the data
concentrates in about eight directions.

Fitted on the training representations and scored on held-out ones, the
cost of that is 8.5 to 11.6 nats per sample:

    spherical, tied - what is used     28.593
    diagonal                           30.573
    low-rank q=4                       37.124   <- wins BIC outright
    low-rank q=8                       39.805
    full                               41.471   overfits: train-test
                                                gap 13.6 against 6.6

Each component's covariance is

    Sigma_k = W_k W_k' + diag(psi_k)

with W_k of rank q. That is `q*d + d` parameters against `d(d+1)/2` for
a full covariance - linear in the dimension rather than quadratic - and
q around four to eight is what Sec.38.2's nine-dimensional manifold
implies a component should need.

**Why this lives in bulkdgd and not in tgmm.** `tgmm` dispatches on
`covariance_type` through an if/elif chain that ends in a ValueError,
so a new type has to be handled before that chain sees it. Everything
here overrides the three methods that chain touches -`_e_step`,
`_m_step` and the density - and delegates every other type to the
parent untouched. `tgmm` itself is not modified.

**The two identities that make it cheap.** Inverting Sigma_k and taking
its log-determinant directly would be O(d^3) per component per
iteration. The Woodbury identity and the matrix determinant lemma turn
both into O(d q^2):

    Sigma^-1  = Psi^-1 - Psi^-1 W (I + W' Psi^-1 W)^-1 W' Psi^-1
    log|Sigma| = log|Psi| + log|I + W' Psi^-1 W|
"""


import logging as log

import torch


# Get the module's logger.
logger = log.getLogger(__name__)


class LowRankMixin:

    """Adds a ``"low_rank"`` covariance type to a ``tgmm`` mixture.

    Mixed into the package's wrapper, which subclasses
    ``tgmm.GaussianMixture``. Every covariance type but the new one is
    handed straight to the parent.
    """

    # The rank, unless the configuration says otherwise. Four is where
    # BIC put the optimum on both the d32 and the d64 models.
    DEFAULT_RANK = 4

    #-----------------------------------------------------------------#

    def __init__(self,
                 *args,
                 rank = None,
                 **kwargs):
        """Take ``rank`` out of the arguments before the parent sees
        it.

        ``tgmm.GaussianMixture.__init__`` raises on any keyword it does
        not recognise, and ``rank`` is one of ours - it describes a
        covariance type the parent has never heard of. The
        configuration hands every latent option to the constructor as a
        keyword, so without this a low-rank model dies at construction
        with

            GaussianMixture.__init__() got unexpected keyword
            argument(s): 'rank'

        rather than anywhere near the covariance code.
        """

        self.rank = rank if rank is not None else self.DEFAULT_RANK

        # The low-rank parameters are allocated lazily, the first time
        # a density is asked for, because their shape needs the
        # dimensionality and the parent has not necessarily settled it
        # yet at this point.
        self.factors_ = None
        self.psi_ = None

        super().__init__(*args, **kwargs)

    #-----------------------------------------------------------------#

    def _as_spherical(self):
        """Present a covariance type the parent recognises, for the
        duration of a call into it.

        ``tgmm`` branches on ``covariance_type`` in TWELVE places
        across three modules - allocation, four initialisers, the
        M-step's two covariance updates, sampling, the Mahalanobis
        distance, BIC and AIC - and every one of them ends in

            ValueError: Unsupported covariance type: low_rank

        Overriding all twelve would mean reimplementing most of the
        parent. The alternative rests on the fact that a low-rank
        mixture never READS ``covariances_``: the covariance lives in
        ``factors_`` and ``psi_``, and the density, the E-step and the
        M-step are all overridden here. So the parent is allowed to
        allocate and maintain a spherical ``covariances_`` it will
        never be asked about, and everything that matters is computed
        from our own parameters.

        The swap is held only around calls INTO the parent. It must
        never wrap our own ``_e_step`` or ``_m_step``, which decide
        whether to delegate by reading exactly this attribute.
        """

        import contextlib

        @contextlib.contextmanager
        def _swap():

            real = getattr(self, "covariance_type", None)
            self.covariance_type = "spherical"

            try:
                yield

            finally:
                self.covariance_type = real

        return _swap()

    #-----------------------------------------------------------------#

    def _allocate_parameters(self, *args, **kwargs):
        """Let the parent set up means, weights and a placeholder
        covariance, then allocate the low-rank parameters."""

        if getattr(self, "covariance_type", None) != "low_rank":
            return super()._allocate_parameters(*args, **kwargs)

        with self._as_spherical():
            out = super()._allocate_parameters(*args, **kwargs)

        n_features = getattr(self, "n_features", None)

        if n_features:
            self._lr_allocate(n_features)

        return out

    #-----------------------------------------------------------------#

    def _build_covariances_for_sampling(self,
                                        indices,
                                        n_samples):
        """Full covariances to draw from, as ``W W' + diag(psi)``, ONE
        PER REQUESTED SAMPLE.

        The parent's contract is per-sample, not per-component: it
        passes the component each of the ``n_samples`` draws belongs to
        and expects a matrix for each of them, which it hands straight
        to ``MultivariateNormal``. Returning one matrix per component
        instead gives a batch of the wrong length and the sampler fails
        with a broadcast error.

        The representation search draws its candidates from the
        components, so this is on the hot path rather than a
        convenience.
        """

        if getattr(self, "covariance_type", None) != "low_rank":
            return super()._build_covariances_for_sampling(
                indices, n_samples)

        if not self._lr_ready():
            self._lr_allocate(self.n_features)

        W = self.factors_[indices]
        psi = self.psi_[indices]

        return W @ W.transpose(-1, -2) + torch.diag_embed(psi)

    #-----------------------------------------------------------------#

    def _lr_ready(self):
        """Whether the low-rank parameters have been allocated."""

        return (getattr(self, "factors_", None) is not None
                and getattr(self, "psi_", None) is not None)

    #-----------------------------------------------------------------#

    def _lr_allocate(self, n_features):
        """Create the factor loadings and the diagonal, from whatever
        the parent has already fitted."""

        K = self.n_components
        q = getattr(self, "rank", None) or self.DEFAULT_RANK
        d = n_features

        dev = self.means_.device if self.means_ is not None else None
        dt = self.means_.dtype if self.means_ is not None \
            else torch.float64

        g = torch.Generator(device = "cpu")
        g.manual_seed(0)

        self.rank = q

        # Small random loadings, and a diagonal seeded from whatever
        # variance the parent's fit already found so that the first
        # E-step is not run against nonsense.
        self.factors_ = \
            (0.1 * torch.randn(K, d, q, generator = g,
                               dtype = torch.float64)).to(
                                   device = dev, dtype = dt)

        base = 1.0

        if getattr(self, "covariances_", None) is not None:

            c = self.covariances_

            if c.dim() == 3:
                base = float(torch.diagonal(c, dim1 = -2,
                                            dim2 = -1).mean())
            else:
                base = float(c.mean())

        self.psi_ = torch.full((K, d), max(base, 1.0e-6),
                               device = dev, dtype = dt)

        logger.info(
            f"The low-rank covariance was allocated with rank {q}: "
            f"{K * (d * q + d)} parameters against "
            f"{K * d * (d + 1) // 2} for a full covariance.")

    #-----------------------------------------------------------------#

    def _lr_solve(self, W, psi, Xc):
        """The log-determinant and the Mahalanobis terms of one
        component, by Woodbury and the matrix determinant lemma."""

        d, q = W.shape

        Pi = 1.0 / psi                                   # (d,)
        A = torch.eye(q, device = W.device, dtype = W.dtype) \
            + (W.T * Pi) @ W                             # (q, q)

        L = torch.linalg.cholesky(A)

        logdet = torch.log(psi).sum() \
            + 2.0 * torch.log(torch.diagonal(L)).sum()

        XP = Xc * Pi                                     # (n, d)
        B = XP @ W                                       # (n, q)
        Y = torch.cholesky_solve(B.T, L).T                # (n, q)

        maha = (Xc * XP).sum(-1) - (B * Y).sum(-1)

        return logdet, maha

    #-----------------------------------------------------------------#

    def _estimate_log_gaussian_low_rank(self, X):
        """``log p(x | z, theta)`` for every component, under
        ``Sigma_k = W_k W_k' + diag(psi_k)``."""

        if not self._lr_ready():
            self._lr_allocate(X.shape[1])

        n, d = X.shape
        out = torch.empty(n, self.n_components,
                          device = X.device, dtype = X.dtype)

        for k in range(self.n_components):

            Xc = X - self.means_[k]

            logdet, maha = self._lr_solve(self.factors_[k],
                                          self.psi_[k], Xc)

            out[:, k] = -0.5 * (d * torch.log(
                torch.tensor(2.0 * torch.pi, device = X.device,
                             dtype = X.dtype)) + logdet + maha)

        return out

    #-----------------------------------------------------------------#

    def _e_step(self, X):
        """Responsibilities. Only the new covariance type is handled
        here; everything else goes to the parent unchanged."""

        if getattr(self, "covariance_type", None) != "low_rank":
            return super()._e_step(X)

        log_prob = self._estimate_log_gaussian_low_rank(X)

        log_prob = log_prob \
            + torch.log(self.weights_ + 1.0e-20).unsqueeze(0)

        log_prob_norm = torch.logsumexp(log_prob, dim = 1)
        resp = torch.exp(log_prob - log_prob_norm.unsqueeze(1))

        return resp, log_prob_norm

    #-----------------------------------------------------------------#

    def _m_step(self, X, resp):
        """One EM update of the weights, means, loadings and diagonal.

        The factor-analysis update is the standard one: form each
        component's responsibility-weighted scatter, take the posterior
        moments of the latent factors under the current loadings, and
        solve for the loadings that best explain the scatter.
        """

        if getattr(self, "covariance_type", None) != "low_rank":
            return super()._m_step(X, resp)

        if not self._lr_ready():
            self._lr_allocate(X.shape[1])

        n, d = X.shape
        q = self.rank

        Nk = resp.sum(0) + 1.0e-10

        self.weights_ = Nk / n
        self.means_ = (resp.T @ X) / Nk.unsqueeze(-1)

        eye_q = torch.eye(q, device = X.device, dtype = X.dtype)

        for k in range(self.n_components):

            Xc = X - self.means_[k]
            r = resp[:, k:k + 1]

            # The weighted scatter of this component.
            Sw = (Xc * r).T @ Xc / Nk[k]

            W = self.factors_[k]
            Pi = 1.0 / self.psi_[k]

            A = eye_q + (W.T * Pi) @ W
            Ainv = torch.linalg.inv(A)

            G = Ainv @ (W.T * Pi)                        # (q, d)

            Ez = G @ Sw                                   # (q, d)
            Ezz = Ainv + G @ Sw @ G.T                     # (q, q)

            self.factors_[k] = torch.linalg.solve(Ezz, Ez).T

            self.psi_[k] = torch.clamp(
                torch.diagonal(Sw - self.factors_[k] @ Ez),
                min = 1.0e-6)

    #-----------------------------------------------------------------#

    def _expected_covar_shape(self):
        """The parent validates ``covariances_`` against this; the new
        type keeps its parameters elsewhere, so report the shape the
        parent will not complain about."""

        if getattr(self, "covariance_type", None) != "low_rank":
            return super()._expected_covar_shape()

        return (self.n_components, self.n_features)

    #-----------------------------------------------------------------#

    def lr_state(self):
        """The low-rank parameters, for saving beside the rest."""

        if not self._lr_ready():
            return {}

        return {"factors_": self.factors_.detach().cpu(),
                "psi_": self.psi_.detach().cpu(),
                "rank": self.rank}

    def load_lr_state(self, state):
        """Restore what :meth:`lr_state` wrote.

        Raises if a low-rank model is handed a state that has no
        low-rank parameters in it, or one whose shapes do not match the
        mixture being loaded into.

        **Why this raises rather than warning.** The parameters live
        outside ``covariances_``, so a checkpoint written by any other
        covariance type simply has no ``factors_`` in it. Loading one
        silently would leave the freshly allocated random loadings in
        place, and the model would go on to compute every density from
        them - loading without complaint, reporting sensible-looking
        losses, and being a different model from the one that was
        trained.
        """

        want_low_rank = \
            getattr(self, "covariance_type", None) == "low_rank"

        if not state:

            if want_low_rank:

                errstr = \
                    "This is a 'low_rank' mixture, but the state " \
                    "being loaded holds no low-rank parameters - it " \
                    "was almost certainly written by a model with a " \
                    "different covariance type. Loading it would " \
                    "leave the covariance at its random " \
                    "initialization while everything else came from " \
                    "the file."
                raise ValueError(errstr)

            return

        missing = [k for k in ("factors_", "psi_") if k not in state]

        if missing:

            errstr = \
                f"The state being loaded is missing " \
                f"{', '.join(repr(m) for m in missing)}, which a " \
                f"'low_rank' mixture needs."
            raise ValueError(errstr)

        factors = torch.as_tensor(state["factors_"])
        psi = torch.as_tensor(state["psi_"])

        # The shapes have to match the mixture this is being loaded
        # into, or the first density evaluation fails somewhere far
        # from here with an error about matrix dimensions.
        K = getattr(self, "n_components", factors.shape[0])

        if factors.shape[0] != K or psi.shape[0] != K:

            errstr = \
                f"The state holds {factors.shape[0]} component(s) of " \
                f"low-rank covariance, but the mixture has {K}."
            raise ValueError(errstr)

        if factors.shape[1] != psi.shape[1]:

            errstr = \
                f"The loadings are {factors.shape[1]}-dimensional " \
                f"and the diagonal is {psi.shape[1]}-dimensional; " \
                f"they must agree."
            raise ValueError(errstr)

        # Onto whatever device and dtype the rest of the mixture is
        # on. A checkpoint is written from the CPU, so loading it into
        # a model on a GPU otherwise leaves the covariance behind and
        # the first density evaluation fails with a device mismatch.
        if getattr(self, "means_", None) is not None:

            factors = factors.to(device = self.means_.device,
                                 dtype = self.means_.dtype)
            psi = psi.to(device = self.means_.device,
                         dtype = self.means_.dtype)

        self.rank = state.get("rank", factors.shape[2])
        self.factors_ = factors
        self.psi_ = psi
