#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    componentfit.py
#
#    Anchored component expansion for fine-tuning.
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
__doc__ = "Anchored component expansion for fine-tuning."


#######################################################################


# Import from the standard library.
import copy
import math

# Import from third-party libraries.
import pandas as pd
import torch


#######################################################################


def _squared_distances(
        points: torch.Tensor,
        centres: torch.Tensor) -> torch.Tensor:
    """Return squared Euclidean distances to every centre."""

    return torch.sum(
        (points.unsqueeze(1) - centres.unsqueeze(0)) ** 2,
        dim = 2)


def _initialize_means(
        points: torch.Tensor,
        old_means: torch.Tensor,
        n_components: int,
        method: str,
        seed: int) -> torch.Tensor:
    """Initialize appended means deterministically."""

    generator = torch.Generator(device = points.device)
    generator.manual_seed(seed)
    selected = []
    centres = old_means

    for _ in range(n_components):

        distances = _squared_distances(
            points = points,
            centres = centres).min(dim = 1).values

        if method == "maxdist":
            index = int(torch.argmax(distances))
        else:
            total = distances.sum()

            if not torch.isfinite(total) or total <= 0:
                index = int(torch.randint(
                    low = 0,
                    high = points.shape[0],
                    size = (1,),
                    generator = generator,
                    device = points.device))
            else:
                index = int(torch.multinomial(
                    distances / total,
                    num_samples = 1,
                    generator = generator))

        selected.append(points[index].clone())
        centres = torch.cat(
            [old_means, torch.stack(selected)], dim = 0)

    return torch.stack(selected)


def _log_joint_tied_spherical(
        points: torch.Tensor,
        means: torch.Tensor,
        weights: torch.Tensor,
        variance: torch.Tensor) -> torch.Tensor:
    """Return log-joint densities for tied spherical covariance."""

    dim = points.shape[1]
    distances = _squared_distances(points = points, centres = means)
    constant = dim * (math.log(2.0 * math.pi) + torch.log(variance))
    log_density = -0.5 * (constant + distances / variance)

    return log_density + torch.log(weights).unsqueeze(0)


#######################################################################


def append_anchored_components(
        parent,
        target_representations: torch.Tensor,
        n_new_components: int,
        new_component_weight: float,
        initialization: str = "kpp",
        max_iter: int = 100,
        tol: float = 1.0e-5,
        reg_covar: float = 1.0e-6,
        seed: int = 37):
    """Append target-fitted components without moving old components.

    Version one supports the tied-spherical TGMM used by the current
    GTEx ensemble.  Old means, covariance, relative weights, and IDs
    remain exact.  The new components share the parent's tied variance.

    Parameters
    ----------
    parent : :class:`bulkdgd.core.latents.GaussianMixtureModelTGMM`
        The fitted parent mixture.

    target_representations : :class:`torch.Tensor`
        Target representations, one row per sample.

    n_new_components : :class:`int`
        The number of components to append.

    new_component_weight : :class:`float`
        The total mixture mass reserved for appended components.

    initialization : :class:`str`
        Either ``"kpp"`` or ``"maxdist"``.

    max_iter : :class:`int`
        The maximum number of anchored EM iterations.

    tol : :class:`float`
        The convergence tolerance on mean log likelihood.

    reg_covar : :class:`float`
        The minimum admissible parent variance.  The tied covariance
        remains fixed, because moving it would also move every parent
        component.

    seed : :class:`int`
        The deterministic initialization seed.

    Returns
    -------
    derived : :class:`bulkdgd.core.latents.GaussianMixtureModelTGMM`
        The expanded mixture.

    component_map : :class:`pandas.DataFrame`
        The stable parent-to-derived component map.

    history : :class:`pandas.DataFrame`
        The anchored EM objective history.
    """

    if parent.covariance_type != "tied_spherical":
        raise NotImplementedError(
            "Anchored component expansion currently supports the "
            "tied-spherical TGMM used by the GTEx ensemble.")

    if target_representations.ndim != 2 or \
            target_representations.shape[0] == 0:
        raise ValueError(
            "Target representations must be a non-empty matrix.")

    if target_representations.shape[1] != parent.n_features:
        raise ValueError(
            "Target representations do not match the latent "
            "dimension.")

    if type(n_new_components) is not int or n_new_components <= 0:
        raise ValueError("n_new_components must be positive.")

    if not 0 < new_component_weight < 1:
        raise ValueError(
            "new_component_weight must be between zero and one.")

    if initialization not in {"kpp", "maxdist"}:
        raise ValueError("initialization must be 'kpp' or 'maxdist'.")

    if reg_covar <= 0:
        raise ValueError("reg_covar must be positive.")

    points = target_representations.to(
        device = parent.means_.device,
        dtype = parent.means_.dtype)
    old_means = parent.means_.detach().clone()
    old_weights = parent.weights_.detach().clone()
    variance = parent.covariances_.detach().clone()

    if torch.any(variance < reg_covar):
        raise ValueError(
            "The fixed parent covariance is below reg_covar. It "
            "cannot be raised without changing parent components.")
    old_count = int(parent.n_components)
    new_means = _initialize_means(
        points = points,
        old_means = old_means,
        n_components = n_new_components,
        method = initialization,
        seed = seed)
    new_weights = torch.full(
        (n_new_components,),
        new_component_weight / n_new_components,
        dtype = points.dtype,
        device = points.device)
    fixed_old_weights = \
        old_weights / old_weights.sum() * (1.0 - new_component_weight)
    rows = []
    previous = None

    for iteration in range(1, max_iter + 1):

        means = torch.cat([old_means, new_means], dim = 0)
        weights = torch.cat(
            [fixed_old_weights, new_weights], dim = 0)
        log_joint = _log_joint_tied_spherical(
            points = points,
            means = means,
            weights = weights,
            variance = variance)
        log_norm = torch.logsumexp(log_joint, dim = 1)
        objective = float(log_norm.mean())
        rows.append(
            {"iteration" : iteration,
             "mean_log_likelihood" : objective})
        responsibilities = torch.softmax(log_joint, dim = 1)
        new_resp = responsibilities[:, old_count:]
        mass = new_resp.sum(dim = 0)
        denominator = mass.clamp_min(torch.finfo(points.dtype).tiny)
        proposed = new_resp.T @ points / denominator.unsqueeze(1)
        has_mass = mass > torch.finfo(points.dtype).eps
        new_means = torch.where(
            has_mass.unsqueeze(1), proposed, new_means)

        if float(mass.sum()) > 0:
            new_weights = mass / mass.sum() * new_component_weight
        else:
            new_weights.fill_(
                new_component_weight / n_new_components)

        if previous is not None and abs(objective - previous) <= tol:
            break

        previous = objective

    derived = copy.deepcopy(parent)
    derived.n_components = old_count + n_new_components
    derived.means_ = torch.cat([old_means, new_means], dim = 0)
    derived.weights_ = torch.cat(
        [fixed_old_weights, new_weights], dim = 0)
    derived.covariances_ = variance
    derived.initial_means_ = derived.means_.clone()
    derived.initial_weights_ = derived.weights_.clone()
    derived.initial_covariances_ = variance.clone()
    derived.n_iter_ = len(rows)
    derived.converged_ = len(rows) < max_iter
    derived.lower_bound_ = rows[-1]["mean_log_likelihood"]
    derived.fitted_ = True

    component_rows = \
        [{"derived_component" : component,
          "status" : "parent",
          "parent_component" : component}
         for component in range(old_count)]
    component_rows.extend(
        {"derived_component" : old_count + component,
         "status" : "appended",
         "parent_component" : pd.NA}
        for component in range(n_new_components))

    return (derived,
            pd.DataFrame(component_rows),
            pd.DataFrame(rows))


#######################################################################
