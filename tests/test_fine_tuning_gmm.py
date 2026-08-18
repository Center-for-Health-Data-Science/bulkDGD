#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    test_fine_tuning_gmm.py
#
#    Tests for anchored fine-tuning of the TGMM.


#######################################################################


# Import from third-party libraries.
import pytest
import torch

# Import from 'bulkdgd'.
from bulkdgd.core import componentfit
from bulkdgd.core import latents


#######################################################################


def _parent_mixture():
    """Build a small fitted tied-spherical mixture."""

    mixture = latents.GaussianMixtureModelTGMM(
        n_components = 2,
        n_features = 2,
        covariance_type = "tied_spherical",
        device = "cpu")
    mixture.weights_ = torch.tensor(
        [0.25, 0.75], dtype = torch.float64)
    mixture.means_ = torch.tensor(
        [[-2.0, -2.0], [2.0, 2.0]], dtype = torch.float64)
    mixture.covariances_ = torch.tensor(
        0.5, dtype = torch.float64)
    mixture.initial_weights_ = mixture.weights_.clone()
    mixture.initial_means_ = mixture.means_.clone()
    mixture.initial_covariances_ = mixture.covariances_.clone()
    mixture.fitted_ = True

    return mixture


def test_appended_components_preserve_parent_exactly():
    """Old means, covariance, relative weights, and IDs are fixed."""

    parent = _parent_mixture()
    old_state = {key : value.clone()
                 for key, value in parent.state_dict().items()
                 if isinstance(value, torch.Tensor)}
    target = torch.tensor(
        [[7.0, 7.0], [7.2, 6.8], [6.8, 7.2],
         [-7.0, -7.0], [-7.2, -6.8], [-6.8, -7.2]],
        dtype = torch.float64)
    derived, component_map, history = \
        componentfit.append_anchored_components(
            parent = parent,
            target_representations = target,
            n_new_components = 2,
            new_component_weight = 0.2,
            initialization = "maxdist",
            max_iter = 20,
            seed = 11)

    assert derived.n_components == 4
    assert torch.equal(derived.means_[:2], parent.means_)
    assert torch.equal(derived.covariances_, parent.covariances_)
    assert torch.allclose(
        derived.weights_[:2] / derived.weights_[:2].sum(),
        parent.weights_ / parent.weights_.sum(),
        rtol = 0.0,
        atol = torch.finfo(torch.float64).eps)
    assert torch.isclose(derived.weights_.sum(),
                         torch.tensor(1.0, dtype = torch.float64))
    assert component_map["status"].tolist() == \
        ["parent", "parent", "appended", "appended"]
    assert not history.empty

    for key, value in old_state.items():
        assert torch.equal(parent.state_dict()[key], value)


def test_appended_components_are_deterministic():
    """A fixed seed gives the same appended mixture."""

    parent = _parent_mixture()
    target = torch.arange(
        20, dtype = torch.float64).reshape(10, 2)
    args = \
        {"parent" : parent,
         "target_representations" : target,
         "n_new_components" : 2,
         "new_component_weight" : 0.1,
         "initialization" : "kpp",
         "max_iter" : 10,
         "seed" : 19}
    first, first_map, first_history = \
        componentfit.append_anchored_components(**args)
    second, second_map, second_history = \
        componentfit.append_anchored_components(**args)

    assert torch.equal(first.means_, second.means_)
    assert torch.equal(first.weights_, second.weights_)
    assert first_map.equals(second_map)
    assert first_history.equals(second_history)


def test_wrong_covariance_is_rejected():
    """Version one does not guess at another covariance contract."""

    parent = _parent_mixture()
    parent.covariance_type = "diag"

    try:
        componentfit.append_anchored_components(
            parent = parent,
            target_representations = torch.ones(
                (2, 2), dtype = torch.float64),
            n_new_components = 1,
            new_component_weight = 0.1)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("An unsupported covariance was accepted.")


def test_parent_covariance_below_floor_is_rejected():
    """The numerical floor cannot silently move parent covariance."""

    parent = _parent_mixture()

    with pytest.raises(ValueError, match="reg_covar"):
        componentfit.append_anchored_components(
            parent = parent,
            target_representations = torch.ones(
                (2, 2), dtype = torch.float64),
            n_new_components = 1,
            new_component_weight = 0.1,
            reg_covar = 1.0)


#######################################################################
