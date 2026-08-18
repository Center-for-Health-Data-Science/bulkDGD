.. _fine_tuning_config_options:

Configuration for fine-tuning a trained model
=============================================

The :meth:`bulkdgd.core.model.BulkDGD.fine_tune` method derives a new
model from a checkpoint-backed parent. It never overwrites the parent
model or its checkpoints. ``output_dir`` must therefore name a new
directory unless an implemented scheme explicitly supports resuming.

The public ``fine_tuning_scheme`` argument and the value in the
configuration must agree. Unknown options are errors; a misspelled
preservation setting is not silently ignored.

Common options
--------------

* ``fine_tuning_scheme`` selects ``add_gmm_components``,
  ``dispersion_only``, ``output_head``, ``low_rank_adapter``, or
  ``joint_replay``. The default is ``add_gmm_components``.

* ``seed`` seeds Python, NumPy, Torch CPU, and every Torch CUDA
  generator. It defaults to ``37``.

* ``deterministic_algorithms`` asks Torch to use deterministic
  operations and raise where none exists. It defaults to ``True``.

* ``target_representation_options.config`` is a representation
  configuration dictionary, path, or shipped name. New analyses
  should use ``sample_keyed`` initialization. Historical positional
  initialization has to be requested explicitly.

* ``replay_options.replay_ratio`` is the number of healthy replay
  samples per target sample. Decoder-moving schemes require a positive
  ratio. ``stratify_by`` names an optional replay metadata column.

* ``anchoring_options.l2_sp`` and ``functional`` control parameter
  and prediction anchoring to the parent. They must be non-negative.

* ``checkpoint_options.resume`` and ``every_n_epochs`` control
  resumable fine-tuning checkpoints.

* ``early_stopping_options.patience`` and ``min_delta`` control
  validation stopping for optimization-based schemes.

* ``reporting_options.fixed_panels`` requests the fixed healthy,
  target, and differential-expression preservation panels.

Anchored component expansion
----------------------------

``add_gmm_components`` is the first implemented and default scheme.
It infers deterministic target representations with the frozen parent,
then appends target-fitted components to the TGMM. The decoder and all
parent component means, covariance, relative weights, and component IDs
remain fixed. New components are placed after the parent IDs.

The scheme options are:

* ``component_fit_seed``: the anchored component-initialization seed,
  which is independent of the representation-initialization seed;
* ``n_new_components``: a positive number of appended components;
* ``new_component_weight``: total mass in ``(0, 1)`` reserved for the
  appended components;
* ``initialization``: ``kpp`` or ``maxdist``;
* ``max_iter`` and ``tol``: the anchored-EM stopping controls;
* ``reg_covar``: the minimum admissible fixed parent variance;
* ``outer_iterations``: currently exactly one.

The current GTEx ensemble uses a tied-spherical TGMM. Its covariance
is shared by every component, so a new component cannot learn a distinct
variance without changing the old components too. The appended
components therefore share the exact parent covariance. A parent whose
fixed variance is below ``reg_covar`` is rejected rather than modified.

The output directory contains the child ``model.yaml``, ``gmm.pth`` and
``dec.pth``. Parent-inferred fitting and comparator coordinates have
``parent_`` prefixes. The public result's training and test
representations and predictions are freshly inferred by the child and
have ``child_`` prefixes on disk. The directory also contains a stable
component map, metrics, loss history, resolved configuration, and a
hash-linked provenance manifest, including a selected-count value hash.

Reserved decoder-moving schemes
-------------------------------

The strict configurations and explicit private dispatch methods for the
remaining four schemes are present so their interfaces cannot drift:

* ``dispersion_only`` moves only supported dispersion parameters;
* ``output_head`` moves the mean head, or the full output head when
  explicitly requested;
* ``low_rank_adapter`` adds a zero-initialized residual adapter;
* ``joint_replay`` moves selected decoder parameters and target and
  replay representations together.

These schemes require healthy replay and preservation testing. Until an
optimizer, serialization path, checkpoint resume, and no-parent-mutation
tests are complete, calling one raises ``NotImplementedError``. This is
intentional: accepting a configuration and silently running a different
adaptation would make the derived model uninterpretable.

Example
-------

.. code-block:: python

   result = model.fine_tune(
       df_samples=counts,
       names_train=train_ids,
       names_test=test_ids,
       fine_tuning_scheme="add_gmm_components",
       config_fine_tune=None,
       output_dir="derived/add_gmm_components")

``result`` is a named
:class:`bulkdgd.core.finetuning.FineTuningResult`; it is not a tuple
whose meaning changes when new outputs are added.
