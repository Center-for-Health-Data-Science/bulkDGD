.. _rep_config_options:

Configuration for the optimization scheme
=========================================

Two optimization schemes to find the representations are implemented:

* ``two_opt``, which consists of two consecutive rounds of optimizations. Indeed, multiple candidate representations per sample are found, optimized, and the best one for each sample is picked from the pool. Then, a second round of optimization is performed on these selected representations. The ``two_opt`` scheme is implemented in the YAML file ``bulkdgd/configs/representations/two_opt.yaml``.

* ``two_opt_multiseed``, which runs ``two_opt`` once per initialization seed and keeps every seed's answer instead of one. It is implemented in the YAML file ``bulkdgd/configs/representations/two_opt_multiseed.yaml``.

.. _rep_multiseed:

Why a multi-seed scheme
-----------------------

The seed decides **where the search starts, and nothing else**. It places the candidate representations drawn from the components of the Gaussian mixture model; the two rounds of descent and the selection between them are deterministic once those candidates are fixed. Two seeds therefore arrive at two different local optima, and on tumour data the genes called from them differ by roughly a fifth of their union.

Neither run can say which of the two is right. What can be said is that the calls the seeds **agree** on are measurably better than the calls only one of them makes: on TCGA, genes called from all three of three seeds overlap a cancer-matched driver catalogue 1.6 times as often as genes called from only one, and that separation survives controlling for effect size, widening to 2.2 times in the largest fold-change bin. The agreement is therefore worth producing deliberately rather than reconstructing from separate runs.

**Each seed is run in sequence, at the tensor shape a standalone run uses.** The seeds could share one optimization and be *mathematically* identical to separate runs, since the loss is summed over candidates, there is no gradient clipping in this code path, and AdamW is elementwise, so no candidate's gradient depends on how many others share the tensor. They would not be *bitwise* identical: the decoder's forward pass is a batched matrix multiply, and changing how many rows pass through it lets the linear algebra library select a different kernel, which moves the last bits and, over hundreds of epochs, can flip a near-tie between two candidates. Running each seed separately makes reproducing a standalone run a property of the arithmetic instead of a hope about kernel selection, at the cost of the efficiency of the larger multiply.

.. note::

   ``two_opt_multiseed`` requires ``latent_type: tgmm``. The legacy Gaussian mixture model places its candidates by a deterministic rule that takes no seed, so every seed would return the same representation and the agreement between them would measure nothing.

   It also requires ``loss_reduction_type: sum``. Under ``mean``, every gradient is divided by the number of candidates, and a seed's run here would no longer correspond to the same seed run on its own.

   The seeds must be **distinct**. Two runs from one seed give the same representation, so the scheme raises rather than reporting an agreement that is an artefact of the configuration.

Outputs
-------

``two_opt_multiseed`` produces one representation per sample **per seed**, together with the matching predicted means and r-values, and one further table that ``two_opt`` has no counterpart for: the loss of each sample's selected representation at the end of each optimization round.

After the run, these are available on the model as ``multiseed_results``, a dictionary with the keys ``"seeds"``, ``"representations"``, ``"pred_means"``, ``"pred_r_values"`` and ``"losses"``. The first four map a seed to its result; ``"losses"`` is a :class:`pandas.DataFrame` with one row per sample, indexed by sample name, and two columns per seed:

.. code-block:: text

   sample                        loss_opt1_seed7  loss_opt2_seed7  loss_opt1_seed13  loss_opt2_seed13
   GTEX-1117F-0226-SM-5GZZ7.1          100322.87         97806.77         105977.95         100584.39
   GTEX-1117F-1326-SM-5EGHH.1           97275.88         92556.16          91213.52          90363.78

``loss_opt1_seed<N>`` is the total loss of the candidate that won seed *N*'s selection, as of the last epoch of the first optimization: the competition's own number for the representation it kept. ``loss_opt2_seed<N>`` is that same representation's loss once the second optimization has finished moving it, and is therefore always the smaller of the two.

The method returns the **first** seed's representations, predicted means and r-values, so that anything downstream expecting the output of a scheme keeps working unchanged.

.. note::

   Tables written to disk are written as Parquet. A float64 written as text does not read back as the number that was written, and these losses are compared between seeds and between runs.

.. note::

   A ``one_opt`` scheme - a single round of optimization over the candidates, with no selection step and no second descent - was previously implemented and has been **retired**. Every result produced with this package used ``two_opt``, so ``one_opt`` was an untested path that still had to be kept working. Configuration files setting ``scheme_type: one_opt`` are no longer valid.

   The dispatch on ``scheme_type`` is deliberately kept, so that adding a further scheme means adding a branch and a template case rather than rebuilding how a scheme is chosen.

The options to customize these schemes can be passed as a nested dictionary or are specified in a YAML configuration file.

The function that loads the configuration file is :func:`bulkdgd.ioutil.load_config_rep`.

The options that can be specified are described below.

* ``"scheme_type"`` is the optimization scheme to use. This can be:

   * ``"two_opt"`` for the optimization scheme with two rounds of optimization.

   * ``"two_opt_multiseed"`` for the same scheme run once per initialization seed. See :ref:`rep_multiseed`. It takes every option ``two_opt`` takes, and reads its seeds from ``scheme_options.initialization.seeds``, a list of distinct integers.

* ``"latent_type"`` is the type of latent space used in the model. This can be:

   * ``"lgmm"`` for the legacy Gaussian Mixture Model implementation.

   * ``"tgmm"`` for the TorchGMM implementation.

* ``"n_rep_per_comp"`` is the number of representations to initialize per component per sample. This is a positive integer and defaults to ``1``.

* ``"data_loader_options"`` is a dictionary of options to initialize the data loader used to load the data for the optimization. It can contain the following options:

   * ``"batch_size"`` is the batch size to use for the data loader. This is a positive integer and defaults to ``128``.

   * ``"shuffle"`` is a boolean that indicates whether to shuffle the data at each epoch. It defaults to ``False``.

* ``"reporting_options"`` is a dictionary of options for reporting during the optimization. It can contain the following options:

   * ``"loss"`` is a dictionary of options to report the loss. It can contain the following options:

      * ``"reduction_type"`` is the reduction method to use for the loss function. This can be:

         * ``"sum"``, which computes the sum of the loss over the batch. This is the default value if not specified.

         * ``"mean"``, which computes the mean of the loss over the batch.
   
      * ``"latent"`` is a dictionary of options for the latent space loss. It can contain the following options:
      
         * ``"norm_type"`` is the method to use to normalize the loss when reporting it. This can be:

            * ``"none"``, which does not normalize the loss. This is the default value if not specified.

            * ``"n_samples"``, which normalizes the loss by the number of samples in the batch.

            * ``"n_samples * latent_dim"``, which normalizes the loss by the number of samples times the latent dimension.

         * ``"lambda"`` is the weight to use for the latent space loss. This is a non-negative float that defaults to ``1.0``.
      
      * ``"decoder"`` is a dictionary of options for the reconstruction loss. It can contain the following options:
      
         * ``"norm_type"`` is the method to use to normalize the loss when reporting it. This can be:

            * ``"none"``, which does not normalize the loss. This is the default value if not specified.

            * ``"n_samples"``, which normalizes the loss by the number of samples in the batch.

            * ``"n_samples * n_genes"``, which normalizes the loss by the number of samples times the number of genes.
      
      * ``"total"`` is a dictionary of options for the total loss. It can contain the following options:
      
         * ``"norm_type"`` is the method to use to normalize the loss when reporting it. This can be:

            * ``"none"``, which does not normalize the loss. This is the default value if not specified.

            * ``"n_samples"``, which normalizes the loss by the number of samples in the batch.

            * ``"n_samples * n_genes"``, which normalizes the loss by the number of samples times the number of genes.

* ``"scheme_options"`` is a dictionary of options specific to the optimization scheme. The options vary depending on the scheme type and the latent space type.

  * ``"initialization"`` controls how candidate representations are
    drawn for a TorchGMM latent space. It contains an integer ``"seed"``
    (or a list of integer ``"seeds"`` for ``two_opt_multiseed``) and a
    ``"mode"``:

    * ``"sample_keyed"`` is the default for new runs. BulkDGD derives
      an independent deterministic stream from the configured seed and
      the exact sample ID. The same sample therefore receives
      bit-for-bit identical initial candidates, representations,
      decoder outputs and DEA results when it is reordered, subsetted,
      placed beside different samples, or processed in a different
      chunk.

    * ``"legacy_positional"`` reproduces the historical behaviour. One
      stream is consumed in input order, so the candidate assigned to a
      sample depends on its position and on the size of the input chunk.
      Published-paper configurations must request this mode explicitly.

    * ``"legacy_indexed"`` reconstructs historical positional
      candidates while allowing the current counts table to be reordered
      or subsetted. In addition to the seed, it requires
      ``"index_file"``,
      ``"original_n_samples"`` and ``"chunk_size"``. ``"index_file"``
      is a one-column CSV whose row names are sample IDs and whose
      integer values are their zero-based absolute positions in the
      historical input::

        sample_id,position
        SRR8261581,0
        SRR8261574,1

      ``"chunk_size"`` is the size of the historical outer input chunk,
      not ``data_loader_options.batch_size``. The total sample count is
      needed because the last historical chunk can be shorter. Sample
      IDs and requested historical positions must both be unique.

  * For the ``two_opt`` scheme with the legacy GMM (``"lgmm"``):

      * ``"loss_reduction_type"`` is the reduction method to use for the loss function. This can be:

         * ``"sum"``, which computes the sum of the loss over the batch. This is the default value if not specified.

         * ``"mean"``, which computes the mean of the loss over the batch.

      * ``"warm_start"`` is an optional dictionary that seeds the search with a data-driven starting point instead of leaving every candidate to a draw from the mixture. It can contain:

         * ``"pth_file"`` is the path to a fitted ridge predictor, written by :func:`bulkdgd.core.warmstart.fit_from_model_dir`. The prediction REPLACES one of the ``"n_rep_per_comp"`` times the number of components candidates of each sample rather than being added to them, so every count the scheme assumes is unchanged. Absent by default, in which case every candidate comes from the mixture as before.

           The predictor maps a sample's counts to its representation and is fitted on the model's own training representations, which it recovers with an :math:`R^2` of about 0.84 - the map is very nearly linear, not because the decoder is linear but because projecting 14,740 genes down to a few dozen numbers is massively overdetermined. It is not an encoder and enters neither the generative model nor the objective; it only chooses where the search starts. Used ALONE it reaches a worse optimum than the candidate competition does, which is why it takes one slot rather than all of them.

      * ``"optimization_1"`` is a dictionary of options for the first optimization round. It can contain the following options:

         * ``"epochs"`` is the number of epochs to run the first optimization for. This is a positive integer and defaults to ``10``.

         * ``"optimizer_type"`` is the type of optimizer to use. This can be:

            * ``"adam"``, which uses the Adam optimizer.

            * ``"adamw"``, which uses the AdamW optimizer. This is the default.

         * ``"optimizer_options"`` is a dictionary of options for the optimizer. It can contain the following options:

            * ``"lr"`` is the learning rate. This is a positive float that defaults to ``0.01``.

            * ``"weight_decay"`` is the weight decay. This is a non-negative float that defaults to ``0.0``.

            * ``"betas"`` is a list of two floats that specify the beta parameters for the optimizer. The defaults are ``[0.9, 0.999]``.


         * ``"noise_type"`` is the type of noise to inject into the representations while they are being optimized. This can be ``"gaussian"``, or absent. It is **off unless set**: the decoder is fixed when representations are found, and a representation is an inference about a sample, so a configuration that says nothing about noise finds the representation it would have found before this option existed. This is the same perturbation, under the same options, that the decoder's training applies to its own representations (see ``"train_noise_type"`` in :ref:`train_config_options`).

         * ``"noise_options"`` is a dictionary of options for that noise, used when ``"noise_type"`` is ``"gaussian"``. It can contain:

            * ``"scale"`` is the base scale of the noise. This is a non-negative float that defaults to ``0.0``, which disables the injection.

            * ``"start"`` and ``"end"`` are the multipliers the scale is cosine-annealed between, from this round's first epoch to its last. These are non-negative floats defaulting to ``1.0`` and ``0.01``.

            * ``"within_radius_prob"`` is the probability defining the hypersphere the noise is normalized against, so that a scale means the same thing whatever the dimensionality of the latent space. This is a float in [0, 1] that defaults to ``0.95``.

            * ``"gain"`` is a final multiplier on the noise. This is a non-negative float that defaults to ``1.0``.

            Each round is annealed over its OWN epochs and is configured separately: the first round explores from many candidates and the second refines the one that won, so they need not be perturbed by the same amount.
      * ``"optimization_2"`` is a dictionary of options for the second optimization round. It has the same structure as ``"optimization_1"`` but with ``"epochs"`` defaulting to ``50``.

   * For the ``two_opt`` scheme with TorchGMM (``"tgmm"``):

      * ``"loss_reduction_type"`` is the reduction method to use for the loss function. This can be:

         * ``"sum"``, which computes the sum of the loss over the batch. This is the default value if not specified.

         * ``"mean"``, which computes the mean of the loss over the batch.

      * ``"latent_loss_calculation"`` is a dictionary of options for the latent space loss calculation. It can contain:

         * ``"lambda"`` is the weight to use for the latent space loss. This is a non-negative float that defaults to ``1.0``.

      * ``"warm_start"`` is an optional dictionary that seeds the search with a data-driven starting point instead of leaving every candidate to a draw from the mixture. It can contain:

         * ``"pth_file"`` is the path to a fitted ridge predictor, written by :func:`bulkdgd.core.warmstart.fit_from_model_dir`. The prediction REPLACES one of the ``"n_rep_per_comp"`` times the number of components candidates of each sample rather than being added to them, so every count the scheme assumes is unchanged. Absent by default, in which case every candidate comes from the mixture as before.

           The predictor maps a sample's counts to its representation and is fitted on the model's own training representations, which it recovers with an :math:`R^2` of about 0.84 - the map is very nearly linear, not because the decoder is linear but because projecting 14,740 genes down to a few dozen numbers is massively overdetermined. It is not an encoder and enters neither the generative model nor the objective; it only chooses where the search starts. Used ALONE it reaches a worse optimum than the candidate competition does, which is why it takes one slot rather than all of them.

      * ``"optimization_1"`` is a dictionary of options for the first optimization round. It can contain the following options:

         * ``"epochs"`` is the number of epochs to run the first optimization for. This is a positive integer and defaults to ``10``.

         * ``"optimizer_type"`` is the type of optimizer to use. This can be:

            * ``"adam"``, which uses the Adam optimizer.

            * ``"adamw"``, which uses the AdamW optimizer. This is the default.

         * ``"optimizer_options"`` is a dictionary of options for the optimizer. It can contain the following options:

            * ``"lr"`` is the learning rate. This is a positive float that defaults to ``0.01``.

            * ``"weight_decay"`` is the weight decay. This is a non-negative float that defaults to ``0.0``.

            * ``"betas"`` is a list of two floats that specify the beta parameters for the optimizer. The defaults are ``[0.9, 0.999]``.


         * ``"noise_type"`` is the type of noise to inject into the representations while they are being optimized. This can be ``"gaussian"``, or absent. It is **off unless set**: the decoder is fixed when representations are found, and a representation is an inference about a sample, so a configuration that says nothing about noise finds the representation it would have found before this option existed. This is the same perturbation, under the same options, that the decoder's training applies to its own representations (see ``"train_noise_type"`` in :ref:`train_config_options`).

         * ``"noise_options"`` is a dictionary of options for that noise, used when ``"noise_type"`` is ``"gaussian"``. It can contain:

            * ``"scale"`` is the base scale of the noise. This is a non-negative float that defaults to ``0.0``, which disables the injection.

            * ``"start"`` and ``"end"`` are the multipliers the scale is cosine-annealed between, from this round's first epoch to its last. These are non-negative floats defaulting to ``1.0`` and ``0.01``.

            * ``"within_radius_prob"`` is the probability defining the hypersphere the noise is normalized against, so that a scale means the same thing whatever the dimensionality of the latent space. This is a float in [0, 1] that defaults to ``0.95``.

            * ``"gain"`` is a final multiplier on the noise. This is a non-negative float that defaults to ``1.0``.

            Each round is annealed over its OWN epochs and is configured separately: the first round explores from many candidates and the second refines the one that won, so they need not be perturbed by the same amount.
      * ``"optimization_2"`` is a dictionary of options for the second
        optimization round. It has the same structure as
        ``"optimization_1"`` but with ``"epochs"`` defaulting to
        ``50``.
