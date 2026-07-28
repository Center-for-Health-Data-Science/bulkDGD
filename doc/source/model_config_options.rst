.. _model_config_options:

Configuration for creating an instance of the BulkDGD model
===========================================================

To create a new instance of :class:`core.model.BulkDGD`, we need to set a number of options.

These options can be passed as a nested dictionary or are specified in a YAML configuration file.

The function that loads the configuration file is :func:`bulkdgd.ioutil.load_config_model`.

The options that can be specified are described below.

* ``"genes_txt_file"`` is the path to the plain text file containing the list of genes used in the model. This file should contain one gene name per line in Ensemble ID format.

* ``"latent_dim"`` is the dimensionality of the latent space. This is a positive integer that specifies the number of dimensions in the latent space.

* ``"latent_type"`` is the type of latent space to use. This is a string that can take one of the following values: ``"lgmm"`` for the legacy Gaussian Mixture Model implementation, and ``"tgmm"`` for the new Gaussian Mixture Model implementeation.

* ``"latent_options"`` is a dictionary containing the options for the GMM. The options that can be specified in this dictionary depend on the type of GMM used.

   * If you are loading a trained model, ``"latent_pth_file"`` is the path to the .pth file containing the state dictionary of the latent space. 

   * For the legacy GMM (``"lgmm"``):

      * ``"n_components"`` is the number of components in the GMM. This is a positive integer that specifies the number of Gaussian components in the mixture.

      * ``"covariance_type"`` is the type of covariance to use. This is a string that can take one of the following values:
         
         * ``"fixed"`` for a fixed covariance matrix.
         * ``"isotropic"`` for an isotropic covariance matrix.
         * ``"diagonal"`` for a diagonal covariance matrix. This is the default.
      
      * ``"means_prior_type"`` is the type of prior distribution used for the means of the GMM components. This is a string that can take one of the following values:
         
         * ``"softball"`` for a softball prior. This is the default.
      
      * ``"means_prior_options"`` is a dictionary containing the options for the means prior distribution. The options that can be specified in this dictionary depend on the type of prior distribution used.

         * For the softball prior (``"softball"``):

            * ``"radius"`` is the radius of the softball prior. This is a positive float that specifies the radius of the sphere on which the means are constrained to lie.
            * ``"sharpness"`` is the sharpness of the softball prior. This is a positive float that specifies how sharply the means are constrained to lie on the sphere.
      
      * ``"weights_prior_type"`` is the type of prior distribution used for the weights of the GMM components. This is a string that can take one of the following values:
         
         * ``"dirichlet"`` for a Dirichlet prior. This is the default.
      
      * ``"weights_prior_options"`` is a dictionary containing the options for the weights prior distribution. The options that can be specified in this dictionary depend on the type of prior distribution used.
         
         * For the Dirichlet prior (``"dirichlet"``):

            * ``"alpha"`` is the concentration parameter of the Dirichlet prior. This is a positive float that specifies the concentration of the Dirichlet distribution.
      
      * ``"log_var_prior_type"`` is the type of prior distribution used for the log-variances of the GMM components. This is a string that can take one of the following values:

         * ``"gaussian"`` for a Gaussian prior. This is the default.
      
      * ``"log_var_prior_options"`` is a dictionary containing the options for the log-variances prior distribution. The options that can be specified in this dictionary depend on the type of prior distribution used.

         * For the Gaussian prior (``"gaussian"``):

            * ``"mean"`` is the mean of the Gaussian prior. This is a float that specifies the mean of the Gaussian distribution.
            * ``"stddev"`` is the standard deviation of the Gaussian prior. This is a positive float that specifies the standard deviation of the Gaussian distribution.
   
   * For the new GMM implementation (``"tgmm"``):

      * ``"n_components"`` is the number of components in the GMM. This is a positive integer that specifies the number of Gaussian components in the mixture.

      * ``"covariance_type"`` is the type of covariance to use. This is a string that can take one of the following values:
         
         * ``"full"`` for a full covariance matrix. This is the default.
         * ``"diag"`` for a diagonal covariance matrix.
         * ``"spherical"`` for a spherical covariance matrix.
         * ``"tied_full"`` for a full tied covariance matrix.
         * ``"tied_diag"`` for a diagonal tied covariance matrix.
         * ``"tied_spherical"`` for a spherical tied covariance matrix.
         * ``"low_rank"`` for a low-rank (factor-analyser) covariance matrix, where each component's covariance is :math:`W_k W_k^\top + \mathrm{diag}(\psi_k)` with :math:`W_k` of rank ``"rank"``. This has ``rank * dim + dim`` free numbers per component instead of the ``dim * (dim + 1) / 2`` a full covariance needs, so it can be anisotropic without being quadratic in the dimensionality. It is the type to use when the components' clouds are much lower-dimensional than the latent space they sit in.

     * ``"rank"`` is the rank of the covariance and is used only when ``"covariance_type"`` is ``"low_rank"``, where it is the number of directions each component is allowed to be broad in. This is a positive integer, and the default is ``4``. Every other covariance type ignores it.

       .. note::
          A ``"low_rank"`` mixture keeps its ``factors_`` and ``psi_`` outside ``covariances_``, so its checkpoint carries keys that no other covariance type writes. Loading a checkpoint written by a different type into a ``"low_rank"`` model raises: the covariance would otherwise stay at its random initialization while every other parameter came from the file, and the model would run without complaint.

     * ``"init_means"`` is the method used to initialize the means of the GMM components. This is a string that can take one of the following values:
         
         * ``"kmeans"`` for K-means initialization. This is the default value if not specified.
         * ``"kpp"`` for K-means++ initialization.
         * ``"random"`` for random initialization.
         * ``"points"`` for initialization using random points from the dataset.
         * ``"maxdist"`` for initialization using the points with the maximum determinant of the covariance matrix.
    
     * ``"init_weights"`` is the method used to initialize the weights of the GMM components. This is a string that can take one of the following values:
         
         * ``"uniform"`` for uniform initialization. This is the default value if not specified.
         * ``"random"`` for random initialization.
         * ``"kmeans"`` for K-means initialization.
   
     * ``"init_covariances"`` is the method used to initialize the covariances of the GMM components. This is a string that can take one of the following values:
         
         * ``"empirical"`` for empirical covariance initialization. This is the default value if not specified.
         * ``"eye"`` for identity covariance initialization.
         * ``"random"`` for random initialization.
         * ``"global"`` for global covariance initialization.
   
     * ``"tol"`` is the convergence threshold based on relative improvement in log-likelihood. If not specified, the default value is ``1e-4``.

     * ``"reg_covar"`` is the non-negative regularization added to the diagonal of covariance. This is used to ensure that the covariance matrices are positive definite. If not specified, the default value is ``1e-6``.

     * ``"n_init"`` is the number of initializations to perform. The default value is 1.
   
     * ``"random_state"`` is the random seed used for initialization. This is an integer that specifies the random seed for reproducibility. If not specified, the default value is None, which means that the random seed will be determined by the PyTorch's internal seed.

       Note that this reaches further than the GMM. When it is set, ``tgmm`` calls :func:`torch.manual_seed` with it as it places the mixture's components, which reseeds PyTorch's global generator for everything that is drawn after - the representations' initialization, the order of the batches, and the noise added during training. Leaving it unset and calling :func:`bulkdgd.reproducibility.set_seeds` once, before the model is built, makes the placement reproducible along with everything else and without the side effect. See :doc:`reproducibility <reproducibility>`.

     * ``"cem"`` is a boolean that specifies whether to use the classification expectation-maximization (CEM) algorithm for fitting the GMM. If not specified, the default value is False, which means that the standard expectation-maximization (EM) algorithm will be used.

* ``"gmm_final"`` is an optional dictionary containing the options for a second Gaussian mixture model, fitted to the representations *after* training and saved to its own file, ``gmm_final.pth``.

  This is not the prior. The mixture in ``"latent_options"`` is the prior: it is what training used, it is what ``gmm.pth`` holds, and it is what finding a representation for a new sample goes through. It is not replaced, and ``gmm_final.pth`` never enters the search for a representation - a mixture fitted to the representations cannot also be what produces them.

  What it is for is every question that is about the *density* of the latent space rather than about producing it: a sample's probability density, which component it belongs to, how atypical it is, and which directions a sampler should draw in. During training the mixture's covariance barely matters, because the mixture contributes a hundredth of a percent of the objective and cannot move a representation wherever its covariance points, so it is given the cheapest covariance that works. Afterwards the covariance is the whole of the answer.

  A model without this section is trained exactly as before and writes no such file.

   * ``"covariance_type"`` is the type of covariance the final mixture should have, from the same set of values as the ``"tgmm"`` covariance types above. There is no default: a model that asks for a final mixture is asking for a covariance the prior did not have, and which one is the whole of the request.

   * ``"shrinkage"`` is how far each component's own covariance is pulled back towards the one shared by all of them, between 0.0 and 1.0. If not specified, the default value is 0.0.

     It does nothing for the tied covariance types, which are already the thing being shrunk towards, and it is **required** for ``"full"``: a per-component full covariance in 32 dimensions is 528 numbers per component, against however many samples that component alone collected to fit them from, and an unshrunk fit comes back with negative variances.

   * ``"reg_covar"`` is the value added to the diagonal of the covariance. If not specified, the trained mixture's own value is used, so that the final mixture is regularized exactly as much as the prior was.

* ``"decoder_options"`` is a dictionary containing the options for the decoder.

   * If you are loading a trained model, ``"decoder_pth_file"`` is the path to the .pth file containing the state dictionary of the decoder.

   * ``"n_units_hidden_layers"`` is a list of positive integers that specifies the number of units in each hidden layer of the decoder. For example, if ``"n_units_hidden_layers"`` is set to ``[128, 64]``, then the decoder will have two hidden layers, with 128 units in the first hidden layer and 64 units in the second hidden layer.

   * ``"activations"`` is a list of strings that specifies the activation function to use in each hidden layer of the decoder. The length of this list should be equal to the length of the ``"n_units_hidden_layers"`` list. Each string in this list can take one of the following values:
         
      * ``"relu"`` for ReLU activation.
      * ``"elu"`` for ELU activation.
      * ``"gelu"`` for GELU activation (smooth everywhere, so the decoder map has no activation boundaries).
      
   * ``"dropout"`` is a float between 0 and 1 that specifies the dropout rate to use in the decoder. This is the fraction of the input units to drop during training. If not specified, the default value is 0, which means that no dropout will be applied.

   * ``"output_module_name"`` is the name of the output module used in the decoder. This is a string that can take one of the following values:
         
      * ``"poisson"`` for the Poisson output module.
      * ``"nb_feature_dispersion"`` for the negative binomial output module with r-values learned per gene.
      * ``"nb_full_dispersion"`` for the negative binomial output module with r-values learned per gene and sample.
      * ``"nb_full_dispersion_shrunk"`` as ``"nb_full_dispersion"``, but with the per-sample dispersion shrunk toward a per-gene baseline (see below).
      * ``"nb_full_dispersion_tied"`` as ``"nb_full_dispersion"``, but with the per-sample dispersion tied to the mean rather than predicted freely (see below).
      * ``"nb_full_dispersion_shrunk_tied"`` tied to the mean and shrunk (see below).
      * ``"nb_full_dispersion_hierarchical"`` as ``"nb_full_dispersion_shrunk"``, but with the strength of the shrinkage **learned per gene** rather than fixed (see below).

  The four variants of ``"nb_full_dispersion"`` exist because the per-sample dispersion it learns is the noisiest thing the model predicts: it is a free linear projection of the decoder's features to one log-r-value per gene per sample, anchored to nothing, and the likelihood constrains it far less than it constrains the mean. Two models trained alike agree closely on the mean but much less on the dispersion, and since a p-value is a tail probability of the negative binomial, the disagreement lands squarely on significance. The variants give the dispersion structure it otherwise lacks, without giving up the per-gene-per-sample flexibility that makes ``"nb_full_dispersion"`` better at differential expression than ``"nb_feature_dispersion"``:

      * ``"nb_full_dispersion_shrunk"`` writes the log-r-value as a per-gene baseline (a parameter fitted across all samples, hence stable) plus a per-sample deviation, and penalizes the deviation - the empirical-Bayes shrinkage of DESeq2 and edgeR.
      * ``"nb_full_dispersion_tied"`` makes the log-r-value a per-gene intercept plus a slope times the log of the predicted mean, and nothing else - the dispersion borrows the mean's stability, varying per sample only through the mean.
      * ``"nb_full_dispersion_shrunk_tied"`` is the mean-trend of the tied variant plus a penalized per-sample deviation from it.
      * ``"nb_full_dispersion_hierarchical"`` gives the per-sample deviation a proper Gaussian prior, ``deviation ~ Normal(0, sigma[gene]^2)``, and **learns** ``sigma`` for each gene.

  The last one is worth a paragraph, because the difference between it and ``"nb_full_dispersion_shrunk"`` is not a bigger or smaller penalty but whether the penalty is estimated at all.

  A penalty of ``shrinkage_lambda * deviation^2`` is the negative log of a Gaussian prior whose width is fixed at ``1/sqrt(2*shrinkage_lambda)``. One number therefore decides how hard **every** gene in **every** sample is pulled back — and that is the observed failure mode of ``"nb_full_dispersion_shrunk"``: it corrects the tail of the null and simultaneously makes the bulk far too conservative, because the pull that is right for an unstable gene is much too strong for a well-behaved one.

  The strength cannot simply be turned into a parameter, because a squared penalty has no normalizing constant: widening the prior would only ever lower the loss, so a free strength runs to no shrinkage at all. ``"nb_full_dispersion_hierarchical"`` adds the missing ``log sigma`` term, which is what makes ``sigma`` estimable, and estimates one per gene. Its two limits are the modules on either side of it — as ``sigma`` goes to zero it becomes ``"nb_feature_dispersion"`` (one dispersion per gene), and as ``sigma`` grows it becomes ``"nb_full_dispersion"`` (a free per-sample dispersion) — and which of those a gene sits nearer is decided by its own data rather than imposed.

  Every term is part of the joint log-likelihood the model already maximizes, so finding representations is unchanged in kind: the same MAP, with one more properly normalized prior in the objective.


   * ``"output_module_options"`` is a dictionary containing the options for the output module. The options that can be specified in this dictionary depend on the type of output module used.

      * For the Poisson output module (``"poisson"``):

         * ``"activation"`` is the activation function to use in the output module. This is a string that can take one of the following values:
               
            * ``"sigmoid"`` for sigmoid activation.
            * ``"softplus"`` for softplus activation.
         
      * For the negative binomial output module with r-values learned per gene (``"nb_feature_dispersion"``):

         * ``"activation"`` is the activation function to use in the output module. This is a string that can take one of the following values:
               
            * ``"sigmoid"`` for sigmoid activation.
            * ``"softplus"`` for softplus activation.
            
         * ``"r_init"`` is the value to use for initializing the r-values in the negative binomial output module. This is a positive integer that specifies the initial value of the r-values in the negative binomial distribution. If not specified, the default value is ``2``.
         
      * For the negative binomial output module with r-values learned per gene and sample (``"nb_full_dispersion"``):

         * ``"activation"`` is the activation function to use in the output module. This is a string that can take one of the following values:

            * ``"sigmoid"`` for sigmoid activation.
            * ``"softplus"`` for softplus activation.

      * For the shrunk and shrunk-and-tied variants (``"nb_full_dispersion_shrunk"``, ``"nb_full_dispersion_shrunk_tied"``), in addition to ``"activation"``:

         * ``"shrinkage_lambda"`` is how hard the per-sample deviation is pulled toward the baseline. It is a non-negative number: ``0`` recovers the plain ``"nb_full_dispersion"`` module, and a large value recovers the per-gene ``"nb_feature_dispersion"`` module. If not specified, the default value is ``0.5``.
         * ``"r_init"`` is the value the per-gene baseline dispersion starts at. If not specified, the default value is ``2``.

      * For the tied variant (``"nb_full_dispersion_tied"``), in addition to ``"activation"``:

         * ``"r_init"`` is the value the per-gene intercept of the dispersion-mean trend starts at. If not specified, the default value is ``2``.

      * For the hierarchical variant (``"nb_full_dispersion_hierarchical"``), in addition to ``"activation"``:

         * ``"sigma_init"`` is the width each gene's prior on the per-sample deviation starts at, in log-r units. It is a positive number, and it should be **small**: training then begins near the stable per-gene dispersion and widens only for the genes whose data ask for it, whereas starting wide begins at the free per-sample dispersion this module exists to move away from. If not specified, the default value is ``0.1``.
         * ``"sigma_min"`` is the smallest width the prior may take. It is needed because the ``log sigma`` term diverges at zero, and a gene whose per-sample deviations all vanish would otherwise send its own width there. It is ``0.01`` rather than something smaller because the floor states how tight a prior is meant to be believed: per-sample log-r-values move by about 0.2 in natural-log units between two runs differing only in a seed, so a width of 1e-3 calls an ordinary deviation a two-hundred-sigma event and returns a penalty near 1e8. If not specified, the default value is ``0.01``.
         * ``"r_init"`` is the value the per-gene baseline dispersion starts at. If not specified, the default value is ``2``.

        Note that there is no ``"shrinkage_lambda"`` here, and its absence is the point: the strength of the pull is what this module estimates rather than what it is told. Note also that its reported loss is **not comparable** to the other modules' — the prior's constant half-log-two-pi term is dropped, since it moves no gradient, but it does move the printed number.

* ``"scaling_factor"`` is how the scaling factor of a sample is computed - the number the decoder's predicted means are multiplied by to put them on the scale of the sample's own counts. This is a string that can take one of the following values:

   * ``"mean"`` for the mean count over all of the sample's genes. This is the default, and it is what every model built before this option existed was trained with.
   * ``"median"`` for the median count over all of the sample's genes.

  The mean is not robust to the few genes that take a large and variable share of a library. In GTEx, the thirteen mitochondrial genes - 0.09% of the 14,895 genes the model is trained on - take 14.49% of all the reads, and the share they take of a single library runs from 0.10% to 90.85%. That moves a sample's mean by up to a factor of eleven, and the mitochondrial fraction is a measure of how the sample was handled rather than of the tissue it came from. Over the same samples, those genes move the median by at most 3.7%.

  Two things are worth knowing before setting this.

  The first is that it belongs to the model rather than to a run of it, which is why it is here and not in the training configuration. The decoder is fitted against the scaling factor, and the median of a sample's counts is about a third of its mean, so a model trained with one and used with the other has every predicted mean wrong by about a factor of three - and nothing fails. The value is read back from this file by :func:`bulkdgd.ioutil.load_config_model` whenever the model is loaded, so that finding representations uses the one the training did.

  The second is that the median is only safe on a gene list that has been filtered to the genes that are expressed in the samples. A median is zero as soon as half of a sample's genes are zero, and a scaling factor of zero makes every predicted mean zero and the negative binomial undefined. :class:`bulkdgd.core.dataclasses.GeneExpressionDataset` raises rather than let a zero through.

  Note also that :meth:`bulkdgd.core.model.BulkDGD.impute` is implemented only for ``"mean"``. The scaling factor of a sample only some of whose genes were measured is solved for, and the equation it is solved from is one only the mean satisfies: a mean is a sum over the genes, so the unmeasured part of the sum can be filled in with the model's own expectation and the factor recovered. A median is not a sum, and there is no closed form.

* ``"dtype"`` is the precision the model's parameters are built in. This is a string that can take one of the following values:

   * ``"float32"`` for single precision. This is the default, and torch's own.
   * ``"float64"`` for double precision.

  This is not a preference that can be applied after the model is built. A module's parameters are made in whatever torch's default dtype is at the moment the module is constructed, and :meth:`torch.nn.Module.load_state_dict` copies a checkpoint *into* the parameters that are already there, casting as it goes - so a float64 checkpoint read into a model built in float32 gives a float32 decoder, and nothing says so. The Gaussian mixture does not go quietly, which is the only luck in it: ``tgmm`` keeps the tensors it is handed rather than copying into its own, so it stays float64 while the decoder becomes float32, and the first matrix multiply of the two raises ``mat1 and mat2 must have the same dtype``.

  So, like ``"scaling_factor"``, this belongs to the model and is read back from this file whenever the model is loaded: a model trained in double is read in double, without the caller having to set torch's default. The default is put back to what it was once the model is built, since it is global and a model asked for in double is not a reason for the rest of the program to be in double.


