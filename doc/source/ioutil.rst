``ioutil`` - utilities for I/O operations
=========================================

.. automodule:: bulkdgd.ioutil
  
   .. autofunction:: bulkdgd.ioutil.load_config_model

   .. autofunction:: bulkdgd.ioutil.load_config_rep

   .. autofunction:: bulkdgd.ioutil.load_config_train

   .. autofunction:: bulkdgd.ioutil.load_config_dim_red

   .. autofunction:: bulkdgd.ioutil.load_config_plot

   .. autofunction:: bulkdgd.ioutil.load_config_genes

   .. autofunction:: bulkdgd.ioutil.load_decoder_outputs

   .. autofunction:: bulkdgd.ioutil.load_samples

   .. autofunction:: bulkdgd.ioutil.load_representations

   .. autofunction:: bulkdgd.ioutil.save_representations

   .. autofunction:: bulkdgd.ioutil.save_samples

   .. autofunction:: bulkdgd.ioutil.save_decoder_outputs

   .. autofunction:: bulkdgd.ioutil.preprocess_samples

   .. autofunction:: bulkdgd.ioutil.save_table

   .. autofunction:: bulkdgd.ioutil.load_table

How a table is written
----------------------

Every table the package writes goes through :func:`bulkdgd.ioutil.save_table`, and the format follows the **suffix of the path it is given**: ``.parquet`` (or ``.pq``) is written as Parquet, and anything else as delimited text. Nothing is silently rewritten under a caller that asked for something else, and a caller that wants the lossless format simply names it.

The decision is in one place because it used to be made file by file, which is how it came to hold for the training outputs and not for the representations, then for the representations and not for the differential expression. Selective application is the failure mode.

Parquet is the format worth asking for when the numbers are going to be compared. These tables carry ``float64`` activations, predicted means and r-values, and a ``float64`` written as text does not come back as the number that was written: the decimal form is rounded, and a round trip moves the value by up to about 1e-12. That is invisible in a printed column and fatal to anything that compares two runs.

The command-line tools keep ``.csv`` in their default output names, so a run that says nothing about the format still gets text and everything that reads data already on disk keeps working. Passing an output name ending in ``.parquet`` is all it takes to get Parquet instead.
