# The command-line interface

`bulkDGD` is structured as an importable Python package.

However, a command-line interface is provided for some of the most common tasks `bulkDGD` is used for.

This interface consists of a series of executables installed together with the package.

## Executables

### `dgd_get_recount3_data`

This executable is devoted to retrieving RNA-seq data (and associated metadata) from the [Recount3 platform](https://rna.recount.bio/).

So far, the program supports the retrieval of data for samples belonging to the [GTEx](https://gtexportal.org/home/) and [TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga) projects.

The executable allows selecting samples for a single tissue (for GTEx data) or cancer type (for TCGA) and filtering the samples according to the associated metadata. The filtering is performed using a query string in the format supported by the [`pandas.DataFrame.query()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html) method, which can be passed from the command line as a string or as a plain text file containing the string. Metadata fields on which it is possible to filter the samples differ between GTEx and TCGA samples. A list of the available fields is available in `bulkDGD/data/recount3/gtex_metadata_fields.txt` for GTEx samples and in `bulkDGD/data/recount3/tcga_metadata_fields.txt` for TCGA samples.

The main output of `dgd_get_recount3_data` is a CSV file containing the RNA-seq data retrieved from Recount3 for the samples of interest. The rows represent the samples, while the columns contain the genes, which are identified by their Ensembl IDs.

The user also has the option to save the original compressed (`.gz`) files containing the RNA-seq data and the metadata associated with the samples.

#### Command line

```
dgd_get_recount3_data [-h] -ip {gtex,tcga} -is INPUT_SAMPLES_CATEGORY [-o OUTPUT_CSV] [-d WORK_DIR] [-sg] [-sm] [-qs QUERY_STRING] [-lf LOG_FILE] [-lc] [-v] [-vv]
```

#### Options

| Option                            | Description                                                  |
| --------------------------------- | ------------------------------------------------------------ |
| `-h`, `--help`                    | Show the help message and exit.                              |
| `-ip`, `--input-project-name`     | The name of the Recount3 project for which samples will be retrieved. The available projects are: `"gtex"`, `"tcga"`. |
| `-is`, `--input-samples-category` | The category of samples for which RNA-seq data will be retrieved. For GTEx data, this is the name of the tissue the samples belong to. For TCGA data, this is the type of cancer the samples are associated with. |
| `-o`, `--output-csv`              | The name of the output CSV file containing the data frame with the RNA-seq data for the samples. The file will be written in the working directory. The default file name is `{input_project_name}_{input_samples_category}.csv`. |
| `-d`, `--work-dir`                | The working directory. The default is the current working directory. |
| `-sg`, `--save-gene-sums`         | Save the original GZ file containing the RNA-seq data for the samples. The file will be saved in the working directory and named `{input_project_name}_{input_samples_category}_gene_sums.gz`. |
| `-sm`, `--save-metadata`          | Save the original GZ file containing the metadata for the samples. The file will be saved in the working directory and named `{input_project_name}_{input_samples_category}_metadata.gz`. |
| `-qs`, `--query-string`           | The string that will be used to filter the samples according to their associated metadata using the `pandas.DataFrame.query()` method. The option also accepts a plain text file containing the string since it can be long for complex queries. |
| `-lf`, `--log-file`               | The name of the log file. The file will be written in the working directory. The default file name is `dgd_get_recount3_data.log`. |
| `-lc`, `--log-console`            | Show log messages also on the console.                       |
| `-v`, `--logging-verbose`         | Enable verbose logging (INFO level).                         |
| `-vv`, `--logging-debug`          | Enable maximally verbose logging for debugging purposes (DEBUG level). |

### `dgd_preprocess_samples`

This executable allows users to preprocess new samples to use within the DGD model.

It expects as input a CSV file containing a data frame with the gene expression data for the new samples.

Each row must represent a sample, while each column must represent a gene identified by its Ensembl ID or additional information about the samples. The first column is expected to contain the unique names, IDs, or indexes of the samples.

In detail, sample preprocessing consists of the following steps:

1. Removing duplicated samples.
2. Removing samples containing missing values for the expression of some genes.
3. Excluding all data for genes that are not included in the DGD model. A plain text file containing the list of the genes used in the model is available in `bulkDGD/data/model/training_genes.txt`.
4. Adding a count of 0 for all genes which are not found in the input samples but are part of the set of genes used to train the DGD model.
5. Sorting the genes in the order expected by the DGD model.

The program will exit with an error if it finds duplicated genes.

Preprocessing new samples is a critical step before trying to find the samples' best representations in latent space using the DGD model (with the [`dgd_get_representations`](#dgd_get_representations) executable).

The main output of `dgd_preprocess_samples` is a CSV file containing the preprocessed samples.

The Ensembl IDs of the genes found in the input samples but not present in the gene set used to train the DGD model, if any, are written in a separate output file in plain text format, with one gene per line. 

Furthermore, if any gene present in the gene set used to train the DGD model is not found in the genes for which counts are available in the input samples, the gene will be written to the output CSV file containing the preprocessed samples with a count of 0 for all samples. These "missing" genes are also written in a separate plain text file containing one gene per line.

#### Command line

```
dgd_preprocess_samples [-h] -i INPUT_CSV [-os OUTPUT_CSV_SAMPLES] [-oe OUTPUT_TXT_GENES_EXCLUDED] [-om OUTPUT_TXT_GENES_MISSING] [-d WORK_DIR] [-lf LOG_FILE] [-lc] [-v] [-vv]
```

#### Options

| Option                               | Description                                                  |
| ------------------------------------ | ------------------------------------------------------------ |
| `-h`, `--help`                       | Show the help message and exit.                              |
| `-i`, `--input-csv`                  | The input CSV file containing a data frame with the samples to be preprocessed. |
| `-os`, `--output-csv-samples`        | The name of the output CSV file containing the data frame with the preprocessed samples. The file will be written in the working directory. The default file name is `samples_preprocessed.csv`. |
| `-oe`, `--output-txt-genes-excluded` | The name of the output plain text file containing the list of genes whose expression data are excluded from the data frame with the preprocessed samples. The file will be written in the working directory. The default file name is `genes_excluded.txt`. |
| `-om`, `--output-txt-genes-missing`  | The name of the output plain text file containing the genes for which no available expression data are found in the input data frame. A default count of 0 is assigned to these genes in the output data frame with the preprocessed samples. The file will be written in the working directory. The default file name is `genes_missing.txt`. |
| `-d`, `--work-dir`                   | The working directory. The default is the current working directory. |
| `-lf`, `--log-file`                  | The name of the log file. The file will be written in the working directory. The default file name is `dgd_preprocess_samples.log`. |
| `-lc`, `--log-console`               | Show log messages also on the console.                       |
| `-v`, `--logging-verbose`            | Enable verbose logging (INFO level).                         |
| `-vv`, `--logging-debug`             | Enable maximally verbose logging for debugging purposes (DEBUG level). |

### `dgd_get_representations`

This executable allows you to get representations in the latent space defined by the DGD model for new samples.

`dgd_get_representations` takes as input a CSV file containing a data frame with the expression data (RNA-seq read counts) for the samples of interest. The program expects the input data frame to display the different samples as rows and the genes as columns so that each cell contains the expression data for a gene in a sample. The genes' names are expected to be their Ensembl IDs.

The samples passed to the program must have expression data only for the genes included in the DGD model, and the genes should be sorted according to the order expected from the DGD model. A file containing the sorted list of these genes can be found in the `bulkDGD/data/model/training_genes.txt` file.

It is recommended to perform this preprocessing step with [`dgd_preprocess_samples`](#dgd_preprocess_samples).

`dgd_get_representations` also needs a YAML configuration file specifying the DGD model parameters, an example of which can be found in `bulkDGD/configs/model`.

The executable also needs a configuration file defining the data loading and optimization options for finding the representations, an example of which is available in `bulkDGD/configs/representations`.

To run the executable, you also need the trained DGD model, which comes in three PyTorch files:

* A file containing the trained Gaussian mixture model. The file `bulkDGD/data/model/gmm.pth` is loaded by default unless a different path is specified in the configuration file. 
* A file containing the trained representation layer. The file `bulkDGD/data/model/rep.pth` is loaded by default unless a different path is specified in the configuration file.
* A file containing the trained deep generative decoder. Since the file is too big to be hosted on GitHub, you can find it [here](https://drive.google.com/file/d/1SZaoazkvqZ6DBF-adMQ3KRcy4Itxsz77/view?usp=sharing). Save it locally and specify its path in the configuration file or in `bulkDGD/data/model` with the name `dec.pth`. This latter strategy would ensure the automatic loading of the file every time you launch the `dgd_get_representations` executable without having to specify its path in the configuration file. 

`dgd_get_representations` produces two output files:

* A CSV file containing a data frame with the best representations found. Here, the rows represent the samples for which the representations were found, while the columns represent the dimensions of the latent space where the representations live, the per-sample loss, and any extra piece of information about the representations that was found in the input data frame.
* A CSV file containing a data frame with the outputs produced by the decoder when the best representations found for the samples are passed through it. Here, the rows represent the samples, and the columns represent the genes (identified by their Ensembl IDs).

#### Command line

```
dgd_get_representations [-h] -i INPUT_CSV [-or OUTPUT_CSV_REP] [-od OUTPUT_CSV_DEC] -cm CONFIG_FILE_MODEL -cr CONFIG_FILE_REP [-d WORK_DIR] [-lf LOG_FILE] [-lc] [-v] [-vv]
```

#### Options

| Option                       | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| `-h`, `--help`               | Show the help message and exit.                              |
| `-i`, `--input-csv`          | The input CSV file containing a data frame with gene expression data for the samples for which a representation in latent space should be found. |
| `-or`, `--output-csv-rep`    | The name of the output CSV file containing a data frame with the representation of each input sample in latent space. The file will be saved in the working directory. The default file name is `representations.csv`. |
| `-od`, `--output-csv-dec`    | The name of the output CSV file containing the data frame with the decoder output for each input sample. The file will be written in the working directory. The default tile name is `decoder_outputs.csv`. |
| `-cm`, `--config-file-model` | The YAML configuration file specifying the DGD model parameters and files containing the trained model. If it is a name without extension, it is assumed to be the name of a configuration file in `$INSTALLDIR/bulkDGD/configs/model`. |
| `-cr`, `--config-file-rep`   | The YAML configuration file containing the options for data loading and optimization when finding the best representations.  If it is a name without extension, it is assumed to be the name of a configuration file in `$INSTALLDIR/bulkDGD/configs/representations`. |
| `-d`, `--work-dir`           | The working directory. The default is the current working directory. |
| `-lf`, `--log-file`          | The name of the log file. The file will be written in the working directory. The default file name is `dgd_get_representations.log`. |
| `-lc`, `--log-console`       | Show log messages also on the console.                       |
| `-v`, `--logging-verbose`    | Enable verbose logging (INFO level).                         |
| `-vv`, `--logging-debug`     | Enable maximally verbose logging for debugging purposes (DEBUG level). |

### `dgd_perform_dea`

This executable can be used to perform differential expression analysis (DEA) of genes between a "treated" sample (for instance, a cancer sample) against an "untreated" or "control" sample.

Within the context of the DGD model, the DEA is intended between a "treated" experimental sample and a "control" sample which is the model's decoder's output for the best representation of the "treated" sample in latent space. The decoder output for the best representation of the "treated" sample, therefore, acts as an *in silico* control sample.

This approach was first presented in the work of Prada-Luengo, Schuster, Liang, and coworkers [^1].

`dgd_perform_dea` expects three inputs. First, a CSV file containing a data frame set of experimental "treated" samples. The program assumes that each row represents a sample, each column represents a gene, and each cell of the data frame contains the read counts for a gene in a specific sample. Then, the program expects a data frame with the *in silico* "control" samples, This data frame is structured as the one containing the "treated" samples and can be obtained by running the [`dgd_get_representations`](#dgd_get_representations) executable on the "treated" sample.

The output of `dgd_perform_dea` is a CSV file containing the results of the differential expression analysis. Here, the p-values, q-values (adjusted p-values), and log2-fold changes relative to each gene's differential expression are reported.

To speed up performing DEA on a set of samples, `dgd_perform_dea` takes advantage of the [Dask](https://www.dask.org/) Python package to parallelize the calculations.

#### Command line

```
[-h] -is INPUT_CSV_SAMPLES -id INPUT_CSV_DEC [-op OUTPUT_CSV_PREFIX] -cm CONFIG_FILE_MODEL [-pr P_VALUES_RESOLUTION] [-qa Q_VALUES_ALPHA] [-qm Q_VALUES_METHOD] [-d WORK_DIR] [-n N_PROC] [-lf LOG_FILE] [-lc] [-v] [-vv]
```

#### Options

| Option                         | Description                                                  |
| ------------------------------ | ------------------------------------------------------------ |
| `-h`, `--help`                 | Show the help message and exit.                              |
| `-is`, `--input-csv-samples`   | The input CSV file containing a data frame with the gene expression data for the samples |
| `-id`, `--input-csv-dec`       | The input CSV file containing the data frame with the decoder output for each sample's best representation found with `dgd_get_representations`. |
| `-op`, `--output-csv-prefix`   | The prefix of the output CSV file(s) that will contain the results of the differential expression analysis. Since the analysis will be performed for each sample, one file per sample will be created. The files' names will have the form `{output_csv_prefix}{sample_name}.csv`. The default prefix is `dea_`. |
| `-cm`, `--config-file-model`   | The YAMl configuration file specifying the DGD model parameters and files containing the trained model. If it is a name without extension, it is assumed to be the name of a configuration file in `$INSTALLDIR/bulkDGD/configs/model`. |
| `-pr`, `--p-values-resolution` | The resolution at which to sum over the probability mass function to compute the p-values. The lower the resolution, the more accurate the calculation. A resolution of 1 corresponds to an exact calculation of the p-values. The default is 1. |
| `-qa`, `--q-values-alpha`      | The alpha value used to calculate the q-values (adjusted p-values). The default is 0.05. |
| `-qm`, `--q-values-method`     | The method used to calculate the q-values (i.e., to adjust the p-values). The default is `"fdr_bh"`. The available methods can be found in the documentation of `statsmodels.stats.multitest.multipletests`, which is used to perform the calculation. |
| `-d`, `--work-dir`             | The working directory. The default is the current working directory. |
| `-n`, `--nproc`                | The number of processes to start (the higher the number, the faster the execution of the program). The default number of processes started is 1. |
| `-lf`, `--log-file`            | The name of the log file. The file will be written in the working directory. The default file name is `dgd_perform_dea.log`. |
| `-lc`, `--log-console`         | Show log messages also on the console.                       |
| `-v`, `--logging-verbose`      | Enable verbose logging (INFO level).                         |
| `-vv`, `--logging-debug`       | Enable maximally verbose logging for debugging purposes (DEBUG level). |

### `dgd_perform_pca`

This executable performs a two-dimensional principal component analysis on a set of given representations found with [ `dgd_get_representations`](#dgd_get_representations).

`dgd_perform_pca` expects in input a CSV file containing a data frame with the representations. Each row of the data frame should contain data for a different representation, and each column should contain either the values of the representations along a dimension of the latent space or additional information about the representations (for instance, the associated loss). The program expects the columns containing the values of the representations to be named `latent_dim_*` where `*` represents an integer from 1 to $$D$$, with $$D$$ being the dimensionality of the latent space the representations live in.

The executable produces two outputs:

* A CSV file containing a data frame with the results of the PCA for the representations.
* A file containing a scatter plot of the results of the PCA.

The executable can also take in input a configuration file specifying the plot's aesthetics and output format. If not provided, the default configuration file `bulkDGD/configs/plot/config_pca_scatter.yaml` is used to generate the plot.

#### Command line

```
dgd_perform_pca [-h] -i INPUT_CSV [-oc OUTPUT_CSV_PCA] [-op OUTPUT_PLOT_PCA] [-cp CONFIG_FILE_PLOT] [-gc GROUPS_COLUMN] [-d WORK_DIR] [-lf LOG_FILE] [-lc] [-v] [-vv]
```

#### Options

| Option                      | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| `-h`, `--help`              | Show the help message and exit.                              |
| `-i`, `--input-csv`         | The input CSV file containing the data frame with the representations. |
| `-oc`, `--output-csv-pca`   | The name of the output CSV file containing the results of the PCA. The file will be written in the working directory. The default file name is `pca.csv`. |
| `-op`, `--output-plot-pca`  | The name of the output file containing the plot displaying the results of the PCA. The file will be written in the working directory. The default file name is `pca.pdf`. |
| `-cp`, `--config-file-plot` | The YAML configuration file specifying the aesthetics of the plot and the plot's output format. If not provided, the default configuration file (`$INSTALLDIR/configs/plot/config_pca_scatter.yaml`) will be used. |
| `-gc`, `--groups-column`    | The name/index of the column in the input data frame containing the groups by which the samples will be colored in the output plot. By default, the program assumes that no such column is present. |
| `-d`, `--work-dir`          | The working directory. The default is the current working directory. |
| `-lf`, `--log-file`         | The name of the log file. The file will be written in the working directory. The default file name is `dgd_perform_pca.log`. |
| `-lc`, `--log-console`      | Show log messages also on the console.                       |
| `-v`, `--logging-verbose`   | Enable verbose logging (INFO level).                         |
| `-vv`, `--logging-debug`    | Enable maximally verbose logging for debugging purposes (DEBUG level). |

### `dgd_get_probability_density`

For a set of given representations and the Gaussian mixture model (GMM) modeling the representation space in the DGD, this executable outputs the probability density of each representation for each GMM component.

This is useful, for instance, to identify a representative sample for each GMM component, namely the one having the highest probability for that component with respect to all samples under consideration.

`dgd_get_probability_density` takes as input a CSV file containing the representations for a set of samples formatted as the one produced by [`dgd_get_representations`](#dgd_get_representations). This CSV file stores a data frame where each row represents a sample's representation, and each column represents a dimension of the latent space where the representations live.

#### Command line

```
[-h] -i INPUT_CSV [-or OUTPUT_CSV_PROB_REP] [-oc OUTPUT_CSV_PROB_COMP] -cm CONFIG_FILE_MODEL [-d WORK_DIR] [-lf LOG_FILE] [-lc] [-v] [-vv]
```

#### Options

| Option                          | Description                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| `-h`, `--help`                  | Show the help message and exit.                              |
| `-i`, `--input-csv`             | The input CSV file containing the data frame with the representations. |
| `-or`, `--output-csv-prob-rep`  | The name of the output CSV file containing, for each representation, its probability density for each of the Gaussian mixture model's components, the maximum probability density found, the component the maximum probability density comes from, and the label of the tissue the input sample belongs to. The file will be written in the working directory. The default file name is `probability_density_representations.csv`. |
| `-oc`, `--output-csv-prob-comp` | The name of the output CSV file containing, for each component of the Gaussian mixture model, the representation(s) having the maximum probability density with respect to it. The file will be written in the working directory. The default file name is `probability_density_components.csv`. |
| `-cm`, `--config-file-model`    | The YAML configuration file specifying the DGD model parameters and files containing the trained model. If it is a name without extension, it is assumed to be the name of a configuration file in `$INSTALLDIR/bulkDGD/configs/model`. |
| `-d`, `--work-dir`              | The working directory. The default is the current working directory. |
| `-lf`, `--log-file`             | The name of the log file. The file will be written in the working directory. The default file name is `dgd_get_probability_density.log`. |
| `-lc`, `--log-console`          | Show log messages also on the console.                       |
| `-v`, `--logging-verbose`       | Enable verbose logging (INFO level).                         |
| `-vv`, `--logging-debug`        | Enable maximally verbose logging for debugging purposes (DEBUG level). |

## References

[^1]: Prada-Luengo, Inigo, et al. "N-of-one differential gene expression without control samples using a deep generative model." *bioRxiv* (2023): 2023-01.