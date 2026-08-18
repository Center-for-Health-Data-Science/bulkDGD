#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    finetuner.py
#
#    Execution of the fine-tuning schemes.
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
__doc__ = "Execution of the fine-tuning schemes."


#######################################################################


# Import from the standard library.
import copy
import hashlib
import os
from pathlib import Path
import subprocess

# Import from third-party libraries.
import pandas as pd
import torch
import yaml

# Import from 'bulkdgd'.
import bulkdgd
from bulkdgd import reproducibility
from bulkdgd.ioutil import configio
from . import componentfit
from . import finetuning


#######################################################################


def _hash_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _hash_items(items: list[str]) -> str:
    """Return a stable SHA-256 digest of ordered strings."""

    payload = "\n".join(items).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_frame(frame: pd.DataFrame) -> str:
    """Hash a numeric data frame's labels, dtypes, and values."""

    digest = hashlib.sha256()
    digest.update(_hash_items(frame.index.astype(str)).encode("ascii"))
    digest.update(
        _hash_items(frame.columns.astype(str)).encode("ascii"))
    digest.update(
        _hash_items(frame.dtypes.astype(str)).encode("ascii"))
    digest.update(str(frame.shape).encode("ascii"))
    digest.update(frame.to_numpy(copy = True).tobytes(order = "C"))

    return digest.hexdigest()


def _source_state() -> dict[str, object]:
    """Return package and source-checkout provenance."""

    repository = Path(__file__).resolve().parents[2]

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd = repository,
            check = True,
            capture_output = True,
            text = True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            cwd = repository,
            check = True,
            capture_output = True,
            text = True).stdout)
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
        dirty = None

    return {
        "bulkdgd_version" : str(bulkdgd.__version__),
        "torch_version" : str(torch.__version__),
        "git_commit" : commit,
        "git_dirty" : dirty}


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a Parquet file atomically."""

    temporary = path.with_name(path.name + ".tmp.parquet")
    df.to_parquet(temporary)
    os.replace(temporary, path)


def _write_yaml(data: dict[str, object], path: Path) -> None:
    """Write a YAML file atomically."""

    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("w") as handle:
        yaml.safe_dump(data, handle, sort_keys = False)

    os.replace(temporary, path)


def _resolve_rep_config(value: object) -> dict[str, object]:
    """Resolve the target representation configuration."""

    if value is None or isinstance(value, str):
        return configio.load_config_rep(config_file = value)

    config, errors, _ = configio.parse_config_rep(config = value)

    if errors:
        raise ValueError(
            "Target representation configuration errors: " +
            " ".join(errors))

    return config


def _validate_inputs(
        model,
        df_samples: pd.DataFrame,
        names_train: list[str],
        names_test: list[str],
        output_dir: Path,
        resume: bool) -> None:
    """Validate inputs before an artifact or optimizer is created."""

    if not model._is_trained:
        raise RuntimeError(
            "Fine-tuning requires a checkpoint-backed trained parent.")

    if model._latent_type != "tgmm":
        raise NotImplementedError(
            "Fine-tuning version one supports TGMM parents only.")

    latent_file = model._latent_initial_options.get("latent_pth_file")
    decoder_file = \
        model._decoder_initial_options.get("decoder_pth_file")

    if latent_file is None or decoder_file is None:
        raise RuntimeError(
            "Fine-tuning requires parent latent and decoder checkpoint "
            "paths in the model configuration.")

    if not df_samples.index.is_unique:
        raise ValueError("Target sample IDs must be unique.")

    if len(names_train) != len(set(names_train)) or \
            len(names_test) != len(set(names_test)):
        raise ValueError("Train and test sample IDs must be unique.")

    overlap = set(names_train) & set(names_test)

    if overlap:
        raise ValueError("Target train and test samples overlap.")

    missing = \
        [name for name in names_train + names_test
         if name not in df_samples.index]

    if missing:
        raise ValueError(
            "Target samples are missing from the count matrix: " +
            ", ".join(missing[:10]))

    genes = \
        [column for column in df_samples.columns
         if column.startswith("ENSG")]

    if genes != list(model.genes):
        raise ValueError(
            "Target gene columns must exactly equal the parent's genes "
            "in identity and order.")

    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"The fine-tuning output directory '{output_dir}' is not "
            "empty. Use a new directory or request resume.")


def _parent_paths(model) -> tuple[Path, Path]:
    """Return the resolved parent latent and decoder paths."""

    latent = Path(
        model._latent_initial_options["latent_pth_file"]).resolve()
    decoder = Path(
        model._decoder_initial_options["decoder_pth_file"]).resolve()

    return latent, decoder


def _build_derived_model(
        model,
        derived_latent,
        output_dir: Path) -> tuple[object, dict[str, object]]:
    """Save and rebuild a derived model without mutating its parent."""

    gmm_path = output_dir / "gmm.pth"
    decoder_path = output_dir / "dec.pth"
    derived_latent.save(str(gmm_path))
    torch.save(model.decoder.state_dict(), decoder_path)
    config = model._get_config_for_rebuilding()
    latent_options = copy.deepcopy(config["latent_options"])
    decoder_options = copy.deepcopy(config["decoder_options"])
    latent_options["n_components"] = \
        int(derived_latent.n_components)
    latent_options["latent_pth_file"] = str(gmm_path)
    decoder_options["decoder_pth_file"] = str(decoder_path)
    config["latent_options"] = latent_options
    config["decoder_options"] = decoder_options
    _write_yaml(data = config, path = output_dir / "model.yaml")
    child = model.__class__(**config, device = str(model.device))

    return child, config


#######################################################################


def run_add_gmm_components(
        model,
        df_samples: pd.DataFrame,
        names_train: list[str],
        names_test: list[str],
        config: dict[str, object],
        output_dir: str) -> finetuning.FineTuningResult:
    """Run anchored appended-component fine-tuning."""

    output = Path(output_dir).resolve()
    resume = config["checkpoint_options"]["resume"]
    _validate_inputs(
        model = model,
        df_samples = df_samples,
        names_train = names_train,
        names_test = names_test,
        output_dir = output,
        resume = resume)

    if resume:
        raise NotImplementedError(
            "Resume is added with the optimization-based schemes; "
            "anchored component fitting is deterministic and short.")

    output.mkdir(parents = True, exist_ok = True)
    seed = config["seed"]
    seed_state = reproducibility.set_seeds(
        seed = seed,
        deterministic = config["deterministic_algorithms"])
    rep_config = _resolve_rep_config(
        config["target_representation_options"]["config"])
    selected = names_train + names_test
    target = df_samples.loc[selected]
    parent_rep, _, _, _ = model.get_representations(
        df_samples = target,
        config_rep = rep_config)
    latent_columns = \
        [column for column in parent_rep.columns
         if column.startswith("latent_dim_")]
    z_train = torch.as_tensor(
        parent_rep.loc[names_train, latent_columns].to_numpy(),
        dtype = getattr(torch, model.dtype),
        device = model.device)
    options = config["scheme_options"]

    if options["outer_iterations"] != 1:
        raise NotImplementedError(
            "Fine-tuning version one supports exactly one anchored "
            "component-fitting iteration.")

    derived_latent, component_map, history = \
        componentfit.append_anchored_components(
            parent = model.latent,
            target_representations = z_train,
            n_new_components = options["n_new_components"],
            new_component_weight = options["new_component_weight"],
            initialization = options["initialization"],
            max_iter = options["max_iter"],
            tol = options["tol"],
            reg_covar = options["reg_covar"],
            seed = options["component_fit_seed"])
    child, _ = _build_derived_model(
        model = model,
        derived_latent = derived_latent,
        output_dir = output)
    child_rep, child_means, child_r_values, _ = \
        child.get_representations(
            df_samples = target,
            config_rep = rep_config)
    parent_rep_train = parent_rep.loc[names_train].copy()
    parent_rep_test = parent_rep.loc[names_test].copy()
    child_rep_train = child_rep.loc[names_train].copy()
    child_rep_test = child_rep.loc[names_test].copy()
    means_test = child_means.loc[names_test].copy()
    r_test = child_r_values.loc[names_test].copy() \
        if child_r_values is not None and \
        set(names_test).issubset(child_r_values.index) \
        else child_r_values
    metrics = pd.DataFrame(
        [{"metric" : "old_component_means_exact",
          "value" : float(torch.equal(
              child.latent.means_[:model.latent.n_components],
              model.latent.means_))},
         {"metric" : "old_covariance_exact",
          "value" : float(torch.equal(
              child.latent.covariances_,
              model.latent.covariances_))},
         {"metric" : "decoder_fixed",
          "value" : 1.0},
         {"metric" : "n_target_train",
          "value" : float(len(names_train))},
         {"metric" : "n_target_test",
          "value" : float(len(names_test))}])
    _write_parquet(parent_rep_train, output / \
                   "parent_representations_target_train.parquet")
    _write_parquet(parent_rep_test, output / \
                   "parent_representations_target_test.parquet")
    _write_parquet(child_rep_train, output / \
                   "child_representations_target_train.parquet")
    _write_parquet(child_rep_test, output / \
                   "child_representations_target_test.parquet")
    _write_parquet(means_test, output / \
                   "predicted_means_target_test.parquet")

    if r_test is not None:
        _write_parquet(r_test, output / \
                       "predicted_r_values_target_test.parquet")

    _write_parquet(history, output / "loss.parquet")
    _write_parquet(metrics, output / "metrics.parquet")
    _write_parquet(component_map, output / "component_map.parquet")
    _write_yaml(config, output / "fine_tuning.yaml")
    parent_latent, parent_decoder = _parent_paths(model)
    artifacts = \
        sorted(path for path in output.iterdir()
               if path.name != "fine_tuning_provenance.yaml")
    provenance = \
        {"fine_tuning_scheme" : "add_gmm_components",
         "parent_latent_path" : str(parent_latent),
         "parent_latent_sha256" : _hash_file(parent_latent),
         "parent_decoder_path" : str(parent_decoder),
         "parent_decoder_sha256" : _hash_file(parent_decoder),
         "sample_order_sha256" : _hash_items(selected),
         "target_train_sha256" : _hash_items(names_train),
         "target_test_sha256" : _hash_items(names_test),
         "target_counts_sha256" : _hash_frame(target),
         "gene_order_sha256" : _hash_items(list(model.genes)),
         "component_fit_seed" : options["component_fit_seed"],
         "result_semantics" :
             {"representations_train" : "child_reinferred",
              "representations_test" : "child_reinferred",
              "predicted_means" : "child_reinferred",
              "predicted_r_values" : "child_reinferred",
              "parent_representations_train" :
                  "parent_inferred_fit_coordinates",
              "parent_representations_test" :
                  "parent_inferred_comparator_coordinates"},
         "dtype" : model.dtype,
         "scaling_factor" : model.scaling_factor,
         "device" : str(model.device),
         "source_state" : _source_state(),
         "seed_state" : seed_state,
         "artifacts" :
             {path.name : _hash_file(path) for path in artifacts}}
    manifest_path = output / "fine_tuning_provenance.yaml"
    _write_yaml(provenance, manifest_path)

    return finetuning.FineTuningResult(
        model = child,
        representations_train = child_rep_train,
        representations_test = child_rep_test,
        replay_representations = None,
        predicted_means = means_test,
        predicted_r_values = r_test,
        loss = history,
        metrics = metrics,
        output_dir = str(output),
        manifest_path = str(manifest_path),
        fine_tuning_scheme = "add_gmm_components",
        stop_epoch = len(history),
        parent_representations_train = parent_rep_train,
        parent_representations_test = parent_rep_test)


def run_unimplemented(scheme: str):
    """Raise a clear error for a not-yet-promoted scheme."""

    raise NotImplementedError(
        f"The '{scheme}' API and configuration are reserved, but its "
        "optimizer is not implemented yet. Implement and test schemes "
        "in the documented promotion order.")


#######################################################################
