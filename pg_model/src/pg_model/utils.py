import shutil

import pandas as pd
import yaml
import json
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse
from loguru import logger


import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

import boto3

s3 = boto3.client("s3")


def get_bucket_and_key(uri: str) -> Tuple[str, str]:
    parsed_uri = urlparse(uri)
    bucket = parsed_uri.netloc
    key = parsed_uri.path.lstrip("/")
    if parsed_uri.query:
        key += f"?{parsed_uri.query}"
    return bucket, key


def download_file_from_s3(uri: str, out_file: str):
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    bucket, key = get_bucket_and_key(uri)
    s3.download_file(bucket, key, str(out_file))


def download_artifacts_from_s3(
    uri: str, path: str, dataset_name: str,
) -> None:
    relative_paths = [
        f"conditional_probs/{dataset_name}.npy",
        f"embeddings/{dataset_name}.h5",
        f"structures/coords/{dataset_name}.npy",
        f"zero_shot_fitness_predictions/{dataset_name}.csv",
    ]
    bucket, base_key = get_bucket_and_key(uri)
    for relative_path in relative_paths:
        file_path = Path(path) / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_key = Path(base_key) / relative_path
        logger.info(f"Downloading {file_key} from {bucket} to {file_path}")
        s3.download_file(bucket, str(file_key), str(file_path))


def prepare_hydra_configs(
    hydra_src_config_path: str, hydra_dest_config_path: str, params_to_update: dict
) -> None:
    shutil.copytree(hydra_src_config_path, hydra_dest_config_path, dirs_exist_ok=True)
    with open(Path(hydra_dest_config_path) / "data/paths.yaml", "r") as fp:
        hydra_paths_config = yaml.safe_load(fp)
    with open(Path(hydra_dest_config_path) / "data/dataset.yaml", "r") as fp:
        hydra_datasets_config = yaml.safe_load(fp)
    with open(Path(hydra_dest_config_path) / "benchmark.yaml", "r") as fp:
        hydra_benchmark_config = yaml.safe_load(fp)

    hydra_paths_config["data_dir"] = params_to_update["data_artifact_path"]
    hydra_paths_config["sequence_col"] = "sequence"
    hydra_paths_config["paths"]["embeddings"] = params_to_update["embedding_path"]
    hydra_paths_config["paths"]["conditional_probs"] = params_to_update["conditional_probs_path"]
    hydra_paths_config["paths"]["DMS_input_folder"] = params_to_update["DMS_input_folder"]
    hydra_paths_config["paths"]["output_folder"] = params_to_update["output_path"]
    hydra_paths_config["target_col"] = params_to_update["target"]

    # WIP: Setting other Kermut params (types of kernels, etc)
    hydra_benchmark_config["optim"]["n_steps"] = params_to_update["n_steps"]
    hydra_benchmark_config["DMS_id"] = params_to_update["dataset_name"]
    hydra_benchmark_config["target_seq"] = params_to_update["reference_sequence"]
    hydra_benchmark_config["use_gpu"] = params_to_update["use_gpu"]
    hydra_benchmark_config["preferential"] = params_to_update["preferential"]
    hydra_benchmark_config["preference_sampling_strategy"] = params_to_update[
        "preference_sampling_strategy"
    ]

    with open(Path(hydra_dest_config_path) / "data/paths.yaml", "w") as fp:
        yaml.dump(hydra_paths_config, fp)
    with open(Path(hydra_dest_config_path) / "data/dataset.yaml", "w") as fp:
        yaml.dump(hydra_datasets_config, fp)
    with open(Path(hydra_dest_config_path) / "benchmark.yaml", "w") as fp:
        yaml.dump(hydra_benchmark_config, fp)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.square((y_true - y_pred)))


def r_square(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return r2_score(y_true=y_true, y_pred=y_pred)
    except ValueError:
        return np.nan


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) <= 1 or len(np.unique(y_pred)) <= 1:
        return np.float64(0.0)
    return spearmanr(y_true, y_pred).correlation


def log_and_save_metrics(results: pd.DataFrame, output_dir: str) -> None:
    metrics = {
        "mse": mse,
        "r_square": r_square,
        "spearman": spearman,
    }
    metric_stats = {}
    for split in results["split"].unique():
        split_df = results[results["split"] == split]
        for metric_name, metric in metrics.items():
            value = metric(split_df["y"].values, split_df["y_pred"].values)
            metric_stats[f"{split}_{metric_name}"] = value
            logger.info(f"{split}_{metric_name}: {value:.4f}")
    with open(Path(output_dir) / "metrics.json", "w") as fh:
        json.dump(metric_stats, fh)


def variant_sequence_to_mutations(variant: str, reference: str) -> str:
    return ":".join(
        [
            f"{aa_ref}{pos + 1}{aa_var}"
            for pos, (aa_ref, aa_var) in enumerate(zip(reference, variant))
            if aa_ref != aa_var
        ]
    )
