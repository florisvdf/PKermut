import tempfile
from pathlib import Path
from typing import Annotated
import typer
from rich.console import Console
import json
import os
from loguru import logger

import pandas as pd
from hydra import initialize_config_dir, compose

import torch

from pg_model.kermut_run import main as kermut_run
from pg_model.scripts.precompute_artifacts import precompute_artifacts
from pg_model.utils import (
    download_file_from_s3,
    download_artifacts_from_s3,
    prepare_hydra_configs,
    log_and_save_metrics,
)
from pg_model.utils import variant_sequence_to_mutations
from pg_model.constants import HYDRA_CONFIG_PATH, HYDRA_TEMP_CONFIG_PATH


CUDA_AVAILABLE = torch.cuda.is_available()


app = typer.Typer(
    help="Kermut model CLI",
    add_completion=True,
)

console = Console()


@app.command()
def train(
    dataset_name: Annotated[
        str,
        typer.Option(
            help="DMS id of the ProteinGym dataset",
        ),
    ],
    target: Annotated[
        str,
        typer.Option(
            help="Name of the column in the dataframe specifying the targets",
        ),
    ],
    data_artifact_path: Annotated[
        str,
        typer.Option(
            help="Path or uri to the directory storing the data artifacts including the dataset in csv format. "
            "If no dataset exists at this path, an uri pointing to the dataset must be provided.",
        ),
    ],
    reference_sequence: Annotated[
        str,
        typer.Option(
            help="Reference sequence of the dataset",
        ),
    ],
    output_path: Annotated[
        str,
        typer.Option(
            help="Path to which results will be saved",
        ),
    ],
    preferential: Annotated[
        bool,
        typer.Option(
            help="Whether to fit Kermut in preferential mode or in the default mode."
        ),
    ] = False,
    preference_sampling_strategy: Annotated[
        str,
        typer.Option(
            help="How to sample preferences when fitting Kermut in preferential mode."
        ),
    ] = "uniform_2.5",
    n_steps: Annotated[
        int,
        typer.Option(
            help="Number of optimization steps of the Gaussian Process",
        ),
    ] = 150,
    device: Annotated[
        str,
        typer.Option(
            help="PyTorch device to use",
        ),
    ] = "cpu",
    pdb_path: Annotated[
        str,
        typer.Option(
            help="Path or uri to a pdb file for the reference sequence. "
            "If provided, artifacts will be computed at data_artifact_path"
        ),
    ] = None,
    dataset_uri: Annotated[
        str,
        typer.Option(
            help="S3 uri to a dataset. If not passed, the dataset will be assumed to be in data_artifact_path."
            "This also means that this parameter must be set when downloading artifacts from s3.",
        ),
    ] = None,
):
    torch.set_default_dtype(torch.float32)
    if preferential:
        torch.set_default_dtype(torch.float64)

    with tempfile.TemporaryDirectory() as temp_dir:
        if data_artifact_path.startswith("s3://"):
            download_artifacts_from_s3(
                data_artifact_path, temp_dir, dataset_name=dataset_name
            )
            data_artifact_path = temp_dir

        data_path = str(Path(data_artifact_path) / f"datasets/{dataset_name}.csv")
        if dataset_uri is not None:
            console.print(f"Downloading dataset from {dataset_uri}")
            download_file_from_s3(dataset_uri, data_path,)
        else:
            console.print(
                f"No dataset uri passed, loading dataset from data artifact path"
            )

        # Make function
        df = pd.read_csv(data_path, index_col=None)
        if "mutant" not in df.columns:
            logger.info("No mutant column in the dataframe. Creating a copy with mutation information")
            df["mutant"] = df["sequence"].map(lambda x: variant_sequence_to_mutations(x, reference_sequence))
            variant_matches_reference = df["sequence"] == reference_sequence
            if variant_matches_reference.any():
                logger.error("Dataframe contains variants identical to reference!")
                logger.error(df[variant_matches_reference])
            df.to_csv(data_path, index=False)

        if pdb_path is not None:
            console.print("PDB file passed, computing necessary artifacts")
            if pdb_path.startswith("s3://"):
                new_pdb_path = str(
                        Path(data_artifact_path) / f"structures/pdbs/{dataset_name}.pdb"
                    )
                download_file_from_s3(
                    pdb_path,
                    new_pdb_path,
                )
                pdb_path = new_pdb_path
            _ = precompute_artifacts(
                dataset_name=dataset_name,
                data_path=data_path,
                pdb_file=pdb_path,
                reference_sequence=reference_sequence,
                artifact_dir=data_artifact_path,
                device=device,
            )

        params_to_update = {
            "dataset_name": dataset_name,
            "target": target,
            "data_artifact_path": data_artifact_path,
            "DMS_input_folder": str(Path(data_artifact_path) / "datasets"),
            "embedding_path": str(Path(data_artifact_path) / "embeddings"),
            "conditional_probs_path": str(Path(data_artifact_path) / "conditional_probs"),
            "reference_sequence": reference_sequence,
            "output_path": output_path,
            "n_steps": n_steps,
            "use_gpu": True if device == "cuda" else False,
            "preferential": True if preferential else False,
            "preference_sampling_strategy": preference_sampling_strategy,
        }
        prepare_hydra_configs(
            HYDRA_CONFIG_PATH, HYDRA_TEMP_CONFIG_PATH, params_to_update
        )

        with initialize_config_dir(config_dir=str(HYDRA_TEMP_CONFIG_PATH)):
            cfg = compose(config_name="benchmark")
            kermut_run(cfg)

    results = pd.read_csv(Path(output_path) / f"{dataset_name}.csv")
    results.rename(columns={"y_var": "y_pred_var"}, inplace=True)
    results[["sequence", "split", "y", "y_pred", "y_pred_var"]].to_csv(
        Path(output_path) / "predictions.csv"
    )
    log_and_save_metrics(results, str(output_path))
    console.print("Finished")


def in_sagemaker() -> bool:
    return os.path.exists("/opt/ml/input/config/hyperparameters.json")


def load_sagemaker_hyperparameters():
    path = "/opt/ml/input/config/hyperparameters.json"

    def auto_cast(x):
        if isinstance(x, str):
            lx = x.lower()
            if lx == "true":
                return True
            if lx == "false":
                return False
        try:
            return int(x)
        except (ValueError, TypeError):
            pass
        try:
            return float(x)
        except (ValueError, TypeError):
            pass
        return x

    with open(path) as f:
        raw_hps = json.load(f, parse_int=int, parse_float=float)
        hyperparameters = {k: auto_cast(v) for k, v in raw_hps.items()}
    return hyperparameters


@app.command()
def ping():
    console.print("pong")


if __name__ == "__main__":
    if in_sagemaker():
        hps = load_sagemaker_hyperparameters()
        train(**hps)
    else:
        app()
