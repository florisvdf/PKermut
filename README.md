# PKermut

PKermut is a preferential version of the Kermut Gaussian process model as reported in 
[Kermut: Composite kernel regression for protein variant effects](https://doi.org/10.48550/arXiv.2407.00002). 
This repository is a fork of the [original repository](https://github.com/petergroth/kermut) 
and contains some modifications that allows users to train and run inference with PKermut: 
Kermut trained with a [preferential objective](https://botorch.readthedocs.io/en/latest/models.html#module-botorch.models.pairwise_gp).


# Structure

This codebase tries to preserve as much as possible of the original implementation, which 
is stored in the `kermut` directory. `pg_model` acts as a wrapper project, providing a 
single entrypoint for training and evaluation, and also contains utilities to provide 
preferential training support, such as preference pair sampling algorithms. 

`pg_model` also contains standalone scripts for precomputing artifacts. See 
`pg_model/src/pg_model/scripts`.

# Installation

```shell
uv sync
```

[ProteinMPNN](https://github.com/dauparas/ProteinMPNN) must be installed to compute 
structure-conditioned amino acid distributions. This can be done by cloning the repository 
and passing its path to an environment variable named `PROTEINMPNN_DIR`.

# Usage

(P)Kermut can be trained providing a dataframe with a `sequence` column storing sequences, 
a `split` column storing `train` and `test` values and an arbitrarily named column storing 
the values to model stored at `data_artifact_path/datasets/<dataset_name>.csv`. In addition, either a PDB structure must be provided, prompting 
(P)Kermut to compute all necessary sequence and structure representations, or those 
representations must be stored at `data_artifact_path`.

```python
from pg_model.__main__ import train


train(
    data_artifact_path="path/to/my/artifacts", # Where should artifacts be stored
    target="my_target", # Name of the column in the dataframe containing values to fit
    dataset_name="my_dataset", # Name of the dataset
    reference_sequence="MYREFERENCESEQWENCE", # Sequence of the reference protein
    pdb_path="path/to/my_structure.pdb", # Path to a .pdb file of the reference protein
    output_path="path/to/my/outputs", # Where should outputs be stored
    n_steps=150, # Number of training iterations
    device="gpu", # Device to use 
    preferential=True, # True -> Train in preferential mode, False -> Train original version of Kermut
    preference_sampling_strategy="uniform_2.5", # Average degree to which to uniformly subsample the preference graph when training in preferential mode. In this case 2.5 
)
```
This wil train (P)Kermut and evaluate in on both the training and test set. Predictions and metrics
are saved to `path/to/my/outputs`.

