# PKermut


# Installation

# Structure

# Usage

(P)Kermut can be trained providing a dataframe with a `sequence` column storing sequences, 
a `split` column storing `train` and `test` values and an arbitrarily named column storing 
the values to model. In addition, either a PDB structure must be provided, prompting 
(P)Kermut to compute all necessary sequence and structure representations, or a directory 
containing those representations must be provided.

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

