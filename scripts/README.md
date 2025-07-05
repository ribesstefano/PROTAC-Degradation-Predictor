# Training Models

## Dataset Specification

From the repository top level directory, run the following command to get the datasets reported in the paper:

```bash
cd scripts
pip install -r requirements.txt
python get_studies_datasets.py
```

For training on custom datasets, please refer to the class `PROTAC_Dataset` in the file [`protac_dataset.py`](../protac_degradation_predictor/protac_dataset.py). The class expects a Pandas dataframe, so plase assemble a file to be parsed into a Pandas DataFrame with the following columns:

| Column Name | Type | Description |
| --- | --- | --- |
| Smiles | str | The SMILES representation of the PROTAC molecule. |
| Uniprot | str | The Uniprot ID of the target protein. |
| E3 Ligase Uniprot | str | The Uniprot ID of the E3 ligase. |
| Cell Line Identifier | str | The cell line identifier as one reported in Cellosaurus. |
| `<active_label>` | bool | The activity label of the PROTAC molecule to be predicted by the model. |

The column `<active_label>` is set _"Active"_ as default in the `PROTAC_Dataset` class and in the `hyperparameter_tuning_and_training` function (see below for how to use it).

## Training on Custom Data

For training on custom datasets, please refer to the function `hyperparameter_tuning_and_training` in [`optuna_utils.py`](../protac_degradation_predictor/optuna_utils.py) and the file [`run_experiments.py`](../scripts/run_experiments.py) for inspiration on how to use the function.

An example of skeleton implementation is as follows:

```python
import protac_degradation_predictor as pdp
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Load train/val and test dataframes
train_val_df = pd.read_csv('path/to/custom_dataset.csv')
test_df = pd.read_csv('path/to/test_dataset.csv') # Load one of our test datasets

# NOTE: Make sure to avoid data leakage by removing leaking data in the train/val
# dataframe. Do NOT do remove/alter the test set, as it would impair comparison
# with our work. Data leakage can occur if the test set contains any combination
# of SMILES, Uniprot, E3 Ligase Uniprot, or Cell Line Identifier that is present
# in the train/val set too.

# Precompute Morgan fingerprints
unique_smiles = pd.concat([train_val_df, test_df])['Smiles'].unique().tolist()
smiles2fp = {s: np.array(pdp.get_fingerprint(s)) for s in unique_smiles}

# Load embedding dictionaries
protein2embedding = pdp.load_protein2embedding()
cell2embedding = pdp.load_cell2embedding()

# Setup Cross-Validation object
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pdp.hyperparameter_tuning_and_training(
    protein2embedding=protein2embedding,
    cell2embedding=cell2embedding,
    smiles2fp=smiles2fp,
    train_val_df=train_val_df,
    test_df=test_df,
    kf=kf,
    n_models_for_test=3,
    n_trials=100,
    max_epochs=20,
    logger_save_dir='../logs',
    logger_name=f'logs_{experiment_name}',
    study_filename=f'../reports/study_{experiment_name}.pkl',
)
```
