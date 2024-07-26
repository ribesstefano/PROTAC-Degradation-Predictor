import os
import sys
from collections import defaultdict
import warnings
import logging
from typing import Literal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import protac_degradation_predictor as pdp

import pytorch_lightning as pl
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from jsonargparse import CLI
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Ignore UserWarning from Matplotlib
warnings.filterwarnings("ignore", ".*FixedLocator*")
# Ignore UserWarning from PyTorch Lightning
warnings.filterwarnings("ignore", ".*does not have many workers.*")


root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

def main(
    active_col: str = 'Active (Dmax 0.6, pDC50 6.0)',
    n_trials: int = 100,
    fast_dev_run: bool = False,
    test_split: float = 0.1,
    cv_n_splits: int = 5,
    max_epochs: int = 100,
    force_study: bool = False,
    experiments: str | Literal['all', 'standard', 'e3_ligase', 'similarity', 'target'] = 'all',
):
    """ Run experiments with the cells one-hot encoding model.

    Args:
        active_col (str): Name of the column containing the active values.
        n_trials (int): Number of hyperparameter optimization trials.
        fast_dev_run (bool): Whether to run a fast development run.
        test_split (float): Percentage of data to use for testing.
        cv_n_splits (int): Number of cross-validation splits.
        max_epochs (int): Maximum number of epochs to train the model.
        force_study (bool): Whether to force the creation of a new study.
        experiments (str): Type of experiments to run. Options are 'all', 'standard', 'e3_ligase', 'similarity', 'target'.
    """

    # Make directory ../reports if it does not exist
    if not os.path.exists('../reports'):
        os.makedirs('../reports')

    # Load embedding dictionaries
    protein2embedding = pdp.load_protein2embedding('../data/uniprot2embedding.h5')
    cell2embedding = pdp.load_cell2embedding('../data/cell2embedding.pkl')

    # Get one-hot encoded embeddings for cell lines
    onehotenc = OneHotEncoder(sparse_output=False)
    cell_embeddings = onehotenc.fit_transform(
        np.array(list(cell2embedding.keys())).reshape(-1, 1)
    )
    cell2embedding = {k: v for k, v in zip(cell2embedding.keys(), cell_embeddings)}

    studies_dir = '../data/studies'
    train_val_perc = f'{int((1 - test_split) * 100)}'
    test_perc = f'{int(test_split * 100)}'
    active_name = active_col.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')

    if experiments == 'all':
        experiments = ['standard', 'similarity', 'target']
    else:
        experiments = [experiments]

    # Cross-Validation Training
    reports = defaultdict(list)
    for split_type in experiments:

        train_val_filename = f'{split_type}_train_val_{train_val_perc}split_{active_name}.csv'
        test_filename = f'{split_type}_test_{test_perc}split_{active_name}.csv'
        
        train_val_df = pd.read_csv(os.path.join(studies_dir, train_val_filename))
        test_df = pd.read_csv(os.path.join(studies_dir, test_filename))

        # Get SMILES and precompute fingerprints dictionary
        unique_smiles = pd.concat([train_val_df, test_df])['Smiles'].unique().tolist()
        smiles2fp = {s: np.array(pdp.get_fingerprint(s)) for s in unique_smiles}

        # Get the CV object
        if split_type == 'standard':
            kf = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=42)
            group = None
        elif split_type == 'e3_ligase':
            kf = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=42)
            group = train_val_df['E3 Group'].to_numpy()
        elif split_type == 'similarity':
            kf = StratifiedGroupKFold(n_splits=cv_n_splits, shuffle=True, random_state=42)
            group = train_val_df['Tanimoto Group'].to_numpy()
        elif split_type == 'target':
            kf = StratifiedGroupKFold(n_splits=cv_n_splits, shuffle=True, random_state=42)
            group = train_val_df['Uniprot Group'].to_numpy()

        # Start the experiment
        experiment_name = f'{active_name}_test_split_{test_split}_{split_type}'
        optuna_reports = pdp.hyperparameter_tuning_and_training( 
            protein2embedding=protein2embedding,
            cell2embedding=cell2embedding,
            smiles2fp=smiles2fp,
            train_val_df=train_val_df,
            test_df=test_df,
            kf=kf,
            groups=group,
            split_type=split_type,
            n_models_for_test=3,
            fast_dev_run=fast_dev_run,
            n_trials=n_trials,
            max_epochs=max_epochs,
            logger_save_dir='../logs',
            logger_name=f'logs_{experiment_name}',
            active_label=active_col,
            study_filename=f'../reports/study_cellsonehot_{experiment_name}.pkl',
            force_study=force_study,
            use_cells_one_hot=True,
        )

        # Save the reports to file
        for report_name, report in optuna_reports.items():
            report.to_csv(f'../reports/cellsonehot_{report_name}_{experiment_name}.csv', index=False)
            reports[report_name].append(report.copy())


if __name__ == '__main__':
    cli = CLI(main)