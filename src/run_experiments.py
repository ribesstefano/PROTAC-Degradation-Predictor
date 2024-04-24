import os
import sys
from collections import defaultdict
import warnings
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import protac_degradation_predictor as pdp

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


def get_random_split_indices(active_df: pd.DataFrame, test_split: float) -> pd.Index:
    """ Get the indices of the test set using a random split.
    
    Args:
        active_df (pd.DataFrame): The DataFrame containing the active PROTACs.
        test_split (float): The percentage of the active PROTACs to use as the test set.
    
    Returns:
        pd.Index: The indices of the test set.
    """
    test_df = active_df.sample(frac=test_split, random_state=42)
    return test_df.index


def get_e3_ligase_split_indices(active_df: pd.DataFrame) -> pd.Index:
    """ Get the indices of the test set using the E3 ligase split.
    
    Args:
        active_df (pd.DataFrame): The DataFrame containing the active PROTACs.
    
    Returns:
        pd.Index: The indices of the test set.
    """
    encoder = OrdinalEncoder()
    active_df['E3 Group'] = encoder.fit_transform(active_df[['E3 Ligase']]).astype(int)
    test_df = active_df[(active_df['E3 Ligase'] != 'VHL') & (active_df['E3 Ligase'] != 'CRBN')]
    return test_df.index


def get_smiles2fp_and_avg_tanimoto(protac_df: pd.DataFrame) -> tuple:
    """ Get the SMILES to fingerprint dictionary and the average Tanimoto similarity.
    
    Args:
        protac_df (pd.DataFrame): The DataFrame containing the PROTACs.
    
    Returns:
        tuple: The SMILES to fingerprint dictionary and the average Tanimoto similarity.
    """
    smiles2fp = {}
    for smiles in tqdm(protac_df['Smiles'].unique().tolist(), desc='Precomputing fingerprints'):
        smiles2fp[smiles] = pdp.get_fingerprint(smiles)

    # Get the pair-wise tanimoto similarity between the PROTAC fingerprints
    tanimoto_matrix = defaultdict(list)
    for i, smiles1 in enumerate(tqdm(protac_df['Smiles'].unique(), desc='Computing Tanimoto similarity')):
        fp1 = smiles2fp[smiles1]
        # TODO: Use BulkTanimotoSimilarity for better performance
        for j, smiles2 in enumerate(protac_df['Smiles'].unique()):
            if j < i:
                continue
            fp2 = smiles2fp[smiles2]
            tanimoto_dist = DataStructs.TanimotoSimilarity(fp1, fp2)
            tanimoto_matrix[smiles1].append(tanimoto_dist)
    avg_tanimoto = {k: np.mean(v) for k, v in tanimoto_matrix.items()}
    protac_df['Avg Tanimoto'] = protac_df['Smiles'].map(avg_tanimoto)

    smiles2fp = {s: np.array(fp) for s, fp in smiles2fp.items()}

    return smiles2fp, protac_df


def get_tanimoto_split_indices(
        active_df: pd.DataFrame,
        active_col: str,
        test_split: float,
        n_bins_tanimoto: int = 200,
) -> pd.Index:
    """ Get the indices of the test set using the Tanimoto-based split.
    
    Args:
        active_df (pd.DataFrame): The DataFrame containing the active PROTACs.
        n_bins_tanimoto (int): The number of bins to use for the Tanimoto similarity.
    
    Returns:
        pd.Index: The indices of the test set.
    """
    tanimoto_groups = pd.cut(active_df['Avg Tanimoto'], bins=n_bins_tanimoto).copy()
    encoder = OrdinalEncoder()
    active_df['Tanimoto Group'] = encoder.fit_transform(tanimoto_groups.values.reshape(-1, 1)).astype(int)
    # Sort the groups so that samples with the highest tanimoto similarity,
    # i.e., the "less similar" ones, are placed in the test set first
    tanimoto_groups = active_df.groupby('Tanimoto Group')['Avg Tanimoto'].mean().sort_values(ascending=False).index

    test_df = []
    # For each group, get the number of active and inactive entries. Then, add those
    # entries to the test_df if: 1) the test_df lenght + the group entries is less
    # 20% of the active_df lenght, and 2) the percentage of True and False entries
    # in the active_col in test_df is roughly 50%.
    for group in tanimoto_groups:
        group_df = active_df[active_df['Tanimoto Group'] == group]
        if test_df == []:
            test_df.append(group_df)
            continue
        
        num_entries = len(group_df)
        num_active_group = group_df[active_col].sum()
        num_inactive_group = num_entries - num_active_group

        tmp_test_df = pd.concat(test_df)
        num_entries_test = len(tmp_test_df)
        num_active_test = tmp_test_df[active_col].sum()
        num_inactive_test = num_entries_test - num_active_test
        
        # Check if the group entries can be added to the test_df
        if num_entries_test + num_entries < test_split * len(active_df):
            # Add anything at the beggining
            if num_entries_test + num_entries < test_split / 2 * len(active_df):
                test_df.append(group_df)
                continue
            # Be more selective and make sure that the percentage of active and
            # inactive is balanced
            if (num_active_group + num_active_test) / (num_entries_test + num_entries) < 0.6:
                if (num_inactive_group + num_inactive_test) / (num_entries_test + num_entries) < 0.6:
                    test_df.append(group_df)
    test_df = pd.concat(test_df)
    return test_df.index


def get_target_split_indices(active_df: pd.DataFrame, active_col: str, test_split: float) -> pd.Index:
    """ Get the indices of the test set using the target-based split.

    Args:
        active_df (pd.DataFrame): The DataFrame containing the active PROTACs.
        active_col (str): The column containing the active/inactive information.
        test_split (float): The percentage of the active PROTACs to use as the test set.

    Returns:
        pd.Index: The indices of the test set.
    """
    encoder = OrdinalEncoder()
    active_df['Uniprot Group'] = encoder.fit_transform(active_df[['Uniprot']]).astype(int)

    test_df = []
    # For each group, get the number of active and inactive entries. Then, add those
    # entries to the test_df if: 1) the test_df lenght + the group entries is less
    # 20% of the active_df lenght, and 2) the percentage of True and False entries
    # in the active_col in test_df is roughly 50%.
    # Start the loop from the groups containing the smallest number of entries.
    for group in reversed(active_df['Uniprot'].value_counts().index):
        group_df = active_df[active_df['Uniprot'] == group]
        if test_df == []:
            test_df.append(group_df)
            continue
        
        num_entries = len(group_df)
        num_active_group = group_df[active_col].sum()
        num_inactive_group = num_entries - num_active_group

        tmp_test_df = pd.concat(test_df)
        num_entries_test = len(tmp_test_df)
        num_active_test = tmp_test_df[active_col].sum()
        num_inactive_test = num_entries_test - num_active_test
        
        # Check if the group entries can be added to the test_df
        if num_entries_test + num_entries < test_split * len(active_df):
            # Add anything at the beggining
            if num_entries_test + num_entries < test_split / 2 * len(active_df):
                test_df.append(group_df)
                continue
            # Be more selective and make sure that the percentage of active and
            # inactive is balanced
            if (num_active_group + num_active_test) / (num_entries_test + num_entries) < 0.6:
                if (num_inactive_group + num_inactive_test) / (num_entries_test + num_entries) < 0.6:
                    test_df.append(group_df)
    test_df = pd.concat(test_df)
    return test_df.index


def main(
    active_col: str = 'Active (Dmax 0.6, pDC50 6.0)',
    n_trials: int = 100,
    fast_dev_run: bool = False,
    test_split: float = 0.1,
    cv_n_splits: int = 5,
    max_epochs: int = 100,
    run_sklearn: bool = False,
):
    """ Train a PROTAC model using the given datasets and hyperparameters.
    
    Args:
        use_ored_activity (bool): Whether to use the 'Active - OR' column.
        n_trials (int): The number of hyperparameter optimization trials.
        n_splits (int): The number of cross-validation splits.
        fast_dev_run (bool): Whether to run a fast development run.
    """
    # Set the Column to Predict
    active_name = active_col.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')

    # Get Dmax_threshold from the active_col
    Dmax_threshold = float(active_col.split('Dmax')[1].split(',')[0].strip('(').strip(')').strip())
    pDC50_threshold = float(active_col.split('pDC50')[1].strip('(').strip(')').strip())

    # Load the PROTAC dataset
    protac_df = pd.read_csv('../data/PROTAC-Degradation-DB.csv')
    # Map E3 Ligase Iap to IAP
    protac_df['E3 Ligase'] = protac_df['E3 Ligase'].str.replace('Iap', 'IAP')
    protac_df[active_col] = protac_df.apply(
        lambda x: pdp.is_active(x['DC50 (nM)'], x['Dmax (%)'], pDC50_threshold=pDC50_threshold, Dmax_threshold=Dmax_threshold), axis=1
    )
    smiles2fp, protac_df = get_smiles2fp_and_avg_tanimoto(protac_df)

    ## Get the test sets
    test_indeces = {}
    active_df = protac_df[protac_df[active_col].notna()].copy()
    test_indeces['random'] = get_random_split_indices(active_df, test_split)
    test_indeces['e3_ligase'] = get_e3_ligase_split_indices(active_df)
    test_indeces['tanimoto'] = get_tanimoto_split_indices(active_df, active_col, test_split)
    test_indeces['uniprot'] = get_target_split_indices(active_df, active_col, test_split)
    
    # Make directory ../reports if it does not exist
    if not os.path.exists('../reports'):
        os.makedirs('../reports')

    # Load embedding dictionaries
    protein2embedding = pdp.load_protein2embedding('../data/uniprot2embedding.h5')
    cell2embedding = pdp.load_cell2embedding('../data/cell2embedding.pkl')

    # Cross-Validation Training
    report = []
    for split_type, indeces in test_indeces.items():
        test_df = active_df.loc[indeces].copy()
        train_val_df = active_df[~active_df.index.isin(test_df.index)].copy()
        
        # Get the CV object
        if split_type == 'random':
            kf = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=42)
            group = None
        elif split_type == 'e3_ligase':
            kf = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=42)
            group = train_val_df['E3 Group'].to_numpy()
        elif split_type == 'tanimoto':
            kf = StratifiedGroupKFold(n_splits=cv_n_splits, shuffle=True, random_state=42)
            group = train_val_df['Tanimoto Group'].to_numpy()
        elif split_type == 'uniprot':
            kf = StratifiedGroupKFold(n_splits=cv_n_splits, shuffle=True, random_state=42)
            group = train_val_df['Uniprot Group'].to_numpy()

        # Start the experiment
        experiment_name = f'{active_name}_test_split_{test_split}_{split_type}'
        reports = pdp.hyperparameter_tuning_and_training(
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
            logger_name=f'logs_{experiment_name}',
            active_label=active_col,
            study_filename=f'../reports/study_{experiment_name}.pkl',
        )
        cv_report, hparam_report, test_report, ablation_report = reports

        # Save the reports to file
        for report, filename in zip([cv_report, hparam_report, test_report, ablation_report], ['cv_train', 'hparams', 'test', 'ablation']):
            report.to_csv(f'../reports/report_{filename}_{experiment_name}.csv', index=False)




        # # Start the CV over the folds
        # X = train_val_df.drop(columns=active_col)
        # y = train_val_df[active_col].tolist()
        # for k, (train_index, val_index) in enumerate(kf.split(X, y, group)):
        #     print('-' * 100)
        #     print(f'Starting CV for group type: {split_type}, fold: {k}')
        #     print('-' * 100)
        #     train_df = train_val_df.iloc[train_index]
        #     val_df = train_val_df.iloc[val_index]

        #     leaking_uniprot = list(set(train_df['Uniprot']).intersection(set(val_df['Uniprot'])))
        #     leaking_smiles = list(set(train_df['Smiles']).intersection(set(val_df['Smiles'])))

        #     stats = {
        #         'fold': k,
        #         'split_type': split_type,
        #         'train_len': len(train_df),
        #         'val_len': len(val_df),
        #         'train_perc': len(train_df) / len(train_val_df),
        #         'val_perc': len(val_df) / len(train_val_df),
        #         'train_active_perc': train_df[active_col].sum() / len(train_df),
        #         'train_inactive_perc': (len(train_df) - train_df[active_col].sum()) / len(train_df),
        #         'val_active_perc': val_df[active_col].sum() / len(val_df),
        #         'val_inactive_perc': (len(val_df) - val_df[active_col].sum()) / len(val_df),
        #         'test_active_perc': test_df[active_col].sum() / len(test_df),
        #         'test_inactive_perc': (len(test_df) - test_df[active_col].sum()) / len(test_df),
        #         'num_leaking_uniprot': len(leaking_uniprot),
        #         'num_leaking_smiles': len(leaking_smiles),
        #         'train_leaking_uniprot_perc': len(train_df[train_df['Uniprot'].isin(leaking_uniprot)]) / len(train_df),
        #         'train_leaking_smiles_perc': len(train_df[train_df['Smiles'].isin(leaking_smiles)]) / len(train_df),
        #     }
        #     if split_type != 'random':
        #         stats['train_unique_groups'] = len(np.unique(group[train_index]))
        #         stats['val_unique_groups'] = len(np.unique(group[val_index]))

        #     # At each fold, train and evaluate the Pytorch model
        #     if split_type != 'tanimoto' or run_sklearn:
        #         logging.info(f'Skipping Pytorch model training on fold {k} with split type {split_type} and test split {test_split}.')
        #         continue
        #     else:
        #         logging.info(f'Starting Pytorch model training on fold {k} with split type {split_type} and test split {test_split}.')
        #         # Train and evaluate the model
        #         model, trainer, metrics = pdp.hyperparameter_tuning_and_training(
        #             protein2embedding,
        #             cell2embedding,
        #             smiles2fp,
        #             train_df,
        #             val_df,
        #             test_df,
        #             fast_dev_run=fast_dev_run,
        #             n_trials=n_trials,
        #             logger_name=f'protac_{active_name}_{split_type}_fold_{k}_test_split_{test_split}',
        #             active_label=active_col,
        #             study_filename=f'../reports/study_{active_name}_{split_type}_fold_{k}_test_split_{test_split}.pkl',
        #         )
        #         hparams = {p.replace('hparam_', ''): v for p, v in stats.items() if p.startswith('hparam_')}
        #         stats.update(metrics)
        #         stats['model_type'] = 'Pytorch'
        #         report.append(stats.copy())
        #         del model
        #         del trainer

        #         # Ablation study: disable embeddings at a time
        #         for disabled_embeddings in [['e3'], ['poi'], ['cell'], ['smiles'], ['e3', 'cell'], ['poi', 'e3', 'cell']]:
        #             print('-' * 100)
        #             print(f'Ablation study with disabled embeddings: {disabled_embeddings}')
        #             print('-' * 100)
        #             stats['disabled_embeddings'] = 'disabled ' + ' '.join(disabled_embeddings)
        #             model, trainer, metrics = pdp.train_model(
        #                 protein2embedding,
        #                 cell2embedding,
        #                 smiles2fp,
        #                 train_df,
        #                 val_df,
        #                 test_df,
        #                 fast_dev_run=fast_dev_run,
        #                 logger_name=f'protac_{active_name}_{split_type}_fold_{k}_disabled-{"-".join(disabled_embeddings)}',
        #                 active_label=active_col,
        #                 disabled_embeddings=disabled_embeddings,
        #                 **hparams,
        #             )
        #             stats.update(metrics)
        #             report.append(stats.copy())
        #             del model
        #             del trainer

        #     # At each fold, train and evaluate sklearn models
        #     if run_sklearn:
        #         for model_type in ['RandomForest', 'SVC', 'LogisticRegression', 'GradientBoosting']:
        #             logging.info(f'Starting sklearn model {model_type} training on fold {k} with split type {split_type} and test split {test_split}.')
        #             # Train and evaluate sklearn models
        #             model, metrics = pdp.hyperparameter_tuning_and_training_sklearn(
        #                 protein2embedding=protein2embedding,
        #                 cell2embedding=cell2embedding,
        #                 smiles2fp=smiles2fp,
        #                 train_df=train_df,
        #                 val_df=val_df,
        #                 test_df=test_df,
        #                 model_type=model_type,
        #                 active_label=active_col,
        #                 n_trials=n_trials,
        #                 study_filename=f'../reports/study_{active_name}_{split_type}_fold_{k}_test_split_{test_split}_{model_type.lower()}.pkl',
        #             )
        #             hparams = {p.replace('hparam_', ''): v for p, v in stats.items() if p.startswith('hparam_')}
        #             stats['model_type'] = model_type
        #             stats.update(metrics)
        #             report.append(stats.copy())

        # # Save the report at the end of each split type
        # report_df = pd.DataFrame(report)
        # report_df.to_csv(
        #     f'../reports/cv_report_hparam_search_{cv_n_splits}-splits_{active_name}_test_split_{test_split}{"_sklearn" if run_sklearn else ""}.csv',
        #     index=False,
        # )


if __name__ == '__main__':
    cli = CLI(main)