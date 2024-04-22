import os
from collections import defaultdict
import warnings

from protac_degradation_predictor.config import config
from protac_degradation_predictor.data_utils import (
    load_protein2embedding,
    load_cell2embedding,
    is_active,
)
from protac_degradation_predictor.pytorch_models import (
    train_model,
)
from protac_degradation_predictor.optuna_utils import (
    hyperparameter_tuning_and_training,
)

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


def main(
    active_col: str = 'Active (Dmax 0.6, pDC50 6.0)',
    n_trials: int = 50,
    fast_dev_run: bool = False,
    test_split: float = 0.2,
    cv_n_splits: int = 5,
):
    """ Train a PROTAC model using the given datasets and hyperparameters.
    
    Args:
        use_ored_activity (bool): Whether to use the 'Active - OR' column.
        n_trials (int): The number of hyperparameter optimization trials.
        n_splits (int): The number of cross-validation splits.
        fast_dev_run (bool): Whether to run a fast development run.
    """
    ## Set the Column to Predict
    active_name = active_col.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')

    # Get Dmax_threshold from the active_col
    Dmax_threshold = float(active_col.split('Dmax')[1].split(',')[0].strip('(').strip(')').strip())
    pDC50_threshold = float(active_col.split('pDC50')[1].strip('(').strip(')').strip())

    ## Load the Data
    protac_df = pd.read_csv('../data/PROTAC-Degradation-DB.csv')

    # Map E3 Ligase Iap to IAP
    protac_df['E3 Ligase'] = protac_df['E3 Ligase'].str.replace('Iap', 'IAP')

    protac_df[active_col] = protac_df.apply(
        lambda x: is_active(x['DC50 (nM)'], x['Dmax (%)'], pDC50_threshold=pDC50_threshold, Dmax_threshold=Dmax_threshold), axis=1
    )

    ## Test Sets

    test_indeces = {}

    ### Random Split

    # Randomly select 20% of the active PROTACs as the test set
    active_df = protac_df[protac_df[active_col].notna()].copy()
    test_df = active_df.sample(frac=test_split, random_state=42)
    test_indeces['random'] = test_df.index

    ### E3-based Split

    encoder = OrdinalEncoder()
    protac_df['E3 Group'] = encoder.fit_transform(protac_df[['E3 Ligase']]).astype(int)
    active_df = protac_df[protac_df[active_col].notna()].copy()
    test_df = active_df[(active_df['E3 Ligase'] != 'VHL') & (active_df['E3 Ligase'] != 'CRBN')]
    test_indeces['e3_ligase'] = test_df.index

    ### Tanimoto-based Split

    #### Precompute fingerprints
    morgan_fpgen = AllChem.GetMorganGenerator(
        radius=config.morgan_radius,
        fpSize=config.fingerprint_size,
        includeChirality=True,
    )

    smiles2fp = {}
    for smiles in tqdm(protac_df['Smiles'].unique().tolist(), desc='Precomputing fingerprints'):
        # Get the fingerprint as a bit vector
        morgan_fp = morgan_fpgen.GetFingerprint(Chem.MolFromSmiles(smiles))
        smiles2fp[smiles] = morgan_fp

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

    # Make the grouping of the PROTACs based on the Tanimoto similarity
    n_bins_tanimoto = 200
    tanimoto_groups = pd.cut(protac_df['Avg Tanimoto'], bins=n_bins_tanimoto).copy()
    encoder = OrdinalEncoder()
    protac_df['Tanimoto Group'] = encoder.fit_transform(tanimoto_groups.values.reshape(-1, 1)).astype(int)
    active_df = protac_df[protac_df[active_col].notna()].copy()
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
    # Save to global dictionary of test indeces
    test_indeces['tanimoto'] = test_df.index

    ### Target-based Split

    encoder = OrdinalEncoder()
    protac_df['Uniprot Group'] = encoder.fit_transform(protac_df[['Uniprot']]).astype(int)
    active_df = protac_df[protac_df[active_col].notna()].copy()

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
    # Save to global dictionary of test indeces
    test_indeces['uniprot'] = test_df.index

    ## Cross-Validation Training
    
    # Make directory ../reports if it does not exist
    if not os.path.exists('../reports'):
        os.makedirs('../reports')

    # Load embedding dictionaries
    protein2embedding = load_protein2embedding('../data/uniprot2embedding.h5')
    cell2embedding = load_cell2embedding('../data/cell2embedding.pkl')

    report = []
    for split_type, indeces in test_indeces.items():
        active_df = protac_df[protac_df[active_col].notna()].copy()
        test_df = active_df.loc[indeces]
        train_val_df = active_df[~active_df.index.isin(test_df.index)]
        
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
        # Start the CV over the folds
        X = train_val_df.drop(columns=active_col)
        y = train_val_df[active_col].tolist()
        for k, (train_index, val_index) in enumerate(kf.split(X, y, group)):
            print('-' * 100)
            print(f'Starting CV for group type: {split_type}, fold: {k}')
            print('-' * 100)
            train_df = train_val_df.iloc[train_index]
            val_df = train_val_df.iloc[val_index]

            leaking_uniprot = list(set(train_df['Uniprot']).intersection(set(val_df['Uniprot'])))
            leaking_smiles = list(set(train_df['Smiles']).intersection(set(val_df['Smiles'])))

            stats = {
                'fold': k,
                'split_type': split_type,
                'train_len': len(train_df),
                'val_len': len(val_df),
                'train_perc': len(train_df) / len(train_val_df),
                'val_perc': len(val_df) / len(train_val_df),
                'train_active_perc': train_df[active_col].sum() / len(train_df),
                'train_inactive_perc': (len(train_df) - train_df[active_col].sum()) / len(train_df),
                'val_active_perc': val_df[active_col].sum() / len(val_df),
                'val_inactive_perc': (len(val_df) - val_df[active_col].sum()) / len(val_df),
                'test_active_perc': test_df[active_col].sum() / len(test_df),
                'test_inactive_perc': (len(test_df) - test_df[active_col].sum()) / len(test_df),
                'num_leaking_uniprot': len(leaking_uniprot),
                'num_leaking_smiles': len(leaking_smiles),
                'train_leaking_uniprot_perc': len(train_df[train_df['Uniprot'].isin(leaking_uniprot)]) / len(train_df),
                'train_leaking_smiles_perc': len(train_df[train_df['Smiles'].isin(leaking_smiles)]) / len(train_df),
            }
            if split_type != 'random':
                stats['train_unique_groups'] = len(np.unique(group[train_index]))
                stats['val_unique_groups'] = len(np.unique(group[val_index]))
            
            print(stats)
        #     # Train and evaluate the model
        #     model, trainer, metrics = hyperparameter_tuning_and_training(
        #         protein2embedding,
        #         cell2embedding,
        #         smiles2fp,
        #         train_df,
        #         val_df,
        #         test_df,
        #         fast_dev_run=fast_dev_run,
        #         n_trials=n_trials,
        #         logger_name=f'protac_{active_name}_{split_type}_fold_{k}_test_split_{test_split}',
        #         active_label=active_col,
        #         study_filename=f'../reports/study_{active_name}_{split_type}_fold_{k}_test_split_{test_split}.pkl',
        #     )
        #     hparams = {p.replace('hparam_', ''): v for p, v in stats.items() if p.startswith('hparam_')}
        #     stats.update(metrics)
        #     report.append(stats.copy())
        #     del model
        #     del trainer

        #     # Ablation study: disable embeddings at a time
        #     for disabled_embeddings in [['e3'], ['poi'], ['cell'], ['smiles'], ['e3', 'cell'], ['poi', 'e3', 'cell']]:
        #         print('-' * 100)
        #         print(f'Ablation study with disabled embeddings: {disabled_embeddings}')
        #         print('-' * 100)
        #         stats['disabled_embeddings'] = 'disabled ' + ' '.join(disabled_embeddings)
        #         model, trainer, metrics = train_model(
        #             protein2embedding,
        #             cell2embedding,
        #             smiles2fp,
        #             train_df,
        #             val_df,
        #             test_df,
        #             fast_dev_run=fast_dev_run,
        #             logger_name=f'protac_{active_name}_{split_type}_fold_{k}_disabled-{"-".join(disabled_embeddings)}',
        #             active_label=active_col,
        #             disabled_embeddings=disabled_embeddings,
        #             **hparams,
        #         )
        #         stats.update(metrics)
        #         report.append(stats.copy())
        #         del model
        #         del trainer

        # report_df = pd.DataFrame(report)
        # report_df.to_csv(
        #     f'../reports/cv_report_hparam_search_{cv_n_splits}-splits_{active_name}_test_split_{test_split}_sklearn.csv',
        #     index=False,
        # )


if __name__ == '__main__':
    cli = CLI(main)