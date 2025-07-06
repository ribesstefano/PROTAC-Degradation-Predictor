import os
import sys
import argparse
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import protac_degradation_predictor as pdp

from collections import defaultdict
import warnings
import logging
from typing import Literal

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from tqdm import tqdm
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from rdkit import DataStructs


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
    unique_smiles = protac_df['Smiles'].unique().tolist()

    smiles2fp = {}
    for smiles in tqdm(unique_smiles, desc='Precomputing fingerprints'):
        smiles2fp[smiles] = pdp.get_fingerprint(smiles)

    # # Get the pair-wise tanimoto similarity between the PROTAC fingerprints
    # tanimoto_matrix = defaultdict(list)
    # for i, smiles1 in enumerate(tqdm(protac_df['Smiles'].unique(), desc='Computing Tanimoto similarity')):
    #     fp1 = smiles2fp[smiles1]
    #     # TODO: Use BulkTanimotoSimilarity for better performance
    #     for j, smiles2 in enumerate(protac_df['Smiles'].unique()[i:]):
    #         fp2 = smiles2fp[smiles2]
    #         tanimoto_dist = 1 - DataStructs.TanimotoSimilarity(fp1, fp2)
    #         tanimoto_matrix[smiles1].append(tanimoto_dist)
    # avg_tanimoto = {k: np.mean(v) for k, v in tanimoto_matrix.items()}
    # protac_df['Avg Tanimoto'] = protac_df['Smiles'].map(avg_tanimoto)


    tanimoto_matrix = defaultdict(list)
    fps = list(smiles2fp.values())

    # Compute all-against-all Tanimoto similarity using BulkTanimotoSimilarity
    for i, (smiles1, fp1) in enumerate(tqdm(zip(unique_smiles, fps), desc='Computing Tanimoto similarity', total=len(fps))):
        similarities = DataStructs.BulkTanimotoSimilarity(fp1, fps[i:])  # Only compute for i to end, avoiding duplicates
        for j, similarity in enumerate(similarities):
            distance = 1 - similarity
            tanimoto_matrix[smiles1].append(distance)  # Store as distance
            if i != i + j:
                tanimoto_matrix[unique_smiles[i + j]].append(distance)  # Symmetric filling

    # Calculate average Tanimoto distance for each unique SMILES
    avg_tanimoto = {k: np.mean(v) for k, v in tanimoto_matrix.items()}
    protac_df['Avg Tanimoto'] = protac_df['Smiles'].map(avg_tanimoto)

    smiles2fp = {s: np.array(fp) for s, fp in smiles2fp.items()}

    return smiles2fp, protac_df


def get_tanimoto_split_indices(
        active_df: pd.DataFrame,
        active_col: str,
        test_split: float,
        n_bins_tanimoto: int = 100, # Original: 200
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
            # inactive is not over-exceeding 60%
            perc_active_group = (num_active_group + num_active_test) / (num_entries_test + num_entries)
            perc_inactive_group = (num_inactive_group + num_inactive_test) / (num_entries_test + num_entries)
            if perc_active_group < 0.6:
                if perc_inactive_group < 0.6:
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


def get_dataframe_stats(
        train_df = None,
        val_df = None,
        test_df = None,
        active_label = 'Active',
    ) -> Dict:
    """ Get some statistics from the dataframes.
    
    Args:
        train_df (pd.DataFrame): The training set.
        val_df (pd.DataFrame): The validation set.
        test_df (pd.DataFrame): The test set.
    """
    stats = {}
    if train_df is not None:
        stats['train_len'] = len(train_df)
        stats['train_active_perc'] = train_df[active_label].sum() / len(train_df)
        stats['train_inactive_perc'] = (len(train_df) - train_df[active_label].sum()) / len(train_df)
        stats['train_avg_tanimoto_dist'] = train_df['Avg Tanimoto'].mean()
    if val_df is not None:
        stats['val_len'] = len(val_df)
        stats['val_active_perc'] = val_df[active_label].sum() / len(val_df)
        stats['val_inactive_perc'] = (len(val_df) - val_df[active_label].sum()) / len(val_df)
        stats['val_avg_tanimoto_dist'] = val_df['Avg Tanimoto'].mean()
    if test_df is not None:
        stats['test_len'] = len(test_df)
        stats['test_active_perc'] = test_df[active_label].sum() / len(test_df)
        stats['test_inactive_perc'] = (len(test_df) - test_df[active_label].sum()) / len(test_df)
        stats['test_avg_tanimoto_dist'] = test_df['Avg Tanimoto'].mean()
    if train_df is not None and val_df is not None:
        leaking_uniprot = list(set(train_df['Uniprot']).intersection(set(val_df['Uniprot'])))
        leaking_smiles = list(set(train_df['Smiles']).intersection(set(val_df['Smiles'])))
        stats['num_leaking_uniprot_train_val'] = len(leaking_uniprot)
        stats['num_leaking_smiles_train_val'] = len(leaking_smiles)
        stats['perc_leaking_uniprot_train_val'] = len(train_df[train_df['Uniprot'].isin(leaking_uniprot)]) / len(train_df)
        stats['perc_leaking_smiles_train_val'] = len(train_df[train_df['Smiles'].isin(leaking_smiles)]) / len(train_df)
        
        key_cols = [
            'Smiles',
            'Uniprot',
            'E3 Ligase Uniprot',
            'Cell Line Identifier',
        ]
        class_cols = ['DC50 (nM)', 'Dmax (%)']
        # Check if there are any entries that are in BOTH train and val sets
        tmp_train_df = train_df[key_cols + class_cols].copy()
        tmp_val_df = val_df[key_cols + class_cols].copy()
        stats['leaking_train_val'] = len(tmp_train_df.merge(tmp_val_df, on=key_cols + class_cols, how='inner'))


    if train_df is not None and test_df is not None:
        leaking_uniprot = list(set(train_df['Uniprot']).intersection(set(test_df['Uniprot'])))
        leaking_smiles = list(set(train_df['Smiles']).intersection(set(test_df['Smiles'])))
        stats['num_leaking_uniprot_train_test'] = len(leaking_uniprot)
        stats['num_leaking_smiles_train_test'] = len(leaking_smiles)
        stats['perc_leaking_uniprot_train_test'] = len(train_df[train_df['Uniprot'].isin(leaking_uniprot)]) / len(train_df)
        stats['perc_leaking_smiles_train_test'] = len(train_df[train_df['Smiles'].isin(leaking_smiles)]) / len(train_df)

        key_cols = [
            'Smiles',
            'Uniprot',
            'E3 Ligase Uniprot',
            'Cell Line Identifier',
        ]
        class_cols = ['DC50 (nM)', 'Dmax (%)']
        # Check if there are any entries that are in BOTH train and test sets
        tmp_train_df = train_df[key_cols + class_cols].copy()
        tmp_test_df = test_df[key_cols + class_cols].copy()
        stats['leaking_train_test'] = len(tmp_train_df.merge(tmp_test_df, on=key_cols + class_cols, how='inner'))

    return stats


def merge_numerical_cols(group: pd.DataFrame) -> pd.DataFrame:
    """ Merge the numerical columns by computing the geometric mean.
    
    Args:
        group (pd.DataFrame): The group to merge.

    Returns:
        pd.DataFrame: The merged group (as a single row).

    """
    key_cols = [
        'Smiles',
        'Uniprot',
        'E3 Ligase Uniprot',
        'Cell Line Identifier',
    ]
    class_cols = ['DC50 (nM)', 'Dmax (%)']
    # Loop over all numerical columns
    for col in group.select_dtypes(include=[np.number]).columns:
        if col == 'Compound ID':
            continue
        # Compute the geometric mean for the column
        values = group[col].dropna()
        if not values.empty:
            group[col] = np.prod(values) ** (1 / len(values))

    row = group.drop_duplicates(subset=key_cols + class_cols).reset_index(drop=True)

    assert len(row) == 1

    return row


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """ Remove duplicates from the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to remove duplicates from.

    Returns:
        pd.DataFrame: The DataFrame without duplicates.
    """
    key_cols = [
        'Smiles',
        'Uniprot',
        'E3 Ligase Uniprot',
        'Cell Line Identifier',
    ]
    class_cols = ['DC50 (nM)', 'Dmax (%)']
    # Check if there are any duplicated entries having the same key columns, if
    # so, merge them by applying a geometric mean to their DC50 and Dmax columns
    duplicated = df[df.duplicated(subset=key_cols, keep=False)]

    # NOTE: Reset index to remove the multi-index
    merged = duplicated.groupby(key_cols).apply(lambda x: merge_numerical_cols(x))
    merged = merged.reset_index(drop=True)

    # Remove the duplicated entries from the original dataframe df
    df = df[~df.duplicated(subset=key_cols, keep=False)]
    # Concatenate the merged dataframe with the original dataframe
    return pd.concat([df, merged], ignore_index=True)


def main(
    active_col: str = 'Active', # (Dmax 0.6, pDC50 6.0)',
    test_split: float = 0.1,
    studies: str | Literal['all', 'standard', 'e3_ligase', 'similarity', 'target'] = 'all',
    cv_n_splits: int = 5,
    data_dir: str = './data/studies',
    Dmax_threshold: float = 0.6,
    pDC50_threshold: float = 6.0,
):
    """ Get and save the datasets for the different studies.
    
    Args:
        active_col (str): The column containing the active/inactive information. It should be in the format 'Active (Dmax N, pDC50 M)', where N and M are the thresholds float values for Dmax and pDC50, respectively.
        test_split (float): The percentage of the active PROTACs to use as the test set.
        studies (str): The type of studies to save dataset for. Options: 'all', 'standard', 'e3_ligase', 'similarity', 'target'.
    """
    pl.seed_everything(42)
    
    parser = argparse.ArgumentParser(description="Get and save the datasets for the different studies.")
    parser.add_argument(
        "--active_col",
        type=str,
        default=active_col,
        help="The column containing the active/inactive information. It should be in the format 'Active (Dmax N, pDC50 M)', where N and M are the thresholds float values for Dmax and pDC50, respectively.",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=test_split,
        help="The percentage of the active PROTACs to use as the test set.",
    )
    parser.add_argument(
        "--studies",
        type=str,
        default=studies,
        choices=['all', 'standard', 'e3_ligase', 'similarity', 'target'],
        help="The type of studies to save dataset for. Options: 'all', 'standard', 'e3_ligase', 'similarity', 'target'.",
    )
    parser.add_argument(
        "--cv_n_splits",
        type=int,
        default=cv_n_splits,
        help="The number of splits for cross-validation.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=data_dir,
        help="The directory to save the datasets.",
    )
    parser.add_argument(
        "--Dmax_threshold",
        type=float,
        default=Dmax_threshold,
        help="The threshold for Dmax to consider a PROTAC as active.",
    )
    parser.add_argument(
        "--pDC50_threshold",
        type=float,
        default=pDC50_threshold,
        help="The threshold for pDC50 to consider a PROTAC as active.",
    )
    args = parser.parse_args()
    active_col = args.active_col
    test_split = args.test_split
    studies = args.studies
    cv_n_splits = args.cv_n_splits
    data_dir = args.data_dir
    Dmax_threshold = args.Dmax_threshold
    pDC50_threshold = args.pDC50_threshold

    # Set the Column to Predict
    active_name = active_col.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')

    # Get Dmax_threshold from the active_col
    if 'Dmax' in active_col and 'pDC50' in active_col:
        Dmax_threshold = float(active_col.split('Dmax')[1].split(',')[0].strip('(').strip(')').strip())
        pDC50_threshold = float(active_col.split('pDC50')[1].strip('(').strip(')').strip())

    # Load the PROTAC dataset
    protac_df = pdp.load_curated_dataset()
    
    # Map E3 Ligase Iap to IAP
    protac_df['E3 Ligase'] = protac_df['E3 Ligase'].str.replace('Iap', 'IAP')

    # Remove duplicates
    protac_df = remove_duplicates(protac_df)

    # Remove legacy columns if they exist
    if 'Active - OR' in protac_df.columns:
        protac_df.drop(columns='Active - OR', inplace=True)
    if 'Active - AND' in protac_df.columns:
        protac_df.drop(columns='Active - AND', inplace=True)
    if 'Active' in protac_df.columns:
        protac_df.drop(columns='Active', inplace=True)
    
    # Calculate Activity and add it as a column
    protac_df[active_col] = protac_df.apply(
        lambda x: pdp.is_active(x['DC50 (nM)'], x['Dmax (%)'], pDC50_threshold=pDC50_threshold, Dmax_threshold=Dmax_threshold), axis=1
    )

    # Precompute fingerprints and average Tanimoto similarity
    _, protac_df = get_smiles2fp_and_avg_tanimoto(protac_df)

    ## Get the test sets
    test_indeces = {}
    active_df = protac_df[protac_df[active_col].notna()].copy()

    if studies == 'standard' or studies == 'all':
        test_indeces['standard'] = get_random_split_indices(active_df, test_split)
    if studies == 'target' or studies == 'all':
        test_indeces['target'] = get_target_split_indices(active_df, active_col, test_split)
    if studies == 'similarity' or studies == 'all':
        test_indeces['similarity'] = get_tanimoto_split_indices(active_df, active_col, test_split)
    # if studies == 'e3_ligase' or studies == 'all':
    #     test_indeces['e3_ligase'] = get_e3_ligase_split_indices(active_df)

    # Make directory for studies datasets if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Open file for reporting
    with open(f'{data_dir}/report_datasets.md', 'w') as f:
        for split_type, indeces in test_indeces.items():
            test_df = active_df.loc[indeces].copy()
            train_val_df = active_df[~active_df.index.isin(test_df.index)].copy()

            # Save the datasets
            train_val_perc = f'{int((1 - test_split) * 100)}'
            test_perc = f'{int(test_split * 100)}'

            train_val_filename = f'{data_dir}/{split_type}_train_val_{train_val_perc}split_{active_name}.csv'
            test_filename = f'{data_dir}/{split_type}_test_{test_perc}split_{active_name}.csv'

            train_val_df.to_csv(train_val_filename, index=False)
            test_df.to_csv(test_filename, index=False)
            
            print(f'Saved {split_type} train/val dataset to: {train_val_filename}')
            print(f'Saved {split_type} test dataset to: {test_filename}')
                
            # Report statistics of the cross-validation training datasets
            # Print statistics on active/inactive percentages
            perc_active = train_val_df[active_col].sum() / len(train_val_df)
            print('-' * 80)
            print(f'{split_type.capitalize()} Split')
            print(f'Len Train/Val:{len(train_val_df)}')
            print(f'Len Test: {len(test_df)}')
            print(f'Percentage Active in Train/Val: {perc_active:.2%}')
            print(f'Percentage Inactive in Train/Val: {1 - perc_active:.2%}')

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
            
            # Get the folds on the train_val_df, then collect statistics on active/inactive percentages
            stats = []
            for i, (train_index, val_index) in enumerate(kf.split(train_val_df, train_val_df[active_col].to_list(), group)):
                train_df = train_val_df.iloc[train_index]
                val_df = train_val_df.iloc[val_index]

                s = get_dataframe_stats(train_df, val_df, test_df, active_col)
                s['fold'] = i + 1
                stats.append(s)
            
            # Append the statistics as markdown to report file f
            stats_df = pd.DataFrame(stats)
            f.write(f'## {split_type.capitalize()} Split\n\n')
            f.write(stats_df.to_markdown(index=False))
            f.write('\n\n')
            print('-' * 80)

    print(f'Wrote statistics to {data_dir}/report_datasets.md')
    print('-' * 80)
    print('All datasets have been saved successfully!')
    return

    # ==========================================================================
    # Regression datasets
    # TODO: Not fully tested yet.
    # ==========================================================================
    # Load the PROTAC dataset
    protac_df = pdp.load_curated_dataset()
    
    # Map E3 Ligase Iap to IAP
    protac_df['E3 Ligase'] = protac_df['E3 Ligase'].str.replace('Iap', 'IAP')

    # Calculate pDC50 on the 'DC50 (nM)' column
    protac_df['pDC50'] = -np.log10(protac_df['DC50 (nM)'] * 1e-9)

    # Precompute fingerprints and average Tanimoto similarity
    _, protac_df = get_smiles2fp_and_avg_tanimoto(protac_df)

    # Get the two datasets
    dmax_df = protac_df[protac_df['Dmax (%)'].notna()].copy()
    pdc50_df = protac_df[protac_df['pDC50'].notna()].copy()

    ## Get the test sets
    test_indeces = {'dmax': {}, 'pdc50': {}}

    if studies == 'standard' or studies == 'all':
        test_indeces['dmax']['standard'] = get_random_split_indices(dmax_df, test_split)
        test_indeces['pdc50']['standard'] = get_random_split_indices(pdc50_df, test_split)
    if studies == 'target' or studies == 'all':
        test_indeces['dmax']['target'] = get_target_split_indices(dmax_df, active_col, test_split)
    if studies == 'similarity' or studies == 'all':
        test_indeces['dmax']['similarity'] = get_tanimoto_split_indices(active_df, active_col, test_split)


if __name__ == '__main__':
    main()