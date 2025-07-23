""" """
import logging
from typing import Dict, Optional, List, Union

import pandas as pd
from torchmetrics import (
    Accuracy,
    AUROC,
    Precision,
    Recall,
    F1Score,
)


def get_dataframe_stats(
        train_df: Optional[pd.DataFrame] = None,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
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
    if train_df is not None and test_df is not None:
        leaking_uniprot = list(set(train_df['Uniprot']).intersection(set(test_df['Uniprot'])))
        leaking_smiles = list(set(train_df['Smiles']).intersection(set(test_df['Smiles'])))
        stats['num_leaking_uniprot_train_test'] = len(leaking_uniprot)
        stats['num_leaking_smiles_train_test'] = len(leaking_smiles)
        stats['perc_leaking_uniprot_train_test'] = len(train_df[train_df['Uniprot'].isin(leaking_uniprot)]) / len(train_df)
        stats['perc_leaking_smiles_train_test'] = len(train_df[train_df['Smiles'].isin(leaking_smiles)]) / len(train_df)
    return stats

def get_majority_vote_metrics(
        test_preds: List,
        test_df: pd.DataFrame,
        active_label: str = 'Active',
) -> Dict:
    """ Get the majority vote metrics. """
    test_preds_mean = np.array(test_preds).mean(axis=0)
    logging.info(f'Test predictions: {test_preds}')
    logging.info(f'Test predictions mean: {test_preds_mean}')
    test_preds = torch.stack(test_preds)
    test_preds, _ = torch.mode(test_preds, dim=0)
    y = torch.tensor(test_df[active_label].tolist())
    # Measure the test accuracy and ROC AUC
    majority_vote_metrics = {
        'test_acc': Accuracy(task='binary')(test_preds, y).item(),
        'test_roc_auc': AUROC(task='binary')(test_preds, y).item(),
        'test_precision': Precision(task='binary')(test_preds, y).item(),
        'test_recall': Recall(task='binary')(test_preds, y).item(),
        'test_f1_score': F1Score(task='binary')(test_preds, y).item(),
    }

    # Get mean predictions
    fp_mean, fn_mean = get_confidence_scores(y, test_preds_mean)
    majority_vote_metrics['test_false_negatives_mean'] = fn_mean
    majority_vote_metrics['test_false_positives_mean'] = fp_mean

    return majority_vote_metrics