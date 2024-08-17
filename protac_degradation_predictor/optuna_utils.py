import os
from typing import Literal, List, Tuple, Optional, Dict, Any
import logging

from .pytorch_models import (
    train_model,
    PROTAC_Model,
    evaluate_model,
    get_confidence_scores,
)
from .protac_dataset import get_datasets

import torch
import optuna
from optuna.samplers import TPESampler, QMCSampler
import joblib
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
)
import numpy as np
import pytorch_lightning as pl
from torchmetrics import (
    Accuracy,
    AUROC,
    Precision,
    Recall,
    F1Score,
)


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

def get_suggestion(trial, dtype, hparams_range):
    if dtype == 'int':
        return trial.suggest_int(**hparams_range)
    elif dtype == 'float':
        return trial.suggest_float(**hparams_range)
    elif dtype == 'categorical':
        return trial.suggest_categorical(**hparams_range)
    else:
        raise ValueError(f'Invalid dtype for trial.suggest: {dtype}')

def pytorch_model_objective(
        trial: optuna.Trial,
        protein2embedding: Dict,
        cell2embedding: Dict,
        smiles2fp: Dict,
        train_val_df: pd.DataFrame,
        kf: StratifiedKFold | StratifiedGroupKFold,
        groups: Optional[np.array] = None,
        test_df: Optional[pd.DataFrame] = None,
        hparams_ranges: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
        fast_dev_run: bool = False,
        active_label: str = 'Active',
        disabled_embeddings: List[str] = [],
        max_epochs: int = 100,
        use_logger: bool = False,
        logger_save_dir: str = 'logs',
        logger_name: str = 'cv_model',
        enable_checkpointing: bool = False,
) -> float:
    """ Objective function for hyperparameter optimization.
    
    Args:
        trial (optuna.Trial): The Optuna trial object.
        train_df (pd.DataFrame): The training set.
        val_df (pd.DataFrame): The validation set.
        hparams_ranges (List[Dict[str, Any]]): NOT IMPLEMENTED YET. Hyperparameters ranges.
            The list must be of a tuple of the type of hparam to suggest ('int', 'float', or 'categorical'), and the dictionary must contain the arguments of the corresponding trial.suggest method.
        fast_dev_run (bool): Whether to run a fast development run.
        active_label (str): The active label column.
        disabled_embeddings (List[str]): The list of disabled embeddings.
    """
    # Set fixed hyperparameters
    batch_size = 128
    apply_scaling = True # It is dynamically disabled for binary data
    use_batch_norm = True

    # Suggest hyperparameters to be used accross the CV folds
    hidden_dim = trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256, 512])
    smote_k_neighbors = trial.suggest_categorical('smote_k_neighbors', [0] + list(range(3, 16)))
    # hidden_dim = trial.suggest_int('hidden_dim', 32, 512, step=32)
    # smote_k_neighbors = trial.suggest_int('smote_k_neighbors', 0, 12)

    # use_smote = trial.suggest_categorical('use_smote', [True, False])
    # smote_k_neighbors = smote_k_neighbors if use_smote else 0
    # dropout = trial.suggest_float('dropout', 0, 0.5)
    # use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])

    # Optimizer parameters
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True)
    beta1 = trial.suggest_float('beta1', 0.1, 0.999)
    beta2 = trial.suggest_float('beta2', 0.1, 0.999)
    eps = trial.suggest_float('eps', 1e-9, 1.0, log=True)

    # Start the CV over the folds
    X = train_val_df.copy().drop(columns=active_label)
    y = train_val_df[active_label].tolist()
    report = []
    val_preds = []
    test_preds = []
    for k, (train_index, val_index) in enumerate(kf.split(X, y, groups)):
        logging.info(f'Fold {k + 1}/{kf.get_n_splits()}')
        # Get the train and val sets
        train_df = train_val_df.iloc[train_index]
        val_df = train_val_df.iloc[val_index]

        # Get some statistics from the dataframes
        stats = {
            'model_type': 'Pytorch',
            'fold': k,
            'train_len': len(train_df),
            'val_len': len(val_df),
            'train_perc': len(train_df) / len(train_val_df),
            'val_perc': len(val_df) / len(train_val_df),
        }
        stats.update(get_dataframe_stats(train_df, val_df, test_df, active_label))
        if groups is not None:
            stats['train_unique_groups'] = len(np.unique(groups[train_index]))
            stats['val_unique_groups'] = len(np.unique(groups[val_index]))

        # At each fold, train and evaluate the Pytorch model
        # Train the model with the current set of hyperparameters
        ret = train_model(
            protein2embedding=protein2embedding,
            cell2embedding=cell2embedding,
            smiles2fp=smiles2fp,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            use_batch_norm=use_batch_norm,
            # dropout=dropout,
            max_epochs=max_epochs,
            smote_k_neighbors=smote_k_neighbors,
            apply_scaling=apply_scaling,
            fast_dev_run=fast_dev_run,
            active_label=active_label,
            return_predictions=True,
            disabled_embeddings=disabled_embeddings,
            use_logger=use_logger,
            logger_save_dir=logger_save_dir,
            logger_name=f'{logger_name}_fold{k}',
            enable_checkpointing=enable_checkpointing,
        )
        if test_df is not None:
            _, _, metrics, val_pred, test_pred = ret
            test_preds.append(test_pred)
        else:
            _, _, metrics, val_pred = ret
        stats.update(metrics)
        report.append(stats.copy())
        val_preds.append(val_pred)

    # Save the report in the trial
    trial.set_user_attr('report', report)

    # Get the majority vote for the test predictions
    if test_df is not None and not fast_dev_run:
        majority_vote_metrics = get_majority_vote_metrics(test_preds, test_df, active_label)
        majority_vote_metrics.update(get_dataframe_stats(train_df, val_df, test_df, active_label))
        trial.set_user_attr('majority_vote_metrics', majority_vote_metrics)
        logging.info(f'Majority vote metrics: {majority_vote_metrics}')

    # Get the average validation accuracy and ROC AUC accross the folds
    val_roc_auc = np.mean([r['val_roc_auc'] for r in report])
    val_acc = np.mean([r['val_acc'] for r in report])
    logging.info(f'Average val accuracy: {val_acc}')
    logging.info(f'Average val ROC AUC: {val_roc_auc}')

    # Optuna aims to minimize the pytorch_model_objective
    return - val_roc_auc


def hyperparameter_tuning_and_training(
        protein2embedding: Dict,
        cell2embedding: Dict,
        smiles2fp: Dict,
        train_val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        kf: StratifiedKFold | StratifiedGroupKFold,
        groups: Optional[np.array] = None,
        split_type: str = 'standard',
        n_models_for_test: int = 3,
        fast_dev_run: bool = False,
        n_trials: int = 50,
        logger_save_dir: str = 'logs',
        logger_name: str = 'protac_hparam_search',
        active_label: str = 'Active',
        max_epochs: int = 100,
        study_filename: Optional[str] = None,
        force_study: bool = False,
) -> tuple:
    """ Hyperparameter tuning and training of a PROTAC model.
    
    Args:
        protein2embedding (Dict): The protein to embedding dictionary.
        cell2embedding (Dict): The cell to embedding dictionary.
        smiles2fp (Dict): The SMILES to fingerprint dictionary.
        train_val_df (pd.DataFrame): The training and validation set.
        test_df (pd.DataFrame): The test set.
        kf (StratifiedKFold | StratifiedGroupKFold): The KFold object.
        groups (np.array): The groups for the StratifiedGroupKFold.
        split_type (str): The split type of the current study. Used for reporting.
        n_models_for_test (int): The number of models to train for the test set.
        fast_dev_run (bool): Whether to run a fast development run.
        n_trials (int): The number of trials for the hyperparameter search.
        logger_save_dir (str): The logger save directory.
        logger_name (str): The logger name.
        active_label (str): The active label column.
        max_epochs (int): The maximum number of epochs.
        study_filename (str): The study filename.
        force_study (bool): Whether to force the study.

    Returns:
        tuple: The trained model, the trainer, and the best metrics.
    """
    pl.seed_everything(42)

    # TODO: Make the following code more modular, i.e., the ranges shall be put
    # in dictionaries or config files or something like that.
    hparams_ranges = None

    # Set the verbosity of Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Set a quasi-random sampler, as suggested in: https://github.com/google-research/tuning_playbook?tab=readme-ov-file#faqs
    # sampler = QMCSampler(qmc_type='halton', scramble=True, seed=42)
    sampler = TPESampler(seed=42, multivariate=True)
    # Create an Optuna study object
    study = optuna.create_study(direction='minimize', sampler=sampler)

    study_loaded = False
    if study_filename and not force_study:
        if os.path.exists(study_filename):
            study = joblib.load(study_filename)
            study_loaded = True
            logging.info(f'Loaded study from {study_filename}')
            logging.info(f'Study best params: {study.best_params}')

    if not study_loaded or force_study:
        study.optimize(
            lambda trial: pytorch_model_objective(
                trial=trial,
                protein2embedding=protein2embedding,
                cell2embedding=cell2embedding,
                smiles2fp=smiles2fp,
                train_val_df=train_val_df,
                kf=kf,
                groups=groups,
                test_df=test_df,
                hparams_ranges=hparams_ranges,
                fast_dev_run=fast_dev_run,
                active_label=active_label,
                max_epochs=max_epochs,
                disabled_embeddings=[],
            ),
            n_trials=n_trials,
        )
        if study_filename:
            joblib.dump(study, study_filename)

    cv_report = pd.DataFrame(study.best_trial.user_attrs['report'])
    hparam_report = pd.DataFrame([study.best_params])

    # Train the best CV models and store their checkpoints by running the objective
    pytorch_model_objective(
        trial=study.best_trial,
        protein2embedding=protein2embedding,
        cell2embedding=cell2embedding,
        smiles2fp=smiles2fp,
        train_val_df=train_val_df,
        kf=kf,
        groups=groups,
        test_df=test_df,
        hparams_ranges=hparams_ranges,
        fast_dev_run=fast_dev_run,
        active_label=active_label,
        max_epochs=max_epochs,
        disabled_embeddings=[],
        use_logger=True,
        logger_save_dir=logger_save_dir,
        logger_name=f'cv_model_{logger_name}',
        enable_checkpointing=True,
    )

    # Retrain N models with the best hyperparameters (measure model uncertainty)
    best_models = []
    test_report = []
    test_preds = []
    dfs_stats = get_dataframe_stats(train_val_df, test_df=test_df, active_label=active_label)
    for i in range(n_models_for_test):
        pl.seed_everything(42 + i + 1)
        model, trainer, metrics, test_pred = train_model(
            protein2embedding=protein2embedding,
            cell2embedding=cell2embedding,
            smiles2fp=smiles2fp,
            train_df=train_val_df,
            val_df=test_df,
            fast_dev_run=fast_dev_run,
            active_label=active_label,
            max_epochs=max_epochs,
            disabled_embeddings=[],
            use_logger=True,
            logger_save_dir=logger_save_dir,
            logger_name=f'best_model_n{i}_{logger_name}',
            enable_checkpointing=True,
            checkpoint_model_name=f'best_model_n{i}_{split_type}',
            return_predictions=True,
            batch_size=128,
            apply_scaling=True,
            # use_batch_norm=True,
            **study.best_params,
        )
        # Rename the keys in the metrics dictionary
        metrics = {k.replace('val_', 'test_'): v for k, v in metrics.items()}
        metrics['model_type'] = 'Pytorch'
        metrics['test_model_id'] = i
        metrics.update(dfs_stats)

        test_report.append(metrics.copy())
        test_preds.append(test_pred)
        best_models.append({'model': model, 'trainer': trainer})
    test_report = pd.DataFrame(test_report)

    # Get the majority vote for the test predictions
    if not fast_dev_run:
        majority_vote_metrics = get_majority_vote_metrics(test_preds, test_df, active_label)
        majority_vote_metrics.update(get_dataframe_stats(train_val_df, test_df=test_df, active_label=active_label))
        majority_vote_metrics_cv = study.best_trial.user_attrs['majority_vote_metrics']
        majority_vote_metrics_cv['cv_models'] = True
        majority_vote_report = pd.DataFrame([
            majority_vote_metrics,
            majority_vote_metrics_cv,
        ])
        majority_vote_report['model_type'] = 'Pytorch'
        majority_vote_report['split_type'] = split_type

    # Ablation study: disable embeddings at a time
    ablation_report = []
    dfs_stats = get_dataframe_stats(train_val_df, test_df=test_df, active_label=active_label)
    disabled_embeddings_combinations = [
        ['e3'],
        ['poi'],
        ['cell'],
        ['smiles'],
        ['e3', 'cell'],
        ['poi', 'e3'],
        ['poi', 'e3', 'cell'],
    ]
    for disabled_embeddings in disabled_embeddings_combinations:
        logging.info('-' * 100)
        logging.info(f'Ablation study with disabled embeddings: {disabled_embeddings}')
        logging.info('-' * 100)
        disabled_embeddings_str = 'disabled ' + ' '.join(disabled_embeddings)
        test_preds = []
        for i, model_trainer in enumerate(best_models):
            logging.info(f'Evaluating model n.{i} on {disabled_embeddings_str}.')
            model = model_trainer['model']
            trainer = model_trainer['trainer']
            _, test_ds, _  = get_datasets(
                protein2embedding=protein2embedding,
                cell2embedding=cell2embedding,
                smiles2fp=smiles2fp,
                train_df=train_val_df,
                val_df=test_df,
                disabled_embeddings=disabled_embeddings,
                active_label=active_label,
                scaler=model.scalers,
                use_single_scaler=model.join_embeddings == 'beginning',
            )
            ret = evaluate_model(model, trainer, test_ds, batch_size=128)
            # NOTE: We are passing the test set as the validation set argument
            # Rename the keys in the metrics dictionary
            test_preds.append(ret['val_pred'])
            ret['val_metrics'] = {k.replace('val_', 'test_'): v for k, v in ret['val_metrics'].items()}
            ret['val_metrics'].update(dfs_stats)
            ret['val_metrics']['majority_vote'] = False
            ret['val_metrics']['model_type'] = 'Pytorch'
            ret['val_metrics']['disabled_embeddings'] = disabled_embeddings_str
            ablation_report.append(ret['val_metrics'].copy())

        # Get the majority vote for the test predictions
        if not fast_dev_run:
            majority_vote_metrics = get_majority_vote_metrics(test_preds, test_df, active_label)
            majority_vote_metrics.update(dfs_stats)
            majority_vote_metrics['majority_vote'] = True
            majority_vote_metrics['model_type'] = 'Pytorch'
            majority_vote_metrics['disabled_embeddings'] = disabled_embeddings_str
            ablation_report.append(majority_vote_metrics.copy())

    ablation_report = pd.DataFrame(ablation_report)

    # Add a column with the split_type to all reports
    for report in [cv_report, hparam_report, test_report, ablation_report]:
        report['split_type'] = split_type

    # Return the reports
    ret = {
        'cv_report': cv_report,
        'hparam_report': hparam_report,
        'test_report': test_report,
        'ablation_report': ablation_report,
    }
    if not fast_dev_run:
        ret['majority_vote_report'] = majority_vote_report
    return ret