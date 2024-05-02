import os
from typing import Literal, List, Tuple, Optional, Dict
import logging

from .pytorch_models import (
    train_model,
    PROTAC_Model,
    evaluate_model,
)
from .protac_dataset import get_datasets

from .sklearn_models import (
    train_sklearn_model,
    suggest_random_forest,
    suggest_logistic_regression,
    suggest_svc,
    suggest_gradient_boosting,
)

import torch
import optuna
from optuna.samplers import TPESampler
import joblib
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
    test_preds = torch.stack(test_preds)
    test_preds, _ = torch.mode(test_preds, dim=0)
    y = torch.tensor(test_df[active_label].tolist())
    # Measure the test accuracy and ROC AUC
    majority_vote_metrics = {
        'test_acc': Accuracy(task='binary')(test_preds, y).item(),
        'test_roc_auc': AUROC(task='binary')(test_preds, y).item(),
        'test_precision': Precision(task='binary')(test_preds, y).item(),
        'test_recall': Recall(task='binary')(test_preds, y).item(),
        'test_f1': F1Score(task='binary')(test_preds, y).item(),
    }
    return majority_vote_metrics


def pytorch_model_objective(
        trial: optuna.Trial,
        protein2embedding: Dict,
        cell2embedding: Dict,
        smiles2fp: Dict,
        train_val_df: pd.DataFrame,
        kf: StratifiedKFold | StratifiedGroupKFold,
        groups: Optional[np.array] = None,
        test_df: Optional[pd.DataFrame] = None,
        hidden_dim_options: List[int] = [256, 512, 768],
        batch_size_options: List[int] = [8, 16, 32],
        learning_rate_options: Tuple[float, float] = (1e-5, 1e-3),
        smote_k_neighbors_options: List[int] = list(range(3, 16)),
        dropout_options: Tuple[float, float] = (0.1, 0.5),
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
        hidden_dim_options (List[int]): The hidden dimension options.
        batch_size_options (List[int]): The batch size options.
        learning_rate_options (Tuple[float, float]): The learning rate options.
        smote_k_neighbors_options (List[int]): The SMOTE k neighbors options.
        dropout_options (Tuple[float, float]): The dropout options.
        fast_dev_run (bool): Whether to run a fast development run.
        active_label (str): The active label column.
        disabled_embeddings (List[str]): The list of disabled embeddings.
    """
    # Suggest hyperparameters to be used accross the CV folds
    hidden_dim = trial.suggest_categorical('hidden_dim', hidden_dim_options)
    batch_size = 128 # trial.suggest_categorical('batch_size', batch_size_options)
    learning_rate = trial.suggest_float('learning_rate', *learning_rate_options, log=True)
    smote_k_neighbors = trial.suggest_categorical('smote_k_neighbors', smote_k_neighbors_options)
    use_smote = trial.suggest_categorical('use_smote', [True, False])
    apply_scaling = True # trial.suggest_categorical('apply_scaling', [True, False])
    dropout = trial.suggest_float('dropout', *dropout_options)
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])

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
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            max_epochs=max_epochs,
            smote_k_neighbors=smote_k_neighbors,
            apply_scaling=apply_scaling,
            use_smote=use_smote,
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
        split_type: str = 'random',
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
        train_df (pd.DataFrame): The training set.
        val_df (pd.DataFrame): The validation set.
        test_df (pd.DataFrame): The test set.
        fast_dev_run (bool): Whether to run a fast development run.
        n_trials (int): The number of hyperparameter optimization trials.
        logger_name (str): The name of the logger.
        active_label (str): The active label column.
        disabled_embeddings (List[str]): The list of disabled embeddings.

    Returns:
        tuple: The trained model, the trainer, and the best metrics.
    """
    pl.seed_everything(42)

    # Define the search space
    hidden_dim_options = [16, 32, 64, 128, 256] #, 512]
    batch_size_options = [128, 128] # [4, 8, 16, 32, 64, 128]
    learning_rate_options = (1e-6, 1e-3) # min and max values for loguniform distribution
    smote_k_neighbors_options = list(range(3, 16))
    # NOTE: We want Optuna to explore the combination (very low dropout, very
    # small hidden_dim)
    dropout_options = (0, 0.5)

    # Set the verbosity of Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Create an Optuna study object
    sampler = TPESampler(seed=42, multivariate=True)
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
                hidden_dim_options=hidden_dim_options,
                batch_size_options=batch_size_options,
                learning_rate_options=learning_rate_options,
                smote_k_neighbors_options=smote_k_neighbors_options,
                dropout_options=dropout_options,
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
        hidden_dim_options=hidden_dim_options,
        batch_size_options=batch_size_options,
        learning_rate_options=learning_rate_options,
        smote_k_neighbors_options=smote_k_neighbors_options,
        dropout_options=dropout_options,
        fast_dev_run=fast_dev_run,
        active_label=active_label,
        max_epochs=max_epochs,
        disabled_embeddings=[],
        use_logger=True,
        logger_save_dir=logger_save_dir,
        logger_name=f'{logger_name}_{split_type}_cv_model',
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
            logger_name=f'{logger_name}_best_model_n{i}',
            enable_checkpointing=True,
            checkpoint_model_name=f'best_model_n{i}_{split_type}',
            return_predictions=True,
            batch_size=128,
            apply_scaling=True,
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

        # _, _, metrics = train_model(
        #     protein2embedding=protein2embedding,
        #     cell2embedding=cell2embedding,
        #     smiles2fp=smiles2fp,
        #     train_df=train_val_df,
        #     val_df=test_df,
        #     fast_dev_run=fast_dev_run,
        #     active_label=active_label,
        #     max_epochs=max_epochs,
        #     use_logger=False,
        #     logger_save_dir=logger_save_dir,
        #     logger_name=f'{logger_name}_disabled-{"-".join(disabled_embeddings)}',
        #     disabled_embeddings=disabled_embeddings,
        #     batch_size=128,
        #     apply_scaling=True,
        #     **study.best_params,
        # )
        # # Rename the keys in the metrics dictionary
        # metrics = {k.replace('val_', 'test_'): v for k, v in metrics.items()}
        # metrics['disabled_embeddings'] = disabled_embeddings_str
        # metrics['model_type'] = 'Pytorch'
        # metrics.update(dfs_stats)

        # # Add the training metrics        
        # train_metrics = {m: v.item() for m, v in trainer.callback_metrics.items() if 'train' in m}
        # metrics.update(train_metrics)
        # ablation_report.append(metrics.copy())

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


def sklearn_model_objective(
        trial: optuna.Trial,
        protein2embedding: Dict,
        cell2embedding: Dict,
        smiles2fp: Dict,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_type: Literal['RandomForest', 'SVC', 'LogisticRegression', 'GradientBoosting'] = 'RandomForest',
        active_label: str = 'Active',
) -> float:
    """ Objective function for hyperparameter optimization.
    
    Args:
        trial (optuna.Trial): The Optuna trial object.
        train_df (pd.DataFrame): The training set.
        val_df (pd.DataFrame): The validation set.
        model_type (str): The model type.
        hyperparameters (Dict): The hyperparameters for the model.
        fast_dev_run (bool): Whether to run a fast development run.
        active_label (str): The active label column.
    """

    # Generate the hyperparameters
    use_single_scaler = trial.suggest_categorical('use_single_scaler', [True, False])
    if model_type == 'RandomForest':
        clf = suggest_random_forest(trial)
    elif model_type == 'SVC':
        clf = suggest_svc(trial)
    elif model_type == 'LogisticRegression':
        clf = suggest_logistic_regression(trial)
    elif model_type == 'GradientBoosting':
        clf = suggest_gradient_boosting(trial)
    else:
        raise ValueError(f'Invalid model type: {model_type}. Available: RandomForest, SVC, LogisticRegression, GradientBoosting.')

    # Train the model with the current set of hyperparameters
    _, metrics = train_sklearn_model(
        clf=clf,
        protein2embedding=protein2embedding,
        cell2embedding=cell2embedding,
        smiles2fp=smiles2fp,
        train_df=train_df,
        val_df=val_df,
        active_label=active_label,
        use_single_scaler=use_single_scaler,
    )
    
    # Metrics is a dictionary containing at least the validation loss
    val_acc = metrics['val_acc']
    val_roc_auc = metrics['val_roc_auc']
    
    # Optuna aims to minimize the sklearn_model_objective
    return - val_acc - val_roc_auc


def hyperparameter_tuning_and_training_sklearn(
        protein2embedding: Dict,
        cell2embedding: Dict,
        smiles2fp: Dict,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        model_type: Literal['RandomForest', 'SVC', 'LogisticRegression', 'GradientBoosting'] = 'RandomForest',
        active_label: str = 'Active',
        n_trials: int = 50,
        logger_name: str = 'protac_hparam_search_sklearn',
        study_filename: Optional[str] = None,
) -> Tuple:
    """ Hyperparameter tuning and training of a PROTAC model.
    
    Args:
        train_df (pd.DataFrame): The training set.
        val_df (pd.DataFrame): The validation set.
        test_df (pd.DataFrame): The test set.
        model_type (str): The model type.
        n_trials (int): The number of hyperparameter optimization trials.
        logger_name (str): The name of the logger. Unused, for compatibility with hyperparameter_tuning_and_training.
        active_label (str): The active label column.

    Returns:
        tuple: The trained model and the best metrics.
    """
    # Set the verbosity of Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Create an Optuna study object
    sampler = TPESampler(seed=42, multivariate=True)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    study_loaded = False
    if study_filename:
        if os.path.exists(study_filename):
            study = joblib.load(study_filename)
            study_loaded = True
            logging.info(f'Loaded study from {study_filename}')
    
    if not study_loaded:
        study.optimize(
            lambda trial: sklearn_model_objective(
                trial=trial,
                protein2embedding=protein2embedding,
                cell2embedding=cell2embedding,
                smiles2fp=smiles2fp,
                train_df=train_df,
                val_df=val_df,
                model_type=model_type,
                active_label=active_label,
            ),
            n_trials=n_trials,
        )
        if study_filename:
            joblib.dump(study, study_filename)
    
    # Retrain the model with the best hyperparameters
    best_hyperparameters = {k.replace('model_', ''): v for k, v in study.best_params.items() if k.startswith('model_')}
    if model_type == 'RandomForest':
        clf = RandomForestClassifier(random_state=42, **best_hyperparameters)
    elif model_type == 'SVC':
        clf = SVC(random_state=42, probability=True, **best_hyperparameters)
    elif model_type == 'LogisticRegression':
        clf = LogisticRegression(random_state=42, max_iter=1000, **best_hyperparameters)
    elif model_type == 'GradientBoosting':
        clf = GradientBoostingClassifier(random_state=42, **best_hyperparameters)
    else:
        raise ValueError(f'Invalid model type: {model_type}. Available: RandomForest, SVC, LogisticRegression, GradientBoosting.')

    model, metrics = train_sklearn_model(
        clf=clf,
        protein2embedding=protein2embedding,
        cell2embedding=cell2embedding,
        smiles2fp=smiles2fp,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        active_label=active_label,
        use_single_scaler=study.best_params['use_single_scaler'],
    )

    # Report the best hyperparameters found
    metrics.update({f'hparam_{k}': v for k, v in study.best_params.items()})

    # Return the best metrics
    return model, metrics