from typing import Optional, Dict
import logging
import os

from .optuna_utils import get_majority_vote_metrics, get_dataframe_stats
from .protac_dataset import get_datasets

import optuna
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import joblib
from optuna.samplers import TPESampler
import torch


xgb.set_config(verbosity=0)


def train_and_evaluate_xgboost(
        protein2embedding: Dict,
        cell2embedding: Dict,
        smiles2fp: Dict,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        params: dict,
        test_df: Optional[pd.DataFrame] = None,
        active_label: str = 'Active',
        num_boost_round: int = 100,
        shuffle_train_data: bool = False,
) -> tuple:
    """
    Train and evaluate an XGBoost model with the given parameters.
    
    Args:
        train_df (pd.DataFrame): The training and validation data.
        test_df (pd.DataFrame): The test data.
        params (dict): Hyperparameters for the XGBoost model.
        active_label (str): The active label column.
        num_boost_round (int): Maximum number of epochs.

    Returns:
        tuple: The trained model, test predictions, and metrics.
    """
    # Get datasets and their numpy arrays
    train_ds, val_ds, test_ds  = get_datasets(
        protein2embedding=protein2embedding,
        cell2embedding=cell2embedding,
        smiles2fp=smiles2fp,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        disabled_embeddings=[],
        active_label=active_label,
        apply_scaling=False,
    )
    X_train, y_train = train_ds.get_numpy_arrays()
    X_val, y_val = val_ds.get_numpy_arrays()

    # Shuffle the training data
    if shuffle_train_data:
        idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]
 
    # Setup training and validation data in XGBoost data format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    evallist = [(dval, 'eval'), (dtrain, 'train')]
 
    # Setup test data
    if test_df is not None:
        X_test, y_test = test_ds.get_numpy_arrays()
        dtest = xgb.DMatrix(X_test, label=y_test)
        evallist.append((dtest, 'test'))

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evallist,
        early_stopping_rounds=10,
        verbose_eval=False,
    )

    # Evaluate model
    val_pred = model.inplace_predict(dval, (model.best_iteration, model.best_iteration))
    val_pred_binary = (val_pred > 0.5).astype(int)
    metrics = {
        'val_acc': accuracy_score(y_val, val_pred_binary),
        'val_roc_auc': roc_auc_score(y_val, val_pred),
        'val_precision': precision_score(y_val, val_pred_binary),
        'val_recall': recall_score(y_val, val_pred_binary),
        'val_f1_score': f1_score(y_val, val_pred_binary),
    }
    preds = {'val_pred': val_pred}

    if test_df is not None:
        test_pred = model.inplace_predict(dtest, (model.best_iteration, model.best_iteration))
        test_pred_binary = (test_pred > 0.5).astype(int)        
        metrics.update({
            'test_acc': accuracy_score(y_test, test_pred_binary),
            'test_roc_auc': roc_auc_score(y_test, test_pred),
            'test_precision': precision_score(y_test, test_pred_binary),
            'test_recall': recall_score(y_test, test_pred_binary),
            'test_f1_score': f1_score(y_test, test_pred_binary),
        })
        preds.update({'test_pred': test_pred})
    
    return model, preds, metrics


def xgboost_model_objective(
        trial: optuna.Trial,
        protein2embedding: Dict,
        cell2embedding: Dict,
        smiles2fp: Dict,
        train_val_df: pd.DataFrame,
        kf: StratifiedKFold,
        groups: Optional[np.array] = None,
        active_label: str = 'Active',
        num_boost_round: int = 100,
        model_name: Optional[str] = None,
) -> float:
    """ Objective function for hyperparameter optimization with XGBoost.
    
    Args:
        trial (optuna.Trial): The Optuna trial object.
        train_val_df (pd.DataFrame): The training and validation data.
        kf (StratifiedKFold): Stratified K-Folds cross-validator.
        test_df (Optional[pd.DataFrame]): The test data.
        active_label (str): The active label column.
        num_boost_round (int): Maximum number of epochs.
        model_name (Optional[str]): The prefix name of the CV models to save, if supplied. Used as: `f"{model_name}_fold_{k}.json"`
    """
    # Suggest hyperparameters to be used across the CV folds
    params = {
        'booster': 'gbtree',
        'tree_method': 'hist', # if torch.cuda.is_available() else 'hist',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': trial.suggest_float('eta', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-4, 1e-1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    
    X = train_val_df.copy().drop(columns=active_label)
    y = train_val_df[active_label].tolist()
    report = []
    val_preds = []

    for k, (train_index, val_index) in enumerate(kf.split(X, y, groups)):
        logging.info(f'Fold {k + 1}/{kf.get_n_splits()}')
        train_df = train_val_df.iloc[train_index]
        val_df = train_val_df.iloc[val_index]

        # Get some statistics from the dataframes
        stats = {
            'model_type': 'XGBoost',
            'fold': k,
            'train_len': len(train_df),
            'val_len': len(val_df),
            'train_perc': len(train_df) / len(train_val_df),
            'val_perc': len(val_df) / len(train_val_df),
        }
        stats.update(get_dataframe_stats(train_df, val_df, active_label=active_label))
        if groups is not None:
            stats['train_unique_groups'] = len(np.unique(groups[train_index]))
            stats['val_unique_groups'] = len(np.unique(groups[val_index]))

        bst, preds, metrics = train_and_evaluate_xgboost(
            protein2embedding=protein2embedding,
            cell2embedding=cell2embedding,
            smiles2fp=smiles2fp,
            train_df=train_df,
            val_df=val_df,
            params=params,
            active_label=active_label,
            num_boost_round=num_boost_round,
        )
        stats.update(metrics)
        report.append(stats.copy())
        val_preds.append(preds['val_pred'])

        if model_name:
            model_filename = f'{model_name}_fold{k}.json'
            bst.save_model(model_filename)
            logging.info(f'CV XGBoost model saved to: {model_filename}')
    
    # Save the report in the trial
    trial.set_user_attr('report', report)
    trial.set_user_attr('val_preds', val_preds)
    trial.set_user_attr('params', params)
    
    # Get the average validation metrics across the folds
    mean_val_roc_auc = np.mean([r['val_roc_auc'] for r in report])
    logging.info(f'\tMean val ROC AUC: {mean_val_roc_auc:.4f}')
    
    # Optuna aims to minimize the objective, so return the negative ROC AUC
    return -mean_val_roc_auc


def xgboost_hyperparameter_tuning_and_training(
        protein2embedding: Dict,
        cell2embedding: Dict,
        smiles2fp: Dict,
        train_val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        kf: StratifiedKFold,
        groups: Optional[np.array] = None,
        split_type: str = 'random',
        n_models_for_test: int = 3,
        n_trials: int = 50,
        active_label: str = 'Active',
        num_boost_round: int = 100,
        study_filename: Optional[str] = None,
        force_study: bool = False,
        model_name: Optional[str] = None,
) -> dict:
    """ Hyperparameter tuning and training of an XGBoost model.
    
    Args:
        train_val_df (pd.DataFrame): The training and validation data.
        test_df (pd.DataFrame): The test data.
        kf (StratifiedKFold): Stratified K-Folds cross-validator.
        groups (Optional[np.array]): Group labels for the samples used while splitting the dataset into train/test set.
        split_type (str): Type of the data split. Used for reporting information.
        n_models_for_test (int): Number of models to train for testing.
        fast_dev_run (bool): Whether to run a fast development run.
        n_trials (int): Number of trials for hyperparameter optimization.
        logger_save_dir (str): Directory to save logs.
        logger_name (str): Name of the logger.
        active_label (str): The active label column.
        num_boost_round (int): Maximum number of epochs.
        study_filename (Optional[str]): File name to save/load the Optuna study.
        force_study (bool): Whether to force the study optimization even if the study file exists.

    Returns:
        dict: A dictionary containing reports from the CV and test.
    """
    # Set the verbosity of Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Create an Optuna study object
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    study_loaded = False
    if study_filename and not force_study:
        if os.path.exists(study_filename):
            study = joblib.load(study_filename)
            study_loaded = True
            logging.info(f'Loaded study from {study_filename}')

    if not study_loaded or force_study:
        study.optimize(
            lambda trial: xgboost_model_objective(
                trial=trial,
                protein2embedding=protein2embedding,
                cell2embedding=cell2embedding,
                smiles2fp=smiles2fp,
                train_val_df=train_val_df,
                kf=kf,
                groups=groups,
                active_label=active_label,
                num_boost_round=num_boost_round,
            ),
            n_trials=n_trials,
        )
        if study_filename:
            joblib.dump(study, study_filename)

    cv_report = pd.DataFrame(study.best_trial.user_attrs['report'])
    hparam_report = pd.DataFrame([study.best_params])

    # Train the best CV models and store their models by running the objective
    if model_name:
        xgboost_model_objective(
            trial=study.best_trial,
            protein2embedding=protein2embedding,
            cell2embedding=cell2embedding,
            smiles2fp=smiles2fp,
            train_val_df=train_val_df,
            kf=kf,
            groups=groups,
            active_label=active_label,
            num_boost_round=num_boost_round,
            model_name=f'{model_name}_cv_model_{split_type}',
        )

    # Retrain N models with the best hyperparameters (measure model uncertainty)
    best_models = []
    test_report = []
    test_preds = []
    for i in range(n_models_for_test):
        logging.info(f'Training best model {i + 1}/{n_models_for_test}')
        model, preds, metrics = train_and_evaluate_xgboost(
            protein2embedding=protein2embedding,
            cell2embedding=cell2embedding,
            smiles2fp=smiles2fp,
            train_df=train_val_df,
            val_df=test_df,
            params=study.best_trial.user_attrs['params'],
            active_label=active_label,
            num_boost_round=num_boost_round,
            shuffle_train_data=True,
        )
        metrics = {k.replace('val_', 'test_'): v for k, v in metrics.items()}
        metrics['model_type'] = 'XGBoost'
        metrics['test_model_id'] = i
        metrics.update(get_dataframe_stats(
            train_val_df,
            test_df=test_df,
            active_label=active_label,
        ))
        test_report.append(metrics.copy())
        test_preds.append(torch.tensor(preds['val_pred']))
        best_models.append(model)

        # Save the trained model
        if model_name:
            model_filename = f'{model_name}_best_model_{split_type}_n{i}.json'
            model.save_model(model_filename)
            logging.info(f'Best XGBoost model saved to: {model_filename}')
    test_report = pd.DataFrame(test_report)

    # Get the majority vote for the test predictions
    majority_vote_metrics = get_majority_vote_metrics(test_preds, test_df, active_label)
    majority_vote_report = pd.DataFrame([majority_vote_metrics])
    majority_vote_report['model_type'] = 'XGBoost'

    # Add a column with the split_type to all reports
    for report in [cv_report, hparam_report, test_report, majority_vote_report]:
        report['split_type'] = split_type

    # Return the reports
    return {
        'cv_report': cv_report,
        'hparam_report': hparam_report,
        'test_report': test_report,
        'majority_vote_report' :majority_vote_report,
    }
