import os
from typing import Literal, List, Tuple, Optional, Dict

from pytorch_models import train_model
from sklearn_models import (
    train_sklearn_model,
    suggest_random_forest,
    suggest_logistic_regression,
    suggest_svc,
    suggest_gradient_boosting,
)

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


def pytorch_model_objective(
        trial: optuna.Trial,
        protein2embedding: Dict,
        cell2embedding: Dict,
        smiles2fp: Dict,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        hidden_dim_options: List[int] = [256, 512, 768],
        batch_size_options: List[int] = [8, 16, 32],
        learning_rate_options: Tuple[float, float] = (1e-5, 1e-3),
        smote_k_neighbors_options: List[int] = list(range(3, 16)),
        dropout_options: Tuple[float, float] = (0.1, 0.5),
        fast_dev_run: bool = False,
        active_label: str = 'Active',
        disabled_embeddings: List[str] = [],
        max_epochs: int = 100,
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
    # Generate the hyperparameters
    hidden_dim = trial.suggest_categorical('hidden_dim', hidden_dim_options)
    batch_size = trial.suggest_categorical('batch_size', batch_size_options)
    learning_rate = trial.suggest_float('learning_rate', *learning_rate_options, log=True)
    join_embeddings = trial.suggest_categorical('join_embeddings', ['beginning', 'concat', 'sum'])
    smote_k_neighbors = trial.suggest_categorical('smote_k_neighbors', smote_k_neighbors_options)
    use_smote = trial.suggest_categorical('use_smote', [True, False])
    apply_scaling = trial.suggest_categorical('apply_scaling', [True, False])
    dropout = trial.suggest_float('dropout', *dropout_options)

    # Train the model with the current set of hyperparameters
    _, _, metrics = train_model(
        protein2embedding,
        cell2embedding,
        smiles2fp,
        train_df,
        val_df,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        join_embeddings=join_embeddings,
        learning_rate=learning_rate,
        dropout=dropout,
        max_epochs=max_epochs,
        smote_k_neighbors=smote_k_neighbors,
        apply_scaling=apply_scaling,
        use_smote=use_smote,
        use_logger=False,
        fast_dev_run=fast_dev_run,
        active_label=active_label,
        disabled_embeddings=disabled_embeddings,
    )

    # Metrics is a dictionary containing at least the validation loss
    val_loss = metrics['val_loss']
    val_acc = metrics['val_acc']
    val_roc_auc = metrics['val_roc_auc']
    
    # Optuna aims to minimize the pytorch_model_objective
    return val_loss - val_acc - val_roc_auc


def hyperparameter_tuning_and_training(
        protein2embedding: Dict,
        cell2embedding: Dict,
        smiles2fp: Dict,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        fast_dev_run: bool = False,
        n_trials: int = 50,
        logger_name: str = 'protac_hparam_search',
        active_label: str = 'Active',
        disabled_embeddings: List[str] = [],
        study_filename: Optional[str] = None,
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
    # Define the search space
    hidden_dim_options = [256, 512, 768]
    batch_size_options = [8, 16, 32]
    learning_rate_options = (1e-5, 1e-3) # min and max values for loguniform distribution
    smote_k_neighbors_options = list(range(3, 16))

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
            print(f'Loaded study from {study_filename}')

    if not study_loaded:
        study.optimize(
            lambda trial: pytorch_model_objective(
                trial=trial,
                protein2embedding=protein2embedding,
                cell2embedding=cell2embedding,
                smiles2fp=smiles2fp,
                train_df=train_df,
                val_df=val_df,
                hidden_dim_options=hidden_dim_options,
                batch_size_options=batch_size_options,
                learning_rate_options=learning_rate_options,
                smote_k_neighbors_options=smote_k_neighbors_options,
                fast_dev_run=fast_dev_run,
                active_label=active_label,
                disabled_embeddings=disabled_embeddings,
            ),
            n_trials=n_trials,
        )
        if study_filename:
            joblib.dump(study, study_filename)

    # Retrain the model with the best hyperparameters
    model, trainer, metrics = train_model(
        protein2embedding=protein2embedding,
        cell2embedding=cell2embedding,
        smiles2fp=smiles2fp,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        use_logger=True,
        logger_name=logger_name,
        fast_dev_run=fast_dev_run,
        active_label=active_label,
        disabled_embeddings=disabled_embeddings,
        **study.best_params,
    )

    # Report the best hyperparameters found
    metrics.update({f'hparam_{k}': v for k, v in study.best_params.items()})

    # Return the best metrics
    return model, trainer, metrics


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
        logger_name: str = 'protac_hparam_search',
        study_filename: Optional[str] = None,
) -> Tuple:
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
            print(f'Loaded study from {study_filename}')
    
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