from typing import Literal, List, Tuple, Optional, Dict

from protac_dataset import PROTAC_Dataset

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import torch
import torch.nn as nn
from torchmetrics import (
    Accuracy,
    AUROC,
    Precision,
    Recall,
    F1Score,
    MetricCollection,
)
import optuna


def train_sklearn_model(
    clf: ClassifierMixin,
    protein2embedding: Dict,
    cell2embedding: Dict,
    smiles2fp: Dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    active_label: str = 'Active',
    use_single_scaler: bool = True,
) -> Tuple[ClassifierMixin, Dict]:
    """ Train a classifier model on train and val sets and evaluate it on a test set.

    Args:
        clf: The classifier model to train and evaluate.
        train_df (pd.DataFrame): The training set.
        val_df (pd.DataFrame): The validation set.
        test_df (Optional[pd.DataFrame]): The test set.

    Returns:
        Tuple[ClassifierMixin, nn.ModuleDict]: The trained model and the metrics.
    """
    # Initialize the datasets
    train_ds = PROTAC_Dataset(
        train_df,
        protein2embedding,
        cell2embedding,
        smiles2fp,
        active_label=active_label,
        use_smote=False,
    )
    scaler = train_ds.fit_scaling(use_single_scaler=use_single_scaler)
    train_ds.apply_scaling(scaler, use_single_scaler=use_single_scaler)
    val_ds = PROTAC_Dataset(
        val_df,
        protein2embedding,
        cell2embedding,
        smiles2fp,
        active_label=active_label,
        use_smote=False,
    )
    val_ds.apply_scaling(scaler, use_single_scaler=use_single_scaler)
    if test_df is not None:
        test_ds = PROTAC_Dataset(
            test_df,
            protein2embedding,
            cell2embedding,
            smiles2fp,
            active_label=active_label,
            use_smote=False,
        )
        test_ds.apply_scaling(scaler, use_single_scaler=use_single_scaler)

    # Get the numpy arrays
    X_train, y_train = train_ds.get_numpy_arrays()
    X_val, y_val = val_ds.get_numpy_arrays()
    if test_df is not None:
        X_test, y_test = test_ds.get_numpy_arrays()

    # Train the model
    clf.fit(X_train, y_train)
    # Define the metrics as a module dict
    stages = ['train_metrics', 'val_metrics', 'test_metrics']
    metrics = nn.ModuleDict({s: MetricCollection({
        'acc': Accuracy(task='binary'),
        'roc_auc': AUROC(task='binary'),
        'precision': Precision(task='binary'),
        'recall': Recall(task='binary'),
        'f1_score': F1Score(task='binary'),
        'opt_score': Accuracy(task='binary') + F1Score(task='binary'),
        'hp_metric': Accuracy(task='binary'),
    }, prefix=s.replace('metrics', '')) for s in stages})

    # Get the predictions
    metrics_out = {}

    y_pred = torch.tensor(clf.predict_proba(X_train)[:, 1])
    y_true = torch.tensor(y_train)
    metrics['train_metrics'].update(y_pred, y_true)
    metrics_out.update(metrics['train_metrics'].compute())

    y_pred = torch.tensor(clf.predict_proba(X_val)[:, 1])
    y_true = torch.tensor(y_val)
    metrics['val_metrics'].update(y_pred, y_true)
    metrics_out.update(metrics['val_metrics'].compute())

    if test_df is not None:
        y_pred = torch.tensor(clf.predict_proba(X_test)[:, 1])
        y_true = torch.tensor(y_test)
        metrics['test_metrics'].update(y_pred, y_true)
        metrics_out.update(metrics['test_metrics'].compute())

    return clf, metrics_out


def suggest_random_forest(
        trial: optuna.Trial,
) -> ClassifierMixin:
    """ Suggest hyperparameters for a Random Forest classifier.

    Args:
        trial (optuna.Trial): The Optuna trial object.

    Returns:
        ClassifierMixin: The Random Forest classifier with the suggested hyperparameters.
    """
    n_estimators = trial.suggest_int('model_n_estimators', 10, 1000)
    max_depth = trial.suggest_int('model_max_depth', 2, 100)
    min_samples_split = trial.suggest_int('model_min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('model_min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('model_max_features', [None, 'sqrt', 'log2'])
    criterion = trial.suggest_categorical('model_criterion', ['gini', 'entropy'])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        random_state=42,
    )

    return clf


def suggest_logistic_regression(
        trial: optuna.Trial,
) -> ClassifierMixin:
    """ Suggest hyperparameters for a Logistic Regression classifier.

    Args:
        trial (optuna.Trial): The Optuna trial object.

    Returns:
        ClassifierMixin: The Logistic Regression classifier with the suggested hyperparameters.
    """
        # Suggest values for the logistic regression hyperparameters
    C = trial.suggest_loguniform('model_C', 1e-4, 1e2)
    penalty = trial.suggest_categorical('model_penalty', ['l1', 'l2', 'elasticnet', None])
    solver = trial.suggest_categorical('model_solver', ['newton-cholesky', 'lbfgs', 'liblinear', 'sag', 'saga'])

    # Check solver compatibility
    if penalty == 'l1' and solver not in ['liblinear', 'saga']:
        raise optuna.exceptions.TrialPruned()
    if penalty == None and solver not in ['newton-cholesky', 'lbfgs', 'sag']:
        raise optuna.exceptions.TrialPruned()

    # Configure the classifier with the trial's suggested parameters
    clf = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=1000,
        random_state=42,
    )

    return clf


def suggest_svc(
        trial: optuna.Trial,
) -> ClassifierMixin:
    """ Suggest hyperparameters for an SVC classifier.

    Args:
        trial (optuna.Trial): The Optuna trial object.

    Returns:
        ClassifierMixin: The SVC classifier with the suggested hyperparameters.
    """
    C = trial.suggest_loguniform('model_C', 1e-4, 1e2)
    kernel = trial.suggest_categorical('model_kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    gamma = trial.suggest_categorical('model_gamma', ['scale', 'auto'])
    degree = trial.suggest_int('model_degree', 2, 5) if kernel == 'poly' else 3
    
    clf = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        probability=True,
        random_state=42,
    )

    return clf


def suggest_gradient_boosting(
        trial: optuna.Trial,
) -> ClassifierMixin:
    """ Suggest hyperparameters for a Gradient Boosting classifier.

    Args:
        trial (optuna.Trial): The Optuna trial object.

    Returns:
        ClassifierMixin: The Gradient Boosting classifier with the suggested hyperparameters.
    """
    n_estimators = trial.suggest_int('model_n_estimators', 50, 500)
    learning_rate = trial.suggest_loguniform('model_learning_rate', 0.01, 1)
    max_depth = trial.suggest_int('model_max_depth', 3, 10)
    min_samples_split = trial.suggest_int('model_min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('model_min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('model_max_features', ['sqrt', 'log2', None])
    
    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
    )

    return clf