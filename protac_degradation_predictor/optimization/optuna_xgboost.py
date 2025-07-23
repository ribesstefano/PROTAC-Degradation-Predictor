""" XGBoost hyperparameter optimization and training for degradation prediction. """
import os
import logging
from typing import Optional, Dict, List, Union, Literal
from pathlib import Path

import optuna
import pandas as pd
import numpy as np
import joblib
import torch
from sklearn.model_selection import StratifiedKFold
from optuna.samplers import TPESampler

from protac_degradation_predictor.data.utils import get_cache_dir
from protac_degradation_predictor.data.datasets import MolPoiE3CellDataset
from protac_degradation_predictor.evaluation import get_confidence_scores
from protac_degradation_predictor.optimization.utils import (
    get_dataframe_stats,
    get_majority_vote_metrics,
)
from protac_degradation_predictor.models.xgboost_model import (
    train_and_eval_xgboost_classifier,
    train_and_eval_xgboost_regressor
)

def xgboost_objective(
        trial: optuna.Trial,
        train_val_df: pd.DataFrame,
        label_columns: Union[str, List[str]],
        dataset_kwargs: Dict = {},
        test_df: Optional[pd.DataFrame] = None,
        kf: Optional[StratifiedKFold] = None,
        groups: Optional[np.array] = None,
        model_name: Optional[str] = None,
        is_regression_task: bool = False,
        log_dir: Optional[str] = None,
) -> float:
    """ Objective function for hyperparameter optimization with XGBoost (classification or regression). """

    # --- Suggest XGBoost hyperparameters ---
    xgb_params = {
        "booster": "gbtree",
        "tree_method": "hist",
        "eta": trial.suggest_float("eta", 1e-4, 1e-1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1e-1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    if is_regression_task:
        xgb_params["objective"] = "reg:squarederror"
        xgb_params["eval_metric"] = "rmse"
    else:
        xgb_params["objective"] = "binary:logistic"
        xgb_params["eval_metric"] = "auc"

    # PCA
    use_pca = trial.suggest_categorical("use_pca", [False, True])
    pca_params = None if not use_pca else {
        "n_components": trial.suggest_float("pca_n_components", 0.8, 0.99, step=0.01),
        "svd_solver": "full",
    }

    # --- Suggest dataset hyperparameters ---
    # Molecule encoding type
    mol_encoding_type = trial.suggest_categorical("mol_encoding_type", ["fingerprint", "transformer"])
    mol_fp_size = trial.suggest_int("mol_fp_size", 128, 2048, step=128) if mol_encoding_type == "fingerprint" else None
    mol_radius = trial.suggest_int("mol_fp_radius", 2, 4) if mol_encoding_type == "fingerprint" else None

    # Protein encoding type
    prot_encoding_type = trial.suggest_categorical("prot_encoding_type", ["esm", "amino_acid_count"])
    count_vect_kwargs = None if prot_encoding_type != "amino_acid_count" else {
        "analyzer": "char",
        "ngram_range": (1, 3),
    }

    # Cell encoding type
    cell_encoding_type = trial.suggest_categorical("cell_encoding_type", ["sentence_transformer", "one_hot"])
    onehot_enc_kwargs = None if cell_encoding_type != "one_hot" else {
        "handle_unknown": "ignore",
    }

    # Scaling
    scaler = trial.suggest_categorical("scaler", ["standard", "minmax", "normalizer", "none"])
    scaler_params = {}
    if scaler == "standard":
        scaler_params = {"with_mean": True, "with_std": True}
    elif scaler == "minmax":
        scaler_params = {"feature_range": (0, 1)}
    elif scaler == "normalizer":
        scaler_params = {"norm": "l2"}
    elif scaler == "none":
        scaler_params = None

    # Imputation
    impute_missing = trial.suggest_categorical("imputer", ["none", "simpler", "knn", "iterative"])
    imputer = None if impute_missing == "none" else impute_missing
    knn_neighbors = trial.suggest_int("knn_neighbors", 3, 10) if impute_missing == "knn" else None
    iterative_max_iter = trial.suggest_int("iterative_max_iter", 10, 100) if impute_missing == "iterative" else None
    imputer_kwargs = {}
    if impute_missing == "simpler":
        imputer_kwargs = {"strategy": "mean"}
    elif impute_missing == "knn":
        imputer_kwargs = {"n_neighbors": knn_neighbors}
    elif impute_missing == "iterative":
        imputer_kwargs = {"max_iter": iterative_max_iter, "random_state": 42}

    # Update dataset_kwargs with suggested hyperparameters
    dataset_kwargs = dataset_kwargs.copy()
    dataset_kwargs.update({
        "mol_embeddings_kwargs": {
            "filename": f"mol_embeddings_type={mol_encoding_type}_fp={mol_fp_size}_r={mol_radius}.npz",
        },
        "mol_embeddings_encode_kwargs": {
            "embeddings_type": mol_encoding_type,
            "fp_size": mol_fp_size,
            "radius": mol_radius,
        },
        "protein_embeddings_kwargs": {
            "count_vect_kwargs": count_vect_kwargs,
            "filename": f"prot_embeddings_type={prot_encoding_type}.npz",
        },
        "protein_embeddings_encode_kwargs": {
            "embeddings_type": prot_encoding_type,
        },
        "cell_embeddings_kwargs": {
            "onehot_enc_kwargs": onehot_enc_kwargs,
            "filename": f"cell_embeddings_type={cell_encoding_type}.npz",
        },
        "cell_embeddings_encode_kwargs": {
            "embeddings_type": cell_encoding_type,
        },
        "label_columns": [label_columns] if isinstance(label_columns, str) else label_columns,
        "save_embeddings_to_cache": True,
        "imputer": imputer,
        "imputer_kwargs": imputer_kwargs,
    })

    report = []
    val_preds = []

    def train_on_fold(train_df, val_df, fold_idx=0):
        train_ds = MolPoiE3CellDataset(df=train_df, **dataset_kwargs)
        val_ds = MolPoiE3CellDataset(df=val_df, **dataset_kwargs)
        X_train, y_train = train_ds.to_numpy()
        X_val, y_val = val_ds.to_numpy()
        stats = {
            "model_type": "XGBoost",
            "fold": fold_idx,
            "train_len": len(train_df),
            "val_len": len(val_df),
            "train_perc": len(train_df) / len(train_val_df),
            "val_perc": len(val_df) / len(train_val_df),
        }
        if is_regression_task:
            model, preds, metrics = train_and_eval_xgboost_regressor(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                xgb_params=xgb_params,
                scaler=scaler,
                scaler_params=scaler_params,
                pca_params=pca_params,
            )
        else:
            model, preds, metrics = train_and_eval_xgboost_classifier(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                xgb_params=xgb_params,
                scaler=scaler,
                scaler_params=scaler_params,
                pca_params=pca_params,
            )
        return model, preds, metrics, stats

    if kf is None:
        if test_df is None:
            raise ValueError("test_df must be provided if kf is not specified (i.e., if no CV is performed).")
        model, preds, metrics, stats = train_on_fold(train_val_df, test_df)
        stats.update(metrics)
        report.append(stats)
        val_preds.append(preds["val_pred"])
        # Save only the best model to a proper location
        if model_name is not None and log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            model_path = os.path.join(log_dir, f"{model_name}_best.joblib")
            model.save(model_path)
            logging.info(f"Best XGBoost model saved to: {model_path}")
    else:
        X = train_val_df.copy().drop(columns=label_columns)
        y = train_val_df[label_columns].tolist()
        for k, (train_index, val_index) in enumerate(kf.split(X, y, groups)):
            logging.info(f"Fold {k + 1}/{kf.get_n_splits()}")
            train_df = train_val_df.iloc[train_index]
            val_df = train_val_df.iloc[val_index]
            model, preds, metrics, stats = train_on_fold(train_df, val_df, fold_idx=k)
            stats.update(metrics)
            report.append(stats.copy())
            val_preds.append(preds["val_pred"])
        # Save only the best model from the last fold
        if model_name:
            save_dir = log_dir or "xgboost_models"
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"{model_name}_best.joblib")
            model.save(model_path)
            logging.info(f"Best XGBoost model saved to: {model_path}")

    # Log results to files
    log_dir = log_dir or "xgboost_logs"
    os.makedirs(log_dir, exist_ok=True)
    pd.DataFrame(report).to_csv(os.path.join(log_dir, f"{model_name}_cv_report.csv"), index=False)
    pd.DataFrame([xgb_params]).to_csv(os.path.join(log_dir, f"{model_name}_hparams.csv"), index=False)
    logging.info(f"Reports and hyperparameters saved to {log_dir}")

    trial.set_user_attr("report", report)
    trial.set_user_attr("val_preds", val_preds)
    trial.set_user_attr("params", xgb_params)
    trial.set_user_attr("dataset_hparams", dataset_kwargs)

    if is_regression_task:
        mean_val_metric = np.mean([np.mean(r["val_r2"]) for r in report])
        logging.info(f"\tMean val R2: {mean_val_metric:.4f}")
        return -mean_val_metric
    else:
        mean_val_metric = np.mean([r["val_roc_auc"] for r in report])
        logging.info(f"\tMean val ROC AUC: {mean_val_metric:.4f}")
        return -mean_val_metric

def xgboost_hyperparameter_tuning_and_training(
        train_val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        label_columns: Union[str, List[str]],
        dataset_kwargs: Optional[Dict],
        kf: Optional[StratifiedKFold] = None,
        groups: Optional[np.array] = None,
        split_type: str = "random",
        n_models_for_test: int = 3,
        n_trials: int = 50,
        num_boost_round: int = 100,
        study_filename: Optional[str] = None,
        force_study: bool = False,
        model_name: Optional[str] = None,
        is_regression_task: bool = False,
        log_dir: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
) -> dict:
    """ Hyperparameter tuning and training of an XGBoost model (classification or regression). """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    if dataset_kwargs is None:
        dataset_kwargs = {}

    # Clean up the cache directory from embeddings
    if cache_dir is None:
        cache_dir = Path(get_cache_dir())

    # Remove old embeddings that start with "mol_*.npz" or "prot_*.npz" or
    # "cell_*.npz"
    for file in cache_dir.glob("mol_*.npz"):
        logging.info(f"Removing old mol embedding cache file: {file}")
        file.unlink(missing_ok=True)
    for file in cache_dir.glob("prot_*.npz"):
        logging.info(f"Removing old protein embedding cache file: {file}")
        file.unlink(missing_ok=True)
    for file in cache_dir.glob("cell_*.npz"):
        logging.info(f"Removing old cell embedding cache file: {file}")
        file.unlink(missing_ok=True)

    # Load the study if it exists
    study_loaded = False
    if study_filename and not force_study:
        if os.path.exists(study_filename):
            study = joblib.load(study_filename)
            study_loaded = True
            logging.info(f"Loaded study from {study_filename}")

    # If the study is not loaded or force_study is True, optimize
    if not study_loaded or force_study:
        study.optimize(
            lambda trial: xgboost_objective(
                trial=trial,
                train_val_df=train_val_df,
                label_columns=label_columns,
                dataset_kwargs=dataset_kwargs,
                test_df=test_df,
                kf=kf,
                groups=groups,
                num_boost_round=num_boost_round,
                model_name=model_name,
                is_regression_task=is_regression_task,
                log_dir=log_dir,
            ),
            n_trials=n_trials,
        )
        if study_filename:
            joblib.dump(study, study_filename)

    cv_report = pd.DataFrame(study.best_trial.user_attrs["report"])
    hparam_report = pd.DataFrame([study.best_trial.user_attrs["params"]])
    dataset_hparam_report = pd.DataFrame([study.best_trial.user_attrs["dataset_hparams"]])

    # Train the best model and save it
    if model_name:
        xgboost_objective(
            trial=study.best_trial,
            train_val_df=train_val_df,
            label_columns=label_columns,
            dataset_kwargs=dataset_kwargs,
            test_df=test_df,
            kf=kf,
            groups=groups,
            num_boost_round=num_boost_round,
            model_name=f"{model_name}_best",
            is_regression_task=is_regression_task,
            log_dir=log_dir,
        )

    # Retrain N models with the best hyperparameters (measure model uncertainty)
    best_models = []
    test_report = []
    test_preds = []
    for i in range(n_models_for_test):
        logging.info(f"Training best model {i + 1}/{n_models_for_test}")
        # Use best params from study for retraining
        xgb_params = study.best_trial.user_attrs["params"]
        dataset_hparams = study.best_trial.user_attrs["dataset_hparams"]

        train_ds = MolPoiE3CellDataset(df=train_val_df, **dataset_hparams)
        test_ds = MolPoiE3CellDataset(df=test_df, **dataset_hparams)
        X_train, y_train = train_ds.to_numpy()
        X_test, y_test = test_ds.to_numpy()

        if is_regression_task:
            model, preds, metrics = train_and_eval_xgboost_regressor(
                X_train=X_train,
                y_train=y_train,
                X_val=X_test,
                y_val=y_test,
                xgb_params=xgb_params,
                scaler=xgb_params.get("scaler", "standard"),
                scaler_params=None,
                shuffle_train_data=True,
            )
            
            # For multitask regression, "explode" the metrics
            metrics_to_update = {}
            metrics_to_delete = []
            for metric in metrics.keys():
                if len(list(metrics[metric])) > 1:
                    metrics_to_delete.append(metric)
                    for j, label in enumerate(label_columns):
                        label = label.strip().replace(" ", "_").lower()
                        metrics_to_update[f"{label}_{metric}"] = metrics[metric][j]

            # Update metrics and remove the original metrics that were exploded
            metrics.update(metrics_to_update)
            for metric in metrics_to_delete:
                del metrics[metric]
        else:
            model, preds, metrics = train_and_eval_xgboost_classifier(
                X_train=X_train,
                y_train=y_train,
                X_val=X_test,
                y_val=y_test,
                xgb_params=xgb_params,
                scaler=xgb_params.get("scaler", "standard"),
                scaler_params=None,
                shuffle_train_data=True,
            )
        metrics = {k.replace("val_", "test_"): v for k, v in metrics.items()}
        metrics["model_type"] = "XGBoost"
        metrics["test_model_id"] = i
        # metrics.update(get_dataframe_stats(train_val_df, test_df=test_df, active_label=label_columns))
        test_report.append(metrics.copy())
        test_preds.append(torch.tensor(preds["val_pred"]))
        best_models.append(model)

    test_report = pd.DataFrame(test_report)

    # Get the majority vote for the test predictions (classification only)
    if not is_regression_task:
        majority_vote_metrics = get_majority_vote_metrics(test_preds, test_df, label_columns)
        # majority_vote_metrics.update(get_dataframe_stats(train_val_df, test_df=test_df, active_label=label_columns))
        majority_vote_report = pd.DataFrame([majority_vote_metrics])
        majority_vote_report["model_type"] = "XGBoost"
    else:
        majority_vote_report = pd.DataFrame()

    # Add a column with the split_type to all reports
    for report in [cv_report, hparam_report, test_report, majority_vote_report, dataset_hparam_report]:
        report["split_type"] = split_type

    # Log all reports to files
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        cv_report.to_csv(os.path.join(log_dir, f"{model_name}_cv_report.csv"), index=False)
        hparam_report.to_csv(os.path.join(log_dir, f"{model_name}_hparam_report.csv"), index=False)
        test_report.to_csv(os.path.join(log_dir, f"{model_name}_test_report.csv"), index=False)
        dataset_hparam_report.to_csv(os.path.join(log_dir, f"{model_name}_dataset_hparam_report.csv"), index=False)
        if not majority_vote_report.empty:
            majority_vote_report.to_csv(os.path.join(log_dir, f"{model_name}_majority_vote_report.csv"), index=False)
        logging.info(f"All reports and hyperparameters saved to {log_dir}")

    # Remove old embeddings that start with "mol_*.npz" or "prot_*.npz" or
    # "cell_*.npz"
    for file in cache_dir.glob("mol_*.npz"):
        logging.info(f"Removing old mol embedding cache file: {file}")
        file.unlink(missing_ok=True)
    for file in cache_dir.glob("prot_*.npz"):
        logging.info(f"Removing old protein embedding cache file: {file}")
        file.unlink(missing_ok=True)
    for file in cache_dir.glob("cell_*.npz"):
        logging.info(f"Removing old cell embedding cache file: {file}")
        file.unlink(missing_ok=True)

    return {
        "cv_report": cv_report,
        "hparam_report": hparam_report,
        "test_report": test_report,
        "majority_vote_report": majority_vote_report,
        "dataset_hparam_report": dataset_hparam_report,
    }