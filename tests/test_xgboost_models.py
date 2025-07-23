""" Test for the XGBoost regression model training and evaluation function with multi-output regression.
"""
import logging

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from protac_degradation_predictor.models.xgboost_model import train_and_eval_xgboost_regressor
from protac_degradation_predictor.optimization.optuna_xgboost import xgboost_hyperparameter_tuning_and_training


def get_random_dataset(n_samples=100, n_features=10, n_targets=3):
    """Generate a random dataset for testing."""
    rng = np.random.RandomState(42)
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_targets=n_targets,
        noise=0.1,
        random_state=rng
    )

    # Scale y to be in the range [0, 1]
    y = (y - y.min(axis=0)) / (y.max(axis=0) - y.min(axis=0))

    # Scale the first target by 10
    y[:, 0] *= 10

    # Randomly select up to n_targets - 1 columns and scale them by 10
    for i in range(1, n_targets - 1):
        if rng.rand() < 0.5:
            y[:, i] *= 10

    # Split into train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=rng)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=rng)
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_random_dataframe(is_regression_task=True):
    # Create a dummy dataframe
    data = {
        "mol": [
            "CC1=C(C2=CC=C(CNC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@@H](NC(=O)CCCNC(=O)C3=CC=C(C4=CN=C(NCC5=CC=CO5)N5C=NN=C45)C=C3)C(C)(C)C)C=C2)SC=N1",
            "CC1=C(C2=CC=C(CNC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@H](C(C)C)N3CC4=CC=CC=C4C3=O)C(OCCNC(=O)C3=CC=CC=C3NC(=O)[C@H](CCCCN)NC(=O)[C@@H]3CCCN3C(=O)CC3=CC=CC4=CC=CC=C34)=C2)SC=N1",
            "CC1=C(C2=CC=C([C@H](C)NC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@@H](NC(=O)CCCCCN3C=C(C4=CC(C5=CC=CC=C5O)=NN=C4N)C=N3)C(C)(C)C)C=C2)SC=N1",
        ],
        "poi": ["AAAAAAAA", "BBBBBBBBBB", "CCCCCCCCCC"],
        "e3": ["DDDDDDDDDD", "EEEEEEEEEE", "FFFFFFFFFF"],
        "cell": ["hela", "RS4; 11", "ramos"],
        "label_bin": [0, 1, 0],
        "label_reg": [0.1, None, 0.3],
        "label_multiclass": [0, 1, 2],
    }
    
    if not is_regression_task:
        data["label_reg"] = [1, 2, 3]

    df = pd.DataFrame(data)

    # Duplicate the dataframe to simulate a larger dataset
    df = pd.concat([df] * 10, ignore_index=True)

    label_columns = ["label_bin", "label_reg", "label_multiclass"]

    return df, label_columns

def test_train_and_eval_xgboost_regressor_multioutput():
    # Synthetic multi-output regression data
    n_samples, n_features, n_targets = 50, 5, 3

    X_train, y_train, X_val, y_val, X_test, y_test = get_random_dataset(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_targets
    )

    # Set some of the y_train data on the n_targets dimension to NaN to simulate
    # missing values
    rng = np.random.RandomState(42)
    random_mask = rng.rand(*y_train.shape) < 0.2  # 20% missing
    y_train[random_mask] = np.nan

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(y_train)

    # Run function
    model, preds, metrics = train_and_eval_xgboost_regressor(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        xgb_params={"n_estimators": 10, "max_depth": 2, "objective": "reg:squarederror"},
        scaler="standard",
        scaler_params={},
        pca_params=None,
        shuffle_train_data=True,
        alpha=0.1
    )

    logging.info(f"Model:\n{model}")
    logging.info(f"Predictions:\n{preds}")
    logging.info(f"Metrics:\n{metrics}")

    # Check preds keys and shapes
    assert "val_pred" in preds
    assert preds["val_pred"].shape == (len(y_val), n_targets)

    # Check metrics keys and values
    for key in ["val_mse", "val_mae", "val_r2", "test_mse", "test_mae", "test_r2"]:
        assert key in metrics
        assert np.isfinite(metrics[key]).all()

    # Check prediction intervals
    if y_train.shape[-1] == 1:
        assert "val_pis" in preds
        assert preds["val_pis"].shape == (5, n_targets, 2) or preds["val_pis"].shape == (5, 2)
        assert "val_pis_lower" in metrics and "val_pis_upper" in metrics
        assert len(metrics["val_pis_lower"]) == 5
        assert len(metrics["val_pis_upper"]) == 5

    # Check that the model can predict on the test set
    if X_test is not None and y_test is not None:
        test_preds = model.predict(X_test)
        assert test_preds.shape == (len(y_test), n_targets)
        assert np.all(np.isfinite(test_preds))
        logging.info(f"Test predictions shape: {test_preds.shape}")
        
def test_optuna_xgboost():
    train_val_df, label_columns = get_random_dataframe()
    test_df, label_columns = get_random_dataframe()
    
    reports = xgboost_hyperparameter_tuning_and_training(
        train_val_df=train_val_df,
        test_df=test_df,
        n_trials=3,
        dataset_kwargs={
            "mol_column": "mol",
            "poi_column": "poi",
            "e3_column": "e3",
            "cell_column": "cell",
            "save_embeddings_to_cache": False,
        },
        label_columns=label_columns,
        is_regression_task=True,
    )

    logging.info(f"CV Report: {reports['cv_report']}")
    logging.info(f"HParam Report: {reports['hparam_report']}")
    logging.info(f"Test Report: {reports['test_report']}")
    logging.info(f"Majority Vote Report: {reports['majority_vote_report']}")
    logging.info(f"Dataset HParam Report: {reports['dataset_hparam_report']}")