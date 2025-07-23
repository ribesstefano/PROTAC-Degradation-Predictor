""" XGBoost model training and evaluation utilities."""
import logging
import joblib
from typing import Dict, Optional, Tuple, Literal

import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from mapie.regression import MapieRegressor
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.multioutput import RegressorChain
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from protac_degradation_predictor.evaluation import get_confidence_scores

from pathlib import Path
from typing import Dict, Optional, Tuple, Literal, Union
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

class XGBoostPipelineBase(BaseEstimator):
    """ Base class for XGBoost pipelines with optional scaling and PCA. """

    def __init__(
        self,
        xgb_params: Optional[Dict] = None,
        scaler: Optional[Literal["standard", "minmax", "normalizer"]] = "standard",
        scaler_params: Optional[Dict] = None,
        pca_params: Optional[Dict] = None,
        use_regressor_chain: bool = False,
    ):
        """ Initialize the XGBoost pipeline base class.
        
        Args:
            xgb_params (Optional[Dict]): Parameters for the XGBoost model.
            scaler (Optional[Literal["standard", "minmax", "normalizer"]]): Type of scaler to use.
            scaler_params (Optional[Dict]): Parameters for the scaler.
            pca_params (Optional[Dict]): Parameters for PCA.
            use_regressor_chain (bool): Whether to use a regressor chain for multi-output regression.
        """
        self.xgb_params = xgb_params
        self.scaler = scaler
        self.scaler_params = scaler_params
        self.pca_params = pca_params
        self.pipeline = None
        self.use_regressor_chain = use_regressor_chain

    def _build_pipeline(self, estimator):
        pipeline_modules = []
        if self.scaler_params is not None:
            if self.scaler == "standard":
                pipeline_modules.append(("scaler", StandardScaler(**self.scaler_params)))
            elif self.scaler == "minmax":
                pipeline_modules.append(("scaler", MinMaxScaler(**self.scaler_params)))
            elif self.scaler == "normalizer":
                pipeline_modules.append(("scaler", Normalizer(**self.scaler_params)))
            else:
                raise ValueError(f"Unsupported scaler: {self.scaler}")
        if self.pca_params is not None:
            pipeline_modules.append(("pca", PCA(**self.pca_params)))
        pipeline_modules.append(("estimator", estimator))
        return Pipeline(pipeline_modules)

    def save(self, path: Union[str, Path]):
        joblib.dump(self, str(path))

    @classmethod
    def load(cls, path: Union[str, Path]):
        return joblib.load(str(path))

class XGBoostPipelineClassifier(XGBoostPipelineBase, ClassifierMixin):
    def fit(self, X, y):
        xgb_params = self.xgb_params or {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "auto",
            "verbosity": 1,
        }
        xgb_params["random_state"] = xgb_params.get("random_state", 42)
        base_classifier = XGBClassifier(**xgb_params)
        self.pipeline = self._build_pipeline(base_classifier)
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

class XGBoostPipelineRegressor(XGBoostPipelineBase, RegressorMixin):
    def fit(self, X, y):
        xgb_params = self.xgb_params or {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "auto",
            "verbosity": 1,
        }
        xgb_params["random_state"] = xgb_params.get("random_state", 42)
        base_regressor = XGBRegressor(**xgb_params)
        if self.use_regressor_chain and y.shape[1] > 1:
            base_regressor = RegressorChain(base_regressor)
        self.pipeline = self._build_pipeline(base_regressor)
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

def train_and_eval_xgboost_classifier(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        xgb_params: Optional[Dict] = None,
        scaler: Optional[Literal["standard", "minmax", "normalizer"]] = "standard",
        scaler_params: Optional[Dict] = None,
        pca_params: Optional[Dict] = None,
        shuffle_train_data: bool = False,
) -> Tuple[XGBoostPipelineClassifier, Dict[str, np.ndarray], Dict[str, float]]:
    """ Train and evaluate an XGBoost classifier with scaling and PCA. """
    if shuffle_train_data:
        idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]

    model = XGBoostPipelineClassifier(
        xgb_params=xgb_params,
        scaler=scaler,
        scaler_params=scaler_params,
        pca_params=pca_params,
    )
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_pred_proba = model.predict_proba(X_val)[:, 1] if len(np.unique(y_train)) == 2 else model.predict_proba(X_val)
    val_pred_binary = (val_pred_proba > 0.5).astype(int) if len(np.unique(y_train)) == 2 else val_pred

    fp_mean, fn_mean = get_confidence_scores(y_val, val_pred_proba)

    metrics = {
        "val_acc": accuracy_score(y_val, val_pred_binary),
        "val_roc_auc": roc_auc_score(y_val, val_pred_proba) if len(np.unique(y_train)) == 2 else None,
        "val_precision": precision_score(y_val, val_pred_binary, average="binary" if len(np.unique(y_train)) == 2 else "weighted"),
        "val_recall": recall_score(y_val, val_pred_binary, average="binary" if len(np.unique(y_train)) == 2 else "weighted"),
        "val_f1_score": f1_score(y_val, val_pred_binary, average="binary" if len(np.unique(y_train)) == 2 else "weighted"),
        "val_false_positives_mean": fp_mean,
        "val_false_negatives_mean": fn_mean,
    }
    preds = {"val_pred": val_pred_proba}

    if X_test is not None and y_test is not None:
        test_pred = model.predict(X_test)
        test_pred_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) == 2 else model.predict_proba(X_test)
        test_pred_binary = (test_pred_proba > 0.5).astype(int) if len(np.unique(y_train)) == 2 else test_pred

        fp_mean, fn_mean = get_confidence_scores(y_test, test_pred_proba)

        metrics.update({
            "test_acc": accuracy_score(y_test, test_pred_binary),
            "test_roc_auc": roc_auc_score(y_test, test_pred_proba) if len(np.unique(y_train)) == 2 else None,
            "test_precision": precision_score(y_test, test_pred_binary, average="binary" if len(np.unique(y_train)) == 2 else "weighted"),
            "test_recall": recall_score(y_test, test_pred_binary, average="binary" if len(np.unique(y_train)) == 2 else "weighted"),
            "test_f1_score": f1_score(y_test, test_pred_binary, average="binary" if len(np.unique(y_train)) == 2 else "weighted"),
            "test_false_positives_mean": fp_mean,
            "test_false_negatives_mean": fn_mean,
        })
        preds.update({"test_pred": test_pred_proba})

    return model, preds, metrics

def train_and_eval_xgboost_regressor(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        xgb_params: Optional[Dict] = None,
        scaler: Optional[Literal["standard", "minmax", "normalizer"]] = "standard",
        scaler_params: Optional[Dict] = None,
        pca_params: Optional[Dict] = None,
        shuffle_train_data: bool = False,
        alpha: float = 0.05
) -> Tuple[XGBoostPipelineRegressor, Dict[str, np.ndarray], Dict[str, float]]:
    """ Train and evaluate an XGBoost regressor with dimensionality reduction and MAPIE. """
    if shuffle_train_data:
        idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]

    if np.isnan(y_train).any():
        imputer = KNNImputer() if y_train.ndim > 1 else SimpleImputer(strategy='mean')
        logging.debug(f"Imputed NaN values in y_train:\n{y_train}")
        y_train = imputer.fit_transform(y_train)
        logging.debug(f"Imputed NaN values in y_train:\n{y_train}")

    model = XGBoostPipelineRegressor(
        xgb_params=xgb_params,
        scaler=scaler,
        scaler_params=scaler_params,
        pca_params=pca_params,
    )
    model.fit(X_train, y_train)

    # MAPIE only for single-output regression
    if y_train.shape[-1] == 1:
        mapie = MapieRegressor(model)
        mapie.fit(X_train, y_train)
        y_pred, y_pis = mapie.predict(X_val, alpha=alpha)
    else:
        y_pred = model.predict(X_val)

    metrics = {
        "val_mse": mean_squared_error(y_val, y_pred, multioutput="raw_values"),
        "val_mae": mean_absolute_error(y_val, y_pred, multioutput="raw_values"),
        "val_r2": r2_score(y_val, y_pred, multioutput="raw_values"),
    }
    preds = {"val_pred": y_pred}

    if y_train.ndim == 1:
        metrics["val_pis_lower"] = y_pis[:, 0]
        metrics["val_pis_upper"] = y_pis[:, 1]
        preds["val_pis"] = y_pis

    if X_test is not None and y_test is not None:
        if y_train.ndim == 1:
            y_test_pred, y_test_pis = mapie.predict(X_test, alpha=alpha)
            metrics.update({
                "test_mse": mean_squared_error(y_test, y_test_pred, multioutput="raw_values"),
                "test_mae": mean_absolute_error(y_test, y_test_pred, multioutput="raw_values"),
                "test_r2": r2_score(y_test, y_test_pred, multioutput="raw_values"),
                "test_pis_lower": y_test_pis[:, 0],
                "test_pis_upper": y_test_pis[:, 1],
            })
            preds.update({"test_pred": y_test_pred, "test_pis": y_test_pis})
        else:
            y_test_pred = model.predict(X_test)
            metrics.update({
                "test_mse": mean_squared_error(y_test, y_test_pred),
                "test_mae": mean_absolute_error(y_test, y_test_pred),
                "test_r2": r2_score(y_test, y_test_pred),
            })
            preds.update({"test_pred": y_test_pred})

    return model, preds, metrics