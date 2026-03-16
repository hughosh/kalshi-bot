"""GBDT point-win probability model.

Wraps XGBoost (primary) and LightGBM (alternative) behind a common interface.
The model predicts P(server wins current point) as a binary classification task.

Training target: 1 if server won the point, 0 if returner won.
Primary metric:  log-loss (binary cross-entropy)
Secondary:       Brier score
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np


class PointWinModel:
    """Gradient-boosted point-win probability estimator.

    The model is backend-agnostic: pass backend="xgboost" (default) or
    backend="lightgbm". Both expose the same interface so swapping them
    is one argument change.
    """

    DEFAULT_XGBOOST_PARAMS = {
        "objective": "binary:logistic",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "gamma": 1.0,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cpu",
        "random_state": 42,
    }

    DEFAULT_LGBM_PARAMS = {
        "objective": "binary",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
    }

    def __init__(self, backend: str = "xgboost", params: Optional[dict] = None):
        """
        Args:
            backend: "xgboost" or "lightgbm"
            params: Override default hyperparameters.
        """
        self.backend = backend.lower()
        self._model = None
        self._feature_names: Optional[list[str]] = None
        self._trained = False

        if self.backend == "xgboost":
            base_params = dict(self.DEFAULT_XGBOOST_PARAMS)
        elif self.backend == "lightgbm":
            base_params = dict(self.DEFAULT_LGBM_PARAMS)
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose 'xgboost' or 'lightgbm'.")

        if params:
            base_params.update(params)
        self._params = base_params

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list[str]] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = True,
    ) -> dict:
        """Fit the model with early stopping on validation log-loss.

        Args:
            X_train: Training features, shape (n_train, n_features).
            y_train: Binary labels (1=server won, 0=returner won), shape (n_train,).
            X_val:   Validation features.
            y_val:   Validation labels.
            feature_names: Optional list of feature names for interpretability.
            early_stopping_rounds: Stop if no improvement after this many rounds.
            verbose: Print training progress.

        Returns:
            Dict with train/val log-loss and Brier score.
        """
        self._feature_names = feature_names

        if self.backend == "xgboost":
            return self._train_xgb(X_train, y_train, X_val, y_val, early_stopping_rounds, verbose)
        else:
            return self._train_lgbm(X_train, y_train, X_val, y_val, early_stopping_rounds, verbose)

    def _train_xgb(self, X_tr, y_tr, X_val, y_val, early_stopping, verbose):
        import xgboost as xgb

        params = dict(self._params)
        n_estimators = params.pop("n_estimators", 500)
        params.pop("device", None)  # may not exist in older xgb

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping,
            **params,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=100 if verbose else False,
        )
        self._model = model
        self._trained = True

        return self._eval_metrics(X_tr, y_tr, X_val, y_val)

    def _train_lgbm(self, X_tr, y_tr, X_val, y_val, early_stopping, verbose):
        import lightgbm as lgb

        params = dict(self._params)
        n_estimators = params.pop("n_estimators", 500)

        callbacks = []
        if verbose:
            callbacks.append(lgb.log_evaluation(period=100))
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping, verbose=verbose))

        model = lgb.LGBMClassifier(n_estimators=n_estimators, **params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )
        self._model = model
        self._trained = True

        return self._eval_metrics(X_tr, y_tr, X_val, y_val)

    def _eval_metrics(self, X_tr, y_tr, X_val, y_val) -> dict:
        from sklearn.metrics import log_loss, brier_score_loss

        p_tr = self.predict_proba(X_tr)
        p_val = self.predict_proba(X_val)

        return {
            "train_logloss": log_loss(y_tr, p_tr),
            "val_logloss": log_loss(y_val, p_val),
            "train_brier": brier_score_loss(y_tr, p_tr),
            "val_brier": brier_score_loss(y_val, p_val),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(server wins point) for each row. Shape: (n_samples,).

        This is the primary inference method. Single-row inference (X.shape=(1,n))
        completes in < 1ms on modern hardware.
        """
        if not self._trained or self._model is None:
            raise RuntimeError("Model has not been trained. Call train() first.")
        if self.backend == "xgboost":
            import xgboost as xgb
            booster = self._model.get_booster()
            dmat = xgb.DMatrix(X, feature_names=self._feature_names)
            return booster.predict(dmat).astype(np.float64)
        else:
            proba = self._model.predict_proba(X)
            return proba[:, 1].astype(np.float64)

    def predict_single(self, x: np.ndarray) -> float:
        """Predict P(server wins point) for a single feature vector.

        Args:
            x: Feature vector, shape (n_features,).

        Returns:
            Float probability ∈ [0, 1].
        """
        return float(self.predict_proba(x.reshape(1, -1))[0])

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> Optional[dict]:
        """Return feature importance dict, or None if not available."""
        if self._model is None:
            return None
        if self.backend == "xgboost":
            imp = self._model.get_booster().get_score(importance_type="gain")
        else:
            import lightgbm as lgb
            imp = dict(zip(
                self._model.booster_.feature_name(),
                self._model.booster_.feature_importance(importance_type="gain"),
            ))
        if self._feature_names:
            return {
                name: imp.get(f"f{i}", imp.get(name, 0.0))
                for i, name in enumerate(self._feature_names)
            }
        return imp

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model to disk.

        XGBoost: saves as .ubj (binary format for speed)
        LightGBM: saves as .pkl
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if self.backend == "xgboost":
            # Save booster in native binary format, plus metadata via pickle
            booster_path = path.replace(".pkl", ".ubj")
            self._model.get_booster().save_model(booster_path)
            meta = {"backend": self.backend, "params": self._params,
                    "feature_names": self._feature_names, "booster_path": booster_path}
            with open(path, "wb") as f:
                pickle.dump(meta, f)
        else:
            with open(path, "wb") as f:
                pickle.dump({
                    "backend": self.backend,
                    "model": self._model,
                    "params": self._params,
                    "feature_names": self._feature_names,
                }, f)

    @classmethod
    def load(cls, path: str) -> PointWinModel:
        """Load a saved model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        backend = data["backend"]
        obj = cls(backend=backend, params=data.get("params"))
        obj._feature_names = data.get("feature_names")

        if backend == "xgboost":
            import xgboost as xgb
            booster_path = data["booster_path"]
            # Reconstruct XGBClassifier from saved booster
            obj._model = xgb.XGBClassifier(**{
                k: v for k, v in data["params"].items()
                if k not in ("eval_metric",)
            })
            # Load booster directly
            booster = xgb.Booster()
            booster.load_model(booster_path)
            obj._model._Booster = booster
        else:
            obj._model = data["model"]

        obj._trained = True
        return obj
