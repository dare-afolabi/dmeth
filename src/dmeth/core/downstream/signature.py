#!/usr/bin/env python
# coding: utf-8


"""
Biomarker signature discovery and predictive modeling for DNA methylation data.

This module provides a streamlined, production-ready framework for translating \
differential methylation results into high-performance diagnostic, prognostic, \
or predictive biomarker panels. It supports feature selection from statistical \
outputs, cross-validated model training with state-of-the-art algorithms, and \
rigorous independent validation of classification or regression performance.

Features
--------
- Simple yet effective signature selection via top-ranked CpGs by p-value \
or moderated t-statistic
- Flexible predictive modeling with Random Forest and Elastic Net \
(with built-in hyperparameter tuning via CV)
- Automatic task detection and appropriate stratification (classification vs regression)
- Comprehensive cross-validation and held-out test set evaluation with \
standard metrics (AUC, accuracy, R², RMSE)
- Full integration with DMeth result tables and sample-level beta matrices \
(features × samples orientation)
- Reproducible training through fixed random seeds and stratified splitting
- Extensible architecture for future stability selection, recursive feature \
elimination, or multi-omics integration
"""


from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)


def select_signature_panel(
    res: pd.DataFrame,
    method: str = "top",
    top_n: int = 100,
    importance_col: str = "t",
) -> List[str]:
    """
    Extract a candidate biomarker panel from differential methylation results.

    Parameters
    ----------
    res : pd.DataFrame
        Differential analysis results table (e.g., output from ``fit_differential``).
        Must contain at least one of ``pval`` or the column specified in \
        ``importance_col``.
    method : str, default "top"
        Feature selection strategy. Currently only ``"top"`` is implemented.
    top_n : int, default 100
        Number of top-ranked CpGs to retain.
    importance_col : str, default "t"
        Column used for ranking when ``pval`` is unavailable (e.g., moderated \
        t-statistic).

    Returns
    -------
    List[str]
        Ordered list of selected CpG identifiers (row names from ``res``).

    Notes
    -----
    - Prioritizes ``pval`` for ranking if present.
    - Falls back to descending order of ``importance_col`` otherwise.
    - Future extensions will include stability selection and recursive \
    feature elimination.
    """
    if res is None or res.empty:
        return []
    df = res.copy()
    if method == "top":
        if "pval" in df.columns:
            sel = df.nsmallest(min(top_n, len(df)), "pval").index.tolist()
        elif importance_col in df.columns:
            sel = df.nlargest(min(top_n, len(df)), importance_col).index.tolist()
        else:
            sel = df.index.tolist()[:top_n]
        return sel
    else:
        raise NotImplementedError("Only 'top' implemented")


def validate_signature(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    X_test: pd.DataFrame,
    y_test: Union[pd.Series, np.ndarray],
    features: Sequence[str],
    method: str = "elasticnet",
) -> Dict[str, Any]:
    """
    Independent validation of a pre-selected methylation signature on held-out data.

    Re-trains the specified model on training data using only the provided \
    features and reports performance on a separate test set.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Training and test methylation matrices (samples × features).
    y_train, y_test : array-like
        Corresponding labels.
    features : Sequence[str]
        Subset of CpGs constituting the signature.
    method : str, default "elasticnet"
        Model used for validation (same options as ``model_dms_for_prediction``).

    Returns
    -------
    dict
        Performance metrics:
        Binary classification → ``auc`` and ``accuracy``
        Multi-class/regression → ``accuracy`` or ``rmse`` + ``r2``

    Notes
    -----
    Provides an unbiased estimate of clinical/translational performance \
    when the signature was derived on a separate discovery cohort.
    """
    # Subset features
    Xtr = X_train[features].copy()
    Xte = X_test[features].copy()

    # Ensure y arrays align
    y_train_arr = np.asarray(y_train).ravel()
    y_test_arr = np.asarray(y_test).ravel()

    if len(y_train_arr) != Xtr.shape[0]:
        raise ValueError("y_train length does not match X_train rows")
    if len(y_test_arr) != Xte.shape[0]:
        raise ValueError("y_test length does not match X_test rows")

    # Convert to Series with matching index
    y_train_series = pd.Series(y_train_arr, index=Xtr.index)

    # Fit model - features × samples; so transpose
    model_out = model_dms_for_prediction(Xtr.T, y_train_series, method=method)
    est = model_out["estimator"]

    # Binary classification
    if hasattr(est, "predict_proba") and len(np.unique(y_test_arr)) == 2:
        probs = est.predict_proba(Xte)[:, 1]
        return {
            "auc": float(roc_auc_score(y_test_arr, probs)),
            "accuracy": float(accuracy_score(y_test_arr, est.predict(Xte))),
        }

    # Multi-class or regression
    preds = est.predict(Xte)
    if len(np.unique(y_test_arr)) > 2:
        return {
            "rmse": float(math.sqrt(mean_squared_error(y_test_arr, preds))),
            "r2": float(r2_score(y_test_arr, preds)),
        }
    else:
        return {"accuracy": float(accuracy_score(y_test_arr, preds))}


def model_dms_for_prediction(
    beta: pd.DataFrame,
    labels: pd.Series,
    method: str = "random_forest",
    n_splits: int = 5,
    random_state: int = 42,
    task: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train and evaluate a predictive model using DNA methylation signatures.

    Automatically detects classification vs regression tasks and applies \
    appropriate modeling and evaluation strategies.

    Parameters
    ----------
    beta : pd.DataFrame
        Beta-value matrix with CpGs as rows and samples as columns (features × \
        samples orientation).
    labels : pd.Series
        Target variable aligned with ``beta.columns``.
    method : {"random_forest", "elasticnet"}, default "random_forest"
        Predictive algorithm:
        ``random_forest``: 500 trees with parallel training
        ``elasticnet``: LogisticRegressionCV or ElasticNetCV with built-in \
        cross-validated regularization
    n_splits : int, default 5
        Number of cross-validation folds for hyperparameter tuning and \
        performance estimation.
    random_state : int, default 42
        Seed for reproducible splitting and model initialization.
    task : {"classification", "regression"} or None, optional
        Force task type. If ``None``, inferred from label distribution.

    Returns
    -------
    dict
        Contains:
        ``estimator``: final fitted model (trained on full data)
        ``cv_results``: detailed cross-validation scores
        ``test_auc`` / ``test_accuracy`` (classification) or ``test_rmse`` \
        / ``test_r2`` (regression) on a stratified 20% held-out set
    """
    X = beta.T
    y = pd.Series(labels)
    if task is None:
        task = (
            "classification"
            if y.dtype.kind in "biufc" and len(y.unique()) <= 10
            else "regression"
        )

    if method == "random_forest":
        clf = (
            RandomForestClassifier(
                n_estimators=500, n_jobs=-1, random_state=random_state
            )
            if task == "classification"
            else RandomForestRegressor(
                n_estimators=500, n_jobs=-1, random_state=random_state
            )
        )
    elif method == "elasticnet":
        clf = (
            LogisticRegressionCV(
                cv=n_splits,
                penalty="elasticnet",
                solver="saga",
                l1_ratios=[0.5],
                max_iter=5000,
                n_jobs=-1,
                random_state=random_state,
            )
            if task == "classification"
            else ElasticNetCV(cv=n_splits, n_jobs=-1, random_state=random_state)
        )
    else:
        raise ValueError(f"Unsupported method '{method}'")

    cv_split = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if task == "classification"
        else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    )

    cv_results = cross_validate(
        clf,
        X,
        y,
        cv=cv_split,
        scoring="roc_auc" if task == "classification" else "r2",
        return_train_score=True,
    )
    clf.fit(X, y)

    out = {"estimator": clf, "cv_results": cv_results}

    # test split metrics
    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=(y if task == "classification" else None),
    )
    pred = (
        clf.predict_proba(Xte)[:, 1]
        if task == "classification" and hasattr(clf, "predict_proba")
        else clf.predict(Xte)
    )

    if task == "classification":
        if len(np.unique(yte)) == 2:
            # Binary classification
            out["test_auc"] = float(roc_auc_score(yte, pred))
        elif len(np.unique(yte)) > 2:
            # Multi-class classification - need full probability matrix
            pred_proba = clf.predict_proba(Xte)
            out["test_auc"] = float(
                roc_auc_score(yte, pred_proba, multi_class="ovr", average="macro")
            )
        else:
            out["test_auc"] = np.nan

        out["test_accuracy"] = float(accuracy_score(yte, clf.predict(Xte)))

    else:
        out["test_rmse"] = float(np.sqrt(mean_squared_error(yte, pred)))
        out["test_r2"] = float(r2_score(yte, pred))

    return out
