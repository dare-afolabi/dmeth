#!/usr/bin/env python
# coding: utf-8


"""
Data preparation and preprocessing utilities for differential methylation analysis.

This module provides robust, production-ready tools for cleaning and imputing
CpG-level methylation matrices prior to statistical modeling. It implements
stringent quality-control filters based on missingness patterns and group
representation, alongside flexible imputation strategies optimized for
high-dimensional epigenomic data.

Features
--------
- Global and per-group missingness filtering with configurable thresholds
- Minimum representation enforcement across experimental groups
- Multiple imputation methods: row-wise mean/median and K-nearest neighbors
- Sample-wise KNN option for dramatic speed gains when samples ≪ CpGs
- Comprehensive input validation and clear diagnostic reporting
- Memory-efficient operations using NumPy-based vectorization
"""


from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from dmeth.utils.logger import logger


def filter_cpgs_by_missingness(
    M: pd.DataFrame,
    max_missing_rate: float = 0.2,
    min_samples_per_group: Optional[int] = None,
    groups: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, int, int]:
    """
    Remove CpGs exceeding a global missingness threshold and/or failing \
    per-group representation.

    Parameters
    ----------
    M : pd.DataFrame
        Methylation matrix (CpGs × samples) containing possible NaN values.
    max_missing_rate : float, default 0.2
        Maximum allowable fraction of missing values across all samples (0–1).
    min_samples_per_group : int or None, optional
        Minimum number of non-missing observations required in every experimental group.
        Ignored if ``None``.
    groups : pd.Series or None, optional
        Sample group labels indexed by ``M.columns``. Required when \
        ``min_samples_per_group`` is set.

    Returns
    -------
    filtered : pd.DataFrame
        Subset of CpGs passing both filters.
    n_removed : int
        Number of CpGs removed.
    n_kept : int
        Number of CpGs retained.

    Raises
    ------
    ValueError
        If ``min_samples_per_group`` is supplied without ``groups`` or if group \
        labels are misaligned.
    """
    if not 0 <= max_missing_rate <= 1:
        raise ValueError("max_missing_rate must be in [0, 1]")

    if min_samples_per_group is not None:
        if groups is None:
            raise ValueError(
                "groups must be provided when min_samples_per_group is set"
            )
        missing_cols = set(M.columns) - set(groups.index)
        if missing_cols:
            raise ValueError(
                f"Missing columns in groups.index: {sorted(missing_cols)[:5]}"
            )

    values = M.values
    n_cpgs, n_samples = values.shape
    keep = np.isnan(values).sum(axis=1) <= max_missing_rate * n_samples

    if min_samples_per_group is not None:
        groups_aligned = groups.reindex(M.columns).values
        not_na = ~np.isnan(values)
        for g in np.unique(groups_aligned):
            col_idx = np.where(groups_aligned == g)[0]
            keep &= not_na[:, col_idx].sum(axis=1) >= min_samples_per_group

    filtered = M.loc[keep]
    n_removed = len(M) - len(filtered)
    n_kept = len(filtered)
    return filtered, n_removed, n_kept


def impute_missing_values(
    M: pd.DataFrame,
    method: str = "mean",
    k: int = 5,
    use_sample_knn: bool = False,
) -> pd.DataFrame:
    """
    Impute missing methylation values using row-wise statistics or K-nearest neighbours.

    Parameters
    ----------
    M : pd.DataFrame
        CpG × sample matrix with possible NaN entries.
    method : {"mean", "median", "knn"}, default "mean"
        Imputation strategy.
    k : int, default 5
        Number of neighbours used for KNN imputation.
    use_sample_knn : bool, default False
        If True, KNN is performed across samples (samples ≪ CpGs) rather \
        than across CpGs.
        Dramatically faster for typical array datasets.

    Returns
    -------
    pd.DataFrame
        Imputed matrix with the same index/columns as input.

    Notes
    -----
    - ``mean`` and ``median`` are applied row-wise (per CpG).
    - ``knn`` uses scikit-learn's ``KNNImputer`` with distance weighting.
    - No imputation is performed if the matrix contains no missing values.
    """
    if method not in {"mean", "median", "knn"}:
        raise ValueError("method must be 'mean', 'median', or 'knn'")

    X = M.values.astype(float)
    mask = np.isnan(X)
    if not mask.any():
        return M.copy()

    if method in {"mean", "median"}:
        stat_func = np.nanmean if method == "mean" else np.nanmedian  # mean or median
        fill_values = stat_func(X, axis=1)
        row_idx, col_idx = np.where(mask)
        X[row_idx, col_idx] = fill_values[row_idx]
    else:  # KNN
        imputer = KNNImputer(n_neighbors=k, weights="distance")
        X = imputer.fit_transform(X.T if use_sample_knn else X)
        if use_sample_knn:
            X = X.T

    return pd.DataFrame(X, index=M.index, columns=M.columns)


def filter_min_per_group(
    M: pd.DataFrame,
    groups: pd.Series,
    min_per_group: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Retain only CpGs with at least ``min_per_group`` non-missing \
    values in one experimental group.

    Useful for ensuring sufficient representation before differential analysis.

    Parameters
    ----------
    M : pd.DataFrame
        Methylation matrix (CpGs × samples) possibly containing NaN.
    groups : pd.Series
        Group membership for each sample (index must match ``M.columns``).
    min_per_group : int, default 5
        Minimum number of observed (non-missing) values required per group.
    verbose : bool, default True
        Log a concise summary of filtering results.

    Returns
    -------
    pd.DataFrame
        Filtered matrix containing only qualifying CpGs.

    Raises
    ------
    ValueError
        If group labels do not cover all samples in ``M``.
    """
    missing_cols = set(M.columns) - set(groups.index)
    if missing_cols:
        raise ValueError(f"Missing columns in groups: {sorted(missing_cols)[:5]}")

    groups_aligned = groups.reindex(M.columns)
    values = M.values
    not_na = ~np.isnan(values)

    # Count non-missing per group
    per_group_counts = {
        g: not_na[:, groups_aligned == g].sum(axis=1) for g in groups_aligned.unique()
    }

    # Keep if ANY group meets the minimum, not EVERY group
    keep = np.zeros(len(M), dtype=bool)
    for _g, counts in per_group_counts.items():
        keep |= counts >= min_per_group

    filtered = M.loc[keep]
    if verbose:
        removed = len(M) - len(filtered)
        logger.info(
            f"filter_min_per_group: kept {len(filtered):,} / {len(M):,} CpGs "
            f"(removed {removed:,})"
        )
    return filtered
