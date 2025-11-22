#!/usr/bin/env python
# coding: utf-8


"""
Cell-type deconvolution utilities for bulk DNA methylation data.

This module provides fast, reference-based estimation of cell-type \
proportions from epigenome-wide methylation profiles using constrained \
non-negative least squares (NNLS). It supports integration of published \
reference matrices (e.g., blood, brain, tumor microenvironment) and enables \
accurate correction for cellular heterogeneity in differential methylation studies.

Features
--------
- High-performance NNLS deconvolution with automatic feature alignment \
between bulk and reference
- Optional parallel processing via joblib for large cohorts (hundreds to \
thousands of samples)
- Robust normalization of estimated proportions to sum-to-1 per sample
- Graceful handling of missing joblib dependency with automatic fallback to \
serial execution
- Clear error reporting when no overlapping CpGs exist between dataset and reference
- Direct compatibility with standard beta-value matrices (CpGs × samples) used \
throughout DMeth)
- Extensible design for future reference-free (e.g., EpiDISH HEpiDISH, MeDeCom) \
or projective methods
"""


import numpy as np
import pandas as pd
from scipy.optimize import nnls

from dmeth.utils.logger import logger

try:
    import joblib
except ImportError:
    joblib = None
    logger.warning("joblib not installed. Parallel computing unavailable.")


def estimate_cell_composition(
    beta: pd.DataFrame,
    ref_profiles: pd.DataFrame,
    method: str = "nnls",
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Perform reference-based cell-type deconvolution of bulk DNA methylation \
    data using constrained non-negative least squares (NNLS).

    Estimates the relative proportions of predefined cell types in each \
    sample by solving the linear mixture model:
        β_bulk ≈ R · w

    where R contains cell-type-specific methylation reference profiles \
    and w ≥ 0 are the unknown proportions.

    Parameters
    ----------
    beta : pd.DataFrame
        Bulk beta-value matrix with CpGs as rows (index) and samples as columns.
        Values must be in [0, 1]; missing values are not supported.
    ref_profiles : pd.DataFrame
        Reference matrix of pure cell-type methylation profiles.
        Must have the same CpG index as ``beta`` (or a subset thereof) and cell \
        types as columns.
    method : str, default "nnls"
        Deconvolution algorithm. Currently only ``"nnls"`` (scipy.optimize.nnls) \
        is implemented.
    n_jobs : int, default 1
        Number of parallel processes for sample-wise deconvolution.
        Requires ``joblib``; automatically falls back to serial execution if \
        unavailable or set to 1.

    Returns
    -------
    pd.DataFrame
        Estimated cell-type proportions with:
        rows = samples (same order and names as ``beta.columns``)
        columns = cell types (same as ``ref_profiles.columns``)
        values constrained to ≥0 and normalized to sum to 1 per sample

    Raises
    ------
    ValueError
        If no overlapping CpG features exist between ``beta`` and ``ref_profiles``.

    Notes
    -----
    - Automatic feature alignment: only CpGs present in both matrices are used.
    - Proportions are forcibly normalized to sum to 1 (with protection against \
    division by zero).
    - Highly efficient for typical blood, brain, or tumor microenvironment references \
    (e.g., FlowSorted.Blood.EPIC, Houseman, etc.).
    - Designed for seamless integration with DMeth preprocessing pipelines \
    (beta-value input expected).
    """
    if beta is None or beta.empty or ref_profiles is None or ref_profiles.empty:
        logger.warning(
            "Returned empty dataframe because either beta or ref_profiles is invalid"
        )
        return pd.DataFrame()
    # align features
    features = beta.index.intersection(ref_profiles.index)
    if len(features) == 0:
        raise ValueError("No overlapping features between beta and reference")
    B = beta.loc[features].values  # features x samples
    R = ref_profiles.loc[features].values  # features x celltypes
    # for each sample solve R * w = b (R: f x k ; b: f)
    celltypes = list(ref_profiles.columns)
    samples = list(beta.columns)
    props = np.zeros((len(samples), len(celltypes)), dtype=float)
    if joblib is not None and n_jobs != 1:

        def _fit_sample(i):
            b = B[:, i]
            w, rnorm = nnls(R, b)
            w = w / max(w.sum(), 1e-12)
            return w

        with joblib.Parallel(n_jobs=n_jobs) as par:
            rows = par(joblib.delayed(_fit_sample)(i) for i in range(B.shape[1]))
        props = np.vstack(rows)
    else:
        logger.warning(
            "Running NNLS in serial mode (joblib not available or n_jobs=1)."
        )
        for i in range(B.shape[1]):
            b = B[:, i]
            w, rnorm = nnls(R, b)
            if w.sum() > 0:
                w = w / max(w.sum(), 1e-12)
            props[i, :] = w
    return pd.DataFrame(props, index=samples, columns=celltypes)
