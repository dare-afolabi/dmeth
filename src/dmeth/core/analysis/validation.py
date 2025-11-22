#!/usr/bin/env python
# coding: utf-8


"""
Input validation and integrity checks for differential methylation analysis.

- This module implements rigorous pre-analysis validation of data matrices, \
experimental designs, contrasts, sample alignment, and system resources. \
- It ensures statistical estimability, correct pairing structure, and sufficient \
memory before launching computationally intensive analyses, preventing \
silent failures and ambiguous results.

Features
--------
- Accurate memory footprint estimation with conservative peak usage \
prediction and automatic MemoryError on insufficient RAM
- Strict validation of two-group design vectors with automatic construction of \
intercept + indicator design matrix
- Comprehensive contrast validation including shape, zero-vector detection, \
and QR-based estimability checking
- Flexible string contrast syntax ("treatment-control") for simple two-column designs
- Thorough alignment verification between methylation data columns, design \
matrix rows, and optional sample identifiers
- Robust paired-sample validation ensuring exactly two observations per \
subject and balanced group representation within pairs
- Clear, actionable error messages and logging for rapid debugging in \
production pipelines
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from dmeth.utils.logger import logger

try:
    import psutil
except Exception:
    psutil = None
    logger.warning(
        "psutil module not installed, `check_analysis_memory` is unavailable."
    )


def check_analysis_memory(M: pd.DataFrame, warn_threshold_gb: float = 8.0):
    """
    Conservatively estimate RAM requirements for a full-matrix differential analysis.

    - Calculates current data footprint, projects peak usage (approximately 4 times \
    input size), and compares against available system memory.
    - Raises MemoryError early if danger is high, forcing use of chunked mode.

    Parameters
    ----------
    M : pd.DataFrame
        Methylation matrix (CpGs × samples).
    warn_threshold_gb : float, default 8.0
        Issue a warning if estimated peak exceeds this value.

    Returns
    -------
    dict
        Keys: ``data_gb``, ``peak_gb``, ``available_gb``.

    Raises
    ------
    MemoryError
        If projected peak consumption exceeds ~80% of available RAM. In such cases \
        ``fit_differential_chunked`` should be used instead.
    """
    if psutil is None:
        logger.debug("psutil not available - skipping memory safety check")
        return {"data_gb": np.nan, "peak_gb": np.nan, "available_gb": np.nan}

    data_gb = M.memory_usage(deep=True).sum() / (1024**3)
    estimated_peak_gb = data_gb * 4.0

    available_gb = psutil.virtual_memory().available / (1024**3)

    logger.info(
        f"Methylation matrix: {data_gb:.2f} GB → estimated peak: \
        {estimated_peak_gb:.2f} GB"
    )

    if estimated_peak_gb > available_gb * 0.85:  # Slightly more conservative
        raise MemoryError(
            f"Projected memory usage (~{estimated_peak_gb:.1f} GB) exceeds "
            f"85% of available RAM ({available_gb:.1f} GB).\n"
            "Use fit_differential_chunked() to process in chunks."
        )
    elif estimated_peak_gb > warn_threshold_gb:
        logger.warning(
            f"Large analysis detected (~{estimated_peak_gb:.1f} GB peak). "
            "Consider preprocessing filters or fit_differential_chunked()."
        )

    return {
        "data_gb": data_gb,
        "peak_gb": estimated_peak_gb,
        "available_gb": available_gb,
    }


def validate_design(design: Sequence) -> pd.DataFrame:
    """
    Convert a simple two-group label vector into a proper design matrix.

    Accepts lists, tuples, arrays, or pandas Series containing exactly two \
    distinct group labels and returns a two-column matrix:

    - Column 0: intercept (all 1s)
    - Column 1: group indicator (0 = reference group, 1 = alternative)

    Parameters
    ----------
    design : Sequence
        Vector of group assignments (length ≥ 2).

    Returns
    -------
    pd.DataFrame
        n_sample × 2 design matrix acceptable for ``fit_differential``.

    Raises
    ------
    TypeError
        If input is not a supported sequence type.
    ValueError
        If fewer than two samples, NaNs present, or more/less than two unique groups.
    """
    if not isinstance(design, (list, tuple, np.ndarray, pd.Series)):
        raise TypeError(
            "`design` must be a list, tuple, numpy array, or pandas Series."
        )

    s = pd.Series(design)

    if s.isna().any():
        raise ValueError("`design` contains NaN or missing values.")
    if len(s) < 2:
        raise ValueError(f"design must contain ≥2 samples (got {len(s)})")

    uniq = s.unique()
    if len(uniq) != 2:
        raise ValueError(
            f"design must have exactly two groups (got {len(uniq)}: {uniq.tolist()})"
        )

    ref = uniq[0]
    indicator = (s != ref).astype(float)

    return pd.DataFrame(
        {
            "intercept": 1.0,
            "group": indicator.values,
        }
    )


def validate_contrast(
    design_matrix: np.ndarray, contrast: Union[np.ndarray, Sequence[float], str]
) -> np.ndarray:
    """
    Validate and normalise a contrast against a given design matrix.

    Supports:

    - Numeric vectors/matrices
    - Simple string syntax ``"treatment-control"`` (only for two-column \
    intercept + group designs)

    Checks dimensionality, non-zero status, and estimability via QR decomposition.

    Parameters
    ----------
    design_matrix : np.ndarray
        Full design matrix (n_samples × n_coefficients).
    contrast : np.ndarray, Sequence[float], or str
        Contrast specification.

    Returns
    -------
    np.ndarray
        Validated contrast vector (float64.

    Raises
    ------
    ValueError
        On shape mismatch, zero contrast, or non-estimable contrast.
    """
    if not isinstance(design_matrix, np.ndarray) or design_matrix.ndim != 2:
        raise TypeError("design_matrix must be a 2-D np.ndarray")
    n_samples, n_coef = design_matrix.shape
    if n_coef < 2:
        raise ValueError("design_matrix must have ≥2 columns")

    # Numeric path
    if not isinstance(contrast, str):
        try:
            cvec = np.asarray(contrast, dtype=float)
        except Exception as e:
            raise TypeError("contrast must be numeric or a string") from e
    else:
        # String path (2-column only)
        if n_coef != 2:
            raise ValueError("String contrast only for intercept+group designs")
        contrast = contrast.replace(" ", "")
        if "-" not in contrast:
            raise ValueError("String contrast must contain '-'")
        left, right = contrast.split("-", 1)

        uniq = np.unique(design_matrix[:, 1])
        # ref, treat = uniq[0], uniq[1]
        ref, _ = uniq[0], uniq[1]

        cvec = np.array([0.0, 1.0])  # treat - ref
        if left.lower() in {str(ref).lower(), "ref", "reference", "control", "0"}:
            cvec[1] = -1.0

    # Shape & zero check
    if cvec.ndim != 1 or cvec.size != n_coef:
        raise ValueError(f"contrast length {cvec.size} ≠ #coefficients {n_coef}")
    if np.allclose(cvec, 0):
        raise ValueError("contrast cannot be the zero vector")

    # Estimability (QR projection)
    qt, _ = np.linalg.qr(design_matrix.T, mode="reduced")
    proj = qt @ (qt.T @ cvec)
    if not np.allclose(proj, cvec, atol=1e-10):
        raise ValueError("contrast lies outside the column space (non-estimable)")

    return cvec.astype(np.float64)


def validate_alignment(
    data: pd.DataFrame,
    design_matrix: np.ndarray,
    sample_names: Optional[Sequence] = None,
    paired_ids: Optional[Sequence] = None,
    group_col_idx: int = 1,
) -> Tuple[pd.Index, np.ndarray, Optional[np.ndarray]]:
    """
    Ensure perfect alignment between methylation data, design matrix, and optional \
    pairing information.

    Verifies sample counts, column ordering, duplicate names, and (if supplied) \
    paired-subject structure.

    Parameters
    ----------
    data : pd.DataFrame
        Methylation matrix with samples in columns.
    design_matrix : np.ndarray
        Corresponding design matrix.
    sample_names : Sequence or None
        Explicit ordered list of sample identifiers (required only \
        if column names differ).
    paired_ids : Sequence or None
        Subject/block identifiers for paired designs; each ID must appear exactly twice.
    group_col_idx : int, default 1
        Column in design_matrix encoding group membership (used to verify \
        balanced pairing).

    Returns
    -------
    tuple
        (ordered_data_columns, validated_design_matrix, validated_paired_array_or_None)

    Raises
    ------
    ValueError
        On any misalignment, duplicate column names, or invalid pairing structure \
        (e.g., a subject with both samples in the same group).
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(design_matrix, np.ndarray) or design_matrix.ndim != 2:
        raise TypeError("design_matrix must be a 2-D np.ndarray")

    n_samples, _ = design_matrix.shape

    # Column ordering
    if sample_names is None:
        if data.shape[1] != n_samples:
            raise ValueError(
                f"design rows ({n_samples}) ≠ data columns ({data.shape[1]})"
            )
        if data.columns.duplicated().any():
            raise KeyError("data contains duplicate column names")
        ordered = data.columns
    else:
        sample_names_arr = np.asarray(sample_names, dtype=str)
        if sample_names_arr.size != n_samples:
            raise KeyError("sample_names length ≠ design rows")
        missing = set(sample_names_arr) - set(data.columns)
        if missing:
            raise KeyError(f"Missing samples in data: {sorted(missing)}")
        if data.columns.duplicated().any():
            raise KeyError("data contains duplicate column names")
        ordered = pd.Index(sample_names_arr)

    # Pairing
    paired_array: Optional[np.ndarray] = None
    if paired_ids is not None:
        paired_array = np.asarray(paired_ids, dtype=object)
        if paired_array.size != n_samples:
            raise ValueError("paired_ids length ≠ design rows")
        if pd.isna(paired_array).any():
            raise ValueError("paired_ids contains NaN")

        counts = pd.Series(paired_array).value_counts()
        bad = counts[counts != 2]
        if not bad.empty:
            raise ValueError(
                f"each subject must appear exactly twice; issues: {bad.to_dict()}"
            )

        group = design_matrix[:, group_col_idx]
        for sid in counts.index:
            idx = np.where(paired_array == sid)[0]
            if group[idx].sum() != 1:
                raise ValueError(f"subject {sid} does not contain one sample per group")

    return ordered, design_matrix, paired_array
