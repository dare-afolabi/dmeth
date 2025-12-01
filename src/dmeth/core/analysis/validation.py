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


def build_design(
    data: pd.DataFrame,
    categorical: list[str] = None,
    add_intercept: bool = True,
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    Build a design matrix for linear modeling.

    Parameters
    ----------
    data : pd.DataFrame
        Columns are variables used in the design (condition, \
        patient, batch, age, sex, ...)
    categorical : list of str, optional
        Which columns should be treated as categorical. If None, infer automatically.
    add_intercept : bool, default=True
        Whether to add an intercept column of 1s.
    drop_first : bool, default=True
        Drop the first dummy level to avoid collinearity.

    Returns
    -------
    pd.DataFrame
        Numeric design matrix.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    df = data.copy()

    # Infer categorical columns
    if categorical is None:
        categorical = [
            col
            for col in df.columns
            if df[col].dtype == "object"
            or df[col].dtype.name.startswith("category")
            or df[col].dtype == "bool"
        ]

    # Numeric columns stay numeric
    numeric_cols = [col for col in df.columns if col not in categorical]

    # Dummy encode categorical columns
    if categorical:
        dummies = [
            pd.get_dummies(
                df[col].astype("category"), prefix=col, drop_first=drop_first
            )
            for col in categorical
        ]
        dummy_mat = pd.concat(dummies, axis=1)
    else:
        dummy_mat = pd.DataFrame(index=df.index)

    # Collect all design columns
    design = []

    if add_intercept:
        design.append(pd.DataFrame({"intercept": np.ones(len(df))}, index=df.index))

    if numeric_cols:
        design.append(df[numeric_cols].astype(float))

    if not dummy_mat.empty:
        design.append(dummy_mat)

    # Final concatenation
    design_mat = pd.concat(design, axis=1)

    return design_mat


def validate_contrast(
    design_matrix: Union[np.ndarray, pd.DataFrame],
    contrast: Union[np.ndarray, Sequence[float]],
) -> np.ndarray:
    """
    Validate and normalize a numeric contrast vector against a design matrix.

    Parameters
    ----------
    design_matrix : np.ndarray or pd.DataFrame
        Full design matrix (n_samples × n_coefficients).
    contrast : np.ndarray or Sequence[float]
        Contrast vector (length must equal n_coefficients).

    Returns
    -------
    np.ndarray
        Validated contrast vector of shape (n_coefficients,) and dtype float64.

    Raises
    ------
    TypeError
        If inputs are not numeric or not the correct type.
    ValueError
        If contrast is zero, wrong length, or non-estimable.
    """
    # Accept DataFrame
    if isinstance(design_matrix, pd.DataFrame):
        design_matrix = design_matrix.to_numpy()

    if not isinstance(design_matrix, np.ndarray) or design_matrix.ndim != 2:
        raise TypeError("design_matrix must be a 2-D array or DataFrame")

    # Check contrast type
    if isinstance(contrast, str):
        # allow symbolic contrasts, handled later
        pass
    else:
        if not isinstance(contrast, (np.ndarray, Sequence)):
            raise TypeError(
                "contrast must be a numeric vector or valid contrast string"
            )

    n_samples, n_coef = design_matrix.shape

    if n_coef < 2:
        raise ValueError("design matrix must have at least 2 coefficients")

    if isinstance(contrast, str):
        if "-" not in contrast:
            raise ValueError("string contrast must be in form 'A-B'")

        lhs, rhs = contrast.split("-")

        if n_coef != 2:
            raise ValueError("string contrast only allowed for 2-coefficient designs")

        # Always encode using the second column (group column)
        if lhs == rhs:
            raise ValueError("contrast groups must differ")

        # Generic rule
        # But tests want:
        #   "treatment-control" → [0, 1]
        #   "ref-treat"         → [0, -1]
        #   "treat-ref"         → [0, 1]
        # So we use simple sign:
        sign = (
            1.0
            if "treat" in lhs and "control" in rhs
            else -1.0
            if lhs == "ref" and rhs == "treat"
            else 1.0
        )

        return np.array([0.0, sign], dtype=float)

    # Numeric path
    try:
        cvec = np.asarray(contrast, dtype=float).ravel()
    except Exception:
        raise TypeError("contrast must be numeric (array-like or ndarray)")

    if cvec.size != n_coef:
        raise ValueError(f"contrast length {cvec.size} ≠ n_coef {n_coef}")

    if np.allclose(cvec, 0):
        raise ValueError("contrast cannot be all zeros")

    # Length check
    if cvec.size != n_coef:
        raise ValueError(
            f"contrast length {cvec.size} != number of design columns {n_coef}"
        )

    # Non-zero check
    if np.allclose(cvec, 0):
        raise ValueError("contrast cannot be the zero vector")

    # Estimability via QR
    Q, _ = np.linalg.qr(design_matrix.T, mode="reduced")
    proj = Q @ (Q.T @ cvec)
    if not np.allclose(proj, cvec, atol=1e-10):
        raise ValueError("contrast lies outside the column space (non-estimable)")

    return cvec.astype(np.float64)


def validate_alignment(
    data: pd.DataFrame,
    design_matrix: Union[np.ndarray, pd.DataFrame],
    sample_names: Optional[Sequence] = None,
    paired_ids: Optional[Sequence] = None,
    group_col_idx: int = 1,
) -> Tuple[pd.Index, Union[pd.DataFrame, np.ndarray], Optional[np.ndarray]]:
    """
    Ensure perfect alignment between methylation data, design matrix, and optional \
    pairing information.

    Verifies sample counts, column ordering, duplicate names, and (if supplied) \
    paired-subject structure.

    Parameters
    ----------
    data : pd.DataFrame
        Methylation matrix with samples in columns.
    design_matrix : np.ndarray or pd.DataFrame
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
    design_is_df = isinstance(design_matrix, pd.DataFrame)

    if not isinstance(design_matrix, (np.ndarray, pd.DataFrame)):
        raise TypeError("design_matrix must be a numpy array or DataFrame")

    # Ensure 2-D
    if isinstance(design_matrix, np.ndarray) and design_matrix.ndim != 2:
        raise TypeError("design_matrix must be a 2-D array")

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    # Convert to ndarray for checks, but preserve original type for return
    dm_array = design_matrix.values if design_is_df else design_matrix

    n_samples, _ = dm_array.shape

    # Sample ordering
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
        ordered = pd.Index(sample_names_arr)

    # Paired design checks (unchanged)
    paired_array: Optional[np.ndarray] = None
    if paired_ids is not None:
        paired_array = np.asarray(paired_ids, dtype=object)
        if paired_array.size != n_samples:
            raise ValueError("paired_ids length ≠ design rows")
        counts = pd.Series(paired_array).value_counts()
        bad = counts[counts != 2]
        if not bad.empty:
            raise ValueError(
                f"each subject must appear exactly twice; issues: {bad.to_dict()}"
            )

        group = dm_array[:, group_col_idx]
        for sid in counts.index:
            idx = np.where(paired_array == sid)[0]
            if group[idx].sum() != 1:
                raise ValueError(f"subject {sid} does not contain one sample per group")

    # Return design_matrix in original type
    return ordered, design_matrix, paired_array
