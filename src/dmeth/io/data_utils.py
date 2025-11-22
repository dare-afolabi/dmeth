#!/usr/bin/env python
# coding: utf-8


"""
Core data structures and validation utilities for the DMeth pipeline.

This module defines the central ProcessedData container used throughout the package
and provides helper functions for ensuring data integrity (index alignment \
and type safety).

Key Components
--------------
ProcessedData

- The standard container holding aligned methylation matrix, sample metadata,
and probe annotation. All analysis functions expect this object.

- Validation helpers:

    - `_ensure_index_alignment(M, pheno, ann)`
        Checks that samples and probes are consistently indexed across components.
    - `_ensure_index_strings(df)`
        Safely converts DataFrame indices, but not columns, to strings (required \
        for HDF5/h5py compatibility).

All ProcessedData objects are automatically validated and string-indexed upon creation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from dmeth.utils.logger import logger


def _ensure_index_alignment(
    M: pd.DataFrame, pheno: pd.DataFrame, ann: Optional[pd.DataFrame]
) -> None:
    """
    Validate that sample and probe indices are consistent across the methylation \
    matrix, phenotype table, and annotation.

    Raises a clear KeyError with the first few mismatched identifiers if \
    any discrepancy is found.

    Parameters
    ----------
    M : pd.DataFrame
        Methylation matrix (CpGs × samples).
    pheno : pd.DataFrame
        Sample phenotype/metadata table (samples must be the index).
    ann : pd.DataFrame or None
        Optional CpG annotation table (CpGs must be the index). If ``None`` or empty,
        probe-level checks are skipped with a warning.

    Raises
    ------
    KeyError
        If samples present in ``M`` are missing from ``pheno`` or probes \
        present in ``M`` are missing from ``ann``.
    """
    # samples
    missing_samples = set(M.columns) - set(pheno.index)
    if missing_samples:
        raise KeyError(
            f"Samples in M but not in pheno: {sorted(list(missing_samples))[:5]}..."
        )

    # If annotation was not provided, skip probe checks
    if ann is None:
        return

    # If annotation is provided but empty index, warn and skip
    if ann.index is None or len(ann.index) == 0:
        # allow empty annotation but warn
        logger.warning(
            "Annotation provided but empty; skipping probe <-> \
            annotation alignment check."
        )
        return

    missing_probes = set(M.index) - set(ann.index)
    if missing_probes:
        raise KeyError(
            f"Probes in M but not in ann: {sorted(list(missing_probes))[:5]}..."
        )


def _ensure_index_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the DataFrame with its index converted to string type.

    Columns are left untouched. This is required for reliable HDF5 storage (h5py) and
    consistent merging of CpG/gene identifiers.

    Parameters
    ----------
    df : pd.DataFrame or None
        Input DataFrame (or ``None``).

    Returns
    -------
    pd.DataFrame or None
        New DataFrame with string index, or ``None`` if the input was ``None``.
    """
    if df is None:
        return df
    df = df.copy()
    df.index = df.index.astype(str)
    return df


@dataclass
class ProcessedData:
    """
    Central container for aligned methylation data used throughout the DMeth pipeline.

    Guarantees that:

    - All components (methylation matrix, sample metadata, probe annotation) \
    share consistent string indices.
    - Sample and probe alignment is validated on construction.
    - A ``meta`` dictionary tracks processing history (normalisation, \
    batch correction, QC metrics, etc.).

    All downstream functions expect or return instances of this class.

    Parameters
    ----------
    M : pd.DataFrame
        Methylation matrix with CpG sites as rows and samples as columns.
        Values are typically beta (0–1) or M-values (-inf to +inf).
    pheno : pd.DataFrame
        Sample metadata table. Must be indexed by sample IDs (strings \
        after construction).
    ann : pd.DataFrame or None
        Optional probe/CpG annotation table (e.g., Illumina manifest). \
        Must be indexed by CpG IDs.
        If ``None``, probe-level alignment checks are skipped.
    meta : dict, optional
        Free-form dictionary storing pipeline provenance and parameters.
        Pre-populated with sensible defaults if not provided.

    Attributes
    ----------
    M : pd.DataFrame
        Methylation matrix (string-indexed rows and columns).
    pheno : pd.DataFrame
        Sample metadata (string-indexed).
    ann : pd.DataFrame or None
        Probe annotation (string-indexed if present).
    meta : dict
        Processing metadata (e.g., ``{"matrix_type": "beta", \
        "normalized": True, ...}``).

    Notes
    -----
    The ``__post_init__`` method automatically:

    - Converts all relevant indices to strings.
    - Validates alignment between components.
    - Raises informative ``KeyError`` on mismatch.
    """

    M: pd.DataFrame
    pheno: pd.DataFrame
    ann: pd.DataFrame
    meta: Dict[str, Any] = field(
        default_factory=lambda: {
            "matrix_type": "beta",
            "normalized": False,
            "batch_corrected": False,
            "qc": {},
        }
    )

    def __post_init__(self) -> None:
        """
        Perform automatic validation and index normalisation immediately \
        after object construction.

        - Converts all relevant indices (M rows/columns, pheno index, \
        ann index) to strings.
        - Calls ``_ensure_index_alignment`` to guarantee consistency.
        - No return value; raised on validation failure.
        """
        self.M.index = self.M.index.astype(str)
        self.M.columns = self.M.columns.astype(str)
        self.pheno.index = self.pheno.index.astype(str)
        if self.ann is not None:
            self.ann.index = self.ann.index.astype(str)
        _ensure_index_alignment(self.M, self.pheno, self.ann)
