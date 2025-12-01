#!/usr/bin/env python
# coding: utf-8


"""
Output utilities for saving processed methylation data and analysis results.

This module provides a unified interface (Python API + Typer CLI) for exporting:

- ProcessedData objects in multiple formats
- Differential methylation results (filtered and clean)
- IDAT-style HDF5 archives (compatible with downstream tools)
- Self-contained HTML analysis reports with embedded plots

All export functions include overwrite protection, automatic directory creation,
and informative logging.

Features
--------
- Smart format detection and suffix handling
- Optional compression (blosc:zstd or gzip)
- Safe export of differential results (core columns + group means)
- IDAT-compatible HDF5 layout via h5py (fallback to pandas HDFStore)
- One-call HTML report generation from summary dict + matplotlib figures
"""


import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from dmeth.io.data_utils import ProcessedData, _ensure_index_strings
from dmeth.utils.logger import logger

try:
    import h5py
except ImportError:
    h5py = None


def export_results(
    res: pd.DataFrame,
    output_path: str,
    format: str = "csv",
    include_all: bool = False,
    verbose: bool = True,
    overwrite: bool = True,
) -> None:
    """
    Export differential methylation results to CSV, TSV, or Excel with \
    optional column filtering.

    - By default only core statistical columns and per-group mean columns are exported.
    - Overwrite protection is applied if ``overwrite=False``.

    Parameters
    ----------
    res : pd.DataFrame
        Differential results table (as returned by the analysis pipeline).
    output_path : str
        Destination file path (extension determines format if ``format`` is omitted).
    format : {"csv", "tsv", "excel"}, default "csv"
        Output file format.
    include_all : bool, default False
        If True, export every column; if False, only core + mean columns are kept.
    verbose : bool, default True
        Log a confirmation message on success.
    overwrite : bool, default True
        Raise FileExistsError if the file exists and this is False.

    Raises
    ------
    FileExistsError
        When ``overwrite=False`` and the target file already exists.
    ValueError
        For unsupported ``format``.
    RuntimeError
        If writing fails (e.g., missing openpyxl for Excel).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _check_overwrite(path, overwrite)

    if not include_all:
        core = ["logFC", "t", "pval", "padj", "F", "df1", "df2"]
        means = [c for c in res.columns if c.startswith(("meanM_", "meanB_"))]
        cols = [c for c in core + means if c in res.columns]
        res_export = res[cols]
    else:
        res_export = res

    fmt = format.lower()
    try:
        if fmt == "csv":
            res_export.to_csv(path, index=True)
        elif fmt == "tsv":
            res_export.to_csv(path, sep="\t", index=True)
        elif fmt in {"xls", "xlsx", "excel"}:
            try:
                # prefer xlsxwriter to avoid openpyxl/lxml disruptions
                try:
                    logger.info("Attempting Excel export with xlsxwriter engine")
                    res_export.to_excel(path, index=True, engine="xlsxwriter")
                except ImportError:
                    logger.warning("xlsxwriter not available; falling back to openpyxl")
                    res_export.to_excel(path, index=True, engine="openpyxl")
            except Exception as e:
                logger.error(f"Excel export failed: {e}")
                raise RuntimeError(f"Export failed: {e}")

        else:
            raise ValueError(f"Unsupported format: {format}")
    except Exception as e:
        raise RuntimeError(f"Export failed: {e}")

    if verbose:
        logger.info(f"Results exported to {output_path}")


def save_processed_data(
    data: "ProcessedData",
    path: Union[str, Path],
    format: str = "hdf5",
    compression: str | None = "blosc:zstd",
    complib_level: int = 9,
    overwrite: bool = True,
) -> None:
    """
    Persist a ProcessedData object to disk in one of several convenient formats.

    Supported formats
    -----------------
    csv / tsv      → three separate files (M, pheno, ann)
    xlsx           → single workbook with sheets "M", "pheno", "ann"
    pickle / pkl   → direct pickle of the ProcessedData instance
    hdf5 / h5      → pandas HDFStore with compressed tables

    Parameters
    ----------
    data : ProcessedData
        The container to save.
    path : str or Path
        Output path; suffix is automatically corrected to match ``format``.
    format : {"csv", "tsv", "xlsx", "pickle", "pkl", "hdf5", "h5"}, default "hdf5"
        Desired export format.
    compression : str or None, default "blosc:zstd"
        Compression library for HDF5 (only used when format is hdf5).
    complib_level : int, default 9
        Compression level (1–9) for HDF5.
    overwrite : bool, default True
        Allow overwriting existing files.

    Raises
    ------
    ValueError
        If an unsupported format is requested.
    FileExistsError
        When ``overwrite=False`` and the target exists.
    """
    ann = data.ann if data.ann is not None else pd.DataFrame(index=data.M.index)

    path = Path(path)
    fmt = format.lower()

    suffix_map = {
        "csv": ".csv",
        "tsv": ".tsv",
        "xlsx": ".xlsx",
        "pickle": ".pkl",
        "pkl": ".pkl",
        "hdf5": ".h5",
        "h5": ".h5",
    }

    if fmt not in suffix_map:
        raise ValueError(
            f"Unsupported format: {format!r}. Choose from {set(suffix_map)}"
        )

    suffix = suffix_map[fmt]
    if path.suffix and path.suffix != suffix:
        path = path.with_suffix(suffix)
    elif not path.suffix:
        path = path.with_suffix(suffix)

    _check_overwrite(path, overwrite)

    if fmt == "csv":
        data.M.to_csv(path)
        data.pheno.to_csv(path.with_name(f"{path.stem}_pheno.csv"))
        if ann is not None:
            ann.to_csv(path.with_name(f"{path.stem}_ann.csv"))
        logger.info("Saved ProcessedData in csv format")

    elif fmt == "tsv":
        sep = "\t"
        data.M.to_csv(path, sep=sep)
        data.pheno.to_csv(path.with_name(f"{path.stem}_pheno.tsv"), sep=sep)
        if ann is not None:
            ann.to_csv(path.with_name(f"{path.stem}_ann.tsv"), sep=sep)
        logger.info("Saved ProcessedData in tsv format")

    elif fmt == "xlsx":
        try:
            # prefer xlsxwriter
            try:
                logger.info("Saving ProcessedData to Excel with xlsxwriter")
                with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
                    data.M.to_excel(writer, sheet_name="M")
                    data.pheno.to_excel(writer, sheet_name="pheno")
                    if ann is not None:
                        ann.to_excel(writer, sheet_name="ann")
            except ImportError:
                logger.warning("xlsxwriter not available; falling back to openpyxl")
                with pd.ExcelWriter(path, engine="openpyxl") as writer:
                    data.M.to_excel(writer, sheet_name="M")
                    data.pheno.to_excel(writer, sheet_name="pheno")
                    if ann is not None:
                        ann.to_excel(writer, sheet_name="ann")
        except Exception as e:
            logger.error(f"Excel save failed: {e}")
            raise RuntimeError(f"Export failed: {e}")

    elif fmt in ("pickle", "pkl"):
        pd.to_pickle(data, path)
        logger.info("Saved ProcessedData in pickle format")

    elif fmt in ("hdf5", "h5"):
        comp_kwargs = {}
        if compression:
            comp_kwargs = {"complevel": complib_level, "complib": compression}

        with pd.HDFStore(path, mode="w") as store:
            store.put("M", data.M, format="table", data_columns=True, **comp_kwargs)
            store.put(
                "pheno", data.pheno, format="table", data_columns=True, **comp_kwargs
            )
            if ann is not None:
                store.put("ann", ann, format="table", data_columns=True, **comp_kwargs)
        logger.info("Saved ProcessedData in HDF format")


def _check_overwrite(path: Path, overwrite: bool) -> None:
    """
    Helper that raises FileExistsError if the path exists and overwriting is disabled.

    Used internally by all export functions.

    Parameters
    ----------
    path : Path
        File path to check.
    overwrite : bool
        Permission flag.

    Raises
    ------
    FileExistsError
        When the file exists and ``overwrite=False``.
    """
    if not overwrite and path.exists():
        raise FileExistsError(f"{path} exists and overwrite=False")


def export_idat_hdf5(
    beta: pd.DataFrame,
    mvals: Optional[pd.DataFrame],
    sample_sheet: Optional[pd.DataFrame],
    filepath: Union[str, Path],
    compress: bool = True,
) -> Path:
    """
    Write methylation data to an IDAT-compatible HDF5 archive using h5py \
    (preferred) or pandas fallback.

    Layout mirrors common downstream tools: groups ``/beta``, ``/mvals`` (optional),
    and ``/samples`` containing one JSON dataset per sample.

    Parameters
    ----------
    beta : pd.DataFrame
        Beta-value matrix (CpGs × samples).
    mvals : pd.DataFrame or None, optional
        M-value matrix (same dimensions).
    sample_sheet : pd.DataFrame or None, optional
        Sample metadata (index = sample IDs).
    filepath : str or Path
        Destination HDF5 file.
    compress : bool, default True
        Apply gzip compression to large datasets.

    Returns
    -------
    Path
        Path to the created HDF5 file.

    Raises
    ------
    RuntimeError
        If neither h5py nor pandas.HDFStore is available.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # Coerce types
    beta = _ensure_index_strings(beta)
    if mvals is not None:
        mvals = _ensure_index_strings(mvals)
    sample_sheet = sample_sheet.copy() if sample_sheet is not None else pd.DataFrame()

    # H5PY path
    if h5py is not None:
        with h5py.File(str(filepath), "w") as h5:
            grp = h5.create_group("beta")
            grp.create_dataset(
                "values",
                data=beta.values,
                compression="gzip" if compress else None,
                chunks=True,
            )  # Auto-chunk for better I/O
            grp.create_dataset("index", data=np.asarray(beta.index.astype("S")))
            grp.create_dataset("columns", data=np.asarray(beta.columns.astype("S")))

            if mvals is not None:
                grp2 = h5.create_group("mvals")
                grp2.create_dataset(
                    "values",
                    data=mvals.values,
                    compression="gzip" if compress else None,
                )
                grp2.create_dataset("index", data=np.asarray(mvals.index.astype("S")))
                grp2.create_dataset(
                    "columns", data=np.asarray(mvals.columns.astype("S"))
                )
            samp_grp = h5.create_group("samples")
            # sample_sheet -> JSON string per row
            for cidx, row in sample_sheet.reset_index().iterrows():
                key = str(row.get(sample_sheet.index.name or "sample", cidx))
                samp_grp.create_dataset(
                    key, data=np.string_(json.dumps(row.dropna().to_dict()))
                )
            # features empty placeholder
            h5.attrs["created_by"] = "methylation_engine"
        return filepath

    # Fallback to pandas HDFStore
    else:
        try:
            store = pd.HDFStore(
                str(filepath), mode="w", complevel=9 if compress else None
            )
            store.put("beta", beta)
            if mvals is not None:
                store.put("mvals", mvals)
            if not sample_sheet.empty:
                store.put("sample_sheet", sample_sheet)
            store.close()
            return filepath
        except Exception as e:
            raise RuntimeError(
                f"No HDF writer available (h5py or pandas.HDFStore failed): {e}"
            )
Report",
) -> Path:
    """
    Generate a self-contained, minimal HTML report embedding a JSON \
    summary and PNG images of supplied figures.

    Useful for quick sharing of results without requiring Jupyter or \
    additional dependencies.

    Parameters
    ----------
    summary : dict
        Arbitrary summary statistics or results (JSON-serialisable).
    plots : dict[str, plt.Figure]
        Mapping from plot name to Matplotlib figure objects.
    outpath : str or Path, default "analysis_report.html"
        Output HTML file location.
    title : str, default "Methylation Analysis Report"
        Top-level heading in the report.

    Returns
    -------
    Path
        Path to the generated HTML file.
    """
    outpath = Path(outpath)
    outdir = outpath.parent
    outdir.mkdir(parents=True, exist_ok=True)
    images = {}
    for name, fig in (plots or {}).items():
        if fig is None:
            continue
        p = outdir / f"{name}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        images[name] = p.name

    html_lines = [
        f"<html><head><meta charset='utf-8'><title>{title}</title></head><body>",
        f"<h1>{title}</h1>",
        "<h2>Summary</h2>",
        "<pre>",
    ]
    html_lines.append(json.dumps(summary, indent=2, default=str))
    html_lines.append("</pre>")
    for name, img in images.items():
        html_lines.append(f"<h3>{name}</h3>")
        html_lines.append(f"<img src='{img}' style='max-width:100%;height:auto'/>")
    html_lines.append("</body></html>")
    outpath.write_text("\n".join(html_lines), encoding="utf-8")
    return outpath
