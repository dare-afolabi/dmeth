# dmeth - documentation for reproducible DNA methylation analysis

## User Guide

<div align="center">
  <a href="https://codecov.io/gh/dare-afolabi/dmeth">
    <img src="https://img.shields.io/codecov/c/github/dare-afolabi/dmeth?style=flat" alt="Coverage">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  </a>
  <a href="https://badge.fury.io/py/dmeth">
    <img src="https://badge.fury.io/py/dmeth.svg" alt="PyPI version">
  </a>
  <a href="https://github.com/sponsors/dare-afolabi">
    <img src="https://img.shields.io/badge/Sponsor-grey?style=flat&logo=github-sponsors" alt="Sponsor">
  </a>
</div>


A fast, statistically rigorous Python framework providing a toolkit for DNA methylation analysis - from raw beta matrices to biomarkers and functional interpretation. **`dmeth`** implements the full modern differential methylation pipeline used in high-impact epigenome-wide association studies (EWAS), with performance and correctness on par with established R/bioconductor tools, all in pure Python.


## Key Features

| Feature                                | Implementation                                      | Performance |
|----------------------------------------|------------------------------------------------------|-------------|
| Empirical Bayes moderated t-tests      | limma-style (Smyth 2004) with exact replication     | Numba-accelerated (10–100× faster) |
| Memory-efficient chunked analysis      | Automatic fallback for >1M probes                   | <4 GB RAM typical |
| Cell-type deconvolution               | Reference-based NNLS (Houseman/Horvath-style)       | Parallel joblib |
| DMR discovery                          | Sliding-window clustering + gap merging             | Vectorized |
| Gene annotation & pathway enrichment   | IntervalTree + Fisher’s exact (FDR)                 | Sub-second on 450k/EPIC |
| Coordinate liftover (hg19 ↔ hg38)     | pyliftover integration                              | Per-region tracking |
| Biomarker panel discovery & validation | RF / Elastic Net + stratified CV                    | Built-in |
| Robust preprocessing & QC              | Missingness, group representation, imputation       | Production-safe |

Fully supports **Illumina 450K**, **EPIC (850K)**, and any custom CpG × sample matrix.


## Quick Start

```bash
pip install "dmeth[full]"
```

```python
import pandas as pd

from dmeth.io.readers import load_methylation_data
from dmeth.core.analysis.preparation import filter_cpgs_by_missingness, impute_missing_values
from dmeth.core.analysis.validation import build_design, validate_contrast
from dmeth.core.analysis.core_analysis import fit_differential
from dmeth.core.downstream.annotation import find_dmrs_by_sliding_window

# 1. Load your data
# beta: CpG x samples matrix
# pheno: sample metadata with a 'group' column
beta = pd.read_csv("beta_matrix.csv", index_col=0)
pheno = pd.read_csv("phenotype.csv", index_col=0)

# 2. Preprocessing
# Drop CpGs with too much missingness
beta_clean, _, _ = filter_cpgs_by_missingness(beta, max_missing_rate=0.2)

# Impute remaining missing values (kNN)
beta_imp = impute_missing_values(beta_clean, method="knn", k=10)

# 3. Differential analysis (case vs control)
# Build design matrix from phenotype
design = build_design(pheno["group"], categorical=["group"])
contrast = [0] * (design.shape[1] - 1) + [1]
contrast = validate_contrast(design, contrast)

# Extract group labels
group_labels = pd.Series(
    np.where(design["group"], "B", "A"),
    index=beta_imp.columns,
    name="group",
)

# Fit - use fit_differential_chunked() for larger datasets
res = fit_differential(
    data=beta_imp,
    design=design,
    contrast=contrast,
    group_labels=group_labels,
    shrink="smyth",
    robust=True,
)

# 4. Discover DMRs
ann = pd.read_csv("cpg_annotation.csv", index_col=0)  # must include chr, pos columns
dmrs = find_dmrs_by_sliding_window(
    dms=res[res["padj"] < 0.05],
    annotation=ann,
    chr_col="CHR",
    pos_col="MAPINFO",
    max_gap=500,
    min_cpgs=3,
)

print(f"Found {len(dmrs)} DMRs")
print(dmrs.head())
```


## Installation

```bash
# Minimal (no speed, annotation, and other extras)
pip install dmeth

# Recommended: full scientific environment
pip install "dmeth[full]"

# Development
pip install "dmeth[full,dev]"
```

Optional extras (dmeth\[full]):

- **speed**: numba, combat (highly recommended)
- **annotation**: intervaltree, pyliftover
- **parallel**: joblib
- **format**: PyYAML, toml, h5py, xlrd
- **plotting**: plotly, umap-learn
- **io**: pyarrow, tables, openpyxl, xlsxwriter

Optional dev extras (dmeth\[dev]):

pytest, pytest-cov, black, isort, flake8, flake8-pyproject, flake8-bugbear, bandit, mkdocs, mkdocs-material



## Reference for all modules, classes, and functions


---

## Study Planning

High-level planning API for DNA methylation studies.

This module provides a clean, user-friendly interface to the full study planning
capabilities of ``DMeth``. It exposes all configuration queries, cost/timeline
estimators, and sample size calculators as simple top-level functions.

- All functions automatically use the global ``PlannerConfig`` singleton
- (managed in `dmeth.config.config_manager`), so configuration loaded once is immediately available everywhere.

#### Features
- Direct function access: ``list_platforms()``, ``calculate_sample_size()``, etc.
- Zero-boilerplate: no need to manually instantiate or pass config objects
- Real-time updates: any config change (e.g., regional pricing) instantly affects all calculations



---

### `add_custom_platform`


`add_custom_platform(platform_id: str, platform_info: Dict[str, Any]) -> None`


Register a new custom platform after schema validation.

#### Parameters
- **platform_id**: `str`
    Unique identifier for the new platform.
- **platform_info**: `dict`
    Platform data conforming to ``PlatformSchema``.

#### Raises
ValueError
    If the provided data fails validation.


---

### `calculate_sample_size`


`calculate_sample_size(*args, **kwargs)`


Compute required sample size per group using power analysis.

- The calculation accounts for the chosen experimental design, platform
- CpG count (for Bonferroni correction), and any design-specific power
adjustment factor.

#### Parameters
- **design_id**: `str`
    Identifier of the experimental design.
- **platform_id**: `str`
    Identifier of the methylation platform.
- **effect_size**: `float`
    Expected Cohen's d.
- **alpha**: `float, default 0.05`
    Nominal type-I error rate.
- **power**: `float, default 0.8`
    Desired statistical power.
- **mcp_method**: `str, optional`
    Multiple-comparison correction method (``bonferroni``, ``fdr``, ``none``).
- **max_n**: `int, default 500`
    Upper bound on calculated sample size per group.

#### Returns
dict
    Contains ``n_per_group``, ``total_samples``, effective alpha, etc.


---

### `estimate_study_timeline`


`estimate_study_timeline(*args, **kwargs)`


Produce a detailed timeline estimate for the entire study.

The estimate respects phase-specific scaling, batch adjustments,
parallelisability, and an optional global contingency buffer.

#### Parameters
- **n_samples**: `int`
    Total number of samples to be processed.
- **platform_id**: `str`
    Platform identifier (affects array-processing duration).
- **contingency**: `bool, default True`
    Apply the global contingency buffer percentage.

#### Returns
DataFrame
    One row per phase plus a summary ``TOTAL`` row.


---

### `estimate_total_cost`


`estimate_total_cost(*args, **kwargs)`


Calculate the complete study cost and provide a detailed breakdown.

Handles per-sample, per-CpG, and fixed cost components, with optional
inclusion of components marked as ``optional``.

#### Parameters
- **n_samples**: `int`
    Number of samples.
- **platform_id**: `str`
    Platform identifier.
- **include_optional**: `bool, default True`
    Include optional cost components.
- **validate**: `bool, default True`
    Perform unit validation (kept for backward compatibility).

#### Returns
dict
    Contains a ``components`` DataFrame, ``total`` cost,
    ``per_sample`` cost, and metadata.


---

### `get_config`


`get_config() -> dmeth.config.config_manager.PlannerConfig`


Get the global PlannerConfig singleton instance.

#### Returns
PlannerConfig
    The global configuration instance.


---

### `get_cost_components`


`get_cost_components(platform: Optional[str] = None, include_optional: bool = True) -> Dict[str, Dict]`


Return cost components, optionally filtered by platform or optionality.

#### Parameters
- **platform**: `str, optional`
    Restrict to components whose ``applies_to`` list contains this platform.
- **include_optional**: `bool, default True`
    Whether to include components marked as optional.

#### Returns
dict
    Mapping from component identifier to component dictionary.


---

### `get_design`


`get_design(design_id: str) -> Dict[str, Any]`


Retrieve the configuration dictionary for a specific experimental design.

#### Parameters
- **design_id**: `str`
    Identifier of the design.

#### Returns
dict
    Design configuration.

#### Raises
KeyError
    If the design identifier is unknown.


---

### `get_platform`


`get_platform(platform_id: str) -> Dict[str, Any]`


Retrieve the configuration dictionary for a specific platform.

#### Parameters
- **platform_id**: `str`
    Identifier of the platform.

#### Returns
dict
    Platform configuration.

#### Raises
KeyError
    If the platform identifier is unknown.


---

### `get_platform_by_budget`


`get_platform_by_budget(max_cost_per_sample: float) -> DataFrame`


Return platforms whose per-sample cost does not exceed a budget limit.

#### Parameters
- **max_cost_per_sample**: `float`
    Maximum acceptable cost per sample.

#### Returns
DataFrame
    Subset of platforms meeting the budget constraint.


---

### `list_designs`


`list_designs() -> DataFrame`


Return a table of all configured experimental designs.

#### Returns
DataFrame
    One row per design.


---

### `list_platforms`


`list_platforms(recommended_only: bool = False) -> DataFrame`


Return a DataFrame containing all configured platforms.

#### Parameters
- **recommended_only**: `bool, default False`
    If True, only platforms with ``recommended=True`` are returned.

#### Returns
DataFrame
    One row per platform, indexed by platform identifier.


---

### `update_platform_cost`


`update_platform_cost(platform_id: str, new_cost: float) -> None`


Change the per-sample cost of an existing platform.

#### Parameters
- **platform_id**: `str`
    Platform identifier.
- **new_cost**: `float`
    New cost per sample (must be non-negative).

#### Raises
KeyError
    If the platform does not exist.


---

### `update_regional_pricing`


`update_regional_pricing(multiplier: float, region: Optional[str] = None) -> None`


Apply a uniform multiplicative adjustment to all platform costs.

#### Parameters
- **multiplier**: `float`
    Factor by which current costs are multiplied (must be > 0).
- **region**: `str, optional`
    Descriptive label for logging purposes.

#### Raises
ValueError
    If ``multiplier`` is not positive.


---

## Input/Output

### Data Utils

Core data structures and validation utilities for the DMeth pipeline.

This module defines the central ProcessedData container used throughout the package
and provides helper functions for ensuring data integrity (index alignment and type safety).

Key Components
--------------
ProcessedData

- The standard container holding aligned methylation matrix, sample metadata,
and probe annotation. All analysis functions expect this object.

- Validation helpers:

    - `_ensure_index_alignment(M, pheno, ann)`
        Checks that samples and probes are consistently indexed across components.
    - `_ensure_index_strings(df)`
        Safely converts DataFrame indices, but not columns, to strings (required         for HDF5/h5py compatibility).

All ProcessedData objects are automatically validated and string-indexed upon creation.



---

### `ProcessedData`


`ProcessedData(M: DataFrame, pheno: DataFrame, ann: DataFrame, meta: Dict[str, Any] = <factory>) -> None`


Central container for aligned methylation data used throughout the DMeth pipeline.

Guarantees that:

- All components (methylation matrix, sample metadata, probe annotation)     share consistent string indices.
- Sample and probe alignment is validated on construction.
- A ``meta`` dictionary tracks processing history (normalisation,     batch correction, QC metrics, etc.).

All downstream functions expect or return instances of this class.

#### Parameters
- **M**: `pd.DataFrame`
    Methylation matrix with CpG sites as rows and samples as columns.
    Values are typically beta (0–1) or M-values (-inf to +inf).
- **pheno**: `pd.DataFrame`
    Sample metadata table. Must be indexed by sample IDs (strings         after construction).
- **ann**: `pd.DataFrame or None`
    Optional probe/CpG annotation table (e.g., Illumina manifest).         Must be indexed by CpG IDs.
    If ``None``, probe-level alignment checks are skipped.
- **meta**: `dict, optional`
    Free-form dictionary storing pipeline provenance and parameters.
    Pre-populated with sensible defaults if not provided.

#### Attributes
- **M**: `pd.DataFrame`
    Methylation matrix (string-indexed rows and columns).
- **pheno**: `pd.DataFrame`
    Sample metadata (string-indexed).
- **ann**: `pd.DataFrame or None`
    Probe annotation (string-indexed if present).
- **meta**: `dict`
    Processing metadata (e.g., ``{"matrix_type": "beta",         "normalized": True, ...}``).

#### Notes
The ``__post_init__`` method automatically:

- Converts all relevant indices to strings.
- Validates alignment between components.
- Raises informative ``KeyError`` on mismatch.

### Readers

Input utilities for loading methylation datasets in the DMeth pipeline.

This module provides a **Python API** for loading
preprocessed or raw methylation data into the standard :class:`ProcessedData` container used throughout the package.

#### Features
- Robust multi-format reading (CSV, TSV, Excel, pickle, HDF5, Parquet, Feather)
- Automatic coercion of methylation matrix to numeric values
- Smart HDF5 key selection (prefers `/M` or `M`)
- Safe sample alignment between methylation matrix and phenotype data
- Two high-level entry points:
    - ``load_processed_data()`` – load a previously saved ``ProcessedData`` object
    - ``load_methylation_data()`` – construct a new ``ProcessedData`` from raw files



---

### `ProcessedData`


`ProcessedData(M: DataFrame, pheno: DataFrame, ann: DataFrame, meta: Dict[str, Any] = <factory>) -> None`


Central container for aligned methylation data used throughout the DMeth pipeline.

Guarantees that:

- All components (methylation matrix, sample metadata, probe annotation)     share consistent string indices.
- Sample and probe alignment is validated on construction.
- A ``meta`` dictionary tracks processing history (normalisation,     batch correction, QC metrics, etc.).

All downstream functions expect or return instances of this class.

#### Parameters
- **M**: `pd.DataFrame`
    Methylation matrix with CpG sites as rows and samples as columns.
    Values are typically beta (0–1) or M-values (-inf to +inf).
- **pheno**: `pd.DataFrame`
    Sample metadata table. Must be indexed by sample IDs (strings         after construction).
- **ann**: `pd.DataFrame or None`
    Optional probe/CpG annotation table (e.g., Illumina manifest).         Must be indexed by CpG IDs.
    If ``None``, probe-level alignment checks are skipped.
- **meta**: `dict, optional`
    Free-form dictionary storing pipeline provenance and parameters.
    Pre-populated with sensible defaults if not provided.

#### Attributes
- **M**: `pd.DataFrame`
    Methylation matrix (string-indexed rows and columns).
- **pheno**: `pd.DataFrame`
    Sample metadata (string-indexed).
- **ann**: `pd.DataFrame or None`
    Probe annotation (string-indexed if present).
- **meta**: `dict`
    Processing metadata (e.g., ``{"matrix_type": "beta",         "normalized": True, ...}``).

#### Notes
The ``__post_init__`` method automatically:

- Converts all relevant indices to strings.
- Validates alignment between components.
- Raises informative ``KeyError`` on mismatch.


---

### `load_methylation_data`


`load_methylation_data(methylation_input: Union[str, pathlib.Path, DataFrame], pheno_input: Union[str, pathlib.Path, DataFrame, NoneType] = None, ann_input: Union[str, pathlib.Path, DataFrame, NoneType] = None, index_col_probe: Optional[int] = 0, index_col_sample: Optional[int] = 0) -> ProcessedData`


High-level constructor that builds a ProcessedData object from     raw matrix/metadata files.

- Accepts file paths, already-loaded DataFrames, or a mix of both.
- Performs automatic type coercion, sample alignment, and validation.

#### Parameters
- **methylation_input**: `str | Path | pd.DataFrame`
    CpG × sample methylation matrix (beta or M-values).
- **pheno_input**: `str | Path | pd.DataFrame, optional`
    Sample phenotype/metadata table. If ``None``, an empty placeholder indexed by
    the matrix columns is created.
- **ann_input**: `str | Path | pd.DataFrame, optional`
    CpG probe annotation (e.g., Illumina manifest). If ``None``,         annotation is omitted.
- **index_col_probe**: `int, default 0`
    Column containing CpG identifiers when reading the methylation matrix.
- **index_col_sample**: `int, default 0`
    Column containing sample identifiers when reading the phenotype table.

#### Returns
ProcessedData
    Fully validated container ready for downstream analysis.

#### Notes
- Non-numeric entries in the methylation matrix are forcibly coerced to NaN
  (with a warning indicating how many values were affected).
- The phenotype table is reindexed to exactly match the sample order in ``M``.
- All indices are converted to strings and alignment is checked by     ``ProcessedData.__post_init__``.


---

### `load_processed_data`


`load_processed_data(path: Union[str, pathlib.Path], trusted: bool = False) -> ProcessedData`


Load a previously saved ProcessedData object from disk.

Supports the two canonical DMeth output formats:
- ``.pkl`` / ``.pickle`` → direct pandas pickle of the ProcessedData instance
- ``.h5`` / ``.hdf5`` → HDF5 store containing the keys ``M``,     ``pheno``, and optionally ``ann``

#### Parameters
- **path**: `str or Path`
    Path to the saved ProcessedData file.
- **trusted**: `bool, default False`
    Only allow ``.pkl`` / ``.pickle`` files from trusted sources

#### Returns
ProcessedData
    Fully reconstructed and validated ProcessedData container.

#### Raises
ValueError
    If pickle file is from an untrusted source
ValueError
    If the file extension is not one of the supported formats.

### Writers

Output utilities for saving processed methylation data and analysis results.

This module provides a unified interface (Python API + Typer CLI) for exporting:

- ProcessedData objects in multiple formats
- Differential methylation results (filtered and clean)
- IDAT-style HDF5 archives (compatible with downstream tools)
- Self-contained HTML analysis reports with embedded plots

All export functions include overwrite protection, automatic directory creation,
and informative logging.

#### Features
- Smart format detection and suffix handling
- Optional compression (blosc:zstd or gzip)
- Safe export of differential results (core columns + group means)
- IDAT-compatible HDF5 layout via h5py (fallback to pandas HDFStore)
- One-call HTML report generation from summary dict + matplotlib figures



---

### `ProcessedData`


`ProcessedData(M: DataFrame, pheno: DataFrame, ann: DataFrame, meta: Dict[str, Any] = <factory>) -> None`


Central container for aligned methylation data used throughout the DMeth pipeline.

Guarantees that:

- All components (methylation matrix, sample metadata, probe annotation)     share consistent string indices.
- Sample and probe alignment is validated on construction.
- A ``meta`` dictionary tracks processing history (normalisation,     batch correction, QC metrics, etc.).

All downstream functions expect or return instances of this class.

#### Parameters
- **M**: `pd.DataFrame`
    Methylation matrix with CpG sites as rows and samples as columns.
    Values are typically beta (0–1) or M-values (-inf to +inf).
- **pheno**: `pd.DataFrame`
    Sample metadata table. Must be indexed by sample IDs (strings         after construction).
- **ann**: `pd.DataFrame or None`
    Optional probe/CpG annotation table (e.g., Illumina manifest).         Must be indexed by CpG IDs.
    If ``None``, probe-level alignment checks are skipped.
- **meta**: `dict, optional`
    Free-form dictionary storing pipeline provenance and parameters.
    Pre-populated with sensible defaults if not provided.

#### Attributes
- **M**: `pd.DataFrame`
    Methylation matrix (string-indexed rows and columns).
- **pheno**: `pd.DataFrame`
    Sample metadata (string-indexed).
- **ann**: `pd.DataFrame or None`
    Probe annotation (string-indexed if present).
- **meta**: `dict`
    Processing metadata (e.g., ``{"matrix_type": "beta",         "normalized": True, ...}``).

#### Notes
The ``__post_init__`` method automatically:

- Converts all relevant indices to strings.
- Validates alignment between components.
- Raises informative ``KeyError`` on mismatch.


---

### `export_idat_hdf5`


`export_idat_hdf5(beta: DataFrame, mvals: Optional[DataFrame], sample_sheet: Optional[DataFrame], filepath: Union[str, pathlib.Path], compress: bool = True) -> pathlib.Path`


Write methylation data to an IDAT-compatible HDF5 archive using h5py     (preferred) or pandas fallback.

Layout mirrors common downstream tools: groups ``/beta``, ``/mvals`` (optional),
and ``/samples`` containing one JSON dataset per sample.

#### Parameters
- **beta**: `pd.DataFrame`
    Beta-value matrix (CpGs × samples).
- **mvals**: `pd.DataFrame or None, optional`
    M-value matrix (same dimensions).
- **sample_sheet**: `pd.DataFrame or None, optional`
    Sample metadata (index = sample IDs).
- **filepath**: `str or Path`
    Destination HDF5 file.
- **compress**: `bool, default True`
    Apply gzip compression to large datasets.

#### Returns
Path
    Path to the created HDF5 file.

#### Raises
RuntimeError
    If neither h5py nor pandas.HDFStore is available.


---

### `export_results`


`export_results(res: DataFrame, output_path: str, format: str = 'csv', include_all: bool = False, verbose: bool = True, overwrite: bool = True) -> None`


Export differential methylation results to CSV, TSV, or Excel with     optional column filtering.

- By default only core statistical columns and per-group mean columns are exported.
- Overwrite protection is applied if ``overwrite=False``.

#### Parameters
- **res**: `pd.DataFrame`
    Differential results table (as returned by the analysis pipeline).
- **output_path**: `str`
    Destination file path (extension determines format if ``format`` is omitted).
- **format**: `{"csv", "tsv", "excel"}, default "csv"`
    Output file format.
- **include_all**: `bool, default False`
    If True, export every column; if False, only core + mean columns are kept.
- **verbose**: `bool, default True`
    Log a confirmation message on success.
- **overwrite**: `bool, default True`
    Raise FileExistsError if the file exists and this is False.

#### Raises
FileExistsError
    When ``overwrite=False`` and the target file already exists.
ValueError
    For unsupported ``format``.
RuntimeError
    If writing fails (e.g., missing openpyxl for Excel).


---

### `save_processed_data`


`save_processed_data(data: 'ProcessedData', path: Union[str, pathlib.Path], format: str = 'hdf5', compression: str | None = 'blosc:zstd', complib_level: int = 9, overwrite: bool = True) -> None`


Persist a ProcessedData object to disk in one of several convenient formats.

Supported formats
-----------------
csv / tsv      → three separate files (M, pheno, ann)
xlsx           → single workbook with sheets "M", "pheno", "ann"
pickle / pkl   → direct pickle of the ProcessedData instance
hdf5 / h5      → pandas HDFStore with compressed tables

#### Parameters
- **data**: `ProcessedData`
    The container to save.
- **path**: `str or Path`
    Output path; suffix is automatically corrected to match ``format``.
- **format**: `{"csv", "tsv", "xlsx", "pickle", "pkl", "hdf5", "h5"}, default "hdf5"`
    Desired export format.
- **compression**: `str or None, default "blosc:zstd"`
    Compression library for HDF5 (only used when format is hdf5).
- **complib_level**: `int, default 9`
    Compression level (1–9) for HDF5.
- **overwrite**: `bool, default True`
    Allow overwriting existing files.

#### Raises
ValueError
    If an unsupported format is requested.
FileExistsError
    When ``overwrite=False`` and the target exists.


---

## Preprocessing Utilities

Core preprocessing utilities for DNA methylation array data.

- This module implements quality control, normalization, batch correction,
and filtering operations for methylation matrices (beta or M-values).
- It provides both standard and high-performance implementations with
memory-efficient processing for large datasets.

#### Features
- Sample-level QC: Flag/remove samples with excessive missing data
- CpG-level QC: Remove probes with high missingness or on sex chromosomes
- Normalization: Beta quantile normalization with optional M-value conversion
- Batch correction: Regression-based or ComBat-style correction
- Filtering: Remove low-variance probes



---

### `ProcessedData`


`ProcessedData(M: DataFrame, pheno: DataFrame, ann: DataFrame, meta: Dict[str, Any] = <factory>) -> None`


Central container for aligned methylation data used throughout the DMeth pipeline.

Guarantees that:

- All components (methylation matrix, sample metadata, probe annotation)     share consistent string indices.
- Sample and probe alignment is validated on construction.
- A ``meta`` dictionary tracks processing history (normalisation,     batch correction, QC metrics, etc.).

All downstream functions expect or return instances of this class.

#### Parameters
- **M**: `pd.DataFrame`
    Methylation matrix with CpG sites as rows and samples as columns.
    Values are typically beta (0–1) or M-values (-inf to +inf).
- **pheno**: `pd.DataFrame`
    Sample metadata table. Must be indexed by sample IDs (strings         after construction).
- **ann**: `pd.DataFrame or None`
    Optional probe/CpG annotation table (e.g., Illumina manifest).         Must be indexed by CpG IDs.
    If ``None``, probe-level alignment checks are skipped.
- **meta**: `dict, optional`
    Free-form dictionary storing pipeline provenance and parameters.
    Pre-populated with sensible defaults if not provided.

#### Attributes
- **M**: `pd.DataFrame`
    Methylation matrix (string-indexed rows and columns).
- **pheno**: `pd.DataFrame`
    Sample metadata (string-indexed).
- **ann**: `pd.DataFrame or None`
    Probe annotation (string-indexed if present).
- **meta**: `dict`
    Processing metadata (e.g., ``{"matrix_type": "beta",         "normalized": True, ...}``).

#### Notes
The ``__post_init__`` method automatically:

- Converts all relevant indices to strings.
- Validates alignment between components.
- Raises informative ``KeyError`` on mismatch.


---

### `batch_correction`


`batch_correction(data: ProcessedData, batch_col: str, covariates: Optional[List[str]] = None, method: str = 'qr', weights: Optional[ndarray] = None, block_size: int = 50000, robust: bool = False, return_diagnostics: bool = False) -> Union[ProcessedData, Tuple[ProcessedData, Dict[str, Any]]]`


Regression-based batch correction using full linear model (intercept +     covariates + batch).

Numerically stable QR/pinv solver with optional per-probe weights and     block-wise processing.

#### Parameters
- **batch_col**: `str`
    Column in ``pheno`` identifying batch.
- **covariates**: `list[str] or None`
    Additional technical or biological covariates to preserve.
- **method**: `{"qr", "pinv"}, default "qr"`
    Solver backend.
- **weights**: `ndarray or None`
    Per-probe inverse-variance weights.
- **robust**: `bool, default False`
    Use Huber robust regression (much slower).
- **return_diagnostics**: `bool, default False`
    Return detailed effect sizes, residuals, and variance explained.

#### Returns
ProcessedData or (ProcessedData, dict)
    Corrected data and optional diagnostics.


---

### `batch_correction_combat`


`batch_correction_combat(data: ProcessedData, batch_col: str, covariates: Optional[List[str]] = None, parametric: bool = True, return_diagnostics: bool = False) -> Union[ProcessedData, Tuple[ProcessedData, Dict[str, Any]]]`


Wrapper for ComBat (or safe fallback) batch correction.

- Uses ``pycombat`` if installed (parametric or nonparametric).
- Falls back to conservative mean-centering + variance rescaling with clear warning.
- Automatically handles beta ↔ M-value conversion.

#### Parameters
- **parametric**: `bool, default True`
    Use parametric ComBat (faster, assumes normality).
- **return_diagnostics**: `bool, default False`
    Return metadata about which method was actually used.

#### Returns
ProcessedData or (ProcessedData, dict)
    Batch-corrected data and diagnostics.


---

### `filter_low_variance_cpgs`


`filter_low_variance_cpgs(data: ProcessedData, min_percentile: float = 10.0, inplace: bool = True) -> ProcessedData`


Remove CpGs below a variance percentile threshold (e.g., bottom 10%).

#### Parameters
- data : ProcessedData
- min_percentile : float, default 10.0
    Keep only probes with variance ≥ this percentile.
- inplace : bool, default True
    Modify input object or return a copy.

#### Returns
- ProcessedData
    Dataset with low-variance probes removed.

#### Notes
- Number of removed probes recorded in ``data.meta["qc"]["low_variance_removed"]``.


---

### `normalize_methylation`


`normalize_methylation(data: ProcessedData, method: str = 'beta_quantile', convert_to: Optional[str] = None, copy: bool = False, q_chunk_threshold: int = 100000000.0, q_block_probes: int = 50000) -> ProcessedData`


Perform quantile normalization across samples with optional beta ↔     M-value conversion.

- Implements memory-efficient beta quantile normalization using column-wise     sorting and block-wise rank mapping.
- Automatically falls back to disk-backed memmap for very large datasets.

#### Parameters
- **data**: `ProcessedData`
    Input methylation data.
- **method**: `{"beta_quantile", "none"}, default "beta_quantile"`
    Normalization method.
- **convert_to**: `{"beta", "m"} or None, optional`
    Convert matrix type after normalization (e.g., "m" for M-values).
- **copy**: `bool, default False`
    Work on a deep copy of the input.
- **q_chunk_threshold**: `int, default 1e8`
    Element count threshold (n_probes × n_samples) above which memmap is used.
- **q_block_probes**: `int, default 50_000`
    Number of probes processed per block during rank-to-target assignment.

#### Returns
ProcessedData
    Normalized dataset with updated ``meta["normalized"]`` and detailed provenance.

#### Notes
Preserves original NaNs. Records full normalization metadata including     memory strategy.


---

### `normalize_methylation_highperf`


`normalize_methylation_highperf(data: ProcessedData, method: str = 'beta_quantile', convert_to: Optional[str] = None, copy: bool = False, memmap_threshold: int = 200000000.0, memmap_dir: Optional[str] = None, n_workers: Optional[int] = None, sample_block: int = 16, random_state: Optional[int] = None) -> ProcessedData`


High-performance quantile normalization with multiprocessing and automatic     memmap handling.

Designed for >850k × 500+ datasets. Uses disk-backed arrays when needed     and parallelizes the rank-mapping stage.

#### Parameters
- **memmap_threshold**: `int, default 2e8`
    Element count above which temporary memmap files are created.
- **memmap_dir**: `str or None`
    Directory for memmap files (defaults to system temp).
- **n_workers**: `int or None`
    Number of processes for parallel assignment (defaults to CPU count – 1).
- **sample_block**: `int, default 16`
    Number of samples per parallel job.
- **random_state**: `int or None`
    Currently unused (kept for API consistency).

#### Returns
ProcessedData
    Quantile-normalized data with comprehensive metadata.

#### Notes
Significantly faster and lower peak RAM than ``normalize_methylation``     on large cohorts.


---

### `qc_cpg_level`


`qc_cpg_level(data: ProcessedData, max_missing_fraction: float = 0.1, drop_sex_chr: bool = True, chr_col: str = 'chromosome') -> ProcessedData`


Remove CpGs with high missingness across samples and optionally drop     sex-chromosome probes.

#### Parameters
- **data**: `ProcessedData`
    Input dataset.
- **max_missing_fraction**: `float, default 0.10`
    Maximum fraction of samples allowed to be missing for a CpG (10% default).
- **drop_sex_chr**: `bool, default True`
    If True and annotation is available, remove all probes on chromosomes X and Y.
- **chr_col**: `str, default "chromosome"`
    Column name in ``data.ann`` containing chromosome information.

#### Returns
ProcessedData
    Updated container with low-quality and/or sex-chromosome CpGs removed.

#### Notes
Summary statistics are recorded in ``data.meta["qc"]["cpg_missing"]``.


---

### `qc_sample_level`


`qc_sample_level(data: ProcessedData, max_missing_fraction: float = 0.05, min_nonmissing_probes: Optional[int] = None, remove_samples: bool = True) -> ProcessedData`


Identify and optionally remove samples with excessive missing methylation data.

#### Parameters
- **data**: `ProcessedData`
    Input methylation dataset.
- **max_missing_fraction**: `float, default 0.05`
    Maximum allowed fraction of missing CpGs per sample (5% by default).
- **min_nonmissing_probes**: `int or None, optional`
    Alternative absolute threshold: minimum number of detected (non-missing)         probes required.
- **remove_samples**: `bool, default True`
    If True, flagged samples are removed from both ``M`` and ``pheno``;         if False, only metadata is recorded.

#### Returns
ProcessedData
    Updated container (samples removed if requested).

#### Notes
Flagged sample list and thresholds are stored in ``data.meta["qc"]    ["sample_missing"]``.


---

## Core Differential Analysis

### Core Analysis

Statistical engines for differential methylation analysis.

This module implements empirical Bayes moderated statistics following
Smyth (2004) limma methodology, with extensions for paired designs,
robust estimation, and Numba-accelerated computation. It provides both
full-matrix and chunked processing for memory-constrained environments.

#### Features
- Empirical Bayes variance shrinkage (Smyth method)
- Robust variance estimation via winsorization
- Numba JIT compilation for 10-100 x speedup
- Memory-efficient chunked processing
- Paired and multi-group designs
- F-tests for multi-coefficient contrasts
- Automatic handling of missing data
- Group mean computation for interpretability



---

### `SmythPrior`


`SmythPrior(df_prior: 'float', var_prior: 'float') -> None`


SmythPrior(df_prior: 'float', var_prior: 'float')


---

### `check_analysis_memory`


`check_analysis_memory(M: 'pd.DataFrame', warn_threshold_gb: 'float' = 8.0)`


Conservatively estimate RAM requirements for a full-matrix differential analysis.

- Calculates current data footprint, projects peak usage (approximately 4 times     input size), and compares against available system memory.
- Raises MemoryError early if danger is high, forcing use of chunked mode.

#### Parameters
- **M**: `pd.DataFrame`
    Methylation matrix (CpGs × samples).
- **warn_threshold_gb**: `float, default 8.0`
    Issue a warning if estimated peak exceeds this value.

#### Returns
dict
    Keys: ``data_gb``, ``peak_gb``, ``available_gb``.

#### Raises
MemoryError
    If projected peak consumption exceeds ~80% of available RAM. In such cases         ``fit_differential_chunked`` should be used instead.


---

### `fit_differential`


`fit_differential(M: 'pd.DataFrame', design: 'pd.DataFrame', contrast: 'Optional[np.ndarray]' = None, contrast_matrix: 'Optional[np.ndarray]' = None, shrink: 'Union[str, float]' = 'auto', robust: 'bool' = True, eps: 'float' = 1e-08, return_residuals: 'bool' = False, min_count: 'int' = 3, max_d0: 'float' = 50.0, winsor_lower: 'float' = 0.05, winsor_upper: 'float' = 0.95, group_labels: 'Optional[pd.Series]' = None, use_numba: 'bool' = True) -> 'Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]'`


Core limma-style differential methylation analysis with empirical Bayes moderation.

- **Features**: `- Numba-accelerated feature-wise fitting (10–100× speedup when no missing data)`
- Automatic handling of missing values
- Robust winsorization of residuals
- Multiple shrinkage strategies (Smyth, median, fixed d₀, none)
- Single-contrast t-tests or multi-coefficient F-tests
- Optional group mean beta/M-value columns for interpretability
- Optional residuals return for diagnostics

#### Parameters
- **M**: `pd.DataFrame`
    CpG × sample matrix of M-values (or beta values if already on log-odds scale).
- **design**: `pd.DataFrame`
    Sample × covariate design matrix (output from ``patsy`` or manual construction).
contrast / contrast_matrix : np.ndarray or None
    Contrast vector (t-test) or matrix (F-test).
- **shrink**: `{"auto", "smyth", "median", "none"} or float`
    Variance shrinkage method or fixed prior df.
- **robust**: `bool, default True`
    Winsorize residuals per feature.
- **group_labels**: `pd.Series, optional`
    Explicit sample grouping for mean columns; otherwise heuristically detected.
- **use_numba**: `bool, default True`
    Enable JIT-compiled loop when possible.

#### Returns
pd.DataFrame
    Results table with columns including ``logFC``, ``t``, ``pval``, ``padj``         (or ``F``, ``df1``, ``df2``), moderated variances (``s2_post``), prior df         (``d0``), and group means.
(pd.DataFrame, pd.DataFrame), optional
    Second element contains per-feature residuals if ``return_residuals=True``.


---

### `fit_differential_chunked`


`fit_differential_chunked(M: 'pd.DataFrame', design: 'pd.DataFrame', chunk_size: 'int' = 10000, verbose: 'bool' = True, group_labels: 'Optional[pd.Series]' = None, **kwargs) -> 'pd.DataFrame'`


Memory-efficient version of ``fit_differential`` that processes features in chunks.

Useful for >1M probe arrays on limited RAM. P-values are combined     globally and FDR-corrected once at the end.

#### Parameters
- **chunk_size**: `int, default 10000`
    Number of CpGs per processing chunk.
**kwargs
    All additional arguments forwarded to ``fit_differential``.

#### Returns
pd.DataFrame
    Complete results table with globally adjusted p-values and group means.

### Postprocessing

Postprocessing and interpretation utilities for differential methylation analysis results.

This module provides robust, publication-ready tools for summarizing, filtering, and extracting biologically meaningful differentially methylated CpGs from statistical output. It supports flexible significance criteria, effect-size thresholding, directional filtering, and delta-beta constraints, with comprehensive summary statistics for reporting.

#### Features
- Comprehensive summary statistics including significance counts, directionality, effect sizes, and empirical Bayes shrinkage metrics
- Flexible extraction of significant CpGs with adjustable logFC, adjusted p-value, and delta-beta thresholds
- Directional filtering for hyper- and hypomethylated sites
- Automatic detection or explicit specification of group mean beta-value columns for delta-beta filtering
- Graceful handling of missing columns and empty result sets with informative warnings
- Optional detailed summary dictionaries for downstream reporting or visualization



---

### `get_significant_cpgs`


`get_significant_cpgs(res: DataFrame, lfc_col: str = 'logFC', pval_col: str = 'padj', lfc_thresh: float = 0.0, pval_thresh: float = 0.05, delta_beta_thresh: Optional[float] = None, direction: Optional[str] = None, delta_beta_cols: Optional[List[str]] = None, return_summary: bool = False, verbose: bool = True) -> Union[List[str], Dict]`


Extract biologically meaningful significant CpGs using flexible,     multi-layer filtering.

- Combines adjusted p-value, log fold-change, directionality, and optional     absolute delta-beta criteria.
- Ideal for generating final candidate lists for downstream validation or     pathway analysis.

#### Parameters
- **res**: `pd.DataFrame`
    Full differential methylation results table.
- **lfc_col**: `str, default "logFC"`
    Column containing log₂ fold change.
- **pval_col**: `str, default "padj"`
    Column containing adjusted p-values.
- **lfc_thresh**: `float, default 0.0`
    Minimum absolute |logFC| required (0 means no LFC filtering).
- **pval_thresh**: `float, default 0.05`
    Maximum adjusted p-value for significance.
- **delta_beta_thresh**: `float or None, optional`
    Minimum absolute difference in mean beta values between groups.
- **direction**: `{"hyper", "hypo", None}, optional`
    Restrict to hypermethylated (logFC > 0), hypomethylated (logFC < 0), or both.
- **delta_beta_cols**: `list[str] or None, optional`
    Exactly two column names containing group mean beta values (e.g.         ``["meanB_case", "meanB_control"]``).
    If ``None`` and delta-beta filtering is requested, automatically detects         columns starting with ``meanB_``.
- **return_summary**: `bool, default False`
    If True, return a detailed dictionary instead of just the CpG list.
- **verbose**: `bool, default True`
    Warn when no significant CpGs are found.

#### Returns
list[str] or dict
    List of significant CpG IDs (default)
    Or a summary dictionary containing counts, directional breakdown,         mean |logFC|, and the CpG list

#### Raises
KeyError
    If required columns (``lfc_col`` or ``pval_col``) are missing.
ValueError
    If ``delta_beta_thresh`` is used but exactly two mean-beta columns         cannot be identified.


---

### `summarize_differential_results`


`summarize_differential_results(res: DataFrame, pval_thresh: float = 0.05, lfc_thresh: float = 0.0, verbose: bool = True) -> Dict[str, Union[int, float]]`


Produce a comprehensive, publication-ready summary of differential     methylation results.

Handles missing columns gracefully and returns key metrics for reporting:
significance counts, directionality, effect-size statistics, and     variance-shrinkage diagnostics.

#### Parameters
- **res**: `pd.DataFrame`
    Differential methylation results containing at minimum ``logFC`` and ``padj``.
    May also include ``pval``, ``s2``, ``s2_post``, and ``d0``         (prior degrees of freedom).
- **pval_thresh**: `float, default 0.05`
    Adjusted p-value threshold defining statistical significance.
- **lfc_thresh**: `float, default 0.0`
    Minimum absolute |logFC| required (0 means no LFC filtering).
- **verbose**: `bool, default True`
    Emit warnings for empty input or missing optional columns.

#### Returns
dict
    Summary dictionary with the following keys:
    ``total_tested``: total CpGs tested
    ``significant``: number of significant CpGs (padj < threshold)
    ``pct_significant``: percentage of significant CpGs
    ``hypermethylated`` / ``hypomethylated``: directional counts
    ``mean_abs_logFC_sig`` / ``median_abs_logFC_sig``:         effect size summaries among significant sites
    ``max_abs_logFC``: largest absolute log fold change
    ``min_pval``: smallest raw p-value
    ``shrinkage_factor``: median ratio of moderated to original variance
    ``d0``: median prior degrees of freedom (empirical Bayes)

#### Notes
Missing optional columns are safely ignored with fallback values.
Empty input returns a zero-filled summary.

### Preparation

Data preparation and preprocessing utilities for differential methylation analysis.

This module provides robust, production-ready tools for cleaning and imputing
CpG-level methylation matrices prior to statistical modeling. It implements
stringent quality-control filters based on missingness patterns and group
representation, alongside flexible imputation strategies optimized for
high-dimensional epigenomic data.

#### Features
- Global and per-group missingness filtering with configurable thresholds
- Minimum representation enforcement across experimental groups
- Multiple imputation methods: row-wise mean/median and K-nearest neighbors
- Sample-wise KNN option for dramatic speed gains when samples ≪ CpGs
- Comprehensive input validation and clear diagnostic reporting
- Memory-efficient operations using NumPy-based vectorization



---

### `filter_cpgs_by_missingness`


`filter_cpgs_by_missingness(M: 'pd.DataFrame', max_missing_rate: 'float' = 0.2, min_samples_per_group: 'Optional[int]' = None, groups: 'Optional[pd.Series]' = None) -> 'Tuple[pd.DataFrame, int, int]'`


Remove CpGs exceeding a global missingness threshold and/or failing     per-group representation.

#### Parameters
- **M**: `pd.DataFrame`
    Methylation matrix (CpGs × samples) containing possible NaN values.
- **max_missing_rate**: `float, default 0.2`
    Maximum allowable fraction of missing values across all samples (0–1).
- **min_samples_per_group**: `int or None, optional`
    Minimum number of non-missing observations required in every experimental group.
    Ignored if ``None``.
- **groups**: `pd.Series or None, optional`
    Sample group labels indexed by ``M.columns``. Required when         ``min_samples_per_group`` is set.

#### Returns
- **filtered**: `pd.DataFrame`
    Subset of CpGs passing both filters.
- **n_removed**: `int`
    Number of CpGs removed.
- **n_kept**: `int`
    Number of CpGs retained.

#### Raises
ValueError
    If ``min_samples_per_group`` is supplied without ``groups`` or if group         labels are misaligned.


---

### `filter_min_per_group`


`filter_min_per_group(M: 'pd.DataFrame', groups: 'pd.Series', min_per_group: 'int' = 5, verbose: 'bool' = True) -> 'pd.DataFrame'`


Retain only CpGs with at least ``min_per_group`` non-missing     values in one experimental group.

Useful for ensuring sufficient representation before differential analysis.

#### Parameters
- **M**: `pd.DataFrame`
    Methylation matrix (CpGs × samples) possibly containing NaN.
- **groups**: `pd.Series`
    Group membership for each sample (index must match ``M.columns``).
- **min_per_group**: `int, default 5`
    Minimum number of observed (non-missing) values required per group.
- **verbose**: `bool, default True`
    Log a concise summary of filtering results.

#### Returns
pd.DataFrame
    Filtered matrix containing only qualifying CpGs.

#### Raises
ValueError
    If group labels do not cover all samples in ``M``.


---

### `impute_missing_values`


`impute_missing_values(M: 'pd.DataFrame', method: 'str' = 'mean', k: 'int' = 5, use_sample_knn: 'bool' = False) -> 'pd.DataFrame'`


Impute missing methylation values using row-wise statistics or K-nearest neighbours.

#### Parameters
- **M**: `pd.DataFrame`
    CpG × sample matrix with possible NaN entries.
- **method**: `{"mean", "median", "knn"}, default "mean"`
    Imputation strategy.
- **k**: `int, default 5`
    Number of neighbours used for KNN imputation.
- **use_sample_knn**: `bool, default False`
    If True, KNN is performed across samples (samples ≪ CpGs) rather         than across CpGs.
    Dramatically faster for typical array datasets.

#### Returns
pd.DataFrame
    Imputed matrix with the same index/columns as input.

#### Notes
- ``mean`` and ``median`` are applied row-wise (per CpG).
- ``knn`` uses scikit-learn's ``KNNImputer`` with distance weighting.
- No imputation is performed if the matrix contains no missing values.

### Validation

Input validation and integrity checks for differential methylation analysis.

- This module implements rigorous pre-analysis validation of data matrices, experimental designs, contrasts, sample alignment, and system resources. - It ensures statistical estimability, correct pairing structure, and sufficient memory before launching computationally intensive analyses, preventing silent failures and ambiguous results.

#### Features
- Accurate memory footprint estimation with conservative peak usage prediction and automatic MemoryError on insufficient RAM
- Strict validation of two-group design vectors with automatic construction of intercept + indicator design matrix
- Comprehensive contrast validation including shape, zero-vector detection, and QR-based estimability checking
- Flexible string contrast syntax ("treatment-control") for simple two-column designs
- Thorough alignment verification between methylation data columns, design matrix rows, and optional sample identifiers
- Robust paired-sample validation ensuring exactly two observations per subject and balanced group representation within pairs
- Clear, actionable error messages and logging for rapid debugging in production pipelines



---

### `build_design`


`build_design(data: 'pd.DataFrame', categorical: 'list[str]' = None, add_intercept: 'bool' = True, drop_first: 'bool' = True) -> 'pd.DataFrame'`


Build a design matrix for linear modeling.

#### Parameters
- **data**: `pd.DataFrame`
    Columns are variables used in the design (condition,         patient, batch, age, sex, ...)
- **categorical**: `list of str, optional`
    Which columns should be treated as categorical. If None, infer automatically.
- **add_intercept**: `bool, default=True`
    Whether to add an intercept column of 1s.
- **drop_first**: `bool, default=True`
    Drop the first dummy level to avoid collinearity.

#### Returns
pd.DataFrame
    Numeric design matrix.


---

### `check_analysis_memory`


`check_analysis_memory(M: 'pd.DataFrame', warn_threshold_gb: 'float' = 8.0)`


Conservatively estimate RAM requirements for a full-matrix differential analysis.

- Calculates current data footprint, projects peak usage (approximately 4 times     input size), and compares against available system memory.
- Raises MemoryError early if danger is high, forcing use of chunked mode.

#### Parameters
- **M**: `pd.DataFrame`
    Methylation matrix (CpGs × samples).
- **warn_threshold_gb**: `float, default 8.0`
    Issue a warning if estimated peak exceeds this value.

#### Returns
dict
    Keys: ``data_gb``, ``peak_gb``, ``available_gb``.

#### Raises
MemoryError
    If projected peak consumption exceeds ~80% of available RAM. In such cases         ``fit_differential_chunked`` should be used instead.


---

### `validate_alignment`


`validate_alignment(data: 'pd.DataFrame', design_matrix: 'Union[np.ndarray, pd.DataFrame]', sample_names: 'Optional[Sequence]' = None, paired_ids: 'Optional[Sequence]' = None, group_col_idx: 'int' = 1) -> 'Tuple[pd.Index, Union[pd.DataFrame, np.ndarray], Optional[np.ndarray]]'`


Ensure perfect alignment between methylation data, design matrix, and optional     pairing information.

Verifies sample counts, column ordering, duplicate names, and (if supplied)     paired-subject structure.

#### Parameters
- **data**: `pd.DataFrame`
    Methylation matrix with samples in columns.
- **design_matrix**: `np.ndarray or pd.DataFrame`
    Corresponding design matrix.
- **sample_names**: `Sequence or None`
    Explicit ordered list of sample identifiers (required only         if column names differ).
- **paired_ids**: `Sequence or None`
    Subject/block identifiers for paired designs; each ID must appear exactly twice.
- **group_col_idx**: `int, default 1`
    Column in design_matrix encoding group membership (used to verify         balanced pairing).

#### Returns
tuple
    (ordered_data_columns, validated_design_matrix, validated_paired_array_or_None)

#### Raises
ValueError
    On any misalignment, duplicate column names, or invalid pairing structure         (e.g., a subject with both samples in the same group).


---

### `validate_contrast`


`validate_contrast(design_matrix: 'Union[np.ndarray, pd.DataFrame]', contrast: 'Union[np.ndarray, Sequence[float]]') -> 'np.ndarray'`


Validate and normalize a numeric contrast vector against a design matrix.

#### Parameters
- **design_matrix**: `np.ndarray or pd.DataFrame`
    Full design matrix (n_samples × n_coefficients).
- **contrast**: `np.ndarray or Sequence[float]`
    Contrast vector (length must equal n_coefficients).

#### Returns
np.ndarray
    Validated contrast vector of shape (n_coefficients,) and dtype float64.

#### Raises
TypeError
    If inputs are not numeric or not the correct type.
ValueError
    If contrast is zero, wrong length, or non-estimable.


---

## Downstream Analysis

### Annotation

Annotation and functional interpretation utilities for DNA methylation analysis.

This module delivers a complete suite of tools for translating CpG- and region-level differential methylation results into biological context. It enables rapid gene annotation, pathway- and gene-set enrichment analysis, computation of pathway-level methylation activity, correlation with gene expression, and reliable coordinate liftover across genome assemblies — all implemented with performance and robustness suitable for genome-wide studies.

#### Features
- Ultra-fast nearest-gene annotation using IntervalTree with graceful fallback to distance-based mapping
- Fisher’s exact test gene-set enrichment with automatic background correction, size filtering, and FDR adjustment
- Flexible pathway-level methylation scoring (mean/median/sum) from user-provided gene-to-pathway mappings
- Sample-aware Pearson/Spearman correlation between CpG methylation and gene expression with robust overlap handling
- High-accuracy genomic liftover via pyliftover (hg19 ↔ hg38 and other builds) with per-region success tracking
- Comprehensive handling of missing dependencies, empty inputs, and chromosome naming conventions
- Preservation of original indices and seamless integration with DMeth result tables



---

### `adjust_pvalues`


`adjust_pvalues(pvals: Union[pandas.core.series.Series, ndarray, List[float]], method: str = 'fdr_bh') -> pandas.core.series.Series`


Apply multiple-testing correction to raw p-values using statsmodels.

Handles NaN values robustly (treated as non-significant) and preserves the     original index when input is a pandas Series.

#### Parameters
- **pvals**: `array-like`
    Raw p-values (0 ≤ p ≤ 1).
- **method**: `str, default "fdr_bh"`
    Correction method passed to ``statsmodels.stats.multitest.multipletests``.
    Supported: ``"bonferroni"``, ``"holm"``, ``"fdr_bh"``, ``"fdr_by"``,         ``"sidak"``, ``"none"``.

#### Returns
pd.Series
    Adjusted p-values with identical index/order as input.


---

### `correlate_methylation_expression`


`correlate_methylation_expression(beta: 'pd.DataFrame', expression: 'pd.DataFrame', gene_map: 'Optional[Dict[str, str]]' = None, method: 'str' = 'pearson', override_index_alignment: 'bool' = False) -> 'pd.DataFrame'`


Compute sample-wise correlation between CpG methylation and gene expression.

Supports both one-to-one (same index) and many-to-one (custom CpG→gene     mapping) scenarios.

#### Parameters
beta, expression : pd.DataFrame
    Methylation (CpGs × samples) and expression (genes × samples) matrices.
- **gene_map**: `dict or None`
    Explicit mapping from CpG ID → gene symbol (required for cis-analysis).
- **method**: `{"pearson", "spearman"}, default "pearson"`
    Correlation coefficient to compute.
- **override_index_alignment**: `bool, default False`
    Proceed even with <2 overlapping samples (useful for exploratory checks).

#### Returns
pd.DataFrame
    Correlation results with columns ``r`` and ``pval`` (indexed by CpG         and optionally gene).


---

### `gene_set_enrichment`


`gene_set_enrichment(gene_list: 'List[str]', background: 'Optional[List[str]]' = None, gene_sets: 'Optional[Dict[str, List[str]]]' = None, method: 'str' = 'fisher', pval_cutoff: 'float' = 0.05, min_set_size: 'int' = 5, max_set_size: 'int' = 500) -> 'pd.DataFrame'`


Perform over-representation analysis (Fisher’s exact test) on a list     of genes against predefined gene sets (e.g., GO, KEGG, Reactome).

Automatically filters gene sets by size and applies FDR correction.

#### Parameters
- **gene_list**: `list[str]`
    Genes of interest (e.g., nearest genes of significant DMS/DMRs).
- **background**: `list[str] or None`
    Background gene universe. Defaults to union of all genes in ``gene_sets``.
- **gene_sets**: `dict[str, list[str]]`
    Mapping from pathway/term name to member genes.
min_set_size, max_set_size : int
    Exclude overly small or large gene sets (default 5–500).
- **pval_cutoff**: `float, default 0.05`
    Return only terms with adjusted p-value ≤ this threshold.

#### Returns
pd.DataFrame
    Enriched terms with columns: ``term``, ``pvalue``, ``padj``,         ``oddsratio``, ``overlap``, ``set_size``.


---

### `liftover_coordinates`


`liftover_coordinates(regions: 'pd.DataFrame', from_build: 'str' = 'hg19', to_build: 'str' = 'hg38', chr_col: 'str' = 'chr', start_col: 'str' = 'start', end_col: 'str' = 'end') -> 'pd.DataFrame'`


Convert genomic regions between genome assemblies using pyliftover     (e.g., hg19 ↔ hg38).

Handles both single-base and interval coordinates with per-row success reporting.

#### Parameters
- **regions**: `pd.DataFrame`
    Table containing chromosome, start, and end columns.
from_build, to_build : str
    Source and target genome builds (default: hg19 → hg38).
chr_col, start_col, end_col : str
    Column names for genomic coordinates.

#### Returns
pd.DataFrame
    Original table augmented with:
    ``lifted_chr``, ``lifted_start``, ``lifted_end``
    ``lifted`` (boolean indicating successful conversion)

#### Raises
RuntimeError
    If ``pyliftover`` is not installed.


---

### `map_dms_to_genes`


`map_dms_to_genes(dms: 'pd.DataFrame', genes: 'pd.DataFrame', cpg_chr_col: 'str' = 'chr', cpg_pos_col: 'str' = 'pos', gene_chr_col: 'str' = 'chr', gene_start_col: 'str' = 'start', gene_end_col: 'str' = 'end', gene_name_col: 'str' = 'gene_symbol', promoter_upstream: 'int' = 1500, proximal_cutoff: 'int' = 200, promoter_downstream: 'int' = 500, distal_promoter_max: 'int' = 5000, max_distance: 'int' = 1000000) -> 'pd.DataFrame'`


Annotate CpGs with nearest gene and regulatory context.

#### Parameters
- **dms**: `pd.DataFrame`
    CpG positions with chromosome and position columns.
- **genes**: `pd.DataFrame`
    Gene annotations with chromosome, start, end, gene_symbol, and optional strand.
cpg_chr_col, cpg_pos_col : str
    Column names for CpG chromosome and position.
gene_chr_col, gene_start_col, gene_end_col, gene_name_col : str
    Column names for gene annotation.
- **promoter_upstream**: `int`
    Upstream region from TSS considered promoter.
- **proximal_cutoff**: `int`
    Distance near TSS classified as proximal promoter.
- **promoter_downstream**: `int`
    Downstream region from TSS considered promoter.
- **distal_promoter_max**: `int`
    Maximum distance upstream of TSS classified as distal promoter.
- **max_distance**: `int`
    Maximum distance to assign nearest gene if no overlap.

#### Returns
pd.DataFrame
    Original `dms` with added columns:
    - nearest_gene : str or NaN
    - relation : {"proximal_promoter", "distal_promoter", "promoter_downstream",
                  "distal_upstream", "gene_body", "intergenic"}
    - distance_bp : int or NaN


---

### `pathway_methylation_scores`


`pathway_methylation_scores(beta: 'pd.DataFrame', annotation: 'pd.DataFrame', pathway_db: 'Dict[str, List[str]]', method: 'str' = 'mean') -> 'pd.DataFrame'`


Collapse CpG-level beta values into pathway-level methylation scores per sample.

Useful for downstream pathway-activity modeling or visualization.

#### Parameters
- **beta**: `pd.DataFrame`
    Beta-value matrix (CpGs × samples).
- **annotation**: `pd.DataFrame`
    Mapping from CpG identifiers to gene symbols (index → gene).
- **pathway_db**: `dict[str, list[str]]`
    Dictionary of pathways → list of associated genes.
- **method**: `{"mean", "median", "sum"}, default "mean"`
    Aggregation function applied across CpGs belonging to each pathway.

#### Returns
pd.DataFrame
    Pathways × samples matrix of aggregated methylation scores.

### Deconvolution

Cell-type deconvolution utilities for bulk DNA methylation data.

This module provides fast, reference-based estimation of cell-type proportions from epigenome-wide methylation profiles using constrained non-negative least squares (NNLS). It supports integration of published reference matrices (e.g., blood, brain, tumor microenvironment) and enables accurate correction for cellular heterogeneity in differential methylation studies.

#### Features
- High-performance NNLS deconvolution with automatic feature alignment between bulk and reference
- Optional parallel processing via joblib for large cohorts (hundreds to thousands of samples)
- Robust normalization of estimated proportions to sum-to-1 per sample
- Graceful handling of missing joblib dependency with automatic fallback to serial execution
- Clear error reporting when no overlapping CpGs exist between dataset and reference
- Direct compatibility with standard beta-value matrices (CpGs × samples) used throughout DMeth)
- Extensible design for future reference-free (e.g., EpiDISH HEpiDISH, MeDeCom) or projective methods



---

### `estimate_cell_composition`


`estimate_cell_composition(beta: DataFrame, ref_profiles: DataFrame, method: str = 'nnls', n_jobs: int = 1) -> DataFrame`


Perform reference-based cell-type deconvolution of bulk DNA methylation     data using constrained non-negative least squares (NNLS).

Estimates the relative proportions of predefined cell types in each     sample by solving the linear mixture model:
    β_bulk ≈ R · w

where R contains cell-type-specific methylation reference profiles     and w ≥ 0 are the unknown proportions.

#### Parameters
- **beta**: `pd.DataFrame`
    Bulk beta-value matrix with CpGs as rows (index) and samples as columns.
    Values must be in [0, 1]; missing values are not supported.
- **ref_profiles**: `pd.DataFrame`
    Reference matrix of pure cell-type methylation profiles.
    Must have the same CpG index as ``beta`` (or a subset thereof) and cell         types as columns.
- **method**: `str, default "nnls"`
    Deconvolution algorithm. Currently only ``"nnls"`` (scipy.optimize.nnls)         is implemented.
- **n_jobs**: `int, default 1`
    Number of parallel processes for sample-wise deconvolution.
    Requires ``joblib``; automatically falls back to serial execution if         unavailable or set to 1.

#### Returns
pd.DataFrame
    Estimated cell-type proportions with:
    rows = samples (same order and names as ``beta.columns``)
    columns = cell types (same as ``ref_profiles.columns``)
    values constrained to ≥0 and normalized to sum to 1 per sample

#### Raises
ValueError
    If no overlapping CpG features exist between ``beta`` and ``ref_profiles``.

#### Notes
- Automatic feature alignment: only CpGs present in both matrices are used.
- Proportions are forcibly normalized to sum to 1 (with protection against     division by zero).
- Highly efficient for typical blood, brain, or tumor microenvironment references     (e.g., FlowSorted.Blood.EPIC, Houseman, etc.).
- Designed for seamless integration with DMeth preprocessing pipelines     (beta-value input expected).

### Downstream Stats

Core statistical utilities for DNA methylation analysis downstream of differential testing.

- This module provides a comprehensive suite of high-performance, publication-grade functions for p-value adjustment, effect-size computation, delta-beta calculation, reproducible DMR calling via sliding-window clustering, regional summarization, and cross-dataset reproducibility assessment.
- Designed for seamless integration into DMeth pipelines, all functions operate efficiently on large epigenome-wide datasets while preserving genomic coordinates and CpG identifiers.

#### Features
- Multiple p-value correction methods via statsmodels with robust NaN handling
- Stouffer’s Z-score method for meta-analysis across studies or batches
- Cohen’s d and Hedges’ g effect size estimation with proper small-sample correction
- Flexible delta-beta computation with optional absolute values and index alignment
- Threshold-based filtering of DMS results supporting logFC, delta-beta, and directional constraints
- Fast, vectorized sliding-window DMR discovery with configurable gap merging and minimum CpG requirements
- Comprehensive DMR summarization and cross-dataset reproducibility metrics (Jaccard, concordance, Spearman correlation)
- Full preservation of CpG identifiers and genomic coordinates throughout all operations



---

### `adjust_pvalues`


`adjust_pvalues(pvals: Union[pandas.core.series.Series, ndarray, List[float]], method: str = 'fdr_bh') -> pandas.core.series.Series`


Apply multiple-testing correction to raw p-values using statsmodels.

Handles NaN values robustly (treated as non-significant) and preserves the     original index when input is a pandas Series.

#### Parameters
- **pvals**: `array-like`
    Raw p-values (0 ≤ p ≤ 1).
- **method**: `str, default "fdr_bh"`
    Correction method passed to ``statsmodels.stats.multitest.multipletests``.
    Supported: ``"bonferroni"``, ``"holm"``, ``"fdr_bh"``, ``"fdr_by"``,         ``"sidak"``, ``"none"``.

#### Returns
pd.Series
    Adjusted p-values with identical index/order as input.


---

### `compute_delta_beta`


`compute_delta_beta(mean_beta_group1: Union[pandas.core.series.Series, ndarray], mean_beta_group2: Union[pandas.core.series.Series, ndarray], as_abs: bool = False) -> pandas.core.series.Series`


Calculate per-CpG difference in mean beta values between two groups.

Automatically aligns inputs by index; supports absolute differences.

#### Parameters
mean_beta_group1, mean_beta_group2 : Series-like
    Group-wise mean beta values.
- **as_abs**: `bool, default False`
    Return absolute delta-beta if True.

#### Returns
pd.Series
    Delta-beta values (positive = group1 > group2).


---

### `compute_dms_reproducibility`


`compute_dms_reproducibility(res1: DataFrame, res2: DataFrame, id_col: Optional[str] = None, effect_col: str = 'logFC', pval_col: str = 'padj', pval_thresh: float = 0.05) -> Dict[str, Any]`


Quantify reproducibility of differential methylation signals across     two independent analyses or cohorts.

Metrics include:

- Total/overlap feature counts
- Jaccard index
- Number of overlapping significant CpGs
- Directional concordance
- Spearman correlation of effect sizes

#### Parameters
res1, res2 : pd.DataFrame
    Two differential result tables (same CpG index preferred).
effect_col, pval_col : str
    Columns for effect size and adjusted p-value.
- **pval_thresh**: `float, default 0.05`

#### Returns
dict
    Comprehensive reproducibility statistics.


---

### `compute_effect_size`


`compute_effect_size(beta_group1: DataFrame, beta_group2: DataFrame, method: str = 'cohens_d') -> pandas.core.series.Series`


Compute standardized effect size (Cohen’s d or Hedges’ g) for each CpG     between two groups.

Uses pooled standard deviation with proper small-sample correction for Hedges’ g.

#### Parameters
beta_group1, beta_group2 : pd.DataFrame
    Beta matrices (CpGs × samples) for each group.
- **method**: `{"cohens_d", "hedges_g"}, default "cohens_d"`

#### Returns
pd.Series
    Effect size per CpG (positive = higher in group1).


---

### `filter_dms`


`filter_dms(res: DataFrame, lfc_col: str = 'logFC', pval_col: str = 'padj', delta_beta_col: Optional[str] = None, pval_thresh: float = 0.05, lfc_thresh: float = 0.0, delta_beta_thresh: Optional[float] = None, direction: Optional[str] = None) -> DataFrame`


Apply multi-criterion filtering to differential methylation results.

- **Supports**: `- Adjusted p-value threshold`
- Minimum |logFC|
- Minimum |Δβ|
- Directional filtering (“hyper” or “hypo”)

#### Parameters
- **res**: `pd.DataFrame`
    Differential results table.
lfc_col, pval_col, delta_beta_col : str
    Column names (defaults: "logFC", "padj", optional).
pval_thresh, lfc_thresh, delta_beta_thresh
    Numeric thresholds.
- **direction**: `{"hyper", "hypo", None}`

#### Returns
pd.DataFrame
    Subset of rows passing all specified criteria.


---

### `find_dmrs_by_sliding_window`


`find_dmrs_by_sliding_window(dms: DataFrame, annotation: DataFrame, chr_col: str = 'chr', pos_col: str = 'pos', pval_col: str = 'padj', pval_thresh: float = 0.05, delta_beta_col: Optional[str] = 'delta_beta', max_gap: int = 500, min_cpgs: int = 3, merge_distance: Optional[int] = None, use_intervaltree: bool = True) -> DataFrame`


Identify differentially methylated regions (DMRs) by clustering spatially     proximate significant CpGs.

Uses a fast sliding-window/gap-merging approach per chromosome.

#### Parameters
- **dms**: `pd.DataFrame`
    Significant DMS results (after filtering).
- **annotation**: `pd.DataFrame`
    CpG annotation with ``chr`` and ``pos`` columns.
- **max_gap**: `int, default 500`
    Maximum distance (bp) to bridge adjacent significant CpGs.
- **min_cpgs**: `int, default 3`
    Minimum number of significant CpGs required to call a DMR.
- **merge_distance**: `int or None`
    If set, merge DMRs closer than this distance.

#### Returns
pd.DataFrame
    One row per DMR with columns:
    ``chr``, ``start``, ``end``, ``n_cpgs``, ``mean_delta_beta``,         ``mean_logFC``, ``min_padj``, ``cpgs`` (list).


---

### `stouffer_combined_pvalue`


`stouffer_combined_pvalue(pvals)`


Combine independent p-values across studies or replicates using Stouffer’s     Z-score method (equal weighting).

#### Parameters
- **pvals**: `array-like`
    List or array of p-values to combine.

#### Returns
float
    Single combined p-value.

#### Raises
ValueError
    If input is empty or contains values outside [0, 1].


---

### `summarize_regions`


`summarize_regions(dmrs: DataFrame, summary_cols: Optional[List[str]] = None) -> DataFrame`


Generate a concise summary table of discovered DMRs.

Reports total count, median length, average CpG density, mean delta-beta,     and strongest significance.

#### Parameters
- **dmrs**: `pd.DataFrame`
    Output from ``find_dmrs_by_sliding_window``.
- **summary_cols**: `list[str] or None`
    Columns to summarize (defaults to key metrics).

#### Returns
pd.DataFrame
    Single-row summary suitable for manuscript tables.

### Helpers

General-purpose utilities for DNA methylation analysis workflows.

This lightweight module contains essential helper functions used across the DMeth package for consistent data handling and summarization. It provides robust chromosome string normalization and flexible group-wise aggregation of beta-value matrices, ensuring compatibility and reproducibility in downstream statistical and annotation steps.

#### Features
- Robust chromosome identifier normalization with automatic 'chr' prefix addition and preservation of X/Y conventions
- Safe handling of mixed chromosome formats (e.g., '1', 'chr1', 'CHR1' → 'chr1', 'chrX')
- Efficient group-wise summarization of per-sample beta values with support for arbitrary aggregation functions
- Automatic computation of both mean and variance per group for interpretability and modeling
- Graceful handling of missing groups, empty inputs, and non-overlapping sample sets
- Index-preserving operations fully compatible with CpG × sample matrix conventions



---

### `summarize_groups`


`summarize_groups(beta: DataFrame, groups: pandas.core.series.Series, summary_func: Callable = <function mean at 0x7c994bb96170>) -> DataFrame`


Compute group-wise summary statistics (mean and variance) across samples     for a beta-value matrix.

Designed for rapid interpretation of methylation levels across     experimental conditions or cell types.

#### Parameters
- **beta**: `pd.DataFrame`
    Beta-value matrix with CpGs as rows (index) and samples as columns.
- **groups**: `pd.Series`
    Sample-to-group mapping. Index must align with ``beta.columns``.
- **summary_func**: `callable, default ``np.mean```
    Aggregation function applied per group (e.g., ``np.mean``, ``np.median``).

#### Returns
pd.DataFrame
    New DataFrame with columns:
    ``mean_{group}`` – group-wise summary using ``summary_func``
    ``var_{group}``  – sample variance within each group (ddof=1)

    Missing groups or empty intersections yield ``NaN`` columns.

#### Notes
- Fully preserves CpG index.
- Handles partial or missing group overlap gracefully.
- Ideal for generating input tables for delta-beta calculation,     visualization, or reporting.

### Signature

Biomarker signature discovery and predictive modeling for DNA methylation data.

This module provides a streamlined, production-ready framework for translating differential methylation results into high-performance diagnostic, prognostic, or predictive biomarker panels. It supports feature selection from statistical outputs, cross-validated model training with state-of-the-art algorithms, and rigorous independent validation of classification or regression performance.

#### Features
- Simple yet effective signature selection via top-ranked CpGs by p-value or moderated t-statistic
- Flexible predictive modeling with Random Forest and Elastic Net (with built-in hyperparameter tuning via CV)
- Automatic task detection and appropriate stratification (classification vs regression)
- Comprehensive cross-validation and held-out test set evaluation with standard metrics (AUC, accuracy, R², RMSE)
- Full integration with DMeth result tables and sample-level beta matrices (features × samples orientation)
- Reproducible training through fixed random seeds and stratified splitting
- Extensible architecture for future stability selection, recursive feature elimination, or multi-omics integration



---

### `model_dms_for_prediction`


`model_dms_for_prediction(beta: 'pd.DataFrame', labels: 'pd.Series', method: 'str' = 'random_forest', n_splits: 'int' = 5, random_state: 'int' = 42, task: 'Optional[str]' = None) -> 'Dict[str, Any]'`


Train and evaluate a predictive model using DNA methylation signatures.

Automatically detects classification vs regression tasks and applies     appropriate modeling and evaluation strategies.

#### Parameters
- **beta**: `pd.DataFrame`
    Beta-value matrix with CpGs as rows and samples as columns (features ×         samples orientation).
- **labels**: `pd.Series`
    Target variable aligned with ``beta.columns``.
- **method**: `{"random_forest", "elasticnet"}, default "random_forest"`
    Predictive algorithm:
    ``random_forest``: 500 trees with parallel training
    ``elasticnet``: LogisticRegressionCV or ElasticNetCV with built-in         cross-validated regularization
- **n_splits**: `int, default 5`
    Number of cross-validation folds for hyperparameter tuning and         performance estimation.
- **random_state**: `int, default 42`
    Seed for reproducible splitting and model initialization.
- **task**: `{"classification", "regression"} or None, optional`
    Force task type. If ``None``, inferred from label distribution.

#### Returns
dict
    Contains:
    ``estimator``: final fitted model (trained on full data)
    ``cv_results``: detailed cross-validation scores
    ``test_auc`` / ``test_accuracy`` (classification) or ``test_rmse``         / ``test_r2`` (regression) on a stratified 20% held-out set


---

### `select_signature_panel`


`select_signature_panel(res: 'pd.DataFrame', method: 'str' = 'top', top_n: 'int' = 100, importance_col: 'str' = 't') -> 'List[str]'`


Extract a candidate biomarker panel from differential methylation results.

#### Parameters
- **res**: `pd.DataFrame`
    Differential analysis results table (e.g., output from ``fit_differential``).
    Must contain at least one of ``pval`` or the column specified in         ``importance_col``.
- **method**: `str, default "top"`
    Feature selection strategy. Currently only ``"top"`` is implemented.
- **top_n**: `int, default 100`
    Number of top-ranked CpGs to retain.
- **importance_col**: `str, default "t"`
    Column used for ranking when ``pval`` is unavailable (e.g., moderated         t-statistic).

#### Returns
List[str]
    Ordered list of selected CpG identifiers (row names from ``res``).

#### Notes
- Prioritizes ``pval`` for ranking if present.
- Falls back to descending order of ``importance_col`` otherwise.
- Future extensions will include stability selection and recursive     feature elimination.


---

### `validate_signature`


`validate_signature(X_train: 'pd.DataFrame', y_train: 'Union[pd.Series, np.ndarray]', X_test: 'pd.DataFrame', y_test: 'Union[pd.Series, np.ndarray]', features: 'Sequence[str]', method: 'str' = 'elasticnet') -> 'Dict[str, Any]'`


Independent validation of a pre-selected methylation signature on held-out data.

Re-trains the specified model on training data using only the provided     features and reports performance on a separate test set.

#### Parameters
X_train, X_test : pd.DataFrame
    Training and test methylation matrices (samples × features).
y_train, y_test : array-like
    Corresponding labels.
- **features**: `Sequence[str]`
    Subset of CpGs constituting the signature.
- **method**: `str, default "elasticnet"`
    Model used for validation (same options as ``model_dms_for_prediction``).

#### Returns
dict
    Performance metrics:
    Binary classification → ``auc`` and ``accuracy``
    Multi-class/regression → ``accuracy`` or ``rmse`` + ``r2``

#### Notes
Provides an unbiased estimate of clinical/translational performance     when the signature was derived on a separate discovery cohort.


---

## Utilities

### Logger

Centralised logging utilities for the DMeth DNA-methylation analysis suite.

This module configures a unified logger with the following features:

#### Features
- Timestamped log files automatically saved to ``<output_dir>/log/``
- Simultaneous console (stdout) and file output
- Consistent formatting across all package modules
- A custom :class:`ProgressAwareLogger` that seamlessly integrates ``tqdm`` progress bars:
    - ``logger.progress("Processing samples", total=n)`` starts a progress bar
    - ``logger.progress_update(k)`` advances it
    - Any regular log call (info/warning/error/etc.) automatically closes the active     bar so that log messages are never corrupted by overlapping tqdm output

All other ``dmeth`` modules import the logger via ``get_logger()``.



---

### `ProgressAwareLogger`


`ProgressAwareLogger(name) -> 'None'`


Custom logger class that supports a temporary progress bar.
The progress bar stays active until the next normal log call.


---

### `get_logger`


`get_logger(name: 'str' = 'dmeth') -> 'logging.Logger'`


Return the central DMeth logger instance.

#### Parameters
- **name**: `str, default "dmeth"`
    Logger name. Typically left as the default.

#### Returns
logging.Logger
    The configured ProgressAwareLogger instance.

### Plotting

Plotting utilities for downstream DNA methylation analysis.

This module provides a comprehensive set of visualisation functions for quality control,
differential methylation analysis, dimensionality reduction, variance shrinkage diagnostics,
and publication-ready summary figures. Functions operate on pandas DataFrames, ProcessedData
objects, or differential methylation results and return standardised matplotlib (or optionally
Plotly) figures.

#### Features
- Standardised figure creation
- Enhanced volcano plots with automatic significance colouring and annotation
- P-value QQ plots and histograms
- Sample-level PCA, t-SNE, and UMAP embeddings (static or interactive)
- Variance shrinkage diagnostics (limma-style)
- Mean-difference bar plots for top loci
- Multi-panel QC reports via plot_stage()
- Combined DMS/DMR summary visualisation (volcano + manhattan + heatmap)



---

### `ProcessedData`


`ProcessedData(M: DataFrame, pheno: DataFrame, ann: DataFrame, meta: Dict[str, Any] = <factory>) -> None`


Central container for aligned methylation data used throughout the DMeth pipeline.

Guarantees that:

- All components (methylation matrix, sample metadata, probe annotation)     share consistent string indices.
- Sample and probe alignment is validated on construction.
- A ``meta`` dictionary tracks processing history (normalisation,     batch correction, QC metrics, etc.).

All downstream functions expect or return instances of this class.

#### Parameters
- **M**: `pd.DataFrame`
    Methylation matrix with CpG sites as rows and samples as columns.
    Values are typically beta (0–1) or M-values (-inf to +inf).
- **pheno**: `pd.DataFrame`
    Sample metadata table. Must be indexed by sample IDs (strings         after construction).
- **ann**: `pd.DataFrame or None`
    Optional probe/CpG annotation table (e.g., Illumina manifest).         Must be indexed by CpG IDs.
    If ``None``, probe-level alignment checks are skipped.
- **meta**: `dict, optional`
    Free-form dictionary storing pipeline provenance and parameters.
    Pre-populated with sensible defaults if not provided.

#### Attributes
- **M**: `pd.DataFrame`
    Methylation matrix (string-indexed rows and columns).
- **pheno**: `pd.DataFrame`
    Sample metadata (string-indexed).
- **ann**: `pd.DataFrame or None`
    Probe annotation (string-indexed if present).
- **meta**: `dict`
    Processing metadata (e.g., ``{"matrix_type": "beta",         "normalized": True, ...}``).

#### Notes
The ``__post_init__`` method automatically:

- Converts all relevant indices to strings.
- Validates alignment between components.
- Raises informative ``KeyError`` on mismatch.


---

### `correlate_methylation_expression`


`correlate_methylation_expression(beta: 'pd.DataFrame', expression: 'pd.DataFrame', gene_map: 'Optional[Dict[str, str]]' = None, method: 'str' = 'pearson', override_index_alignment: 'bool' = False) -> 'pd.DataFrame'`


Compute sample-wise correlation between CpG methylation and gene expression.

Supports both one-to-one (same index) and many-to-one (custom CpG→gene     mapping) scenarios.

#### Parameters
beta, expression : pd.DataFrame
    Methylation (CpGs × samples) and expression (genes × samples) matrices.
- **gene_map**: `dict or None`
    Explicit mapping from CpG ID → gene symbol (required for cis-analysis).
- **method**: `{"pearson", "spearman"}, default "pearson"`
    Correlation coefficient to compute.
- **override_index_alignment**: `bool, default False`
    Proceed even with <2 overlapping samples (useful for exploratory checks).

#### Returns
pd.DataFrame
    Correlation results with columns ``r`` and ``pval`` (indexed by CpG         and optionally gene).


---

### `methylation_expression_heatmap`


`methylation_expression_heatmap(beta: DataFrame, expression: DataFrame, genes: Optional[List[str]] = None, top_n: int = 50, sample_metadata: Optional[DataFrame] = None, save_path: Union[str, pathlib.Path, NoneType] = None, method: str = 'pearson') -> matplotlib.figure.Figure`


Heatmap of methylation-expression correlation coefficients for selected genes/CpGs.

#### Parameters
- **beta**: `pd.DataFrame`
    Methylation beta matrix (features × samples).
- **expression**: `pd.DataFrame`
    Gene expression matrix (genes × samples).
- **genes**: `list[str], optional`
    Specific features to display; if None the top_n most correlated are used.
- **top_n**: `int, default 50`
    Number of top correlations shown when ``genes`` is None.
- **sample_metadata**: `pd.DataFrame, optional`
    Currently unused (reserved for future sample-side annotation).
- **save_path**: `str or Path, optional`
    Destination for saving the figure.
- **method**: `{"pearson", "spearman"}, default "pearson"`
    Correlation method.

#### Returns
plt.Figure
    Horizontal heatmap of correlation coefficients.


---

### `pca_plot`


`pca_plot(data: ProcessedData, color_col: str, title: str = 'PCA of Methylation Data', figsize: tuple[int, int] = (7, 6)) -> None`


Display a simple PCA projection of samples coloured by a phenotype column.

#### Parameters
- **data**: `ProcessedData`
    Container holding the methylation matrix and phenotype table.
- **color_col**: `str`
    Column in ``data.pheno`` used for colouring points.
- **title**: `str, default "PCA of Methylation Data"`
    Title shown on the plot.
- **figsize**: `tuple[int, int], default (7, 6)`
    Size of the figure in inches.

#### Notes
The plot is shown immediately with ``plt.show()`` and is not returned.


---

### `plot_mean_difference`


`plot_mean_difference(beta_group1: DataFrame, beta_group2: DataFrame, top_n: int = 50, save_path: Union[str, pathlib.Path, NoneType] = None) -> matplotlib.figure.Figure`


Bar plot of the largest absolute mean beta differences between two groups.

#### Parameters
beta_group1, beta_group2 : pd.DataFrame
    Beta-value matrices (features × samples) for each group.
- **top_n**: `int, default 50`
    Number of top differentially methylated loci to display.
- **save_path**: `str or Path, optional`
    Destination for saving the figure.

#### Returns
plt.Figure
    Bar chart of absolute mean differences.


---

### `plot_pvalue_qq`


`plot_pvalue_qq(res: DataFrame, pval_col: str = 'pval', dpi: int = 300, save_path: Union[str, pathlib.Path, NoneType] = None) -> matplotlib.figure.Figure`


Produce a quantile-quantile plot comparing observed versus     expected -log10(p-values).

#### Parameters
- **res**: `pd.DataFrame`
    DataFrame containing the p-value column.
- **pval_col**: `str, default "pval"`
    Column name of the p-values.
- **dpi**: `int, default 300`
    Figure resolution.
- **save_path**: `str or Path, optional`
    Destination path for saving the figure.

#### Returns
plt.Figure
    Q-Q plot figure.


---

### `plot_shrinkage_diagnostics`


`plot_shrinkage_diagnostics(s2: pandas.core.series.Series, s2_post: pandas.core.series.Series, d0: Optional[float] = None, save_path: Union[str, pathlib.Path, NoneType] = None) -> matplotlib.figure.Figure`


Visualise the effect of empirical Bayes variance shrinkage.

#### Parameters
- **s2**: `pd.Series`
    Original (unmoderated) variance estimates per feature.
- **s2_post**: `pd.Series`
    Moderated (shrunk) variance estimates.
- **d0**: `float, optional`
    Prior degrees of freedom from the shrinkage procedure (displayed         in title if given).
- **save_path**: `str or Path, optional`
    If provided, the figure is saved to this location.

#### Returns
plt.Figure
    Scatter plot comparing log10(original) vs log10(shrunk) variances.


---

### `plot_stage`


`plot_stage(stage: str, M: DataFrame, res: Optional[DataFrame] = None, metadata: Optional[DataFrame] = None, groups_col: str = 'Type', top_n: int = 10, embedding: str = 'pca', interactive: bool = False, save_path: Union[str, pathlib.Path, Mapping[str, str], NoneType] = None, dpi: int = 300, **kwargs: Any) -> Dict[str, Union[matplotlib.figure.Figure, ForwardRef('PlotlyFigure')]]`


Orchestrate a complete multi-panel QC or analysis visualisation for     a given processing stage.

Supported stages: "qc", "variance", "differential", "top_hits", "correlation".

#### Parameters
- **stage**: `str`
    One of the recognised analysis stages.
- **M**: `pd.DataFrame`
    Methylation matrix (features × samples), typically M-values or beta-values.
- **res**: `pd.DataFrame, optional`
    Differential methylation results (required for variance,         differential, top_hits stages).
- **metadata**: `pd.DataFrame, optional`
    Sample annotation table; must contain the grouping column.
- **groups_col**: `str, default "Type"`
    Column in metadata used for colour-coding samples.
- **top_n**: `int, default 10`
    Number of top features to display in relevant panels.
- **embedding**: `{"pca", "tsne", "umap"}, default "pca"`
    Dimensionality-reduction method for sample embedding.
- **interactive**: `bool, default False`
    Return Plotly interactive figures where possible.
- **save_path**: `str, Path or dict, optional`
    Path or mapping of figure names to paths for automatic saving.
- **dpi**: `int, default 300`
    Resolution for static figures.
**kwargs
    Additional keyword arguments passed to the underlying reducer.

#### Returns
dict[str, plt.Figure | plotly.graph_objects.Figure]
    Mapping from panel name to the generated figure(s).


---

### `plot_volcano`


`plot_volcano(res: DataFrame, lfc_col: str = 'logFC', pval_col: str = 'pval', alpha: float = 0.7, lfc_thresh: float = 1.0, pval_thresh: float = 0.05, top_n: int = 10, dpi: int = 300, save_path: Union[str, pathlib.Path, NoneType] = None) -> matplotlib.figure.Figure`


Create an enhanced volcano plot with top hits annotated.

#### Parameters
- **res**: `pd.DataFrame`
    Differential results with logFC and p-values
- **lfc_col**: `str`
    Column name for log fold change
- **pval_col**: `str`
    Column name for p-value
- **alpha**: `float`
    Point transparency
- **lfc_thresh**: `float`
    Threshold for logFC significance
- **pval_thresh**: `float`
    Threshold for p-value significance
- **top_n**: `int`
    Number of top hits to annotate
- **dpi**: `int`
    Figure resolution
- **save_path**: `str or Path, optional`
    Path to save figure

#### Returns
plt.Figure
    Volcano plot figure


---

### `pvalue_histogram`


`pvalue_histogram(pvals: Union[pandas.core.series.Series, ndarray[Any, numpy.dtype[numpy.float64]], List[float]], bins: int = 50, save_path: Union[str, pathlib.Path, NoneType] = None) -> matplotlib.figure.Figure`


Histogram of p-values with a flat null-expectation line for QC assessment.

#### Parameters
- **pvals**: `array-like`
    Collection of p-values.
- **bins**: `int, default 50`
    Number of histogram bins.
- **save_path**: `str or Path, optional`
    Path to save the figure.

#### Returns
plt.Figure
    Histogram figure.


---

### `visualize_dms`


`visualize_dms(res: DataFrame, beta: Optional[DataFrame] = None, top_n: int = 50, volcano: bool = True, manhattan: bool = True, heatmap: bool = True, sample_metadata: Optional[DataFrame] = None, save_dir: Union[str, pathlib.Path, NoneType] = None) -> Dict[str, Optional[matplotlib.figure.Figure]]`


Produce a standard set of summary plots for differentially methylated sites/regions.

Optionally creates volcano, Manhattan, and heatmap visualisations.

#### Parameters
- **res**: `pd.DataFrame`
    Differential methylation results.
- **beta**: `pd.DataFrame, optional`
    Beta matrix required for the heatmap panel.
- **top_n**: `int, default 50`
    Number of top CpGs shown in the heatmap.
volcano / manhattan / heatmap : bool, default True
    Toggle creation of each panel.
- **sample_metadata**: `pd.DataFrame, optional`
    Sample annotation (currently unused but reserved for future clustering).
- **save_dir**: `str or Path, optional`
    Directory where individual PNG files are written.

#### Returns
dict[str, plt.Figure | None]
    Figures for "volcano", "manhattan", and "heatmap" (None if not generated).


---

## Configuration

### Config Manager

Planner configuration manager for DNA methylation studies.

Provides a thread-safe singleton that centralizes platforms, experimental
designs, cost components, timeline phases, and global settings with full
validation and multiple format support.



---

### `CostComponentSchema`


`CostComponentSchema(*, cost: Annotated[float, Ge(ge=0.0)], unit: Annotated[str, _PydanticGeneralMetadata(pattern='^(per_sample|per_cpg|fixed)$')], description: Optional[str] = None, optional: bool = False, applies_to: Optional[List[str]] = None) -> None`


Configuration schema for cost components.


---

### `DesignSchema`


`DesignSchema(*, name: str, description: Optional[str] = None, n_groups: Annotated[int, Ge(ge=1)], paired: bool = False, complexity: Optional[str] = None, min_n_recommended: Annotated[int, Ge(ge=1)], power_adjustment: Annotated[float, Gt(gt=0.0), Le(le=2.0)], analysis_method: Optional[str] = None, example_uses: Optional[List[str]] = None) -> None`


Configuration schema for experimental design.


---

### `GlobalSettingsSchema`


`GlobalSettingsSchema(*, contingency_buffer_percent: Annotated[float, Ge(ge=0.0), Le(le=100.0)] = 10.0, default_mcp_method: Annotated[str, _PydanticGeneralMetadata(pattern='^(bonferroni|fdr|none)$')] = 'fdr') -> None`


Global configuration settings.


---

### `PlannerConfig`


`PlannerConfig(config_file: Union[str, pathlib.Path, NoneType] = None)`


Thread-safe singleton for planner configuration management.

This class manages all configuration for DNA methylation study planning,
including platforms, experimental designs, costs, and timelines.

#### Parameters
- **config_file**: `str or Path, optional`
    Configuration file to load on initialization.


---

### `PlannerConfigModel`


`PlannerConfigModel(*, platforms: Dict[str, dmeth.config.config_manager.PlatformSchema], designs: Dict[str, dmeth.config.config_manager.DesignSchema], cost_components: Dict[str, dmeth.config.config_manager.CostComponentSchema], timeline_phases: Dict[str, dmeth.config.config_manager.TimelinePhaseSchema], global_settings: dmeth.config.config_manager.GlobalSettingsSchema = <factory>) -> None`


Complete planner configuration model.


---

### `PlatformSchema`


`PlatformSchema(*, name: str, manufacturer: Optional[str] = None, n_cpgs: Annotated[int, Ge(ge=1)], cost_per_sample: Annotated[float, Ge(ge=0.0)], processing_days: Annotated[int, Ge(ge=0)], dna_required_ng: Annotated[Optional[float], Ge(ge=0)] = None, coverage: Optional[str] = None, release_year: Annotated[Optional[int], Ge(ge=1990), Le(le=2030)] = None, status: Optional[str] = None, recommended: bool = False, notes: Optional[str] = None) -> None`


Configuration schema for a methylation array platform.


---

### `TimelinePhaseSchema`


`TimelinePhaseSchema(*, name: Optional[str] = None, base_duration_days: Annotated[float, Gt(gt=0)], scaling_factor: Annotated[float, Ge(ge=0.0)] = 0.0, batch_adjustment: Annotated[float, Ge(ge=0.0)] = 0.0, description: Optional[str] = None, critical: bool = False, optional: bool = False, parallelizable: bool = False) -> None`


Configuration schema for timeline phases.


---

### `get_config`


`get_config() -> dmeth.config.config_manager.PlannerConfig`


Get the global PlannerConfig singleton instance.

#### Returns
PlannerConfig
    The global configuration instance.


---

### `load_file`


`load_file(path: Union[str, pathlib.Path], allow_excel: bool = True) -> None`


Load configuration from a file. See PlannerConfig.load_file().


---

### `migrate`


`migrate(migrator: Callable[[Dict], Dict]) -> None`


Apply a migration function. See PlannerConfig.migrate().


---

### `reload`


`reload() -> None`


Reload configuration. See PlannerConfig.reload().


---

### `reset_config`


`reset_config() -> None`


Reset the global configuration instance (primarily for testing).


---

### `save_file`


`save_file(path: Union[str, pathlib.Path], fmt: Optional[str] = None) -> None`


Save configuration to a file. See PlannerConfig.save_file().

## Citation

If you use `dmeth` in your research, please cite:

```bibtex
@software{dmeth2025,
  author = {Afolabi, Dare},
  title = {dmeth: A comprehensive Python toolkit for differential DNA methylation analysis with empirical Bayes moderation and biomarker discovery},
  version = {0.2.0},
  year = {2025},
  publisher = {GitHub},
  doi = {10.5281/zenodo.17777501},
  url = {https://doi.org/10.5281/zenodo.17777501},
}
```

### References

- Smyth, G. K. (2004). Linear models and empirical bayes methods for assessing differential expression in microarray experiments. *Statistical Applications in Genetics and Molecular Biology*, 3(1).
- Liu, P., & Hwang, J.T.G. (2007). Quick calculation for sample size while controlling false discovery rate with application to microarray analysis. *Bioinformatics*, 23(6), 739–746.
- Du, P., Zhang, X., Huang, C.-C., Jafari, N., Kibbe, W.A., Hou, L., & Lin, S. (2010). Comparison of Beta-value and M-value methods for quantifying methylation levels by microarray analysis. *BMC Bioinformatics*, 11:587.
- Jung, S.H., Young, S.S. (2012). Power and sample size calculation for microarray studies. *Journal of Biopharmaceutical Statistics*, 22(1):30-42.
- Phipson, B. et al. (2016). missMethyl: an R package for analyzing data from Illumina’s HumanMethylation450 platform. *Bioinformatics*, 32(2), 286-288.


## Support

- **Issues**: [GitHub Issues](https://github.com/dare-afolabi/dmeth/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dare-afolabi/dmeth/discussions/1)
- **Email**: [dare.afolabi@outlook.com](mailto:dare.afolabi@outlook.com)
ook.com)



---

> **Auto-generated** on 2025-12-13 02:56:16
