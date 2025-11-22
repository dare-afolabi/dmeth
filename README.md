# `dmeth`: Differential Methylation Analysis Toolkit

A fast, statistically rigorous Python framework providing a toolkit for DNA methylation analysis - from raw beta matrices to biomarkers and functional interpretation.

<div align="center">
  <a href="https://github.com/dare-afolabi/dmeth/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/dare-afolabi/dmeth/ci.yml?branch=main&style=flat" alt="Build Status">
  </a>
  <a href="https://codecov.io/gh/dare-afolabi/dmeth">
    <img src="https://img.shields.io/codecov/c/github/dare-afolabi/dmeth?style=flat" alt="Coverage">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  </a>
  <a href="https://badge.fury.io/py/dmeth">
    <img src="https://badge.fury.io/py/dmeth.svg" alt="PyPI version">
  </a>
  <a href="https://github.com/sponsors/dmeth">
    <img src="https://img.shields.io/badge/Sponsor-grey?style=flat&logo=github-sponsors" alt="Sponsor">
  </a>
</div>

**`dmeth`** implements the full modern differential methylation pipeline used in high-impact epigenome-wide association studies (EWAS), with performance and correctness on par with established R/bioconductor tools, all in pure Python.

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
from dmeth.core.analysis.validation import validate_design, validate_contrast
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
design = validate_design(pheno["group"])
contrast = validate_contrast(design, "case-control")

# Fit
res = fit_differential(
    M=beta_imp,
    design=pd.DataFrame(design, index=beta_imp.columns),
    contrast=contrast,
    shrink="smyth",
    robust=True,
)

# 4. Discover DMRs
annotation = pd.read_csv("cpg_annotation.csv", index_col=0)  # must include chr, pos columns
dmrs = find_dmrs_by_sliding_window(
    dms=res[res["padj"] < 0.05],
    annotation=annotation,
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

pytest, pytest-cov, black, isort, flake8, flake8-pyproject, flake8-bugbear, bandit, mypy, mkdocs, mkdocs-material

## Documentation

Full documentation with tutorials, API reference, and reproducibility examples:
[User Guide](https://github.com/dare-afolabi/dmeth/blob/main/docs/UserGuide.md)

## Citation

If you use `dmeth` in your research, please cite:

```bibtex
@software{dmeth2025,
  author = {Afolabi, Dare},
  title = {dmeth: A comprehensive Python toolkit for differential DNA methylation analysis with empirical Bayes moderation and biomarker discovery},
  version = {0.1.0},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/dare-afolabi/dmeth},
  doi = {10.5281/zenodo.XXXXXXX},  # will be assigned upon release
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
