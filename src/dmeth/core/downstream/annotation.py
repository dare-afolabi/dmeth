#!/usr/bin/env python
# coding: utf-8


"""
Annotation and functional interpretation utilities for DNA methylation analysis.

This module delivers a complete suite of tools for translating CpG- and region-level \
differential methylation results into biological context. It enables rapid gene \
annotation, pathway- and gene-set enrichment analysis, computation of pathway-\
level methylation activity, correlation with gene expression, and reliable \
coordinate liftover across genome assemblies — all implemented with performance \
and robustness suitable for genome-wide studies.

Features
--------
- Ultra-fast nearest-gene annotation using IntervalTree with graceful \
fallback to distance-based mapping
- Fisher’s exact test gene-set enrichment with automatic background correction, \
size filtering, and FDR adjustment
- Flexible pathway-level methylation scoring (mean/median/sum) from user-\
provided gene-to-pathway mappings
- Sample-aware Pearson/Spearman correlation between CpG methylation and gene \
expression with robust overlap handling
- High-accuracy genomic liftover via pyliftover (hg19 ↔ hg38 and other builds) \
with per-region success tracking
- Comprehensive handling of missing dependencies, empty inputs, \
and chromosome naming conventions
- Preservation of original indices and seamless integration with DMeth result tables
"""


from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from dmeth.core.downstream.downstream_stats import adjust_pvalues
from dmeth.core.downstream.helpers import _clean_chr
from dmeth.utils.logger import logger

# Safe IntervalTree import
try:
    from intervaltree import Interval, IntervalTree

    use_tree = True
except ImportError:
    logger.warning("IntervalTree not installed. Using distance-based fallback.")
    use_tree = False

try:
    from pyliftover import LiftOver
except ImportError:
    LiftOver = None


def map_dms_to_genes(
    dms: pd.DataFrame,
    genes: pd.DataFrame,
    cpg_chr_col: str = "chr",
    cpg_pos_col: str = "pos",
    gene_chr_col: str = "chr",
    gene_start_col: str = "start",
    gene_end_col: str = "end",
    gene_name_col: str = "gene_symbol",
    promoter_upstream: int = 1500,
    proximal_cutoff: int = 200,
    promoter_downstream: int = 500,
    distal_promoter_max: int = 5000,
    max_distance: int = 1000000,
) -> pd.DataFrame:
    """
    Annotate CpGs with nearest gene and regulatory context.

    Parameters
    ----------
    dms : pd.DataFrame
        CpG positions with chromosome and position columns.
    genes : pd.DataFrame
        Gene annotations with chromosome, start, end, gene_symbol, and optional strand.
    cpg_chr_col, cpg_pos_col : str
        Column names for CpG chromosome and position.
    gene_chr_col, gene_start_col, gene_end_col, gene_name_col : str
        Column names for gene annotation.
    promoter_upstream : int
        Upstream region from TSS considered promoter.
    proximal_cutoff : int
        Distance near TSS classified as proximal promoter.
    promoter_downstream : int
        Downstream region from TSS considered promoter.
    distal_promoter_max : int
        Maximum distance upstream of TSS classified as distal promoter.
    max_distance : int
        Maximum distance to assign nearest gene if no overlap.

    Returns
    -------
    pd.DataFrame
        Original `dms` with added columns:
        - nearest_gene : str or NaN
        - relation : {"proximal_promoter", "distal_promoter", "promoter_downstream",
                      "distal_upstream", "gene_body", "intergenic"}
        - distance_bp : int or NaN
    """
    genes = genes.copy()
    dms = dms.copy()
    genes[gene_chr_col] = genes[gene_chr_col].astype(str).apply(_clean_chr)
    dms[cpg_chr_col] = dms[cpg_chr_col].astype(str).apply(_clean_chr)

    # Helper: classify CpG relative to gene
    def classify_cpg(pos, tss, start, end, strand):
        # 1— immediate check: is inside gene body?
        if start <= pos <= end:
            return "gene_body", 0

        # signed distance to TSS
        dist_signed = pos - tss

        # Adjust promoter logic by strand
        if strand == "+":
            if -proximal_cutoff <= dist_signed <= 0:
                return "proximal_promoter", abs(dist_signed)
            if 0 < dist_signed <= promoter_downstream:
                return "promoter_downstream", dist_signed
            if -promoter_upstream <= dist_signed < -proximal_cutoff:
                return "distal_promoter", abs(dist_signed)
            if -distal_promoter_max <= dist_signed < -promoter_upstream:
                return "distal_upstream", abs(dist_signed)

        else:  # minus strand flips directions
            if 0 <= dist_signed <= proximal_cutoff:
                return "proximal_promoter", dist_signed
            if -promoter_downstream <= dist_signed < 0:
                return "promoter_downstream", abs(dist_signed)
            if proximal_cutoff < dist_signed <= promoter_upstream:
                return "distal_promoter", dist_signed
            if promoter_upstream < dist_signed <= distal_promoter_max:
                return "distal_upstream", dist_signed

        return "intergenic", abs(dist_signed)

    # Handle empty inputs
    if dms is None or dms.empty:
        logger.warning("DMS table empty")
        return pd.DataFrame()
    if genes is None or genes.empty:
        out = dms.copy()
        out["nearest_gene"] = np.nan
        out["relation"] = np.nan
        out["distance_bp"] = np.nan
        return out

    # Build IntervalTree if available
    trees = {}
    if use_tree:
        for chrom, grp in genes.groupby(gene_chr_col):
            t = IntervalTree()
            for _, r in grp.iterrows():
                t.add(
                    Interval(
                        int(r[gene_start_col]),
                        int(r[gene_end_col]) + 1,
                        {
                            "gene": r[gene_name_col],
                            "start": int(r[gene_start_col]),
                            "end": int(r[gene_end_col]),
                            "strand": r.get("strand", "+"),
                        },
                    )
                )
            trees[chrom] = t

    # Prepare output lists
    nearest_gene = []
    relation = []
    distance = []

    # Main loop
    for _, row in dms.iterrows():
        chrom = _clean_chr(row[cpg_chr_col])
        pos = int(row[cpg_pos_col])

        best_gene = None
        best_rel = None
        best_dist = None

        # IntervalTree lookup
        t = trees.get(chrom)
        hits = t.at(pos) if t is not None else []

        if hits:
            for h in hits:
                g = h.data
                start, end = g["start"], g["end"]
                gene = g["gene"]
                strand = g.get("strand", "+")
                tss = start if strand == "+" else end

                rel, dist = classify_cpg(pos, tss, start, end, strand)

                if best_dist is None or dist < best_dist:
                    best_gene, best_rel, best_dist = gene, rel, dist

            nearest_gene.append(best_gene)
            relation.append(best_rel)
            distance.append(best_dist)
            continue

        # Distance-based fallback
        gchr = genes[genes[gene_chr_col] == chrom]
        if gchr.empty:
            nearest_gene.append(np.nan)
            relation.append(np.nan)
            distance.append(np.nan)
            continue

        starts = gchr[gene_start_col].to_numpy()
        ends = gchr[gene_end_col].to_numpy()
        inside = (starts <= pos) & (pos <= ends)

        if inside.any():
            idx = np.where(inside)[0][0]
            gene = gchr.iloc[idx][gene_name_col]
            nearest_gene.append(gene)
            relation.append("gene_body")
            distance.append(0)
            continue

        d_up = np.where(pos < starts, starts - pos, np.inf)
        d_down = np.where(pos > ends, pos - ends, np.inf)
        d = np.minimum(d_up, d_down)

        min_idx = np.argmin(d)
        min_dist = d[min_idx]

        if min_dist <= max_distance:
            r = gchr.iloc[min_idx]
            gene = r[gene_name_col]
            start, end = int(r[gene_start_col]), int(r[gene_end_col])
            strand = r.get("strand", "+")
            tss = start if strand == "+" else end

            rel, dist = classify_cpg(pos, tss, start, end, strand)
            nearest_gene.append(gene)
            relation.append(rel)
            distance.append(dist)
        else:
            nearest_gene.append(np.nan)
            relation.append("intergenic")
            distance.append(np.nan)

    out = dms.copy()
    out["nearest_gene"] = nearest_gene
    out["relation"] = relation
    out["distance_bp"] = distance
    return out


def gene_set_enrichment(
    gene_list: List[str],
    background: Optional[List[str]] = None,
    gene_sets: Optional[Dict[str, List[str]]] = None,
    method: str = "fisher",
    pval_cutoff: float = 0.05,
    min_set_size: int = 5,
    max_set_size: int = 500,
) -> pd.DataFrame:
    """
    Perform over-representation analysis (Fisher’s exact test) on a list \
    of genes against predefined gene sets (e.g., GO, KEGG, Reactome).

    Automatically filters gene sets by size and applies FDR correction.

    Parameters
    ----------
    gene_list : list[str]
        Genes of interest (e.g., nearest genes of significant DMS/DMRs).
    background : list[str] or None
        Background gene universe. Defaults to union of all genes in ``gene_sets``.
    gene_sets : dict[str, list[str]]
        Mapping from pathway/term name to member genes.
    min_set_size, max_set_size : int
        Exclude overly small or large gene sets (default 5–500).
    pval_cutoff : float, default 0.05
        Return only terms with adjusted p-value ≤ this threshold.

    Returns
    -------
    pd.DataFrame
        Enriched terms with columns: ``term``, ``pvalue``, ``padj``, \
        ``oddsratio``, ``overlap``, ``set_size``.
    """
    if not gene_list:
        return pd.DataFrame(
            columns=["term", "pvalue", "oddsratio", "overlap", "set_size"]
        )

    background = set(background or sum(gene_sets.values(), []))
    genes = set(gene_list)
    results = []

    for term, members in (gene_sets or {}).items():
        members_set = set(members) & background
        m = len(members_set)
        if m < min_set_size or m > max_set_size:
            continue
        overlap = len(genes & members_set)
        a = overlap
        b = m - a
        c = len(genes) - a
        d = len(background) - (a + b + c)
        if any(x < 0 for x in (a, b, c, d)):
            continue
        _, p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
        oddsratio = (a * d) / (b * c) if b * c != 0 else np.inf
        results.append(
            {
                "term": term,
                "pvalue": p,
                "oddsratio": oddsratio,
                "overlap": a,
                "set_size": m,
            }
        )

    if not results:
        return pd.DataFrame(
            columns=["term", "pvalue", "oddsratio", "overlap", "set_size"]
        )

    df = pd.DataFrame(results).sort_values("pvalue")
    df["padj"] = adjust_pvalues(df["pvalue"].values, method="fdr_bh")
    return df[df["padj"] <= pval_cutoff]


def pathway_methylation_scores(
    beta: pd.DataFrame,
    annotation: pd.DataFrame,
    pathway_db: Dict[str, List[str]],
    method: str = "mean",
) -> pd.DataFrame:
    """
    Collapse CpG-level beta values into pathway-level methylation scores per sample.

    Useful for downstream pathway-activity modeling or visualization.

    Parameters
    ----------
    beta : pd.DataFrame
        Beta-value matrix (CpGs × samples).
    annotation : pd.DataFrame
        Mapping from CpG identifiers to gene symbols (index → gene).
    pathway_db : dict[str, list[str]]
        Dictionary of pathways → list of associated genes.
    method : {"mean", "median", "sum"}, default "mean"
        Aggregation function applied across CpGs belonging to each pathway.

    Returns
    -------
    pd.DataFrame
        Pathways × samples matrix of aggregated methylation scores.
    """
    if beta is None or beta.empty:
        logger.warning("Returned empty dataframe because beta is invalid")
        return pd.DataFrame()

    res = {}
    for pathway, genes in pathway_db.items():
        cpgs_for_pathway = annotation.index[annotation["gene"].isin(genes)]
        if len(cpgs_for_pathway) == 0:
            continue
        sub = beta.loc[cpgs_for_pathway]
        if method == "mean":
            res[pathway] = sub.mean(axis=0)
        elif method == "median":
            res[pathway] = sub.median(axis=0)
        elif method == "sum":
            res[pathway] = sub.sum(axis=0)
        else:
            raise ValueError(f"Unsupported method '{method}'")

    if not res:
        logger.warning("Returned empty dataframe because pathway_db is invalid")
        return pd.DataFrame()
    return pd.DataFrame(res)


def correlate_methylation_expression(
    beta: pd.DataFrame,
    expression: pd.DataFrame,
    gene_map: Optional[Dict[str, str]] = None,
    method: str = "pearson",
    override_index_alignment: bool = False,
) -> pd.DataFrame:
    """
    Compute sample-wise correlation between CpG methylation and gene expression.

    Supports both one-to-one (same index) and many-to-one (custom CpG→gene \
    mapping) scenarios.

    Parameters
    ----------
    beta, expression : pd.DataFrame
        Methylation (CpGs × samples) and expression (genes × samples) matrices.
    gene_map : dict or None
        Explicit mapping from CpG ID → gene symbol (required for cis-analysis).
    method : {"pearson", "spearman"}, default "pearson"
        Correlation coefficient to compute.
    override_index_alignment : bool, default False
        Proceed even with <2 overlapping samples (useful for exploratory checks).

    Returns
    -------
    pd.DataFrame
        Correlation results with columns ``r`` and ``pval`` (indexed by CpG \
        and optionally gene).
    """
    if beta is None or beta.empty or expression is None or expression.empty:
        return pd.DataFrame(columns=["r", "pval"])

    samples = beta.columns.intersection(expression.columns)
    if len(samples) < 2 and not override_index_alignment:
        logger.warning(
            f"Only {len(samples)} overlapping samples; correlation skipped. "
            "Set override_index_alignment=True to force computation."
        )
        return pd.DataFrame(columns=["r", "pval"])

    beta = beta[samples]
    expression = expression[samples]
    records = []

    if gene_map:
        for cpg, gene in gene_map.items():
            if cpg not in beta.index or gene not in expression.index:
                continue
            x = beta.loc[cpg].values
            y = expression.loc[gene].values
            r, p = (
                stats.pearsonr(x, y) if method == "pearson" else stats.spearmanr(x, y)
            )
            records.append(
                {"feature": cpg, "gene": gene, "r": float(r), "pval": float(p)}
            )
        return pd.DataFrame(records).set_index(["feature", "gene"])
    else:
        common = beta.index.intersection(expression.index)
        for f in common:
            x = beta.loc[f].values
            y = expression.loc[f].values
            r, p = (
                stats.pearsonr(x, y) if method == "pearson" else stats.spearmanr(x, y)
            )
            records.append({"feature": f, "r": float(r), "pval": float(p)})
        return pd.DataFrame(records).set_index("feature")


def liftover_coordinates(
    regions: pd.DataFrame,
    from_build: str = "hg19",
    to_build: str = "hg38",
    chr_col: str = "chr",
    start_col: str = "start",
    end_col: str = "end",
) -> pd.DataFrame:
    """
    Convert genomic regions between genome assemblies using pyliftover \
    (e.g., hg19 ↔ hg38).

    Handles both single-base and interval coordinates with per-row success reporting.

    Parameters
    ----------
    regions : pd.DataFrame
        Table containing chromosome, start, and end columns.
    from_build, to_build : str
        Source and target genome builds (default: hg19 → hg38).
    chr_col, start_col, end_col : str
        Column names for genomic coordinates.

    Returns
    -------
    pd.DataFrame
        Original table augmented with:
        ``lifted_chr``, ``lifted_start``, ``lifted_end``
        ``lifted`` (boolean indicating successful conversion)

    Raises
    ------
    RuntimeError
        If ``pyliftover`` is not installed.
    """
    if LiftOver is None:
        raise RuntimeError("pyliftover is not installed; cannot perform liftover")
    if regions is None or regions.empty:
        logger.warning("Returned empty dataframe because regions is invalid")
        return pd.DataFrame()

    lifter = LiftOver(from_build, to_build)
    out_rows = []

    for _, r in regions.iterrows():
        chrom = _clean_chr(r[chr_col]).replace("chr", "")
        start = int(r[start_col])
        end = int(r[end_col])

        lifted_start = lifter.liftover(chrom, start)
        lifted_end = lifter.liftover(chrom, end)

        if lifted_start and lifted_end:
            s_chrom, s_pos, _, _ = lifted_start[0]
            e_chrom, e_pos, _, _ = lifted_end[0]
            out_rows.append(
                {
                    "lifted_chr": f"chr{s_chrom}",
                    "lifted_start": int(s_pos),
                    "lifted_end": int(e_pos),
                    "lifted": True,
                }
            )
        else:
            out_rows.append(
                {
                    "lifted_chr": np.nan,
                    "lifted_start": np.nan,
                    "lifted_end": np.nan,
                    "lifted": False,
                }
            )

    result = pd.DataFrame(out_rows, index=regions.index)
    return pd.concat([regions, result], axis=1)
