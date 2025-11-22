#!/usr/bin/env python
# coding: utf-8


"""
Plotting utilities for downstream DNA methylation analysis.

This module provides a comprehensive set of visualisation functions for quality control,
differential methylation analysis, dimensionality reduction, \
variance shrinkage diagnostics,
and publication-ready summary figures. Functions operate on pandas \
DataFrames, ProcessedData
objects, or differential methylation results and return standardised \
matplotlib (or optionally
Plotly) figures.

Features
--------
- Standardised figure creation
- Enhanced volcano plots with automatic significance colouring and annotation
- P-value QQ plots and histograms
- Sample-level PCA, t-SNE, and UMAP embeddings (static or interactive)
- Variance shrinkage diagnostics (limma-style)
- Mean-difference bar plots for top loci
- Multi-panel QC reports via plot_stage()
- Combined DMS/DMR summary visualisation (volcano + manhattan + heatmap)
"""


from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# from sklearn.preprocessing import quantile_transform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

from dmeth.core.downstream.annotation import correlate_methylation_expression
from dmeth.core.downstream.helpers import _clean_chr
from dmeth.io.data_utils import ProcessedData
from dmeth.utils.logger import logger

try:
    import plotly.graph_objects as go
except ImportError:
    go = None
    plotly = None
    logger.warning(
        "plotly not installed. \
        plotly interactive figures for embedding disabled."
    )

try:
    import umap
except ImportError:
    umap = None
    logger.warning("umap-learn not installed. UMAP QC embedding disabled.")

try:
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None
    logger.warning("sklearn.manifold not installed. TSNE QC embedding disabled.")

sns.set(style="whitegrid")


def _new_fig(figsize: tuple = (10, 6), dpi: int = 300) -> plt.Figure:
    """
    Create a new matplotlib figure with standardised DMeth plotting settings.

    Parameters
    ----------
    figsize : tuple[int, int], default (10, 6)
        Width and height of the figure in inches.
    dpi : int, default 300
        Resolution of the figure in dots per inch.

    Returns
    -------
    plt.Figure
        A fresh matplotlib Figure instance with the requested dimensions and DPI.
    """
    return plt.figure(figsize=figsize, dpi=dpi)


def plot_volcano(
    res: pd.DataFrame,
    lfc_col: str = "logFC",
    pval_col: str = "pval",
    alpha: float = 0.7,
    lfc_thresh: float = 1.0,
    pval_thresh: float = 0.05,
    top_n: int = 10,
    dpi: int = 300,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Create an enhanced volcano plot with top hits annotated.

    Parameters
    ----------
    res : pd.DataFrame
        Differential results with logFC and p-values
    lfc_col : str
        Column name for log fold change
    pval_col : str
        Column name for p-value
    alpha : float
        Point transparency
    lfc_thresh : float
        Threshold for logFC significance
    pval_thresh : float
        Threshold for p-value significance
    top_n : int
        Number of top hits to annotate
    dpi : int
        Figure resolution
    save_path : str or Path, optional
        Path to save figure

    Returns
    -------
    plt.Figure
        Volcano plot figure
    """
    res = res.copy()
    pvals = res[pval_col].clip(lower=np.nextafter(0, 1))
    res["neg_log10_p"] = -np.log10(pvals)

    conditions = [
        (res[pval_col] < pval_thresh) & (res[lfc_col] >= lfc_thresh),
        (res[pval_col] < pval_thresh) & (res[lfc_col] <= -lfc_thresh),
    ]
    choices = ["Hypermethylated", "Hypomethylated"]
    res["Group"] = np.select(conditions, choices, default="Not significant")
    color_map = {
        "Hypermethylated": "red",
        "Hypomethylated": "blue",
        "Not significant": "grey",
    }

    fig = _new_fig((10, 7), dpi)
    ax = fig.add_subplot(111)

    for group, color in color_map.items():
        subset = res[res["Group"] == group]
        ax.scatter(
            subset[lfc_col],
            subset["neg_log10_p"],
            c=color,
            alpha=alpha,
            s=30,
            edgecolor="k",
            linewidth=0.3,
            label=group,
        )

    ax.axvline(-lfc_thresh, color="black", linestyle="--", alpha=0.6)
    ax.axvline(lfc_thresh, color="black", linestyle="--", alpha=0.6)
    ax.axhline(-np.log10(pval_thresh), color="black", linestyle="--", alpha=0.6)

    if top_n > 0:
        sig = res[res["Group"] != "Not significant"]
        if not sig.empty:
            hits = sig.nsmallest(top_n, pval_col)
            for idx, row in hits.iterrows():
                ax.annotate(
                    idx,
                    (row[lfc_col], row["neg_log10_p"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.8,
                    ha="left",
                )

    ax.set_xlabel("Log2 Fold Change")
    ax.set_ylabel("-log₁₀(p-value)")
    ax.set_title("Volcano Plot")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_pvalue_qq(
    res: pd.DataFrame,
    pval_col: str = "pval",
    dpi: int = 300,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Produce a quantile-quantile plot comparing observed versus \
    expected -log10(p-values).

    Parameters
    ----------
    res : pd.DataFrame
        DataFrame containing the p-value column.
    pval_col : str, default "pval"
        Column name of the p-values.
    dpi : int, default 300
        Figure resolution.
    save_path : str or Path, optional
        Destination path for saving the figure.

    Returns
    -------
    plt.Figure
        Q-Q plot figure.
    """
    pvals = res[pval_col].dropna()
    if pvals.empty:
        raise ValueError("No p-values to plot")
    observed = -np.log10(np.sort(pvals))
    n = len(pvals)
    expected = -np.log10(np.linspace(1 / (n + 1), 1 - 1 / (n + 1), n))

    fig = _new_fig((6, 6), dpi)
    ax = fig.add_subplot(111)
    ax.scatter(expected, observed, alpha=0.6, s=15, edgecolor="k", linewidth=0.3)
    ax.plot([0, expected.max()], [0, expected.max()], "r--", lw=2, label="y=x")
    ax.set_xlabel("Expected -log₁₀(p)")
    ax.set_ylabel("Observed -log₁₀(p)")
    ax.set_title("P-value Q-Q Plot")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_stage(
    stage: str,
    M: pd.DataFrame,
    res: Optional[pd.DataFrame] = None,
    metadata: Optional[pd.DataFrame] = None,
    groups_col: str = "Type",
    top_n: int = 10,
    embedding: str = "pca",
    interactive: bool = False,
    save_path: Optional[Union[str, Dict[str, str]]] = None,
    dpi: int = 300,
    **kwargs,
) -> Dict[str, Union[plt.Figure, "plotly.graph_objects.Figure"]]:
    """
    Orchestrate a complete multi-panel QC or analysis visualisation for \
    a given processing stage.

    Supported stages: "qc", "variance", "differential", "top_hits", "correlation".

    Parameters
    ----------
    stage : str
        One of the recognised analysis stages.
    M : pd.DataFrame
        Methylation matrix (features × samples), typically M-values or beta-values.
    res : pd.DataFrame, optional
        Differential methylation results (required for variance, \
        differential, top_hits stages).
    metadata : pd.DataFrame, optional
        Sample annotation table; must contain the grouping column.
    groups_col : str, default "Type"
        Column in metadata used for colour-coding samples.
    top_n : int, default 10
        Number of top features to display in relevant panels.
    embedding : {"pca", "tsne", "umap"}, default "pca"
        Dimensionality-reduction method for sample embedding.
    interactive : bool, default False
        Return Plotly interactive figures where possible.
    save_path : str, Path or dict, optional
        Path or mapping of figure names to paths for automatic saving.
    dpi : int, default 300
        Resolution for static figures.
    **kwargs
        Additional keyword arguments passed to the underlying reducer.

    Returns
    -------
    dict[str, plt.Figure | plotly.graph_objects.Figure]
        Mapping from panel name to the generated figure(s).
    """
    figs: Dict[str, Union[plt.Figure, "plotly.graph_objects.Figure"]] = {}

    # Metadata alignment
    if metadata is not None:
        if groups_col not in metadata.columns:
            raise KeyError(f"groups_col '{groups_col}' not found in metadata")
        metadata = metadata.reindex(M.columns)
        if metadata.isna().any().any():
            raise ValueError("metadata contains NaN after alignment")

    # QC STAGE
    if stage == "qc":
        # Missing per sample
        fig1 = _new_fig((10, 5), dpi)
        ax = fig1.add_subplot(111)
        missing_pct = M.isna().sum(axis=0) / len(M) * 100
        palette = sns.color_palette("husl", n_colors=metadata[groups_col].nunique())
        color_map = dict(zip(metadata[groups_col].unique(), palette))
        colors = metadata[groups_col].map(color_map)
        ax.bar(range(len(missing_pct)), missing_pct.values, color=colors)
        ax.set_ylabel("% Missing")
        ax.set_title("Missing Data per Sample")
        ax.set_xticks(range(len(M.columns)))
        ax.set_xticklabels(M.columns, rotation=90, fontsize=7)
        ax.axhline(10, color="red", linestyle="--", label="10% threshold")
        ax.legend()
        figs["missing"] = fig1

        # Mean M-value
        fig2 = _new_fig((6, 5), dpi)
        ax = fig2.add_subplot(111)
        mean_M = M.mean(axis=0)
        for group in metadata[groups_col].unique():
            mask = metadata[groups_col] == group
            ax.hist(
                mean_M[mask], alpha=0.7, bins=20, label=group, color=color_map[group]
            )
        ax.set_xlabel("Mean M-value")
        ax.set_ylabel("Frequency")
        ax.set_title("Mean M-value Distribution")
        ax.legend()
        figs["mean_dist"] = fig2

        # Variance
        fig3 = _new_fig((10, 5), dpi)
        ax = fig3.add_subplot(111)
        var_M = M.var(axis=0)
        ax.bar(range(len(var_M)), var_M.values, color=colors)
        ax.set_ylabel("Variance")
        ax.set_title("Within-Sample Variance")
        ax.set_xticks(range(len(M.columns)))
        ax.set_xticklabels(M.columns, rotation=90, fontsize=7)
        figs["variance"] = fig3

        # Embedding
        fig4 = _new_fig((7, 6), dpi)
        ax = fig4.add_subplot(111)
        M_emb = M.fillna(M.mean(axis=1), axis=0).T

        if embedding.lower() == "pca":
            reducer = PCA(n_components=2, **kwargs)
            coords = reducer.fit_transform(M_emb)
            xlabel = f"PC1 ({reducer.explained_variance_ratio_[0]*100:.1f}%)"
            ylabel = f"PC2 ({reducer.explained_variance_ratio_[1]*100:.1f}%)"
        elif embedding.lower() == "tsne":
            if TSNE is None:
                raise ImportError("tsne requires `sklearn.manifold`")
            reducer = TSNE(n_components=2, random_state=42, **kwargs)
            coords = reducer.fit_transform(M_emb)
            xlabel, ylabel = "t-SNE1", "t-SNE2"
        elif embedding.lower() == "umap":
            if umap is None:
                raise ImportError("UMAP requires `umap-learn`")
            reducer = umap.UMAP(**kwargs)
            coords = reducer.fit_transform(M_emb)
            xlabel, ylabel = "UMAP1", "UMAP2"
        else:
            raise ValueError("embedding must be 'pca', 'tsne', or 'umap'")

        if interactive and go is not None:
            fig_plotly = go.Figure()
            for group in metadata[groups_col].unique():
                mask = metadata[groups_col] == group
                fig_plotly.add_trace(
                    go.Scatter(
                        x=coords[mask, 0], y=coords[mask, 1], mode="markers", name=group
                    )
                )
            fig_plotly.update_layout(
                title=f"{embedding.upper()} of Samples",
                xaxis_title=xlabel,
                yaxis_title=ylabel,
            )
            figs["embedding"] = fig_plotly
        else:
            for group in metadata[groups_col].unique():
                mask = metadata[groups_col] == group
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    label=group,
                    color=color_map[group],
                    alpha=0.8,
                    s=80,
                    edgecolor="k",
                )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{embedding.upper()} of Samples")
            ax.legend()
            figs["embedding"] = fig4

        # Save all
        if save_path:
            base = Path(save_path) if isinstance(save_path, str) else None
            save_dir = base.parent / f"{base.stem}_plots" if base else None
            if save_dir:
                save_dir.mkdir(exist_ok=True)
                for name, fig in figs.items():
                    try:
                        if interactive and isinstance(fig, go.Figure):
                            fig.write_html(save_dir / f"{name}.html")
                        else:
                            fig.savefig(
                                save_dir / f"{name}.png", dpi=dpi, bbox_inches="tight"
                            )
                    except Exception as e:
                        logger.warning(f"Could not save {name}: {e}")
        if not interactive:
            for fig in figs.values():
                plt.close(fig)
        return figs

    # VARIANCE STAGE
    elif stage == "variance":
        if res is None:
            raise ValueError("res required")
        fig = _new_fig((12, 5), dpi)
        ax1, ax2 = fig.subplots(1, 2)

        mean_expr = res.filter(like="meanM_").mean(axis=1)
        log_s2 = np.log2(np.maximum(res["s2"].values, np.finfo(float).eps))
        log_s2_post = np.log2(res["s2_post"].replace(0, np.finfo(float).eps))

        ax1.scatter(mean_expr, log_s2, alpha=0.3, s=10, label="Raw")
        ax1.scatter(mean_expr, log_s2_post, alpha=0.3, s=10, label="Moderated")
        ax1.set_xlabel("Mean M-value")
        ax1.set_ylabel("log2(variance)")
        ax1.set_title("Mean-Variance Trend")
        ax1.legend()
        ax1.grid(alpha=0.3)

        rank = np.arange(len(res))
        ax2.scatter(rank, np.sqrt(res["s2"]), alpha=0.3, s=10, label="Raw SD")
        ax2.scatter(
            rank, np.sqrt(res["s2_post"]), alpha=0.3, s=10, label="Moderated SD"
        )
        ax2.set_xlabel("Rank")
        ax2.set_ylabel("SD")
        ax2.set_title("Shrinkage")
        ax2.legend()
        ax2.grid(alpha=0.3)

        figs["variance"] = fig
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return figs

    # DIFFERENTIAL STAGE
    elif stage == "differential":
        if res is None:
            raise ValueError("res required")
        figs["volcano"] = plot_volcano(res, top_n=top_n, dpi=dpi, **kwargs)
        figs["qq"] = plot_pvalue_qq(res, dpi=dpi, **kwargs)
        return figs

    # TOP HITS
    elif stage == "top_hits":
        if res is None:
            raise ValueError("res required")
        top_cpgs = res.nsmallest(top_n, "pval").index
        fig = _new_fig((12, 6), dpi)
        ax = fig.add_subplot(111)
        sns.heatmap(
            M.loc[top_cpgs], cmap="RdBu_r", center=0, ax=ax, cbar_kws={"shrink": 0.8}
        )
        ax.set_title(f"Top {top_n} Differentially Methylated CpGs")
        figs["top_hits"] = fig
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return figs

    # CORRELATION / CLUSTERING
    elif stage == "correlation":
        if len(M.columns) > 200:
            logger.warning(
                "Sample correlation on >200 samples may be slow; consider PCA embedding"
            )

        fig1 = _new_fig((10, 8), dpi)
        ax1 = fig1.add_subplot(111)
        corr = M.corr()
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax1, cbar_kws={"shrink": 0.8})
        ax1.set_title("Sample Correlation")
        figs["correlation"] = fig1

        fig2 = _new_fig((12, 6), dpi)
        ax2 = fig2.add_subplot(111)
        linked = linkage(M.T.fillna(M.mean(axis=1)), method="ward")
        dendrogram(linked, labels=M.columns, leaf_rotation=90, ax=ax2)
        ax2.set_title("Sample Clustering")
        figs["dendrogram"] = fig2

        if save_path:
            base = Path(save_path) if isinstance(save_path, str) else None
            save_dir = base.parent / f"{base.stem}_plots" if base else None
            if save_dir:
                save_dir.mkdir(exist_ok=True)
                for name, fig in figs.items():
                    fig.savefig(save_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")

        for f in figs.values():
            plt.close(f)
        return figs

    else:
        raise NotImplementedError(f"Stage '{stage}' not implemented")


def pca_plot(
    data: ProcessedData,
    color_col: str,
    title: str = "PCA of Methylation Data",
    figsize: tuple[int, int] = (7, 6),
) -> None:
    """
    Display a simple PCA projection of samples coloured by a phenotype column.

    Parameters
    ----------
    data : ProcessedData
        Container holding the methylation matrix and phenotype table.
    color_col : str
        Column in ``data.pheno`` used for colouring points.
    title : str, default "PCA of Methylation Data"
        Title shown on the plot.
    figsize : tuple[int, int], default (7, 6)
        Size of the figure in inches.

    Notes
    -----
    The plot is shown immediately with ``plt.show()`` and is not returned.
    """
    if PCA is None:
        raise RuntimeError("scikit-learn required for PCA.")
    if color_col not in data.pheno.columns:
        raise KeyError(f"Color column '{color_col}' not in pheno")

    X = data.M.T.fillna(data.M.median(axis=1))
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)

    categories = data.pheno[color_col].astype("category")
    colors = categories.cat.codes

    plt.figure(figsize=figsize)
    plt.scatter(pcs[:, 0], pcs[:, 1], c=colors, cmap="tab10", alpha=0.8)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.title(title)

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=plt.cm.tab10(i / len(categories.cat.categories)),
            markersize=8,
        )
        for i in range(len(categories.cat.categories))
    ]
    plt.legend(
        handles,
        categories.cat.categories,
        title=color_col,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.tight_layout()
    plt.show()


def plot_shrinkage_diagnostics(
    s2: pd.Series,
    s2_post: pd.Series,
    d0: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Visualise the effect of empirical Bayes variance shrinkage.

    Parameters
    ----------
    s2 : pd.Series
        Original (unmoderated) variance estimates per feature.
    s2_post : pd.Series
        Moderated (shrunk) variance estimates.
    d0 : float, optional
        Prior degrees of freedom from the shrinkage procedure (displayed \
        in title if given).
    save_path : str or Path, optional
        If provided, the figure is saved to this location.

    Returns
    -------
    plt.Figure
        Scatter plot comparing log10(original) vs log10(shrunk) variances.
    """
    s2 = np.asarray(s2, dtype=float)
    s2_post = np.asarray(s2_post, dtype=float)
    fig = _new_fig((7, 5))
    ax = fig.add_subplot(111)
    ax.scatter(np.log10(s2 + 1e-12), np.log10(s2_post + 1e-12), s=6, alpha=0.6)
    ax.plot(ax.get_xlim(), ax.get_xlim(), linestyle="--", color="k", alpha=0.6)
    ax.set_xlabel("log10(original s2)")
    ax.set_ylabel("log10(shrunk s2)")
    if d0 is not None:
        ax.set_title(f"Shrinkage diagnostics (d0={d0:.2f})")
    else:
        ax.set_title("Shrinkage diagnostics")
    plt.tight_layout()
    return fig


def plot_mean_difference(
    beta_group1: pd.DataFrame,
    beta_group2: pd.DataFrame,
    top_n: int = 50,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Bar plot of the largest absolute mean beta differences between two groups.

    Parameters
    ----------
    beta_group1, beta_group2 : pd.DataFrame
        Beta-value matrices (features × samples) for each group.
    top_n : int, default 50
        Number of top differentially methylated loci to display.
    save_path : str or Path, optional
        Destination for saving the figure.

    Returns
    -------
    plt.Figure
        Bar chart of absolute mean differences.
    """
    means1 = beta_group1.mean(axis=1)
    means2 = beta_group2.mean(axis=1)
    diff = (means1 - means2).abs().sort_values(ascending=False).head(top_n)
    fig = _new_fig((8, 5))
    ax = fig.add_subplot(111)
    ax.bar(range(len(diff)), diff.values, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(diff)))
    ax.set_xticklabels(diff.index, rotation=90)
    ax.set_ylabel("Absolute mean difference")
    ax.set_title("Top Beta Differences Between Groups")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def pvalue_histogram(
    pvals: Union[pd.Series, np.ndarray, List[float]],
    bins: int = 50,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Histogram of p-values with a flat null-expectation line for QC assessment.

    Parameters
    ----------
    pvals : array-like
        Collection of p-values.
    bins : int, default 50
        Number of histogram bins.
    save_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    plt.Figure
        Histogram figure.
    """
    p = np.asarray(pvals, dtype=float)
    fig = _new_fig((6, 4))
    ax = fig.add_subplot(111)
    ax.hist(p[~np.isnan(p)], bins=bins, density=False, alpha=0.8)
    ax.plot(
        [0, 1], [len(p[~np.isnan(p)]) / bins] * 2, color="k", linestyle="--", alpha=0.6
    )
    ax.set_xlabel("p-value")
    ax.set_ylabel("count")
    ax.set_title("P-value histogram")
    plt.tight_layout()
    return fig


def visualize_dms(
    res: pd.DataFrame,
    beta: Optional[pd.DataFrame] = None,
    top_n: int = 50,
    volcano: bool = True,
    manhattan: bool = True,
    heatmap: bool = True,
    sample_metadata: Optional[pd.DataFrame] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Optional[plt.Figure]]:
    """
    Produce a standard set of summary plots for differentially methylated sites/regions.

    Optionally creates volcano, Manhattan, and heatmap visualisations.

    Parameters
    ----------
    res : pd.DataFrame
        Differential methylation results.
    beta : pd.DataFrame, optional
        Beta matrix required for the heatmap panel.
    top_n : int, default 50
        Number of top CpGs shown in the heatmap.
    volcano / manhattan / heatmap : bool, default True
        Toggle creation of each panel.
    sample_metadata : pd.DataFrame, optional
        Sample annotation (currently unused but reserved for future clustering).
    save_dir : str or Path, optional
        Directory where individual PNG files are written.

    Returns
    -------
    dict[str, plt.Figure | None]
        Figures for "volcano", "manhattan", and "heatmap" (None if not generated).
    """
    figs: Dict[str, Optional[plt.Figure]] = {
        "volcano": None,
        "manhattan": None,
        "heatmap": None,
    }
    if res is None or res.empty:
        logger.warning("Empty results provided to visualize_dms()")
        return figs

    if volcano:
        try:
            fig = plot_volcano(res)
            figs["volcano"] = fig
            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                fig.savefig(
                    Path(save_dir) / "volcano.png", dpi=300, bbox_inches="tight"
                )
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to create volcano plot: {e}")

    if manhattan:
        try:
            # Simple manhattan: position by chromosomal coordinate
            df = res.copy()
            if {"chr", "pos"}.issubset(df.columns):
                df["chr"] = df["chr"].apply(_clean_chr)
                # map chromosomes to numeric order
                chr_order = {
                    c: i
                    for i, c in enumerate(
                        sorted(
                            df["chr"].unique(), key=lambda x: (x.replace("chr", ""), x)
                        )
                    )
                }
                df["_chr_ord"] = df["chr"].map(chr_order)
                df["_plot_x"] = df["_chr_ord"] * 1e9 + df["pos"]
                fig = _new_fig((12, 4))
                ax = fig.add_subplot(111)
                ax.scatter(
                    df["_plot_x"],
                    -np.log10(df["pval"].clip(lower=np.nextafter(0, 1))),
                    s=6,
                    alpha=0.6,
                )
                ax.set_xlabel("Genomic position (chr concatenated)")
                ax.set_ylabel("-log10(p-value)")
                figs["manhattan"] = fig
                if save_dir:
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    fig.savefig(
                        Path(save_dir) / "manhattan.png", dpi=300, bbox_inches="tight"
                    )
                    plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to create manhattan plot: {e}")

    if heatmap and beta is not None:
        try:
            top = res.nsmallest(min(len(res), top_n), "pval").index.tolist()
            sub = beta.loc[beta.index.intersection(top)]
            if sub.empty:
                logger.warning("Top features not present in beta matrix for heatmap")
            else:
                fig = _new_fig((10, max(4, sub.shape[0] / 4)))
                ax = fig.add_subplot(111)
                # z-score rows
                mat = (sub - sub.mean(axis=1).values.reshape(-1, 1)) / (
                    sub.std(axis=1).replace(0, 1).values.reshape(-1, 1)
                )
                im = ax.imshow(mat, aspect="auto", interpolation="nearest")
                ax.set_yticks(np.arange(mat.shape[0]))
                ax.set_yticklabels(sub.index)
                ax.set_xticks([])
                fig.colorbar(im, ax=ax)
                figs["heatmap"] = fig
                if save_dir:
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    fig.savefig(
                        Path(save_dir) / "heatmap.png", dpi=300, bbox_inches="tight"
                    )
                    plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to create heatmap: {e}")

    return figs


def methylation_expression_heatmap(
    beta: pd.DataFrame,
    expression: pd.DataFrame,
    genes: Optional[List[str]] = None,
    top_n: int = 50,
    sample_metadata: Optional[pd.DataFrame] = None,
    save_path: Optional[Union[str, Path]] = None,
    method: str = "pearson",
) -> plt.Figure:
    """
    Heatmap of methylation-expression correlation coefficients for selected genes/CpGs.

    Parameters
    ----------
    beta : pd.DataFrame
        Methylation beta matrix (features × samples).
    expression : pd.DataFrame
        Gene expression matrix (genes × samples).
    genes : list[str], optional
        Specific features to display; if None the top_n most correlated are used.
    top_n : int, default 50
        Number of top correlations shown when ``genes`` is None.
    sample_metadata : pd.DataFrame, optional
        Currently unused (reserved for future sample-side annotation).
    save_path : str or Path, optional
        Destination for saving the figure.
    method : {"pearson", "spearman"}, default "pearson"
        Correlation method.

    Returns
    -------
    plt.Figure
        Horizontal heatmap of correlation coefficients.
    """
    corr = correlate_methylation_expression(beta, expression, method=method)
    if corr.empty:
        raise RuntimeError("No correlations computed")

    features = (
        genes or corr["r"].abs().sort_values(ascending=False).head(top_n).index.tolist()
    )
    sel = corr.loc[corr.index.intersection(features)]
    fig = _new_fig((10, max(5, len(sel) / 5)))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        sel["r"].values.reshape(-1, 1), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1
    )
    ax.set_yticks(np.arange(len(sel)))
    ax.set_yticklabels(sel.index)
    ax.set_xticks([0])
    ax.set_xticklabels([f"Correlation ({method})"])
    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig
