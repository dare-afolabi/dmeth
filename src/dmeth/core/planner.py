#!/usr/bin/env python
# coding: utf-8


"""
High-level planning API for DNA methylation studies.

This module provides a clean, user-friendly interface to the full study planning
capabilities of ``DMeth``. It exposes all configuration queries, cost/timeline
estimators, and sample size calculators as simple top-level functions.

- All functions automatically use the global ``PlannerConfig`` singleton
- (managed in `dmeth.config.config_manager`), so configuration loaded once is \
immediately available everywhere.

Features
--------
- Direct function access: ``list_platforms()``, ``calculate_sample_size()``, etc.
- Zero-boilerplate: no need to manually instantiate or pass config objects
- Real-time updates: any config change (e.g., regional pricing) instantly \
affects all calculations
"""


from typing import Any, Dict, Optional

import pandas as pd

from dmeth.config.config_manager import get_config


# Convenience top-level wrappers
def list_platforms(recommended_only: bool = False) -> pd.DataFrame:
    """
    Return a DataFrame containing all configured platforms.

    Parameters
    ----------
    recommended_only : bool, default False
        If True, only platforms with ``recommended=True`` are returned.

    Returns
    -------
    pandas.DataFrame
        One row per platform, indexed by platform identifier.
    """
    return get_config().list_platforms(recommended_only=recommended_only)


def list_designs() -> pd.DataFrame:
    """
    Return a table of all configured experimental designs.

    Returns
    -------
    pandas.DataFrame
        One row per design.
    """
    return get_config().list_designs()


def get_platform(platform_id: str) -> Dict[str, Any]:
    """
    Retrieve the configuration dictionary for a specific platform.

    Parameters
    ----------
    platform_id : str
        Identifier of the platform.

    Returns
    -------
    dict
        Platform configuration.

    Raises
    ------
    KeyError
        If the platform identifier is unknown.
    """
    return get_config().get_platform(platform_id)


def get_design(design_id: str) -> Dict[str, Any]:
    """
    Retrieve the configuration dictionary for a specific experimental design.

    Parameters
    ----------
    design_id : str
        Identifier of the design.

    Returns
    -------
    dict
        Design configuration.

    Raises
    ------
    KeyError
        If the design identifier is unknown.
    """
    return get_config().get_design(design_id)


def get_cost_components(
    platform: Optional[str] = None, include_optional: bool = True
) -> Dict[str, Dict]:
    """
    Return cost components, optionally filtered by platform or optionality.

    Parameters
    ----------
    platform : str, optional
        Restrict to components whose ``applies_to`` list contains this platform.
    include_optional : bool, default True
        Whether to include components marked as optional.

    Returns
    -------
    dict
        Mapping from component identifier to component dictionary.
    """
    return get_config().get_cost_components(
        platform=platform, include_optional=include_optional
    )


def update_platform_cost(platform_id: str, new_cost: float) -> None:
    """
    Change the per-sample cost of an existing platform.

    Parameters
    ----------
    platform_id : str
        Platform identifier.
    new_cost : float
        New cost per sample (must be non-negative).

    Raises
    ------
    KeyError
        If the platform does not exist.
    """
    return get_config().update_platform_cost(platform_id, new_cost)


def add_custom_platform(platform_id: str, platform_info: Dict[str, Any]) -> None:
    """
    Register a new custom platform after schema validation.

    Parameters
    ----------
    platform_id : str
        Unique identifier for the new platform.
    platform_info : dict
        Platform data conforming to ``PlatformSchema``.

    Raises
    ------
    ValueError
        If the provided data fails validation.
    """
    return get_config().add_custom_platform(platform_id, platform_info)


def update_regional_pricing(multiplier: float, region: Optional[str] = None) -> None:
    """
    Apply a uniform multiplicative adjustment to all platform costs.

    Parameters
    ----------
    multiplier : float
        Factor by which current costs are multiplied (must be > 0).
    region : str, optional
        Descriptive label for logging purposes.

    Raises
    ------
    ValueError
        If ``multiplier`` is not positive.
    """
    return get_config().update_regional_pricing(multiplier, region)


def get_platform_by_budget(max_cost_per_sample: float) -> pd.DataFrame:
    """
    Return platforms whose per-sample cost does not exceed a budget limit.

    Parameters
    ----------
    max_cost_per_sample : float
        Maximum acceptable cost per sample.

    Returns
    -------
    pandas.DataFrame
        Subset of platforms meeting the budget constraint.
    """
    return get_config().get_platform_by_budget(max_cost_per_sample)


def calculate_sample_size(*args, **kwargs):
    """
    Compute required sample size per group using power analysis.

    - The calculation accounts for the chosen experimental design, platform
    - CpG count (for Bonferroni correction), and any design-specific power
    adjustment factor.

    Parameters
    ----------
    design_id : str
        Identifier of the experimental design.
    platform_id : str
        Identifier of the methylation platform.
    effect_size : float
        Expected Cohen's d.
    alpha : float, default 0.05
        Nominal type-I error rate.
    power : float, default 0.8
        Desired statistical power.
    mcp_method : str, optional
        Multiple-comparison correction method (``bonferroni``, ``fdr``, ``none``).
    max_n : int, default 500
        Upper bound on calculated sample size per group.

    Returns
    -------
    dict
        Contains ``n_per_group``, ``total_samples``, effective alpha, etc.
    """
    return get_config().calculate_sample_size(*args, **kwargs)


def estimate_study_timeline(*args, **kwargs):
    """
    Produce a detailed timeline estimate for the entire study.

    The estimate respects phase-specific scaling, batch adjustments,
    parallelisability, and an optional global contingency buffer.

    Parameters
    ----------
    n_samples : int
        Total number of samples to be processed.
    platform_id : str
        Platform identifier (affects array-processing duration).
    contingency : bool, default True
        Apply the global contingency buffer percentage.

    Returns
    -------
    pandas.DataFrame
        One row per phase plus a summary ``TOTAL`` row.
    """
    return get_config().estimate_study_timeline(*args, **kwargs)


def estimate_total_cost(*args, **kwargs):
    """
    Calculate the complete study cost and provide a detailed breakdown.

    Handles per-sample, per-CpG, and fixed cost components, with optional
    inclusion of components marked as ``optional``.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    platform_id : str
        Platform identifier.
    include_optional : bool, default True
        Include optional cost components.
    validate : bool, default True
        Perform unit validation (kept for backward compatibility).

    Returns
    -------
    dict
        Contains a ``components`` DataFrame, ``total`` cost,
        ``per_sample`` cost, and metadata.
    """
    return get_config().estimate_total_cost(*args, **kwargs)
