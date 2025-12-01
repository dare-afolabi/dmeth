#!/usr/bin/env python
# coding: utf-8

"""
Planner configuration manager for DNA methylation studies.

Provides a thread-safe singleton that centralizes platforms, experimental
designs, cost components, timeline phases, and global settings with full
validation and multiple format support.
"""

import json
import os
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, validator

from dmeth.utils.logger import logger

# Optional dependency imports with better error messages
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False
    logger.warning("PyYAML not installed. Install with: pip install pyyaml")

try:
    import toml

    TOML_AVAILABLE = True
except ImportError:
    toml = None
    TOML_AVAILABLE = False
    logger.warning("toml not installed. Install with: pip install toml")

try:
    from statsmodels.stats.power import TTestIndPower, TTestPower

    POWER_ANALYSIS_AVAILABLE = True
except ImportError:
    TTestPower = None
    TTestIndPower = None
    POWER_ANALYSIS_AVAILABLE = False
    logger.warning(
        "statsmodels not installed. Power analysis unavailable. "
        "Install with: pip install statsmodels"
    )


# Pydantic Schemas with Enhanced Validation


class PlatformSchema(BaseModel):
    """Configuration schema for a methylation array platform."""

    name: str
    manufacturer: Optional[str] = None
    n_cpgs: int = Field(..., ge=1, description="Number of CpG sites")
    cost_per_sample: float = Field(..., ge=0.0, description="Cost per sample in USD")
    processing_days: int = Field(..., ge=0, description="Processing time in days")
    dna_required_ng: Optional[float] = Field(None, ge=0)
    coverage: Optional[str] = None
    release_year: Optional[int] = Field(None, ge=1990, le=2030)
    status: Optional[str] = None
    recommended: bool = False
    notes: Optional[str] = None

    @validator("cost_per_sample")
    def validate_cost(cls, v) -> float:
        if v > 1_000_000:
            raise ValueError("Cost per sample seems unrealistic (>$1M)")
        return v


class DesignSchema(BaseModel):
    """Configuration schema for experimental design."""

    name: str
    description: Optional[str] = None
    n_groups: int = Field(..., ge=1, description="Number of experimental groups")
    paired: bool = False
    complexity: Optional[str] = None
    min_n_recommended: int = Field(..., ge=1)
    power_adjustment: float = Field(..., gt=0.0, le=2.0)
    analysis_method: Optional[str] = None
    example_uses: Optional[List[str]] = None

    @validator("power_adjustment")
    def validate_power_adjustment(cls, v) -> float:
        if v > 1.5:
            logger.warning(f"Unusually high power_adjustment: {v}")
        return v


class CostComponentSchema(BaseModel):
    """Configuration schema for cost components."""

    cost: float = Field(..., ge=0.0)
    unit: str = Field(..., regex="^(per_sample|per_cpg|fixed)$")
    description: Optional[str] = None
    optional: bool = False
    applies_to: Optional[List[str]] = None


class TimelinePhaseSchema(BaseModel):
    """Configuration schema for timeline phases."""

    name: Optional[str] = None
    base_duration_days: float = Field(..., gt=0)
    scaling_factor: float = Field(0.0, ge=0.0)
    batch_adjustment: float = Field(0.0, ge=0.0)
    description: Optional[str] = None
    critical: bool = False
    optional: bool = False
    parallelizable: bool = False


class GlobalSettingsSchema(BaseModel):
    """Global configuration settings."""

    contingency_buffer_percent: float = Field(10.0, ge=0.0, le=100.0)
    default_mcp_method: str = Field("fdr", regex="^(bonferroni|fdr|none)$")


class PlannerConfigModel(BaseModel):
    """Complete planner configuration model."""

    platforms: Dict[str, PlatformSchema]
    designs: Dict[str, DesignSchema]
    cost_components: Dict[str, CostComponentSchema]
    timeline_phases: Dict[str, TimelinePhaseSchema]
    global_settings: GlobalSettingsSchema = Field(default_factory=GlobalSettingsSchema)

    @validator("platforms")
    def validate_platforms(cls, v) -> Dict[str, PlatformSchema]:
        if not v:
            raise ValueError("At least one platform must be defined")
        return v

    @validator("designs")
    def validate_designs(cls, v) -> Dict[str, DesignSchema]:
        if not v:
            raise ValueError("At least one design must be defined")
        return v


# Utility Functions


def _atomic_write(path: Union[str, Path], content: Union[str, bytes]) -> None:
    """
    Write content to a file atomically using a temporary file.

    Parameters
    ----------
    path : str or Path
        Destination file path.
    content : str or bytes
        Content to write.

    Raises
    ------
    OSError
        If the write operation fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to bytes if needed
    content_bytes = content.encode("utf-8") if isinstance(content, str) else content

    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=str(path.parent), delete=False
        ) as tmp:
            tmp_file = Path(tmp.name)
            tmp.write(content_bytes)
            tmp.flush()
            os.fsync(tmp.fileno())

        # Atomic replace
        os.replace(str(tmp_file), str(path))
        logger.debug(f"Successfully wrote file: {path}")

    except Exception as e:
        logger.error(f"Failed to write file {path}: {e}")
        if tmp_file and tmp_file.exists():
            try:
                tmp_file.unlink()
            except Exception as e:
                raise ValueError(
                    f"Failed to write file {path} and unlink temp file: {e}"
                )


def _read_python_literal(path: Path) -> Dict[str, Any]:
    """
    Parse a Python literal dictionary from a file.

    Parameters
    ----------
    path : Path
        Path to the file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    ValueError
        If the file doesn't contain a valid dictionary.
    """
    import ast

    try:
        text = path.read_text(encoding="utf-8")
        obj = ast.literal_eval(text)
    except Exception as e:
        raise ValueError(f"Failed to parse Python literal from {path}: {e}")

    if not isinstance(obj, dict):
        raise ValueError(f"File {path} must contain a dictionary at top level")

    return obj


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update base dictionary with values from updates.

    Parameters
    ----------
    base : dict
        Base dictionary (modified in place).
    updates : dict
        Updates to apply.

    Returns
    -------
    dict
        The updated base dictionary.
    """
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


# PlannerConfig Singleton Class
class PlannerConfig:
    """
    Thread-safe singleton for planner configuration management.

    This class manages all configuration for DNA methylation study planning,
    including platforms, experimental designs, costs, and timelines.

    Parameters
    ----------
    config_file : str or Path, optional
        Configuration file to load on initialization.
    """

    _instance: Optional["PlannerConfig"] = None
    _lock = RLock()

    def __new__(cls, config_file: Optional[Union[str, Path]] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        # Prevent re-initialization
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._cfg_path: Optional[Path] = None
        self._data_lock = RLock()
        self._raw: Dict[str, Any] = {}
        self.model: Optional[PlannerConfigModel] = None

        # Try to load and merge sidecar config file
        self._load_sidecar_config()

        # Load user config if provided (overrides sidecar config)
        if config_file is not None:
            try:
                self.load_file(config_file)
                logger.info(f"Loaded user config from {config_file}")
            except Exception as e:
                logger.error(f"Failed to load config file {config_file}: {e}")
                raise

        # Validate and set
        self._validate_and_set(self._raw)
        self._initialized = True

    def _load_sidecar_config(self) -> None:
        """Load configuration from a sidecar file if it exists."""
        candidates = [
            files(__package__) / "defaults.json",
            Path.cwd() / "defaults.json",
        ]
        for sidecar_path in candidates:
            if sidecar_path.exists():
                loaded = self._load_by_format(sidecar_path, ".json", allow_excel=False)
                self._raw = loaded
                logger.info(f"Loaded sidecar config from {sidecar_path}")
                return
        raise FileNotFoundError(
            "No defaults.json found in either config or working directory"
        )

    @contextmanager
    def _transaction(self):
        """
        Context manager for atomic configuration updates.

        Creates a backup of the current configuration state. If an exception
        occurs during the transaction, the backup is restored, ensuring
        atomic updates.

        Yields
        ------
        None

        Examples
        --------
        >>> with config._transaction():
        ...     config.platforms['new_platform'] = {...}
        ...     config._validate_and_set(config._raw)
        """
        with self._data_lock:
            backup = deepcopy(self._raw)
            backup_platforms = (
                deepcopy(self.platforms) if hasattr(self, "platforms") else None
            )
            backup_designs = (
                deepcopy(self.designs) if hasattr(self, "designs") else None
            )
            backup_costs = (
                deepcopy(self.cost_components)
                if hasattr(self, "cost_components")
                else None
            )
            backup_timeline = (
                deepcopy(self.timeline_phases)
                if hasattr(self, "timeline_phases")
                else None
            )
            backup_settings = (
                deepcopy(self.global_settings)
                if hasattr(self, "global_settings")
                else None
            )

            try:
                yield
            except Exception:
                # Restore all state on error
                self._raw = backup
                if backup_platforms is not None:
                    self.platforms = backup_platforms
                if backup_designs is not None:
                    self.designs = backup_designs
                if backup_costs is not None:
                    self.cost_components = backup_costs
                if backup_timeline is not None:
                    self.timeline_phases = backup_timeline
                if backup_settings is not None:
                    self.global_settings = backup_settings
                raise

    # Loading and Saving
    def load_file(self, path: Union[str, Path], allow_excel: bool = True) -> None:
        """
        Load configuration from a file.

        Supported formats: JSON, YAML, TOML, Python literal, Excel, CSV.

        Parameters
        ----------
        path : str or Path
            Path to configuration file.
        allow_excel : bool, default True
            Whether to allow Excel/CSV files.

        Raises
        ------
        FileNotFoundError
            If file doesn't exist.
        ValueError
            If file format is unsupported or invalid.
        ValidationError
            If configuration fails validation.
        """
        path = Path(path).resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}\n"
                f"Current directory: {Path.cwd()}"
            )

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        if not os.access(path, os.R_OK):
            raise PermissionError(f"Cannot read file: {path}")

        logger.info(f"Loading configuration from {path}")

        ext = path.suffix.lower()
        loaded = self._load_by_format(path, ext, allow_excel)

        if not isinstance(loaded, dict):
            raise ValueError(f"Configuration must be a dictionary, got {type(loaded)}")

        with self._transaction():
            merged = deepcopy(self._raw)
            _deep_update(merged, loaded)
            self._validate_and_set(merged)
            self._cfg_path = path

        logger.info(f"Successfully loaded configuration from {path}")

    def _load_by_format(
        self, path: Path, ext: str, allow_excel: bool
    ) -> Dict[str, Any]:
        """Load configuration based on file format."""

        if ext == ".json":
            return json.loads(path.read_text(encoding="utf-8"))

        elif ext in (".yml", ".yaml"):
            if not YAML_AVAILABLE:
                raise RuntimeError("PyYAML required to load YAML files")
            return yaml.safe_load(path.read_text(encoding="utf-8"))

        elif ext == ".toml":
            if not TOML_AVAILABLE:
                raise RuntimeError("toml required to load TOML files")
            return toml.loads(path.read_text(encoding="utf-8"))

        elif ext in (".py", ".txt"):
            return _read_python_literal(path)

        elif ext in (".xlsx", ".xls", ".csv"):
            if not allow_excel:
                raise RuntimeError("Excel/CSV loading is disabled")
            return self._load_from_table_file(path)

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def _load_from_table_file(self, path: Path) -> Dict[str, Any]:
        """
        Load configuration from Excel or CSV table file.

        Parameters
        ----------
        path : Path
            Path to table file.

        Returns
        -------
        dict
            Loaded configuration sections.
        """
        ext = path.suffix.lower()
        result: Dict[str, Any] = {}

        if ext == ".csv":
            result = self._load_csv_config(path)
        else:
            result = self._load_excel_config(path)

        if not result:
            raise ValueError(f"No valid configuration tables found in {path}")

        return result

    def _load_csv_config(self, path: Path) -> Dict[str, Any]:
        """Load configuration from a CSV file."""
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower().str.strip()

        # Detect table type by columns
        if {"platform_id", "n_cpgs", "cost_per_sample"}.issubset(df.columns):
            df = df.set_index("platform_id")
            return {"platforms": df.to_dict(orient="index")}

        elif {"design_id", "n_groups"}.issubset(df.columns):
            df = df.set_index("design_id")
            return {"designs": df.to_dict(orient="index")}

        elif {"component_id", "cost", "unit"}.issubset(df.columns):
            df = df.set_index("component_id")
            return {"cost_components": df.to_dict(orient="index")}

        else:
            raise ValueError("CSV format not recognized")

    def _load_excel_config(self, path: Path) -> Dict[str, Any]:
        """Load configuration from an Excel file."""
        xl = pd.ExcelFile(path)
        result = {}

        # Map sheet names to config sections
        sheet_mapping = {
            "platforms": ("platforms", "platform_id"),
            "designs": ("designs", "design_id"),
            "costs": ("cost_components", "component_id"),
            "timeline": ("timeline_phases", "phase_id"),
        }

        sheets_lower = {name.lower(): name for name in xl.sheet_names}

        for sheet_key, (config_key, id_col) in sheet_mapping.items():
            if sheet_key in sheets_lower:
                sheet_name = sheets_lower[sheet_key]
                df = pd.read_excel(xl, sheet_name=sheet_name)
                df = self._normalize_dataframe(df)

                if id_col in df.columns:
                    df = df.set_index(id_col)

                result[config_key] = df.to_dict(orient="index")

        return result

    @staticmethod
    def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and data types."""
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # Coerce numeric columns
        numeric_cols = [
            "n_cpgs",
            "cost_per_sample",
            "cost",
            "base_duration_days",
            "processing_days",
            "n_groups",
            "min_n_recommended",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def save_file(self, path: Union[str, Path], fmt: Optional[str] = None) -> None:
        """
        Save current configuration to a file.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        fmt : str, optional
            Format (json, yaml, toml). Inferred from extension if not provided.

        Raises
        ------
        ValueError
            If format is unsupported.
        """
        path = Path(path)
        fmt = (fmt or path.suffix.lstrip(".")).lower()

        with self._data_lock:
            data = self._export_config()

        if fmt in ("json", ""):
            content = json.dumps(data, indent=2)
            _atomic_write(path, content)

        elif fmt in ("yml", "yaml"):
            if not YAML_AVAILABLE:
                raise RuntimeError("PyYAML required to write YAML")
            content = yaml.safe_dump(data, sort_keys=False)
            _atomic_write(path, content)

        elif fmt == "toml":
            if not TOML_AVAILABLE:
                raise RuntimeError("toml required to write TOML")
            content = toml.dumps(data)
            _atomic_write(path, content)

        else:
            raise ValueError(f"Unsupported format: {fmt}")

        logger.info(f"Configuration saved to {path}")
        self._cfg_path = path

    def _export_config(self) -> Dict[str, Any]:
        """Export current configuration as dictionary."""
        return {
            "platforms": self.platforms,
            "designs": self.designs,
            "cost_components": self.cost_components,
            "timeline_phases": self.timeline_phases,
            "global_settings": self.global_settings,
        }

    # Validation
    def _validate_and_set(self, raw: Dict[str, Any]) -> None:
        """
        Validate configuration and update internal state.

        Parameters
        ----------
        raw : dict
            Raw configuration dictionary.

        Raises
        ------
        ValidationError
            If validation fails.
        """
        try:
            validated = PlannerConfigModel(**raw)
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

        with self._data_lock:
            self.model = validated
            self.platforms = {k: v.dict() for k, v in validated.platforms.items()}
            self.designs = {k: v.dict() for k, v in validated.designs.items()}
            self.cost_components = {
                k: v.dict() for k, v in validated.cost_components.items()
            }
            self.timeline_phases = {
                k: v.dict() for k, v in validated.timeline_phases.items()
            }
            self.global_settings = validated.global_settings.dict()
            self._raw = raw

    # Query Methods
    def list_platforms(self, recommended_only: bool = False) -> pd.DataFrame:
        """
        List all platforms as a DataFrame.

        Parameters
        ----------
        recommended_only : bool, default False
            Return only recommended platforms.

        Returns
        -------
        pd.DataFrame
            Platform information.
        """
        with self._data_lock:
            df = pd.DataFrame.from_dict(self.platforms, orient="index")
            if recommended_only and "recommended" in df.columns:
                df = df[df["recommended"]]
            return df.copy()

    def get_platform(self, platform_id: str) -> Dict[str, Any]:
        """
        Get configuration for a specific platform.

        Parameters
        ----------
        platform_id : str
            Platform identifier.

        Returns
        -------
        dict
            Platform configuration.

        Raises
        ------
        KeyError
            If platform not found.
        """
        with self._data_lock:
            if platform_id not in self.platforms:
                available = ", ".join(self.platforms.keys())
                raise KeyError(
                    f"Platform '{platform_id}' not found. "
                    f"Available platforms: {available}"
                )
            return self.platforms[platform_id].copy()

    def list_designs(self) -> pd.DataFrame:
        """List all experimental designs as a DataFrame."""
        with self._data_lock:
            return pd.DataFrame.from_dict(self.designs, orient="index")

    def get_design(self, design_id: str) -> Dict[str, Any]:
        """
        Get configuration for a specific design.

        Parameters
        ----------
        design_id : str
            Design identifier.

        Returns
        -------
        dict
            Design configuration.

        Raises
        ------
        KeyError
            If design not found.
        """
        with self._data_lock:
            if design_id not in self.designs:
                available = ", ".join(self.designs.keys())
                raise KeyError(
                    f"Design '{design_id}' not found. "
                    f"Available designs: {available}"
                )
            return self.designs[design_id].copy()

    def get_cost_components(
        self, platform: Optional[str] = None, include_optional: bool = True
    ) -> Dict[str, Dict]:
        """
        Get cost components, optionally filtered.

        Parameters
        ----------
        platform : str, optional
            Filter by platform applicability.
        include_optional : bool, default True
            Include optional components.

        Returns
        -------
        dict
            Cost components.
        """
        with self._data_lock:
            result = {}
            for k, v in self.cost_components.items():
                if not include_optional and v.get("optional", False):
                    continue

                applies_to = v.get("applies_to")
                if platform and applies_to and platform not in applies_to:
                    continue

                result[k] = v.copy()

            return result

    # Modification Methods
    def update_platform_cost(self, platform_id: str, new_cost: float) -> None:
        """
        Update the cost per sample for a platform.

        Parameters
        ----------
        platform_id : str
            Platform identifier.
        new_cost : float
            New cost per sample.

        Raises
        ------
        KeyError
            If platform not found.
        ValueError
            If cost is negative.
        """
        if new_cost < 0:
            raise ValueError("Cost must be non-negative")

        with self._transaction():
            if platform_id not in self.platforms:
                raise KeyError(f"Platform '{platform_id}' not found")

            self.platforms[platform_id]["cost_per_sample"] = float(new_cost)
            self._raw.setdefault("platforms", {})[platform_id] = self.platforms[
                platform_id
            ]

            logger.info(f"Updated cost for {platform_id}: ${new_cost:.2f}")

    def add_custom_platform(
        self, platform_id: str, platform_info: Dict[str, Any]
    ) -> None:
        """
        Add a new custom platform.

        Parameters
        ----------
        platform_id : str
            Unique identifier.
        platform_info : dict
            Platform configuration.

        Raises
        ------
        ValueError
            If validation fails or platform already exists.
        """
        with self._transaction():
            if platform_id in self.platforms:
                raise ValueError(f"Platform '{platform_id}' already exists")

            # Validate schema
            try:
                PlatformSchema(**platform_info)
            except ValidationError as e:
                raise ValueError(f"Invalid platform configuration: {e}")

            self.platforms[platform_id] = platform_info
            self._raw.setdefault("platforms", {})[platform_id] = platform_info

            logger.info(f"Added custom platform: {platform_id}")

    def update_regional_pricing(
        self, multiplier: float, region: Optional[str] = None
    ) -> None:
        """
        Apply a pricing multiplier to all platforms.

        Parameters
        ----------
        multiplier : float
            Pricing multiplier (must be > 0).
        region : str, optional
            Region name for logging.

        Raises
        ------
        ValueError
            If multiplier is not positive.
        """
        if multiplier <= 0:
            raise ValueError("Multiplier must be positive")

        with self._transaction():
            for _platform_id, platform in self.platforms.items():
                old_cost = float(platform.get("cost_per_sample", 0.0))
                new_cost = round(old_cost * multiplier, 2)
                platform["cost_per_sample"] = new_cost

            region_str = f" for {region}" if region else ""
            logger.info(f"Applied {multiplier:.2f}x pricing multiplier{region_str}")

    def get_platform_by_budget(self, max_cost_per_sample: float) -> pd.DataFrame:
        """
        Find platforms within budget.

        Parameters
        ----------
        max_cost_per_sample : float
            Maximum cost per sample.

        Returns
        -------
        pd.DataFrame
            Platforms within budget.
        """
        df = self.list_platforms()
        return df[df["cost_per_sample"] <= max_cost_per_sample]

    # Analysis Methods
    def calculate_sample_size(
        self,
        design_id: str,
        platform_id: str,
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.8,
        mcp_method: Optional[str] = None,
        max_n: int = 500,
    ) -> Dict[str, Any]:
        """
        Calculate required sample size using power analysis.

        Parameters
        ----------
        design_id : str
            Experimental design identifier.
        platform_id : str
            Platform identifier.
        effect_size : float
            Expected effect size (Cohen's d).
        alpha : float, default 0.05
            Type I error rate.
        power : float, default 0.8
            Desired statistical power.
        mcp_method : str, optional
            Multiple comparison method (bonferroni, fdr, none).
        max_n : int, default 500
            Maximum sample size per group.

        Returns
        -------
        dict
            Sample size calculation results including n_per_group, total_samples,
            effective alpha, etc.

        Raises
        ------
        ValueError
            If effect_size is invalid.
        RuntimeError
            If power analysis is not available.
        """
        if not POWER_ANALYSIS_AVAILABLE:
            raise RuntimeError(
                "Power analysis requires statsmodels. \
                Install with: pip install statsmodels"
            )

        if effect_size <= 0:
            raise ValueError("Effect size must be positive")

        if effect_size > 2.0:
            logger.warning(
                f"Large effect size ({effect_size:.2f}). \
                Verify this is realistic for your study."
            )

        with self._data_lock:
            design = self.get_design(design_id)
            platform = self.get_platform(platform_id)

            paired = bool(design.get("paired", False))
            min_n = int(design.get("min_n_recommended", 3))
            adj_factor = float(design.get("power_adjustment", 1.0))
            n_groups = int(design.get("n_groups", 2))

            n_cpgs = int(platform.get("n_cpgs", 0))
            mcp = mcp_method or self.global_settings.get("default_mcp_method", "fdr")

        # Compute effective alpha
        alpha_eff = alpha
        if mcp == "bonferroni" and n_cpgs > 0:
            alpha_eff = alpha / n_cpgs
            if alpha_eff < 1e-12:
                logger.warning(
                    "Bonferroni correction resulted in extremely small alpha. "
                    "Consider using FDR correction instead."
                )
        elif mcp.lower() == "fdr":
            # For FDR, approximate effective alpha; don't inflate sample size
            alpha_eff = alpha

        # Power calculation
        try:
            solver = TTestPower() if paired else TTestIndPower()
            n_raw = solver.solve_power(
                effect_size=effect_size,
                alpha=alpha_eff,
                power=power,
                alternative="two-sided",
            )
            n_adjusted = n_raw * adj_factor
            n_per_group = int(np.clip(np.ceil(n_adjusted), min_n, max_n))

        except Exception as e:
            logger.warning(f"Power calculation failed: {e}. Using fallback.")
            n_per_group = int(np.clip(np.ceil(min_n * adj_factor), min_n, max_n))

        total_samples = n_per_group * n_groups

        logger.info(
            f"Sample size calc: design={design_id}, platform={platform_id}, "
            f"effect_size={effect_size:.2f}, alpha={alpha:.3f}, "
            f"alpha_eff={alpha_eff:.3e}, "
            f"paired={paired}, n_per_group={n_per_group}, total={total_samples}"
        )

        return {
            "n_per_group": n_per_group,
            "total_samples": total_samples,
            "alpha_nominal": alpha,
            "alpha_effective": alpha_eff,
            "mcp_method": mcp,
            "effect_size": effect_size,
            "power": power,
            "design_id": design_id,
            "platform_id": platform_id,
        }

    def estimate_study_timeline(
        self, n_samples: int, platform_id: str, contingency: bool = True
    ) -> pd.DataFrame:
        """
        Estimate study timeline across all phases.

        Parameters
        ----------
        n_samples : int
            Total number of samples.
        platform_id : str
            Platform identifier.
        contingency : bool, default True
            Apply contingency buffer.

        Returns
        -------
        pd.DataFrame
            Timeline breakdown by phase with total.
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        with self._data_lock:
            platform = self.get_platform(platform_id)
            phases = deepcopy(self.timeline_phases)
            buffer_pct = (
                self.global_settings.get("contingency_buffer_percent", 10.0) / 100.0
            )

        rows = []
        for phase_id, info in phases.items():
            base = float(info.get("base_duration_days", 0.0))
            scaling = float(info.get("scaling_factor", 0.0))
            batch_adj = float(info.get("batch_adjustment", 0.0))

            # Special handling for array processing
            if phase_id == "array_processing":
                base = float(platform.get("processing_days", base))

            # Calculate duration
            duration = base + scaling * n_samples

            # Add batch adjustment
            n_batches = max(1, int(np.ceil(n_samples / 96.0)))
            duration += batch_adj * max(0, n_batches - 1)

            # Apply contingency
            if contingency:
                duration *= 1.0 + buffer_pct

            rows.append(
                {
                    "phase_id": phase_id,
                    "phase_name": info.get("name", phase_id.replace("_", " ").title()),
                    "estimated_days": np.ceil(duration),
                    "critical": info.get("critical", False),
                    "parallelizable": info.get("parallelizable", False),
                }
            )

        df = pd.DataFrame(rows)

        # Calculate total considering parallelization
        serial_days = df.loc[~df["parallelizable"], "estimated_days"].sum()
        parallel_days = df.loc[df["parallelizable"], "estimated_days"].sum()
        total_days = serial_days + parallel_days * 0.5

        # Add total row
        df.loc[len(df)] = {
            "phase_id": "TOTAL",
            "phase_name": "TOTAL",
            "estimated_days": np.ceil(total_days),
            "critical": False,
            "parallelizable": False,
        }

        logger.info(
            f"Study timeline calc: n_samples={n_samples}, \
            platform_id={platform_id}, estimated_days={np.ceil(total_days)}"
        )

        return df

    def estimate_total_cost(
        self,
        n_samples: int,
        platform_id: str,
        include_optional: bool = True,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Estimate total study cost with detailed breakdown.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        platform_id : str
            Platform identifier.
        include_optional : bool, default True
            Include optional cost components.
        validate : bool, default True
            Validate inputs (kept for compatibility).

        Returns
        -------
        dict
            Cost breakdown with total and per-sample costs.

        Raises
        ------
        ValueError
            If n_samples is invalid.
        """
        if not isinstance(n_samples, (int, float)) or n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        n_samples = int(n_samples)

        with self._data_lock:
            platform = self.get_platform(platform_id)
            components = self.get_cost_components(
                platform=platform_id, include_optional=include_optional
            )

        rows = []

        # Platform base cost
        base_cost = float(platform.get("cost_per_sample", 0.0))
        rows.append(
            {
                "component": f"platform:{platform_id}",
                "unit": "per_sample",
                "cost": base_cost * n_samples,
                "description": platform.get("name", platform_id),
            }
        )

        # Additional components
        for comp_id, comp in components.items():
            unit = comp.get("unit", "per_sample")
            cost_val = float(comp.get("cost", 0.0))

            if unit == "per_sample":
                total = cost_val * n_samples
            elif unit == "per_cpg":
                n_cpgs = float(platform.get("n_cpgs", 0))
                total = cost_val * n_cpgs * n_samples
            elif unit == "fixed":
                total = cost_val
            else:
                logger.warning(f"Unknown cost unit '{unit}' for {comp_id}")
                continue

            rows.append(
                {
                    "component": comp_id,
                    "unit": unit,
                    "cost": total,
                    "description": comp.get("description", ""),
                }
            )

        df = pd.DataFrame(rows)
        total_cost = df["cost"].sum()
        per_sample_cost = total_cost / n_samples

        logger.info(
            f"Total cost calc: n_samples={n_samples}, "
            f"platform_id={platform_id}, total_cost="
            f"{total_cost}, per_sample_cost={per_sample_cost}"
        )

        return {
            "components": df,
            "total": total_cost,
            "per_sample": per_sample_cost,
            "n_samples": n_samples,
            "platform": platform_id,
        }

    # Utility Methods
    def reload(self) -> None:
        """Reload configuration from the last loaded file."""
        if self._cfg_path:
            self.load_file(self._cfg_path)
            logger.info(f"Reloaded configuration from {self._cfg_path}")
        else:
            self._validate_and_set(self._raw)
            logger.info("Re-validated configuration (no file path)")

    def migrate(self, migrator: Callable[[Dict], Dict]) -> None:
        """
        Apply a migration function to the configuration.

        Parameters
        ----------
        migrator : callable
            Function that takes current config dict and returns updated config.
        """
        with self._transaction():
            updated = migrator(deepcopy(self._raw))
            _deep_update(self._raw, updated)
            self._validate_and_set(self._raw)
            logger.info("Configuration migrated successfully")

    def to_dict(self) -> Dict[str, Any]:
        """Export current configuration as a dictionary."""
        with self._data_lock:
            return self._export_config()


# Global Singleton Access
_global_planner_config: Optional[PlannerConfig] = None


def get_config() -> PlannerConfig:
    """
    Get the global PlannerConfig singleton instance.

    Returns
    -------
    PlannerConfig
        The global configuration instance.
    """
    global _global_planner_config
    if _global_planner_config is None:
        _global_planner_config = PlannerConfig()
    return _global_planner_config


def reset_config() -> None:
    """Reset the global configuration instance (primarily for testing)."""
    global _global_planner_config
    _global_planner_config = None
    logger.debug("Global configuration reset")


# Convenience Functions


def load_file(path: Union[str, Path], allow_excel: bool = True) -> None:
    """Load configuration from a file. See PlannerConfig.load_file()."""
    get_config().load_file(path=path, allow_excel=allow_excel)


def save_file(path: Union[str, Path], fmt: Optional[str] = None) -> None:
    """Save configuration to a file. See PlannerConfig.save_file()."""
    get_config().save_file(path=path, fmt=fmt)


def reload() -> None:
    """Reload configuration. See PlannerConfig.reload()."""
    get_config().reload()


def migrate(migrator: Callable[[Dict], Dict]) -> None:
    """Apply a migration function. See PlannerConfig.migrate()."""
    get_config().migrate(migrator)