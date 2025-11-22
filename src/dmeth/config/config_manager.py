#!/usr/bin/env python
# coding: utf-8


"""
Planner configuration manager for DNA methylation studies.

- Provides a thread-safe singleton (`PlannerConfig`) that centralizes \
platforms, experimental designs, cost components, timeline phases, and global settings.
- All configurations are strictly validated with Pydantic models and can be \
atomically loaded from or saved to multiple formats (JSON, YAML, TOML, Python \
literals, Excel/CSV tables).

Features
--------
- Global singleton accessible via `get_config()`
- Atomic file load/save with automatic merging over built-in defaults
- Full Pydantic validation on every load and modification
- Excel/CSV sheet-based overrides (Platforms, Designs, Costs, Timeline)
- Power-analysis-based sample size calculation with Bonferroni/FDR correction
- Timeline and cost estimation helpers
- In-place configuration migration facility
"""


import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from dmeth.utils.logger import logger

try:
    import yaml
except Exception:
    yaml = None
    logger.warning("PyYAML not installed, YAML config files will not be supported.")

try:
    import toml
except Exception:
    toml = None
    logger.warning(
        "toml module not installed, TOML config files will not be supported."
    )

try:
    from statsmodels.stats.power import TTestIndPower, TTestPower
except ImportError:
    TTestPower = None
    TTestIndPower = None


# Pydantic Schemas (strict)


class PlatformSchema(BaseModel):
    name: str
    manufacturer: Optional[str] = None
    n_cpgs: int = Field(..., ge=1)
    cost_per_sample: float = Field(..., ge=0.0)
    processing_days: int = Field(..., ge=0)
    dna_required_ng: Optional[float] = None
    coverage: Optional[str] = None
    release_year: Optional[int] = None
    status: Optional[str] = None
    recommended: bool = False
    notes: Optional[str] = None


class DesignSchema(BaseModel):
    name: str
    description: Optional[str] = None
    n_groups: int = Field(..., ge=1)
    paired: bool = False
    complexity: Optional[str] = None
    min_n_recommended: int = Field(..., ge=1)
    power_adjustment: float = Field(..., gt=0.0)
    analysis_method: Optional[str] = None
    example_uses: Optional[List[str]] = None


class CostComponentSchema(BaseModel):
    cost: float = Field(..., ge=0.0)
    unit: str = Field(..., regex="^(per_sample|per_cpg|fixed)$")
    description: Optional[str] = None
    optional: bool = False
    applies_to: Optional[List[str]] = None


class TimelinePhaseSchema(BaseModel):
    name: Optional[str] = None
    base_duration_days: float = Field(..., gt=0)
    scaling_factor: Optional[float] = 0.0
    batch_adjustment: Optional[float] = 0.0
    description: Optional[str] = None
    critical: bool = False
    optional: bool = False
    parallelizable: bool = False


class GlobalSettingsSchema(BaseModel):
    contingency_buffer_percent: float = Field(10.0, ge=0.0, le=100.0)
    default_mcp_method: str = Field("fdr", regex="^(bonferroni|fdr|none)$")


class PlannerConfigModel(BaseModel):
    platforms: Dict[str, PlatformSchema]
    designs: Dict[str, DesignSchema]
    cost_components: Dict[str, CostComponentSchema]
    timeline_phases: Dict[str, TimelinePhaseSchema]
    global_settings: Optional[GlobalSettingsSchema] = GlobalSettingsSchema()

    @staticmethod
    def _ensure_non_empty_platforms(v):
        if not v or not isinstance(v, dict):
            raise ValueError("platforms must be a non-empty mapping")
        return v


# Defaults (minimal; can be overwritten by file)
DEFAULT_CONFIG = {
    "platforms": {
        "EPIC": {
            "name": "MethylationEPIC",
            "manufacturer": "Illumina",
            "n_cpgs": 866895,
            "cost_per_sample": 500.0,
            "processing_days": 7,
            "dna_required_ng": 500.0,
            "coverage": "Enhanced",
            "release_year": 2016,
            "status": "Current standard",
            "recommended": True,
        },
        "450K": {
            "name": "HumanMethylation450",
            "manufacturer": "Illumina",
            "n_cpgs": 485512,
            "cost_per_sample": 400.0,
            "processing_days": 7,
            "dna_required_ng": 500.0,
            "coverage": "Genome-wide",
            "release_year": 2011,
            "status": "Mature",
            "recommended": True,
        },
    },
    "designs": {
        "two_group": {
            "name": "Two-Group Comparison",
            "description": "Compare two independent groups (e.g., case vs control)",
            "n_groups": 2,
            "paired": False,
            "complexity": "Simple",
            "min_n_recommended": 12,
            "power_adjustment": 1.0,
            "analysis_method": "Two-sample t-test / Linear model",
        },
        "paired": {
            "name": "Paired Design",
            "description": "Compare matched samples",
            "n_groups": 2,
            "paired": True,
            "complexity": "Simple",
            "min_n_recommended": 10,
            "power_adjustment": 0.71,
            "analysis_method": "Paired t-test / Linear mixed model",
        },
    },
    "cost_components": {
        "dna_extraction": {
            "cost": 50.0,
            "unit": "per_sample",
            "description": "DNA extraction",
            "optional": False,
        },
        "qc": {
            "cost": 30.0,
            "unit": "per_sample",
            "description": "Quality control",
            "optional": False,
        },
        "bioinformatics": {
            "cost": 100.0,
            "unit": "per_sample",
            "description": "Bioinformatics",
            "optional": False,
        },
    },
    "timeline_phases": {
        "planning_irb": {
            "name": "Planning & IRB",
            "base_duration_days": 30,
            "description": "Design & IRB",
            "critical": True,
        },
        "sample_collection": {
            "name": "Sample collection",
            "base_duration_days": 14,
            "description": "Collect samples",
            "critical": True,
        },
        "array_processing": {
            "name": "Array processing",
            "base_duration_days": 7,
            "description": "Hybridization & scan",
            "critical": False,
            "parallelizable": True,
        },
        "analysis": {
            "name": "Analysis",
            "base_duration_days": 14,
            "description": "Differential analysis",
            "critical": True,
        },
    },
    "global_settings": {
        "contingency_buffer_percent": 10.0,
        "default_mcp_method": "fdr",
    },
}


# Utility helpers
def _atomic_write(path: Union[str, Path], content: Union[str, bytes]) -> None:
    """
    Write content to a file atomically.

    The function writes to a temporary file in the same directory as the target
    and replaces the target file in a single atomic operation, ensuring that
    the destination file is never left in a partially written state.

    Parameters
    ----------
    path : str or Path
        Path of the destination file.
    content : str or bytes
        Data to write to the file.

    Raises
    ------
    OSError
        If the write operation fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure content is bytes
    if isinstance(content, str):
        content_bytes = content.encode("utf-8")
    else:
        content_bytes = content

    # Use NamedTemporaryFile in same dir for atomic replace
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
    except Exception as e:
        logger.warning(f"Warning: Failed to write to file: {e}")
        # try clean up and raise
        if tmp_file and tmp_file.exists():
            try:
                tmp_file.unlink()
            except Exception as e:
                logger.warning(f"Warning: Failed to remove temporary file: {e}")
        raise


def _read_python_literal(path: Path) -> Dict[str, Any]:
    """
    Parse a Python literal expression from a file and return it as a dictionary.

    The file must contain a single top-level dictionary expression that can be
    safely evaluated with ``ast.literal_eval``.

    Parameters
    ----------
    path : Path
        Path to the file containing the literal.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    ValueError
        If the file does not contain a valid top-level dictionary.
    """
    import ast

    text = path.read_text()
    try:
        obj = ast.literal_eval(text)
    except Exception as e:
        raise ValueError(f"Python literal file must contain a top-level dict: {e}")

    if not isinstance(obj, dict):
        raise ValueError("Python literal file must contain a top-level dict")
    return obj


# PlannerConfig (singleton)
class PlannerConfig:
    """
    Thread-safe singleton holding the complete planner configuration.

    The singleton pattern ensures that only one configuration instance exists
    throughout the application lifetime. Access is recommended via the
    ``get_config()`` function.

    Parameters
    ----------
    config_file : str or Path, optional
        Configuration file to load immediately after instantiation.
    """

    _instance: Optional["PlannerConfig"] = None
    _lock = RLock()

    def __new__(cls, config_file: Optional[Union[str, Path]] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        if getattr(self, "_initialized", False):
            return
        self._cfg_path: Optional[Path] = None
        self._data_lock = RLock()
        self._raw: Dict[str, Any] = {}
        self.model: Optional[PlannerConfigModel] = None
        try:
            self._raw = deepcopy(DEFAULT_CONFIG)
            self._validate_and_set(self._raw)
            logger.info("Loaded built-in default configuration.")
        except ValidationError as e:
            logger.warning(f"Default configuration failed validation: {e}")
            raise
        if config_file:
            self.load_file(config_file)
        self._initialized = True

    # Loading
    def load_file(self, path: Union[str, Path], allow_excel: bool = True) -> None:
        """
        Load and validate a configuration file, merging it with built-in defaults.

        Supported formats are JSON, YAML, TOML, Python literal dictionaries,
        and Excel/CSV table files. The loaded data is deeply merged over the
        defaults and then validated against the Pydantic schema.

        Parameters
        ----------
        path : str or Path
            Path to the configuration file.
        allow_excel : bool, default True
            Permit loading from Excel (.xlsx, .xls) or CSV files.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        PermissionError
            If the file cannot be read.
        ValueError
            If the file format is unsupported or malformed.
        ValidationError
            If the merged configuration fails schema validation.
        """
        try:
            path = Path(path).resolve()
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid path provided: {path}") from e

        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}\n"
                f"Current working directory: {Path.cwd()}"
            )

        if not path.is_file():
            raise ValueError(f"Path exists but is not a file: {path}")

        # Check file is readable
        if not os.access(path, os.R_OK):
            raise PermissionError(f"Cannot read file (permission denied): {path}")

        logger.info(f"Loading config from {path}")
        ext = path.suffix.lower()
        loaded: Dict[str, Any] = {}
        try:
            if ext in (".json",):
                loaded = json.loads(path.read_text())
            elif ext in (".yml", ".yaml"):
                if yaml is None:
                    raise RuntimeError("PyYAML required to load YAML files")
                loaded = yaml.safe_load(path.read_text())
            elif ext in (".toml",):
                if toml is None:
                    raise RuntimeError("toml required to load TOML files")
                loaded = toml.loads(path.read_text())
            elif ext in (".py", ".txt"):
                loaded = _read_python_literal(path)
            elif ext in (".xlsx", ".xls", ".csv"):
                if not allow_excel:
                    raise RuntimeError("Excel/CSV loading is disabled")
                try:
                    loaded = self._load_from_table_file(path)
                except Exception as e:
                    logger.warning(
                        f"Warning: failed to parse table-file \
                        config {path}, skipping: {e}"
                    )
                    loaded = {}
            else:
                raise ValueError(f"Unsupported config file extension: {ext}")

            if not isinstance(loaded, dict):
                raise ValueError("Top-level config must be a mapping/dict")

            merged = deepcopy(DEFAULT_CONFIG)
            self._deep_update(merged, loaded)
            self._validate_and_set(merged)
            self._cfg_path = path
            logger.info(f"Configuration loaded and validated from {path}")
        except Exception:
            logger.warning(f"Failed to load configuration from {path}, using defaults")
            raise

    def _load_from_table_file(self, path: Path) -> Dict[str, Any]:
        """
        Parse configuration fragments from an Excel workbook or CSV file.

        Recognised sheets/tables are:

        - Platforms
        - Designs
        - Costs (mapped to ``cost_components``)
        - Timeline (mapped to ``timeline_phases``)

        The function returns a dictionary containing only the sections that
        could be successfully interpreted.

        Parameters
        ----------
        path : Path
            Path to the Excel or CSV file.

        Returns
        -------
        dict
            Partial configuration with keys ``platforms``, ``designs``,
            ``cost_components``, and/or ``timeline_phases``.

        Raises
        ------
        ValueError
            If no recognised configuration tables are found.
        """
        ext = path.suffix.lower()
        result: Dict[str, Any] = {}

        def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
            # normalize columns to lower_snake_case (best-effort)
            df = df.copy()
            df.columns = [str(c).strip() for c in df.columns]
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            # attempt to coerce typical numeric columns
            for col in [
                "n_cpgs",
                "cost_per_sample",
                "cost",
                "base_duration_days",
                "processing_days",
            ]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df

        if ext == ".csv":
            df = pd.read_csv(path)
            df.columns = df.columns.str.lower()
            if {"platform_id", "n_cpgs", "cost_per_sample"}.issubset(df.columns):
                df = df.set_index("platform_id")
                result["platforms"] = df.to_dict(orient="index")
            elif {"design_id", "n_groups"}.issubset(df.columns):
                df = df.set_index("design_id")
                result["designs"] = df.to_dict(orient="index")
            elif {"component_id", "cost", "unit"}.issubset(df.columns):
                df = df.set_index("component_id")
                result["cost_components"] = df.to_dict(orient="index")
            else:
                raise ValueError("CSV appears not to be a recognized config table")
        else:
            xl = pd.ExcelFile(path)
            sheets = {name.lower(): name for name in xl.sheet_names}
            # Platforms sheet
            for name_candidate, out_key, id_col in [
                ("platforms", "platforms", "platform_id"),
                ("designs", "designs", "design_id"),
                ("costs", "cost_components", "component_id"),
                ("timeline", "timeline_phases", "phase_id"),
            ]:
                if name_candidate in sheets:
                    df = pd.read_excel(xl, sheet_name=sheets[name_candidate])
                    df = _normalize_df(df)
                    if id_col in df.columns:
                        df = df.set_index(id_col)
                    result[out_key] = df.to_dict(orient="index")
            # allow some sheet name variants (e.g., "Platform" singular)
            if not result:
                # treat each sheet as table and infer by columns
                for sheet in xl.sheet_names:
                    df = pd.read_excel(xl, sheet_name=sheet)
                    df = _normalize_df(df)
                    if {"platform_id", "n_cpgs"}.issubset(df.columns):
                        df = df.set_index("platform_id")
                        result.setdefault("platforms", {}).update(
                            df.to_dict(orient="index")
                        )
                    elif {"design_id", "n_groups"}.issubset(df.columns):
                        df = df.set_index("design_id")
                        result.setdefault("designs", {}).update(
                            df.to_dict(orient="index")
                        )
                    elif {"component_id", "cost", "unit"}.issubset(df.columns):
                        df = df.set_index("component_id")
                        result.setdefault("cost_components", {}).update(
                            df.to_dict(orient="index")
                        )
                    elif {"phase_id", "base_duration_days"}.issubset(df.columns):
                        df = df.set_index("phase_id")
                        result.setdefault("timeline_phases", {}).update(
                            df.to_dict(orient="index")
                        )

        if not result:
            raise ValueError(f"No valid tables found in {path}")
        return result

    # Validate & set
    def _validate_and_set(self, raw: Dict[str, Any]) -> None:
        """
        Validate a raw configuration dictionary and update the internal state.

        On success the validated Pydantic model is stored and convenient
        attribute dictionaries (``platforms``, ``designs``, etc.) are created.

        Parameters
        ----------
        raw : dict
            Complete raw configuration dictionary.

        Raises
        ------
        ValidationError
            If the configuration does not conform to the schema.
        """
        with self._data_lock:
            try:
                validated = PlannerConfigModel(**raw)
            except ValidationError as e:
                logger.warning(f"Config validation error: {e}")
                raise
            self.model = validated
            self.platforms = {k: v.dict() for k, v in validated.platforms.items()}
            self.designs = {k: v.dict() for k, v in validated.designs.items()}
            self.cost_components = {
                k: v.dict() for k, v in validated.cost_components.items()
            }
            self.timeline_phases = {
                k: v.dict() for k, v in validated.timeline_phases.items()
            }
            self.global_settings = (
                validated.global_settings.dict() if validated.global_settings else {}
            )
            self._raw = raw

    # Utility merge
    @staticmethod
    def _deep_update(base: Dict[str, Any], extra: Dict[str, Any]) -> None:
        """
        Recursively merge ``extra`` into ``base`` in place.

        Nested dictionaries are updated recursively; other values are overwritten.

        Parameters
        ----------
        base : dict
            Dictionary modified in place.
        extra : dict
            Dictionary whose contents take precedence.
        """
        for k, v in extra.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                PlannerConfig._deep_update(base[k], v)
            else:
                base[k] = v

    # Save/Export
    def save_file(self, path: Union[str, Path], fmt: Optional[str] = None) -> None:
        """
        Atomically save the current configuration to a file.

        The format is inferred from the file extension unless ``fmt`` is provided.

        Supported formats: JSON (default), YAML, TOML.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        fmt : str, optional
            Explicit format (``json``, ``yaml``, ``toml``).

        Raises
        ------
        ValueError
            If the requested format is not supported.
        RuntimeError
            If a required library (PyYAML or toml) is missing.
        """
        path = Path(path)
        fmt = (fmt or path.suffix.lstrip(".")).lower()
        data = self._raw.copy()
        data["platforms"] = self.platforms
        data["designs"] = self.designs
        data["cost_components"] = self.cost_components
        data["timeline_phases"] = self.timeline_phases
        data["global_settings"] = self.global_settings

        if fmt in ("json", ""):
            content = json.dumps(data, indent=2)
            _atomic_write(path, content)
        elif fmt in ("yml", "yaml"):
            if yaml is None:
                raise RuntimeError("PyYAML required to write YAML")
            _atomic_write(path, yaml.safe_dump(data, sort_keys=False).encode("utf-8"))
        elif fmt == "toml":
            if toml is None:
                raise RuntimeError("toml required to write TOML")
            _atomic_write(path, toml.dumps(data).encode("utf-8"))
        else:
            raise ValueError("Unsupported save format: %s" % fmt)
        logger.info(f"Configuration saved to {path}")
        self._cfg_path = path

    # Reload / Migrate
    def reload(self) -> None:
        """
        Reload the currently-loaded configuration file.

        If no file was loaded earlier, this re-validates the in-memory config.
        """
        if self._cfg_path:
            self.load_file(self._cfg_path)
        else:
            self._validate_and_set(self._raw)
            logger.info("Configuration re-validated (no file path)")

    def migrate(self, migrator_callable) -> None:
        """
        Apply an in-place migration to the active configuration.

        The provided callable receives a copy of the current raw configuration
        and must return an updated dictionary. The result is merged, validated,
        and activated.

        Parameters
        ----------
        migrator_callable : callable
            Function ``raw_config -> updated_config``.
        """
        with self._data_lock:
            new = migrator_callable(self._raw.copy())
            self._deep_update(self._raw, new)
            self._validate_and_set(self._raw)
            logger.info("Configuration migrated using provided migrator.")

    # Query API (user-facing)
    def list_platforms(self, recommended_only: bool = False) -> pd.DataFrame:
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
        with self._data_lock:
            df = pd.DataFrame.from_dict(self.platforms, orient="index")
            if recommended_only:
                df = df[df["recommended"]]
            return df.copy()

    def get_platform(self, platform_id: str) -> Dict[str, Any]:
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
        with self._data_lock:
            try:
                return self.platforms[platform_id]
            except KeyError as e:
                raise KeyError(f"Platform '{platform_id}' not found") from e

    def list_designs(self) -> pd.DataFrame:
        """
        Return a table of all configured experimental designs.

        Returns
        -------
        pandas.DataFrame
            One row per design.
        """
        with self._data_lock:
            return pd.DataFrame.from_dict(self.designs, orient="index")

    def get_design(self, design_id: str) -> Dict[str, Any]:
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
        with self._data_lock:
            try:
                return self.designs[design_id]
            except KeyError as e:
                raise KeyError(f"Design '{design_id}' not found") from e

    def get_cost_components(
        self, platform: Optional[str] = None, include_optional: bool = True
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
        with self._data_lock:
            comps = {}
            for k, v in self.cost_components.items():
                if not include_optional and v.get("optional", False):
                    continue
                if (
                    platform
                    and v.get("applies_to")
                    and platform not in v.get("applies_to")
                ):
                    continue
                comps[k] = v
            return comps

    def update_platform_cost(self, platform_id: str, new_cost: float) -> None:
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
        with self._data_lock:
            if platform_id not in self.platforms:
                raise KeyError(platform_id)
            self.platforms[platform_id]["cost_per_sample"] = float(new_cost)
            self._raw.setdefault("platforms", {})[platform_id] = self.platforms[
                platform_id
            ]
            logger.info(f"Updated cost_per_sample for {platform_id}: {new_cost}")

    def add_custom_platform(
        self, platform_id: str, platform_info: Dict[str, Any]
    ) -> None:
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
        with self._data_lock:
            try:
                PlatformSchema(**platform_info)
            except ValidationError as e:
                raise ValueError(f"Invalid platform_info: {e}") from e
            self.platforms[platform_id] = platform_info
            self._raw.setdefault("platforms", {})[platform_id] = platform_info
            logger.info(f"Added custom platform {platform_id}")

    def update_regional_pricing(
        self, multiplier: float, region: Optional[str] = None
    ) -> None:
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
        if multiplier <= 0:
            raise ValueError("multiplier must be > 0")
        with self._data_lock:
            for p in self.platforms.values():
                old = float(p.get("cost_per_sample", 0.0))
                p["cost_per_sample"] = float(round(old * float(multiplier), 2))
            logger.info(f"Applied regional multiplier {multiplier} (region={region})")

    def get_platform_by_budget(self, max_cost_per_sample: float) -> pd.DataFrame:
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
        with self._data_lock:
            df = self.list_platforms()
            return df[df["cost_per_sample"] <= float(max_cost_per_sample)]

    # Higher-level helpers using config
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
        Compute required sample size per group using power analysis.

        The calculation accounts for the chosen experimental design, platform
        CpG count (for Bonferroni correction), and any design-specific power
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
        with self._data_lock:
            design = self.get_design(design_id)
            platform = self.get_platform(platform_id)
            paired = bool(design.get("paired", False))
            min_n = int(design.get("min_n_recommended", 3))
            adj_factor = float(design.get("power_adjustment", 1.0))
            mcp = mcp_method or self.global_settings.get("default_mcp_method", "fdr")

            if effect_size <= 0:
                raise ValueError("effect_size must be > 0 (Cohen's d)")

            if paired:
                min_n = max(min_n, 2)

        alpha_eff = alpha
        n_cpgs = int(platform.get("n_cpgs", 0) or 0)
        if mcp == "bonferroni":
            if n_cpgs <= 0:
                logger.warning(
                    "Warning: platform reports n_cpgs=0, \
                    skipping bonferroni alpha adjustment"
                )
            else:
                alpha_eff = alpha / float(platform["n_cpgs"])
                if alpha_eff < 1e-12:
                    logger.warning(
                        "Warning: Bonferroni-adjusted alpha extremely small \
                        (alpha_eff < 1e-12); result may be impractical"
                    )

        n_per_group = None
        try:
            if paired:
                if TTestPower is None:
                    raise RuntimeError(
                        "statsmodels.stats.power.TTestPower not available; \
                        cannot compute paired sample size."
                    )
                solver = TTestPower()
                n = solver.solve_power(
                    effect_size=effect_size,
                    alpha=alpha_eff,
                    power=power,
                    alternative="two-sided",
                )
            else:
                if TTestIndPower is None:
                    raise RuntimeError(
                        "statsmodels.stats.power.TTestIndPower not available; \
                        cannot compute independent sample size."
                    )
                solver = TTestIndPower()
                n = solver.solve_power(
                    effect_size=effect_size,
                    alpha=alpha_eff,
                    power=power,
                    alternative="two-sided",
                )

            n_adjusted = float(n) * adj_factor
            n_adjusted = max(n_adjusted, min_n)
            n_adjusted = min(n_adjusted, max_n)
            n_per_group = int(np.ceil(n_adjusted))
        except Exception as e:
            logger.warning(
                f"Power solver failed ({e}), using fallback min_n calculation"
            )
            # ensure n is defined even on failure (use fallback)
            fallback = min_n / max(effect_size, 1e-3)
            fallback *= adj_factor
            fallback = max(fallback, min_n)
            fallback = min(fallback, max_n)
            n_per_group = int(np.ceil(fallback))
            # set n to the fallback (float)
            n = float(n_per_group)

        # Compute total sample size
        total = n_per_group * int(design.get("n_groups", 2))
        return {
            "n_per_group": int(n),
            "total_samples": int(total),
            "alpha_nominal": alpha,
            "alpha_effective": float(alpha_eff),
            "mcp_method": mcp,
            "effect_size": float(effect_size),
            "power": float(power),
        }

    def estimate_study_timeline(
        self, n_samples: int, platform_id: str, contingency: bool = True
    ) -> pd.DataFrame:
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
        with self._data_lock:
            platform = self.get_platform(platform_id)
            phases = self.timeline_phases.copy()
            buffer_pct = (
                float(self.global_settings.get("contingency_buffer_percent", 10.0))
                / 100.0
            )

        rows = []
        for phase_id, info in phases.items():
            base = float(info.get("base_duration_days", 0.0))
            scaling = float(info.get("scaling_factor", 0.0) or 0.0)
            batch_adj = float(info.get("batch_adjustment", 0.0) or 0.0)
            if phase_id == "array_processing":
                base = float(platform.get("processing_days", base))
            est = base + scaling * float(n_samples)
            n_batches = max(1, int(np.ceil(n_samples / 96.0)))
            est += batch_adj * max(0, n_batches - 1)
            if contingency:
                est *= 1.0 + buffer_pct
            rows.append(
                {
                    "phase_id": phase_id,
                    "phase_name": info.get("name")
                    or phase_id.replace("_", " ").title(),
                    "estimated_days": float(np.ceil(est)),
                    "critical": bool(info.get("critical", False)),
                    "parallelizable": bool(info.get("parallelizable", False)),
                }
            )
        df = pd.DataFrame(rows)
        serial = df.loc[~df["parallelizable"], "estimated_days"].sum()
        parallel = df.loc[df["parallelizable"], "estimated_days"].sum()
        total = float(np.ceil(serial + parallel * 0.5))
        df.loc[len(df)] = {
            "phase_id": "TOTAL",
            "phase_name": "TOTAL",
            "estimated_days": total,
            "critical": False,
            "parallelizable": False,
        }
        return df

    def estimate_total_cost(
        self,
        n_samples: int,
        platform_id: str,
        include_optional: bool = True,
        validate: bool = True,
    ) -> Dict[str, Any]:
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
        with self._data_lock:
            platform = self.get_platform(platform_id)
            comps = self.get_cost_components(
                platform=platform_id, include_optional=include_optional
            )

        if not isinstance(n_samples, (int, float)) or n_samples <= 0:
            raise ValueError(f"n_samples must be a positive integer, got {n_samples}")

        rows = []
        base_platform_cost = float(platform.get("cost_per_sample", 0.0))
        rows.append(
            {
                "component": f"platform:{platform_id}",
                "unit": "per_sample",
                "cost": float(n_samples) * base_platform_cost,
            }
        )

        for comp_id, comp in comps.items():
            unit = comp.get("unit", "per_sample")
            try:
                cost_val = float(comp.get("cost", 0.0))
            except Exception:
                raise ValueError(
                    f"Invalid cost for component '{comp_id}': {comp.get('cost')}"
                )

            if unit == "per_sample":
                total = cost_val * float(n_samples)
            elif unit == "per_cpg":
                n_cpgs = float(platform.get("n_cpgs", 0))
                total = cost_val * n_cpgs * float(n_samples)
            elif unit == "fixed":
                total = cost_val
            else:
                raise ValueError(
                    f"Unknown cost unit '{unit}' for component '{comp_id}'"
                )
            rows.append({"component": comp_id, "unit": unit, "cost": float(total)})

        df = pd.DataFrame(rows)
        total = float(df["cost"].sum())
        per_sample = float(total / float(n_samples))

        return {
            "components": df,
            "total": total,
            "per_sample": per_sample,
            "n_samples": int(n_samples),
            "platform": platform_id,
        }


# Global getter
_global_planner_config = None


def get_config() -> PlannerConfig:
    """
    Retrieve the global singleton instance of ``PlannerConfig``.

    The instance is created on first access if necessary.

    Returns
    -------
    PlannerConfig
        The active configuration object.
    """
    global _global_planner_config
    if _global_planner_config is None:
        _global_planner_config = PlannerConfig()
    return _global_planner_config


def reset_config():
    """Destroy the current singleton instance (primarily for testing)."""
    global _global_planner_config
    _global_planner_config = None


# Convenience top-level wrappers
def load_file(path: Union[str, Path], allow_excel: bool = True) -> None:
    """
    Load and validate a configuration file, merging it with built-in defaults.

    Supported formats are JSON, YAML, TOML, Python literal dictionaries,
    and Excel/CSV table files. The loaded data is deeply merged over the
    defaults and then validated against the Pydantic schema.

    Parameters
    ----------
    path : str or Path
        Path to the configuration file.
    allow_excel : bool, default True
        Permit loading from Excel (.xlsx, .xls) or CSV files.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    PermissionError
        If the file cannot be read.
    ValueError
        If the file format is unsupported or malformed.
    ValidationError
        If the merged configuration fails schema validation.
    """
    return get_config().load_file(path=path, allow_excel=allow_excel)


def load_from_table_file(path: Path) -> Dict[str, Any]:
    """
    Parse configuration fragments from an Excel workbook or CSV file.

    Recognised sheets/tables are:

    - Platforms
    - Designs
    - Costs (mapped to ``cost_components``)
    - Timeline (mapped to ``timeline_phases``)

    The function returns a dictionary containing only the sections that
    could be successfully interpreted.

    Parameters
    ----------
    path : Path
        Path to the Excel or CSV file.

    Returns
    -------
    dict
        Partial configuration with keys ``platforms``, ``designs``,
        ``cost_components``, and/or ``timeline_phases``.

    Raises
    ------
    ValueError
        If no recognised configuration tables are found.
    """
    return get_config()._load_from_table_file(path)


def save_file(path: Union[str, Path], fmt: Optional[str] = None) -> None:
    """
    Atomically save the current configuration to a file.

    The format is inferred from the file extension unless ``fmt`` is provided.

    Supported formats: JSON (default), YAML, TOML.

    Parameters
    ----------
    path : str or Path
        Destination file path.
    fmt : str, optional
        Explicit format (``json``, ``yaml``, ``toml``).

    Raises
    ------
    ValueError
        If the requested format is not supported.
    RuntimeError
        If a required library (PyYAML or toml) is missing.
    """
    return get_config().save_file(path=path, fmt=fmt)


def reload() -> None:
    """
    Reload the currently-loaded configuration file.

    If no file was loaded earlier, this re-validates the in-memory config.
    """
    return get_config().reload()


def migrate(migrator_callable) -> None:
    """
    Apply an in-place migration to the active configuration.

    The provided callable receives a copy of the current raw configuration
    and must return an updated dictionary. The result is merged, validated,
    and activated.

    Parameters
    ----------
    migrator_callable : callable
        Function ``raw_config -> updated_config``.
    """
    return get_config().migrate(migrator_callable)
