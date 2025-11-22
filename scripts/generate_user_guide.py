#!/usr/bin/env python


"""
Ultimate User Guide Generator for dmeth

Generates a hierarchical User Guide (docs/UserGuide.md) by combining:
- README sections
- all dmeth package/module/class/function docstrings

Includes signature shortening and docstring normalization.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pathlib
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

# Setup
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

README_PATH = PROJECT_ROOT / "README.md"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "UserGuide.md"

TOP_LEVEL_PACKAGES = [
    "dmeth.core.data_preprocessing",
    "dmeth.core.planner",
    "dmeth.core.analysis",
    "dmeth.core.downstream",
    "dmeth.config",
    "dmeth.io",
    "dmeth.utils",
]

PACKAGE_DISPLAY_NAMES = {
    "dmeth.core.data_preprocessing": "Preprocessing Utilities",
    "dmeth.core.planner": "Study Planning",
    "dmeth.core.analysis": "Core Differential Analysis",
    "dmeth.core.downstream": "Downstream Analysis",
    "dmeth.config": "Configuration",
    "dmeth.io": "Input/Output",
    "dmeth.utils": "Utilities",
}

TOP_SECTIONS = ["Installation", "Quick Start", "Key Features"]
BOTTOM_SECTIONS = ["Citation", "References", "Support", "License", "Contributing"]


# Utility: Shorten long dotted names in type hints
def shorten_type_paths(text: str) -> str:
    """Shorten fully qualified type names in signatures."""
    replacements = {
        r"dmeth\.[\w\.]+\.ProcessedData": "ProcessedData",
        r"numpy\.ndarray": "ndarray",
        r"pandas\.DataFrame": "DataFrame",
        r"pandas\.core\.frame\.DataFrame": "DataFrame",
        r"typing\.": "",
        r"builtins\.": "",
    }
    for pat, rep in replacements.items():
        text = re.sub(pat, rep, text)
    return text


# Signature extractor (with shortening)
def get_clean_signature(obj: Any) -> str:
    """Extract clean signature"""
    try:
        sig = str(inspect.signature(obj))
    except Exception:
        return ""

    return shorten_type_paths(sig)


# Docstring cleaner (lightweight)
def clean_docstring(doc: Optional[str]) -> str:
    """Normalize docstrings and shorten type names inside them."""
    if not doc:
        return "_No docstring provided._"

    text = inspect.cleandoc(doc)

    # shorten fully-qualified types in docstrings
    text = shorten_type_paths(text)

    # Convert NumPy-style sections to ### headers
    section_headers = [
        "Parameters",
        "Returns",
        "Notes",
        "Features",
        "Examples",
        "Attributes",
        "Raises",
        "See Also",
    ]
    for sec in section_headers:
        text = re.sub(rf"^{sec}\s*\n[-=]+\s*$", f"### {sec}", text, flags=re.MULTILINE)

    # Convert "param : type" lines to bullets
    text = re.sub(r"^(\w+)\s*:\s*(.+)$", r"- **\1**: `\2`", text, flags=re.MULTILINE)

    return text.strip()


# Markdown heading detection
def _heading_info(line: str) -> Optional[Tuple[int, str]]:
    """Detect headings in markdown"""
    m = re.match(r"^\s{0,3}(#{1,6})\s+(.*)$", line)
    if not m:
        return None
    return len(m.group(1)), m.group(2).strip()


# README extraction
def extract_readme_intro() -> str:
    """Return text under first H1 until first H2/H3."""
    if not README_PATH.exists():
        return ""

    lines = README_PATH.read_text().splitlines()
    out = []
    found_h1 = False

    for line in lines:
        hi = _heading_info(line)

        if hi and hi[0] == 1:
            found_h1 = True
            continue

        if found_h1 and hi and hi[0] in (2, 3):
            break

        if found_h1:
            out.append(line)

    return "\n".join(out).strip() + "\n\n"


def extract_readme_sections(targets: Set[str]) -> str:
    """Extract exact H2 sections (excluding subsections)."""
    if not README_PATH.exists():
        return ""

    lines = README_PATH.read_text().splitlines()
    out_sections: Dict[str, List[str]] = {}
    order = []

    current = None
    buf = []
    in_code = False

    for line in lines:
        if line.strip().startswith(("```", "~~~")):
            in_code = not in_code
            if current:
                buf.append(line)
            continue

        hi = _heading_info(line) if not in_code else None

        if hi:
            level, title = hi
            if level == 2:
                if current and buf:
                    out_sections[current] = buf
                buf = []

                if title in targets:
                    current = title
                    order.append(title)
                    buf.append(line)
                else:
                    current = None
                continue

            if level < 2 and current:
                out_sections[current] = buf
                current, buf = None, []
                continue

        if current:
            buf.append(line)

    if current and buf:
        out_sections[current] = buf

    result = []
    for title in order:
        if title in out_sections:
            result.append("\n".join(out_sections[title]))

    return ("\n\n".join(result) + "\n\n") if result else ""


# Package/module member discovery
def get_public_members(obj: Any):
    """Extract modules"""
    for name in dir(obj):
        if name.startswith("_"):
            continue

        try:
            attr = getattr(obj, name)
        except Exception:
            continue

        if inspect.isfunction(attr) or inspect.isclass(attr):
            if getattr(attr, "__module__", "").startswith("dmeth"):
                yield name, attr


def discover_submodules(pkg: Any):
    """Extract submodules"""
    import pkgutil

    if not hasattr(pkg, "__path__"):
        return

    for _, name, is_pkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        if not is_pkg:
            try:
                yield importlib.import_module(name)
            except Exception as e:
                logger.warning(f"Failed to import {name}: {e}")


# User Guide generator
def generate_user_guide() -> str:
    """Generate complete user guide"""
    out = []

    out.append("# dmeth - documentation for reproducible DNA methylation analysis\n")
    out.append("## User Guide\n")

    intro = extract_readme_intro()
    if intro:
        out.append(intro)

    top = extract_readme_sections(set(TOP_SECTIONS))
    if top:
        out.append(top)

    out.append("## Reference for all modules, classes, and functions\n")

    for pkg_name in TOP_LEVEL_PACKAGES:
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception as e:
            out.append(f"## {pkg_name} — import failed: {e}\n\n")
            continue

        display = PACKAGE_DISPLAY_NAMES.get(pkg_name, pkg_name.split(".")[-1].title())

        out.append("\n---\n")
        out.append(f"## {display}\n")

        if pkg.__doc__:
            out.append(clean_docstring(pkg.__doc__) + "\n\n")

        # package-level functions/classes
        for name, obj in sorted(get_public_members(pkg)):
            sig = get_clean_signature(obj)
            out.append("\n---\n")
            out.append(f"`{name}{sig}`\n\n")
            out.append(clean_docstring(obj.__doc__ or "") + "\n")

        # submodules
        for mod in sorted(discover_submodules(pkg), key=lambda m: m.__name__):
            title = mod.__name__.split(".")[-1].replace("_", " ").title()
            out.append(f"### {title}\n")

            if mod.__doc__:
                out.append(clean_docstring(mod.__doc__) + "\n\n")

            for name, obj in sorted(get_public_members(mod)):
                sig = get_clean_signature(obj)
                out.append("\n---\n")
                out.append(f"`{name}{sig}`\n\n")
                out.append(clean_docstring(obj.__doc__ or "") + "\n")

    bottom = extract_readme_sections(set(BOTTOM_SECTIONS))
    if bottom:
        out.append(bottom)

    out.append(f"\n---\n\n> **Auto-generated** on {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    return "\n".join(out)


# CLI
def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating User Guide…")
    content = generate_user_guide()
    OUTPUT_PATH.write_text(content, encoding="utf-8")

    logger.info(f"✓ User Guide saved to {OUTPUT_PATH}")
    logger.info(f"  Size: {len(content):,} characters")


if __name__ == "__main__":
    main()
