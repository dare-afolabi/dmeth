#!/usr/bin/env python3


"""
Update CHANGELOG.md with commits since the last tag.

- Reads version from pyproject.toml
- Finds the latest git tag
- Collects commit messages since that tag
- Prepends a new section to CHANGELOG.md
"""


import re
import subprocess
from pathlib import Path
from datetime import datetime


def get_version() -> str:
    text = Path("pyproject.toml").read_text()
    m = re.search(r'version\s*=\s*"([^"]+)"', text)
    if not m:
        raise RuntimeError("Version not found in pyproject.toml")
    return m.group(1)


def get_latest_tag() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return None


def get_commits(since_tag: str | None) -> list[str]:
    cmd = ["git", "log", "--pretty=format:%h %s"]
    if since_tag:
        cmd.insert(2, f"{since_tag}..HEAD")
    return subprocess.check_output(cmd, text=True).splitlines()


def main():
    version = get_version()
    date = datetime.utcnow().strftime("%Y-%m-%d")
    latest_tag = get_latest_tag()
    commits = get_commits(latest_tag)

    header = f"## {version} - {date}\n"
    body = "\n".join(f"- {c}" for c in commits) or "- No changes recorded"

    changelog_path = Path("CHANGELOG.md")
    if changelog_path.exists():
        existing = changelog_path.read_text()
        if not existing.startswith("# Changelog"):
            existing = "# Changelog\n\n" + existing
    else:
        existing = "# Changelog\n\n"

    new_content = existing + header + body + "\n\n"
    changelog_path.write_text(new_content)
    print(f"CHANGELOG.md updated for version {version}")


if __name__ == "__main__":
    main()
