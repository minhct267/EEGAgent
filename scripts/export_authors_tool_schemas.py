"""Freeze the tool schemas registered by the authors' commit for the authors prompt.

The current tools keep the channel-index fix and clearer descriptions. The
authors prompt must show the original <Tools> text, so this script imports the
tools package from `git archive <commit>` in a temp dir and writes its schemas.

Usage: python scripts/export_authors_tool_schemas.py [--commit acd2e6a]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "config" / "protocols" / "authors_tool_schemas.json"

DUMP_CODE = (
    "import json, sys\n"
    "from tools import function_register\n"
    "json.dump(function_register.export_tool_schemas(), sys.stdout, ensure_ascii=False)\n"
)


def export(commit: str) -> list:
    with tempfile.TemporaryDirectory(prefix="authors_tools_") as tmp:
        archive = subprocess.run(
            ["git", "archive", commit], cwd=PROJECT_ROOT, check=True, capture_output=True
        ).stdout
        subprocess.run(["tar", "-x", "-C", tmp], input=archive, check=True)
        dumped = subprocess.run(
            [sys.executable, "-c", DUMP_CODE], cwd=tmp, check=True, capture_output=True, text=True
        ).stdout
    return json.loads(dumped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--commit", default="acd2e6a")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    commit = subprocess.check_output(
        ["git", "rev-parse", args.commit], cwd=PROJECT_ROOT, text=True
    ).strip()
    schemas = export(commit)
    payload = {"commit": commit, "schemas": schemas}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    names = [item["function"]["name"] for item in schemas]
    print(f"Wrote {len(names)} schemas from {commit[:7]} to {out}: {', '.join(names)}")


if __name__ == "__main__":
    main()
