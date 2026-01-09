from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def sanitize(obj: Any) -> Any:
    """Redact absolute paths and data-bearing strings."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, str):
        s = obj
        # redact absolute paths and anything that looks data-bearing
        if s.startswith("/") or "data" in s.lower() or "parquet" in s.lower() or "csv" in s.lower():
            return "<REDACTED>"
        return s
    return obj


def write_run_doc(
    stage: str,
    run_name: str,
    command: str,
    cfg: dict,
    artifacts: dict[str, str],
) -> Path:
    """Write a sanitized run documentation entry."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    docs_dir = Path("docs") / "runs" / stage
    docs_dir.mkdir(parents=True, exist_ok=True)
    out = docs_dir / f"{ts}_{run_name}.md"

    cfg_s = json.dumps(sanitize(cfg), indent=2, sort_keys=True)
    lines = []
    lines.append(f"# Run: {stage} — {ts}_{run_name}\n\n")
    lines.append(f"- Git commit: `{git_head()}`\n")
    lines.append(f"- Command: `{command}`\n\n")
    lines.append("## Artifacts\n")
    for k, v in artifacts.items():
        lines.append(f"- {k}: `{v}`\n")
    lines.append("\n## Config (sanitized)\n")
    lines.append("```json\n" + cfg_s + "\n```\n")

    out.write_text("".join(lines))
    return out
