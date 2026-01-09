from __future__ import annotations

from pathlib import Path

INCLUDE_TOP = {
    "README.md", "AGENTS.md", "CLAUDE.md", "Makefile", ".gitignore",
    "CHANGELOG.md", "CONTRIBUTING.md"
}

STAGES = [
    "stage1_dataset",
    "stage2_door_encoder",
    "stage3_learning",
    "stage4_connectome",
    "stage5_imaging",
]

def main() -> None:
    root = Path(".")
    docs_dir = root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Repo Map\n\n")
    lines.append("Auto-generated index of key files and entrypoints.\n\n")

    lines.append("## Key files\n")
    for name in sorted(INCLUDE_TOP):
        p = root / name
        if p.exists():
            lines.append(f"- `{p.as_posix()}`\n")

    lines.append("\n## Directories\n")
    for d in [".github", ".claude", "docs", "scripts"]:
        p = root / d
        if p.exists():
            lines.append(f"- `{p.as_posix()}/`\n")

    lines.append("\n## Stages\n")
    for stage in STAGES:
        sp = root / stage
        if not sp.exists():
            continue
        lines.append(f"\n### `{stage}/`\n")
        for candidate in ["pyproject.toml", "README.md", "configs", "src"]:
            cp = sp / candidate
            if cp.exists():
                lines.append(f"- `{cp.as_posix()}`\n")

    out = docs_dir / "repo_map.md"
    out.write_text("".join(lines))

if __name__ == "__main__":
    main()
