"""CLI entrypoint for Stage 1 pipeline."""

from __future__ import annotations

import sys
import typer
from pathlib import Path

from .config import load_config
from .pipeline import run_stage1, audit_stage1_output

app = typer.Typer(add_completion=False)


@app.command()
def build(config: str = typer.Argument(..., help="Path to YAML config")) -> None:
    """Build Stage 1 dataset: segment trials and extract features."""
    try:
        cfg = load_config(config)
        run_stage1(cfg)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        sys.exit(1)


@app.command()
def audit(config: str = typer.Argument(..., help="Path to YAML config")) -> None:
    """Audit Stage 1 output for data quality and completeness."""
    try:
        cfg = load_config(config)
        audit_stage1_output(cfg)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    app()
