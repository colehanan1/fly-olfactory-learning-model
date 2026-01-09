"""Stage 1 pipeline orchestration."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

from .schema import standardize_trials
from .features import compute_features_from_wide, apply_filters
from .qc import make_qc_plots, write_qc_report
from .protocol_map_builder import (
    build_protocol_map_for_training,
    build_protocol_map_for_testing,
    merge_protocol_maps,
)


def _ensure_protocol_map(cfg: dict) -> Path:
    """
    Ensure protocol map exists. If not, auto-generate from training/testing CSVs.
    Returns the path to the protocol map CSV.
    """
    protocol_map_path = Path(cfg['paths'].get('protocol_map_csv', '')).expanduser().resolve()
    
    if protocol_map_path.exists():
        print(f"✓ Protocol map exists: {protocol_map_path}")
        return protocol_map_path
    
    print(f"\n--- Auto-generating Protocol Map ---")
    training_csv = Path(cfg['paths'].get('training_csv', '')).expanduser().resolve()
    testing_csv = Path(cfg['paths'].get('testing_csv', '')).expanduser().resolve()
    model_pred_csv = cfg['paths'].get('model_predictions_csv', '')
    if model_pred_csv:
        model_pred_csv = Path(model_pred_csv).expanduser().resolve()
    
    if not training_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {training_csv}")
    if not testing_csv.exists():
        raise FileNotFoundError(f"Testing CSV not found: {testing_csv}")
    
    # Create temp protocol maps
    training_map = protocol_map_path.parent / "protocol_map_training.csv"
    testing_map = protocol_map_path.parent / "protocol_map_testing.csv"
    
    # Build training and testing maps
    build_protocol_map_for_training(str(training_csv), str(training_map))
    build_protocol_map_for_testing(
        str(testing_csv),
        str(testing_map),
        str(model_pred_csv) if model_pred_csv.exists() else None
    )
    
    # Merge them
    merge_protocol_maps(str(training_map), str(testing_map), str(protocol_map_path))
    
    print(f"✓ Generated protocol map: {protocol_map_path}")
    return protocol_map_path


def run_stage1(cfg: dict) -> None:
    """Execute Stage 1 pipeline: load, standardize, extract features, filter, and report.
    
    Args:
        cfg: Configuration dictionary (should come from config.load_config)
        
    Raises:
        FileNotFoundError: If input CSV not found
        ValueError: If required columns or processing steps fail
    """
    # Set random seed
    random_seed = int(cfg['run'].get('random_seed', 1337))
    np.random.seed(random_seed)
    
    # Resolve paths
    input_csv = Path(cfg['paths']['input_csv']).expanduser().resolve()
    out_dir = Path(cfg['paths']['output_dir']).expanduser().resolve()
    reports_dir = Path(cfg['paths']['reports_dir']).expanduser().resolve()
    qc_dir = Path(cfg['paths']['qc_dir']).expanduser().resolve()
    
    # Ensure output directories exist
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    
    # Load input CSV
    print(f"\n{'='*70}")
    print(f"Stage 1 Pipeline")
    print(f"{'='*70}")
    print(f"\n✓ Input CSV: {input_csv}")
    
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    
    df = pd.read_csv(input_csv)
    n_initial = len(df)
    print(f"✓ Loaded {n_initial} trials from input CSV")
    
    # Ensure protocol map exists (auto-generate if missing)
    protocol_map_path = _ensure_protocol_map(cfg)
    cfg['paths']['protocol_map_csv'] = str(protocol_map_path)  # Update config
    
    # Standardize trials (includes protocol join if configured)
    print(f"\n--- Trial Standardization ---")
    trials = standardize_trials(df, cfg)
    print(f"✓ Standardized trials with fly_id, phase, pulse_idx, protocol fields")
    
    # Compute features from wide trace format
    print(f"\n--- Feature Extraction ---")
    features = compute_features_from_wide(trials, cfg)
    print(f"✓ Computed features: per, latency_s, duration_s, baseline_mean, baseline_std, threshold, peak, auc_pos_s")
    
    # Apply optional filters
    print(f"\n--- Filtering ---")
    features_before = len(features)
    features = apply_filters(features, cfg)
    n_final = len(features)
    print(f"✓ Kept {n_final} trials (dropped {features_before - n_final})")
    
    # Output parquet files
    print(f"\n--- Writing Outputs ---")
    trials_path = out_dir / "trials.parquet"
    features_path = out_dir / "features.parquet"
    
    trials.to_parquet(trials_path, index=False)
    print(f"✓ Written trials: {trials_path}")
    
    features.to_parquet(features_path, index=False)
    print(f"✓ Written features: {features_path}")
    
    # QC plots and report
    print(f"\n--- QC Plots & Report ---")
    qc_paths = make_qc_plots(features, cfg, qc_dir)
    print(f"✓ Generated QC plots in {qc_dir}")
    
    report_path = reports_dir / "feature_extraction_qc.md"
    write_qc_report(report_path, cfg, trials_path, features_path, qc_paths, n_initial, n_final)
    print(f"✓ Written QC report: {report_path}")
    
    print(f"\n{'='*70}")
    print(f"Stage 1 Complete!")
    print(f"{'='*70}\n")
    
    # Write run documentation (sanitized, no data paths)
    _write_run_log(cfg, trials_path, features_path, qc_dir, report_path)

def _write_run_log(cfg: dict, trials_path: Path, features_path: Path, qc_dir: Path, report_path: Path) -> None:
    """Write a sanitized run log entry. Imports log_run module via subprocess to avoid circular deps."""
    import json
    import subprocess
    from datetime import datetime
    
    # Prepare sanitized artifacts dict (no absolute paths)
    artifacts = {
        "trials": "data/trials.parquet",
        "features": "data/features.parquet",
        "qc_dir": "data/qc_plots/",
        "qc_report": "reports/feature_extraction_qc.md",
    }
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    docs_dir = Path("docs") / "runs" / "stage1"
    docs_dir.mkdir(parents=True, exist_ok=True)
    out = docs_dir / f"{ts}_stage1.md"
    
    # Sanitize config (redact data-like paths)
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, str):
            s = obj
            if s.startswith("/") or any(x in s.lower() for x in ["data", "parquet", "csv"]):
                return "<REDACTED>"
            return s
        return obj
    
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_head = "UNKNOWN"
    
    cfg_s = json.dumps(sanitize(cfg), indent=2, sort_keys=True)
    lines = []
    lines.append(f"# Run: stage1 — {ts}_stage1\n\n")
    lines.append(f"- Git commit: `{git_head}`\n")
    lines.append(f"- Command: `fly-olf-stage1 build --config stage1_dataset/configs/default.yaml`\n\n")
    lines.append("## Artifacts\n")
    for k, v in artifacts.items():
        lines.append(f"- {k}: `{v}`\n")
    lines.append("\n## Config (sanitized)\n")
    lines.append("```json\n" + cfg_s + "\n```\n")
    
    out.write_text("".join(lines))
    print(f"✓ Wrote run log: {out}")