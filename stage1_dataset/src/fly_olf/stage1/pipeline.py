"""Stage 1 pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .schema import standardize_trials
from .features import compute_features_from_wide, apply_filters
from .qc import make_qc_plots, write_qc_report


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
