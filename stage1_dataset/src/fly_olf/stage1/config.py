"""Configuration loading and validation for Stage 1 pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required config keys are missing
    """
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Validate required top-level sections
    required_sections = ['paths', 'schema', 'trace', 'windowing', 'detection', 'features', 'qc']
    missing = [s for s in required_sections if s not in cfg]
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")
    
    # Validate paths section
    if 'input_csv' not in cfg['paths']:
        raise ValueError("Config paths.input_csv is required")
    
    # Set defaults for optional keys
    cfg.setdefault('run', {})
    cfg['run'].setdefault('random_seed', 1337)
    
    cfg['paths'].setdefault('protocol_map_csv', '')
    cfg['paths'].setdefault('output_dir', 'stage1_dataset/data')
    cfg['paths'].setdefault('reports_dir', 'stage1_dataset/reports')
    cfg['paths'].setdefault('qc_dir', 'stage1_dataset/data/qc_plots')
    
    cfg['schema'].setdefault('required_cols', ['dataset', 'fly', 'fly_number', 'trial_type', 'trial_label'])
    
    cfg['trace'].setdefault('trace_prefix', 'dir_val_')
    cfg['trace'].setdefault('fps_col', 'fps')
    
    cfg['windowing'].setdefault('baseline_frac', 0.2)
    
    cfg['detection'].setdefault('k_std', 4.5)
    cfg['detection'].setdefault('min_duration_s', 0.05)
    
    cfg['features'].setdefault('export_cols', [
        'per', 'latency_s', 'duration_s', 'baseline_mean', 
        'baseline_std', 'threshold', 'peak', 'auc_pos_s'
    ])
    
    cfg['qc'].setdefault('n_random_trials', 30)
    cfg['qc'].setdefault('dist_cols', ['latency_s', 'duration_s', 'auc_pos_s', 'peak', 'per'])
    
    cfg.setdefault('filters', {})
    cfg['filters'].setdefault('max_nan_frac', None)
    cfg['filters'].setdefault('drop_if_non_reactive_flag', False)
    cfg['filters'].setdefault('drop_if_tracking_flagged', False)
    
    return cfg
