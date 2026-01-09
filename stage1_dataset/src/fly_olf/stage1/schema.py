"""Trial schema standardization and protocol mapping."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from .protocol_map_builder import TESTING_DATASET_ALIAS


def standardize_trials(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Standardize trial schema and optionally join protocol metadata.
    
    Args:
        df: Raw trial dataframe from wide CSV
        cfg: Configuration dictionary
        
    Returns:
        Standardized trials dataframe with fly_id, phase, pulse_idx, and protocol fields
        
    Raises:
        ValueError: If required columns are missing or protocol map is invalid
    """
    # Check required columns
    required = cfg['schema']['required_cols']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")
    
    out = df.copy()
    
    # Canonical identifiers
    out["fly_id"] = out["dataset"].astype(str) + "::" + out["fly"].astype(str)
    
    # Phase: training/testing inferred from trial_type
    tt = out["trial_type"].astype(str).str.lower()
    train_kw = cfg['schema'].get('phase_rules', {}).get('training_contains', ['train'])
    test_kw = cfg['schema'].get('phase_rules', {}).get('testing_contains', ['test'])
    
    def infer_phase(s: str) -> str:
        for kw in train_kw:
            if kw in s:
                return "training"
        for kw in test_kw:
            if kw in s:
                return "testing"
        return "unknown"
    
    out["phase"] = tt.map(infer_phase)
    
    # Pulse index: parse trailing integer from trial_label
    pulse_regex = cfg['schema'].get('pulse_idx_regex', r'(\d+)\s*$')
    out["pulse_idx"] = (
        out["trial_label"].astype(str)
        .str.extract(pulse_regex, expand=False)
        .astype("float")
    )
    out["pulse_idx"] = out["pulse_idx"].fillna(-1).astype(int)
    
    # Compute dataset_key for join: apply TESTING_DATASET_ALIAS for testing phase
    out["dataset_key"] = out.apply(
        lambda row: TESTING_DATASET_ALIAS.get(row["dataset"], row["dataset"])
        if row["phase"] == "testing"
        else row["dataset"],
        axis=1
    )
    
    # Protocol map join (optional)
    protocol_map_path = cfg['paths'].get('protocol_map_csv', '')
    if protocol_map_path and protocol_map_path.strip():
        protocol_path = Path(protocol_map_path).expanduser().resolve()
        if protocol_path.exists():
            proto_df = pd.read_csv(protocol_path)
            
            # Validate required protocol columns
            proto_required = ['dataset_key', 'phase', 'pulse_idx', 'odor_name', 'reward', 'cs_type']
            proto_missing = [c for c in proto_required if c not in proto_df.columns]
            if proto_missing:
                raise ValueError(
                    f"Protocol map CSV missing required columns: {proto_missing}. "
                    f"Required: {proto_required}. "
                    f"Path: {protocol_path}"
                )
            
            # Join on (dataset_key, phase, pulse_idx)
            join_keys = ['dataset_key', 'phase', 'pulse_idx']
            pre_merge_count = len(out)
            out = out.merge(
                proto_df[join_keys + ['odor_name', 'reward', 'cs_type']],
                on=join_keys,
                how='left',
                suffixes=('', '_proto')
            )
            
            # Hard-fail if any protocol fields are null (join failed)
            null_odor = out["odor_name"].isna().sum()
            null_reward = out["reward"].isna().sum()
            null_cs = out["cs_type"].isna().sum()
            
            if null_odor > 0 or null_reward > 0 or null_cs > 0:
                raise ValueError(
                    f"Protocol join produced {null_odor} null odor_name, "
                    f"{null_reward} null reward, {null_cs} null cs_type out of {pre_merge_count} rows. "
                    f"Join keys {join_keys} did not match. "
                    f"This indicates a mismatch between input data and protocol_map. "
                    f"Sample input dataset_key: {out['dataset_key'].unique()[:5]}, "
                    f"sample input phase: {out['phase'].unique()}, "
                    f"sample input pulse_idx: {sorted(out['pulse_idx'].unique())[:10]}"
                )
            
            print(f"✓ Joined protocol map from {protocol_path} on {join_keys}")
        else:
            print(f"⚠ Protocol map path specified but not found: {protocol_path}")
            _add_placeholder_protocol_fields(out)
    else:
        _add_placeholder_protocol_fields(out)
    
    return out


def _add_placeholder_protocol_fields(df: pd.DataFrame) -> None:
    """Add placeholder protocol fields if not present."""
    if "odor_name" not in df.columns:
        df["odor_name"] = "UNKNOWN"
    if "reward" not in df.columns:
        df["reward"] = -1
    if "cs_type" not in df.columns:
        df["cs_type"] = "UNKNOWN"
