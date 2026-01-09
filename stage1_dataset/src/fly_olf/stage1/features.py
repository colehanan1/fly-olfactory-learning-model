"""Feature extraction from wide trace format."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _get_trace_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    """Extract and sort trace columns by numeric suffix."""
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"HARD-FAIL: No trace columns found with prefix '{prefix}'")
    
    # Ensure numeric order if suffix is integer
    def key(c: str) -> int:
        try:
            return int(c.replace(prefix, ''))
        except Exception:
            return 10**9
    return sorted(cols, key=key)


def compute_features_from_wide(trials: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute behavioral features from wide trace format.
    
    Args:
        trials: DataFrame with trial metadata and trace columns
        cfg: Configuration dictionary
        
    Returns:
        DataFrame with all trial columns plus computed features
        
    Raises:
        ValueError: If no trace columns found or required config missing
    """
    prefix = cfg['trace']['trace_prefix']
    fps_col = cfg['trace']['fps_col']
    baseline_frac = float(cfg['windowing']['baseline_frac'])
    k_std = float(cfg['detection']['k_std'])
    min_duration_s = float(cfg['detection']['min_duration_s'])

    trace_cols = _get_trace_cols(trials, prefix)
    X = trials[trace_cols].to_numpy(dtype=float)
    fps = trials[fps_col].to_numpy(dtype=float) if fps_col in trials.columns else np.full(len(trials), np.nan)

    n = X.shape[1]
    b_n = max(5, int(np.floor(baseline_frac * n)))

    # Baseline statistics
    baseline = X[:, :b_n]
    base_mean = np.nanmean(baseline, axis=1)
    base_std = np.nanstd(baseline, axis=1) + 1e-9
    thr = base_mean + k_std * base_std

    # Detect above-threshold samples
    above = X > thr[:, None]

    # Latency: first index above threshold
    first_idx = np.argmax(above, axis=1)
    never = ~np.any(above, axis=1)
    first_idx[never] = -1

    # Duration: total above-threshold time
    above_count = np.sum(above, axis=1).astype(float)
    duration_s = np.where(np.isfinite(fps) & (fps > 0), above_count / fps, np.nan)

    # PER binary: above-threshold for at least min_duration_s
    per = (duration_s >= min_duration_s).astype(int)
    per[~np.isfinite(duration_s)] = 0

    # Peak, AUC relative to baseline
    peak = np.nanmax(X, axis=1)
    auc = np.nansum(np.maximum(0.0, X - base_mean[:, None]), axis=1)
    auc_s = np.where(np.isfinite(fps) & (fps > 0), auc / fps, np.nan)

    latency_s = np.where((first_idx >= 0) & np.isfinite(fps) & (fps > 0), first_idx / fps, np.nan)

    # Build output dataframe
    out = trials.copy()
    out["baseline_mean"] = base_mean
    out["baseline_std"] = base_std
    out["threshold"] = thr
    out["per"] = per
    out["latency_s"] = latency_s
    out["duration_s"] = duration_s
    out["peak"] = peak
    out["auc_pos_s"] = auc_s

    return out


def apply_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply optional filters to remove low-quality trials.
    
    Args:
        df: Features dataframe
        cfg: Configuration dictionary
        
    Returns:
        Filtered dataframe
        
    Raises:
        ValueError: If required filter column is missing (when filter is enabled)
    """
    filters = cfg.get('filters', {})
    n_initial = len(df)

    # Filter by NaN fraction
    max_nan_frac = filters.get('max_nan_frac')
    if max_nan_frac is not None:
        prefix = cfg['trace']['trace_prefix']
        trace_cols = [c for c in df.columns if c.startswith(prefix)]
        if trace_cols:
            nan_frac = df[trace_cols].isna().mean(axis=1)
            mask = nan_frac <= max_nan_frac
            n_dropped = (~mask).sum()
            df = df[mask].copy()
            print(f"  Dropped {n_dropped} trials with NaN fraction > {max_nan_frac}")

    # Filter by non-reactive flag
    if filters.get('drop_if_non_reactive_flag', False):
        col = filters.get('non_reactive_flag_col', 'non_reactive_flag')
        if col in df.columns:
            mask = ~df[col].astype(bool)
            n_dropped = (~mask).sum()
            df = df[mask].copy()
            print(f"  Dropped {n_dropped} non-reactive trials (column: {col})")
        else:
            raise ValueError(
                f"HARD-FAIL: Filter 'drop_if_non_reactive_flag' enabled but column '{col}' not found"
            )

    # Filter by tracking flag
    if filters.get('drop_if_tracking_flagged', False):
        col = filters.get('tracking_flag_col', 'tracking_flagged')
        if col in df.columns:
            mask = ~df[col].astype(bool)
            n_dropped = (~mask).sum()
            df = df[mask].copy()
            print(f"  Dropped {n_dropped} tracking-flagged trials (column: {col})")
        else:
            raise ValueError(
                f"HARD-FAIL: Filter 'drop_if_tracking_flagged' enabled but column '{col}' not found"
            )

    n_final = len(df)
    print(f"  Filter summary: {n_initial} trials → {n_final} kept ({n_initial - n_final} dropped)")

    return df
