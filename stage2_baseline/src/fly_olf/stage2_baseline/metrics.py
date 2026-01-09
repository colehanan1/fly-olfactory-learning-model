"""Metrics computation for Stage 2 baseline."""
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from typing import Dict, List


def compute_fold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_prob_baseline: np.ndarray,
) -> Dict[str, float]:
    """
    Compute metrics for a single fold.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities (model)
        y_prob_baseline: Predicted probabilities (baseline)
        
    Returns:
        Dictionary with ROC-AUC and log loss for model and baseline
    """
    metrics = {}
    
    # Model metrics
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError as e:
        # Handle case where fold has only one class
        metrics["roc_auc"] = np.nan
        print(f"Warning: ROC-AUC could not be computed: {e}")
    
    try:
        metrics["log_loss"] = log_loss(y_true, y_prob)
    except ValueError as e:
        metrics["log_loss"] = np.nan
        print(f"Warning: Log loss could not be computed: {e}")
    
    # Baseline metrics
    try:
        metrics["baseline_roc_auc"] = roc_auc_score(y_true, y_prob_baseline)
    except ValueError:
        metrics["baseline_roc_auc"] = np.nan
    
    try:
        metrics["baseline_log_loss"] = log_loss(y_true, y_prob_baseline)
    except ValueError:
        metrics["baseline_log_loss"] = np.nan
    
    return metrics


def aggregate_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across folds (mean/std).
    
    Args:
        fold_metrics: List of per-fold metric dictionaries
        
    Returns:
        Dictionary with mean and std for each metric
    """
    # Collect values per metric
    metric_values = {}
    for fold_dict in fold_metrics:
        for key, value in fold_dict.items():
            if key not in metric_values:
                metric_values[key] = []
            if not np.isnan(value):
                metric_values[key].append(value)
    
    # Compute mean/std
    aggregated = {}
    for metric_name, values in metric_values.items():
        if len(values) > 0:
            aggregated[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n_folds": len(values),
            }
        else:
            aggregated[metric_name] = {
                "mean": np.nan,
                "std": np.nan,
                "n_folds": 0,
            }
    
    return aggregated
