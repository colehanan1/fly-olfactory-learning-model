"""Train/eval pipeline for Stage 2 baseline."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.dummy import DummyClassifier
from typing import Dict, List, Tuple, Optional

from .door_features import DoorOdorEncoder
from .metrics import compute_fold_metrics, aggregate_metrics


def load_and_filter_data(
    features_path: str,
    phases: Optional[List[str]] = None,
    target_column: str = "per",
) -> pd.DataFrame:
    """
    Load features.parquet and filter by phase.
    
    Args:
        features_path: Path to features.parquet
        phases: List of phases to keep (None = all)
        target_column: Name of target column
        
    Returns:
        Filtered DataFrame
    """
    df = pd.read_parquet(features_path)
    
    # Security: Only show shape, never print rows
    print(f"✓ Loaded features: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Filter phases
    if phases and len(phases) > 0:
        df = df[df["phase"].isin(phases)].copy()
        print(f"✓ Filtered to phases {phases}: {len(df)} rows")
    
    # Check required columns
    required = ["fly_id", "odor_name", "phase", target_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Drop rows with null target
    n_before = len(df)
    df = df.dropna(subset=[target_column])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"✓ Dropped {n_dropped} rows with null target")
    
    return df


def run_cv_scheme(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    metadata: pd.DataFrame,
    scheme_name: str,
    n_splits: int,
    model_params: Dict,
    standardize: bool = True,
) -> Tuple[pd.DataFrame, List[Dict[str, float]]]:
    """
    Run GroupKFold CV for one scheme (odor-holdout or fly-holdout).
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        groups: Group labels for GroupKFold (n_samples,)
        metadata: DataFrame with fly_id, odor_name, phase for predictions
        scheme_name: "odor_holdout" or "fly_holdout"
        n_splits: Number of CV folds
        model_params: Model hyperparameters
        standardize: Whether to standardize features
        
    Returns:
        (predictions_df, fold_metrics)
    """
    print(f"\n--- {scheme_name.upper()} ---")
    
    gkf = GroupKFold(n_splits=n_splits)
    
    all_predictions = []
    fold_metrics = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standardize features
        if standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Train model
        model = LogisticRegression(
            solver=model_params.get("solver", "liblinear"),
            C=model_params.get("C", 1.0),
            max_iter=model_params.get("max_iter", 1000),
            random_state=model_params.get("random_seed", 1337),
        )
        model.fit(X_train, y_train)
        
        # Baseline (majority class predictor)
        baseline = DummyClassifier(strategy="prior")
        baseline.fit(X_train, y_train)
        
        # Predictions
        y_prob = model.predict_proba(X_test)[:, 1]
        y_prob_baseline = baseline.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = compute_fold_metrics(y_test, y_prob, y_prob_baseline)
        fold_metrics.append(metrics)
        
        # Store predictions with metadata
        fold_preds = metadata.iloc[test_idx].copy()
        fold_preds["y_true"] = y_test
        fold_preds["y_prob"] = y_prob
        fold_preds["y_prob_baseline"] = y_prob_baseline
        fold_preds["fold_id"] = fold_idx
        fold_preds["scheme"] = scheme_name
        all_predictions.append(fold_preds)
        
        print(f"  Fold {fold_idx}: ROC-AUC={metrics['roc_auc']:.3f}, LogLoss={metrics['log_loss']:.3f}")
    
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    return predictions_df, fold_metrics


def run_stage2_pipeline(cfg: dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Run full Stage 2 baseline pipeline.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        (predictions_df, metrics_dict)
    """
    print("\n" + "=" * 70)
    print("Stage 2 Baseline Training")
    print("=" * 70)
    
    # Load data
    df = load_and_filter_data(
        cfg["paths"]["features_parquet"],
        phases=cfg["data"].get("phases"),
        target_column=cfg["data"]["target_column"],
    )
    
    # Encode odors with DoOR
    print("\n--- DoOR Odor Encoding ---")
    encoder = DoorOdorEncoder(
        cache_dir=cfg["door"]["cache_dir"],
        fill_missing=cfg["door"]["fill_missing"],
        cache_enabled=cfg["door"]["cache_enabled"],
    )
    X = encoder.encode_dataframe(df, odor_column="odor_name")
    print(f"✓ Encoded odors: X.shape={X.shape}")
    print(f"✓ Cached {encoder.cache_size} unique odors")
    
    # Target
    y = df[cfg["data"]["target_column"]].values
    print(f"✓ Target distribution: {np.mean(y):.3f} (mean PER)")
    
    # Metadata for predictions
    metadata = df[["fly_id", "odor_name", "phase"]].reset_index(drop=True)
    
    # Run CV schemes
    all_predictions = []
    all_metrics = {}
    
    for scheme_cfg in cfg["cv"]["schemes"]:
        scheme_name = scheme_cfg["name"]
        group_col = scheme_cfg["group_by"]
        
        # Extract groups
        groups = df[group_col].values
        
        # Run CV
        preds_df, fold_metrics = run_cv_scheme(
            X=X,
            y=y,
            groups=groups,
            metadata=metadata,
            scheme_name=scheme_name,
            n_splits=cfg["cv"]["n_splits"],
            model_params=cfg["model"],
            standardize=cfg["model"].get("standardize_features", True),
        )
        
        all_predictions.append(preds_df)
        
        # Aggregate metrics
        aggregated = aggregate_metrics(fold_metrics)
        all_metrics[scheme_name] = aggregated
        
        # Print summary
        print(f"\nSummary ({scheme_name}):")
        for metric_name, stats in aggregated.items():
            print(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Combine predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    print("\n" + "=" * 70)
    print("Stage 2 Complete!")
    print("=" * 70)
    
    return predictions_df, all_metrics
