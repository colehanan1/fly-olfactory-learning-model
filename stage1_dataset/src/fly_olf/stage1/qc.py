"""QC plots and reporting."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def make_qc_plots(features: pd.DataFrame, cfg: dict, qc_dir: Path) -> dict[str, str]:
    """Generate QC plots for feature distributions and sample traces.
    
    Args:
        features: Features dataframe
        cfg: Configuration dictionary
        qc_dir: Directory to save plots
        
    Returns:
        Dictionary mapping plot type names to file paths
    """
    qc_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    # Distribution histograms for key features
    dist_cols = cfg['qc']['dist_cols']
    for col in dist_cols:
        if col not in features.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        x = features[col].dropna()
        ax.hist(x, bins=50, edgecolor='black', alpha=0.7)
        ax.set_title(f"Distribution: {col}", fontsize=12, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)
        p = qc_dir / f"dist_{col}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths[f"dist_{col}"] = str(p)

    # Random trace panels (requires trace columns)
    prefix = cfg['trace']['trace_prefix']
    trace_cols = [c for c in features.columns if c.startswith(prefix)]
    if trace_cols:
        n_show = int(cfg['qc']['n_random_trials'])
        idx = np.random.choice(len(features), size=min(n_show, len(features)), replace=False)

        for i, row_idx in enumerate(idx):
            row = features.iloc[row_idx]
            y = row[trace_cols].to_numpy(dtype=float)
            thr = float(row.get("threshold", np.nan))

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(y, linewidth=1)
            if np.isfinite(thr):
                ax.axhline(thr, linestyle="--", color='red', label=f'threshold={thr:.2f}')
            
            fly_id = row.get('fly_id', '?')
            phase = row.get('phase', '?')
            label = row.get('trial_label', '?')
            per = row.get('per', '?')
            
            ax.set_title(
                f"{fly_id} | {phase} | label={label} | per={per}",
                fontsize=10, fontweight='bold'
            )
            ax.set_xlabel("frame")
            ax.set_ylabel("trace (dir velocity)")
            ax.grid(alpha=0.3)
            if np.isfinite(thr):
                ax.legend()

            p = qc_dir / f"trace_{i:03d}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)

        paths["random_traces_dir"] = str(qc_dir)

    return paths


def write_qc_report(
    report_path: Path,
    cfg: dict,
    trials_path: Path,
    features_path: Path,
    qc_paths: dict[str, str],
    n_initial: int = 0,
    n_final: int = 0,
) -> None:
    """Write markdown QC report summarizing Stage 1 run.
    
    Args:
        report_path: Path to output .md file
        cfg: Configuration dictionary
        trials_path: Path to trials.parquet
        features_path: Path to features.parquet
        qc_paths: Dictionary of QC plot paths
        n_initial: Number of trials before filters
        n_final: Number of trials after filters
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    lines.append("# Feature Extraction QC Report\n")
    lines.append("## Run Summary\n")
    
    lines.append("### Input / Output Paths\n")
    lines.append(f"- **Input CSV**: `{cfg['paths']['input_csv']}`\n")
    lines.append(f"- **Trials parquet**: `{trials_path.name}`\n")
    lines.append(f"- **Features parquet**: `{features_path.name}`\n")
    
    lines.append("\n### Row Counts\n")
    if n_initial > 0 and n_final > 0:
        lines.append(f"- Initial trials loaded: {n_initial}\n")
        lines.append(f"- Trials after filters: {n_final}\n")
        if n_initial > n_final:
            lines.append(f"- Dropped: {n_initial - n_final} ({100*(n_initial-n_final)/n_initial:.1f}%)\n")
    
    lines.append("\n### Configuration Snapshot\n")
    lines.append("```yaml\n")
    import yaml
    lines.append(yaml.dump(cfg, default_flow_style=False))
    lines.append("```\n")
    
    lines.append("\n### Required Columns Check\n")
    required = cfg['schema']['required_cols']
    lines.append(f"- Required columns in input: {required}\n")
    
    lines.append("\n### QC Artifacts Generated\n")
    if qc_paths:
        for k, v in qc_paths.items():
            lines.append(f"- **{k}**: `{Path(v).name}`\n")
    else:
        lines.append("- No QC artifacts generated\n")
    
    report_path.write_text("".join(lines))
