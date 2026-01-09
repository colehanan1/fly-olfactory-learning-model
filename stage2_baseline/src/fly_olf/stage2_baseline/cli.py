"""CLI for Stage 2 baseline."""
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
import typer
from typing_extensions import Annotated

from .train_eval import run_stage2_pipeline


app = typer.Typer(help="Stage 2 Baseline - Fly Olfactory Learning Model", add_completion=False)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def write_run_log(cfg: dict, metrics: dict, predictions_path: str, metrics_path: str) -> None:
    """
    Write sanitized run log to docs/runs/stage2/.
    
    Args:
        cfg: Configuration dictionary
        metrics: Metrics dictionary
        predictions_path: Path to predictions file
        metrics_path: Path to metrics file
    """
    # Create docs/runs/stage2/ if needed
    log_dir = Path("docs/runs/stage2")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Timestamp and run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg["run"]["name"]
    log_path = log_dir / f"{timestamp}_{run_name}.md"
    
    # Format metrics for display
    metrics_text = []
    for scheme_name, scheme_metrics in metrics.items():
        metrics_text.append(f"### {scheme_name}")
        for metric_name, stats in scheme_metrics.items():
            metrics_text.append(
                f"- **{metric_name}**: {stats['mean']:.4f} ± {stats['std']:.4f} "
                f"(n={stats['n_folds']})"
            )
        metrics_text.append("")
    
    # Write log
    with open(log_path, "w") as f:
        f.write(f"# Stage 2 Run: {run_name}\n\n")
        f.write(f"**Timestamp**: {timestamp}\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"```yaml\n")
        f.write(yaml.dump(cfg, default_flow_style=False))
        f.write(f"```\n\n")
        f.write(f"## Outputs\n\n")
        f.write(f"- **Predictions**: `{predictions_path}`\n")
        f.write(f"- **Metrics**: `{metrics_path}`\n\n")
        f.write(f"## Metrics Summary\n\n")
        f.write("\n".join(metrics_text))
    
    print(f"\n✓ Wrote run log: {log_path}")


@app.command()
def train(
    config: Annotated[str, typer.Argument(help="Path to YAML config file")],
) -> None:
    """Train Stage 2 baseline models with dual CV schemes."""
    try:
        # Load config
        cfg = load_config(config)
        
        # Ensure output directory exists
        output_dir = Path(cfg["paths"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run pipeline
        predictions_df, metrics_dict = run_stage2_pipeline(cfg)
        
        # Write predictions
        predictions_path = cfg["paths"]["predictions_file"]
        predictions_df.to_parquet(predictions_path, index=False)
        print(f"\n✓ Written predictions: {predictions_path}")
        print(f"  Shape: {predictions_df.shape}")
        
        # Write metrics
        metrics_path = cfg["paths"]["metrics_file"]
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"✓ Written metrics: {metrics_path}")
        
        # Write run log
        write_run_log(cfg, metrics_dict, predictions_path, metrics_path)
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    app()
