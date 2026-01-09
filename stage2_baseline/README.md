# Stage 2 Baseline - Fly Olfactory Learning Model

Baseline modeling with DoOR odor encoding and dual cross-validation schemes.

## Installation

```bash
python -m pip install -e stage2_baseline
```

## Usage

```bash
# Train baseline models
fly-olf-stage2 stage2_baseline/configs/default.yaml

# Check CLI help
fly-olf-stage2 --help
```

## Pipeline

1. Load features from Stage 1 (features.parquet)
2. Encode odors using DoOR ORN response vectors
3. Train logistic regression with two CV schemes:
   - **Odor-holdout**: GroupKFold on odor_name
   - **Fly-holdout**: GroupKFold on fly_id
4. Compute ROC-AUC and log loss per fold
5. Output predictions, metrics, and run documentation

## Outputs

- `outputs/stage2/predictions.parquet`: Fold predictions with y_true, y_prob
- `outputs/stage2/metrics.json`: Aggregated metrics (mean/std)
- `docs/runs/stage2/<timestamp>_<run_name>.md`: Sanitized run log

## Configuration

Edit `configs/default.yaml` to customize:
- DoOR encoding (fill_missing value)
- Model hyperparameters (C, solver, max_iter)
- CV splits (n_splits)
- Phase filtering (training-only, testing-only, or both)
