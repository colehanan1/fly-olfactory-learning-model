# Contributing

This repo is public and must not contain raw data, derived data, plots, logs, or secrets.

## Rules
- Never commit datasets (csv/parquet/h5/etc.).
- Never commit run artifacts (plots/logs).
- All outputs must go to gitignored paths.
- If you change code, update README.md. CI enforces this.

## Dev setup
Install hooks:
```bash
pip install pre-commit detect-secrets ruff
detect-secrets scan > .secrets.baseline
pre-commit install
```

## Commands
- `make stage1` — run Stage 1 dataset builder and update docs
- `make docs` — update docs/repo_map.md
