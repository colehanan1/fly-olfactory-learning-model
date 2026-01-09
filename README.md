# fly-olfactory-learning-model
A behavior-first, imaging-ready framework for modeling Drosophila olfactory learning. Builds trial-level datasets from proboscis traces, encodes odors via DoOR, and implements online, biologically inspired plasticity to infer and control learning dynamics in closed-loop experiments.
## Stage 1: Dataset Builder

### Quick Start

```bash
python -m pip install -e stage1_dataset
fly-olf-stage1 build --config stage1_dataset/configs/default.yaml
```

Or use:
```bash
make stage1
```

This runs the full Stage 1 pipeline and updates documentation.

### Protocol Map Auto-Generation

Stage 1 automatically generates protocol maps from training and testing CSVs if not present. The pipeline:

1. **Reads training CSV** → extracts `dataset`, `trial_label`, and determines odor schedules
2. **Reads testing CSV** → applies alias mapping and assigns odor names
3. **Merges maps** → combines training/testing into a unified protocol map with `phase` field
4. **Joins with features** → enriches output with `odor_name`, `odor_display`, `reward`, `cs_type`

The protocol map is written to `stage1_dataset/data/protocol_map.csv` (gitignored, auto-generated).

## Security / Data Policy
This repository is public. Do not commit raw data, derived data (csv/parquet/h5), plots, logs, or secrets.
All outputs must be written to gitignored paths.

## Docs & Run Logs
- Run logs are written to `docs/runs/<stage>/` with sanitized config and artifact paths.
- The repo navigation index is auto-generated at `docs/repo_map.md`.
- Use:
  - `make docs` to update documentation
  - `make stage1` to run Stage 1 and update documentation

For more details, see [AGENTS.md](AGENTS.md), [CLAUDE.md](CLAUDE.md), and [CONTRIBUTING.md](CONTRIBUTING.md).