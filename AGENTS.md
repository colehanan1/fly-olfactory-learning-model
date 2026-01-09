# AGENTS.md — Fly Olfactory Learning Model (Umbrella Repo)

## Objective
Build a behavior-first, imaging-ready, closed-loop learning system for Drosophila olfactory conditioning.
Current priority: Stage 1 dataset builder (traces → trials/features parquet + QC).

## Security / privacy (strict)
- Never print or paste raw dataset rows (CSV/parquet) into chat/logs.
- Never dump full DataFrames. Only show shapes, column names, dtypes, and small aggregated stats.
- Never commit data artifacts, plots, logs, or secrets. This repo is public.

## Repo structure
- `stage1_dataset/`: dataset builder package + CLI (fly-olf-stage1)
- `docs/`: documentation (runs, design notes, repo map)
- `scripts/`: repo automation scripts (docs + repo map)
- `.github/`: CI policies and instructions
- `.claude/`: Claude Code settings

## Stage 1 acceptance criteria
Commands that must work:
- `python -m pip install -e stage1_dataset`
- `fly-olf-stage1 build --config stage1_dataset/configs/default.yaml`
- `make stage1`

Artifacts produced by Stage 1 must be written to gitignored locations.
After each run, a docs entry must be written under `docs/runs/stage1/`.

## Docs discipline
- Every pipeline run writes: `docs/runs/<stage>/<timestamp>_<run_name>.md`
- Every code change updates: `docs/repo_map.md` (auto-generated)
- If code changes, README.md must be updated with usage/notes.

## Workflow for agents
1. Plan (≤10 lines)
2. Verify repo state (list files, inspect configs)
3. Make minimal change
4. Run acceptance checks
5. Summarize changes + where they live
