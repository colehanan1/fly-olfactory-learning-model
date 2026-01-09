# CLAUDE.md — Project Instructions

## Priority
Stage 1 security + docs integration. Do not implement later stages unless asked.

## Strict privacy
- Do not print/paste raw data rows, ever.
- Only output schema/shapes/aggregated metrics.
- Ensure outputs go to gitignored folders only.

## Must-have behaviors
- Every pipeline run creates a docs entry under `docs/runs/`.
- After any code change, update `docs/repo_map.md` (via `make docs`).
- If code changes, README.md must be updated.

## Stage 1 requirements
- CLI: `fly-olf-stage1 build --config stage1_dataset/configs/default.yaml`
- Write run doc: `docs/runs/stage1/<timestamp>_<run_name>.md`
- Config snapshots in docs must redact absolute paths and any data-like paths.
