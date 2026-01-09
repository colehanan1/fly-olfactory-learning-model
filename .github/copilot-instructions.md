# Copilot instructions — Fly OLM

- This repo is public: never add/commit raw data, derived data, plots, logs, or secrets.
- Do not print dataset rows. Only schema/shapes/aggregate stats.
- Any pipeline run must write docs under `docs/runs/<stage>/` with sanitized config + artifact paths.
- After code changes, update `docs/repo_map.md`.
- If code changes, update README.md. CI enforces this.
- Prefer minimal diffs; do not refactor unrelated code.
