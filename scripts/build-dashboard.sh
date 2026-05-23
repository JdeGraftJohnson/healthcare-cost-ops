#!/usr/bin/env bash
# One-command pipeline: ingest → silver → forecast → audit-plan → judges → merge
# Usage: scripts/build-dashboard.sh examples/medicaid_sdud_2026/dashboard_spec.yml
set -euo pipefail

SPEC="${1:-examples/medicaid_sdud_2026/dashboard_spec.yml}"
RUN_NAME="$(python -c "import yaml,sys; print(yaml.safe_load(open('$SPEC'))['name'])")"
OUT="examples/${RUN_NAME}/out"
mkdir -p "${OUT}/judges"

echo "[1/5] Ingest bronze (skipped here — run sdud_pull.py manually for Azure cost reasons)"
echo "[2/5] Silver (skipped — requires bronze)"
echo "[3/5] Forecast (skipped — requires silver)"

echo "[4/5] Plan judges"
python -m services.judges.audit_runner --plan --spec "${SPEC}" --out "${OUT}"

echo "[5/5] Merge (run individual judges first, or invoke the judge-orchestrator subagent)"
python -m services.judges.audit_runner --merge --out "${OUT}"

echo "DONE. See ${OUT}/judges/audit.md and ${OUT}/judges/composite_scorecard.md"
