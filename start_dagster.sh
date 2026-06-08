#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
if [[ ! -f ".venv/bin/activate" ]]; then
    echo "ERROR: .venv not found. Run: uv venv --python 3.11 && uv pip install -r requirements.txt"
    exit 1
fi
source .venv/bin/activate

# Persistent Dagster home — avoids losing run history between sessions
export DAGSTER_HOME="${DAGSTER_HOME:-$HOME/dagster_home}"
mkdir -p "$DAGSTER_HOME"
echo "DAGSTER_HOME: $DAGSTER_HOME"

# Load .env so OPENAI_KEY (and optional overrides) are available to Dagster workers
if [[ -f ".env" ]]; then
    set -a
    source .env
    set +a
else
    echo "WARNING: .env not found — OPENAI_KEY must already be in the environment"
fi

echo ""
echo "Starting Dagster UI → http://127.0.0.1:3000"
echo ""
echo "Jobs:"
echo "  nightly_job     — synopsis + cast + directors (02:00 daily, schedules off by default)"
echo "  film_meta_job   — film_meta only              (03:00 Sundays, schedules off by default)"
echo "  comscore_job    — comscore_match only          (ad-hoc)"
echo "  full_refresh_job — everything                 (ad-hoc)"
echo ""
echo "CLI bypass (same checkpoints, no Dagster overhead):"
echo "  python refresh.py                        # all four LLM paths"
echo "  python refresh.py --only synopsis cast   # subset"
echo "  python rematch_comscore.py               # comscore only"
echo ""

dagster dev -f dagster_defs.py
