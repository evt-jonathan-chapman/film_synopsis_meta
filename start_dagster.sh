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

PORT=3000
URL="http://127.0.0.1:${PORT}"

echo ""
echo "Starting Dagster UI (LLM extraction paths) → $URL"
echo ""
echo "Jobs:"
echo "  nightly_job     — synopsis + cast + directors (02:00 daily, schedules off by default)"
echo "  film_meta_job   — film_meta only              (03:00 Sundays, schedules off by default)"
echo "  full_refresh_job — everything                 (ad-hoc)"
echo ""
echo "CLI bypass (same checkpoints, no Dagster overhead):"
echo "  python refresh.py                        # all four LLM paths"
echo "  python refresh.py --only synopsis cast   # subset"
echo ""
echo "Comscore/Gower matching (no LLM, no API) is a separate Dagster process —"
echo "run ./start_dagster_matching.sh alongside this one, or bypass it entirely:"
echo "  python rematch_comscore.py / python rematch_gower.py / python id_bridge.py"
echo ""

dagster dev -f dagster_defs.py -p "$PORT" &
DAGSTER_PID=$!
trap 'kill "$DAGSTER_PID" 2>/dev/null' EXIT

# Open the browser as soon as the UI responds, so you're not stuck copying the URL manually
( for _ in $(seq 1 60); do
    if curl -s -o /dev/null "$URL"; then
        open "$URL"
        break
    fi
    sleep 0.5
  done ) &

wait "$DAGSTER_PID"
