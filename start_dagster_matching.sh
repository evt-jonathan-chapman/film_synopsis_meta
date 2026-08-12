#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv — requirements-matching.txt is enough for this process (no
# litellm/openai needed; see DAGSTER.md for why this is a separate script).
if [[ ! -f ".venv/bin/activate" ]]; then
    echo "ERROR: .venv not found. Run: uv venv --python 3.11 && uv pip install --system-certs -r requirements-matching.txt"
    exit 1
fi
source .venv/bin/activate

# Persistent Dagster home — shared with the main process is fine, they're
# separate code locations/ports.
export DAGSTER_HOME="${DAGSTER_HOME:-$HOME/dagster_home}"
mkdir -p "$DAGSTER_HOME"
echo "DAGSTER_HOME: $DAGSTER_HOME"

# Load .env — matching doesn't need OPENAI_KEY, but harmless if present.
if [[ -f ".env" ]]; then
    set -a
    source .env
    set +a
fi

PORT=3001
URL="http://127.0.0.1:${PORT}"

echo ""
echo "Starting Dagster UI (Comscore/Gower matching only) → $URL"
echo "No litellm/openai in this process — pure CPU fuzzy matching."
echo ""
echo "Jobs:"
echo "  comscore_job — comscore_match + gower_match + id_bridge (ad-hoc)"
echo ""
echo "Requires film_meta_enriched.parquet to already exist (for the concert-film"
echo "filter) — materialise film_meta via ./start_dagster.sh or"
echo "'python refresh.py --only film_meta' first if this is a fresh data dir."
echo ""
echo "CLI bypass (same checkpoints, no Dagster overhead):"
echo "  python rematch_comscore.py               # comscore only"
echo "  python rematch_gower.py                  # gower only"
echo "  python id_bridge.py                      # join comscore_cache + gower_cache"
echo ""

dagster dev -f dagster_matching_defs.py -p "$PORT" &
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
