
#!/usr/bin/env bash
set -e

LOGFILE=$(mktemp)

# Run stax2aws, tee output to both terminal and a log file, in background
stax2aws login -i stax-au1 -o event --session-duration 3600 2>&1 | tee "$LOGFILE" &
PID=$!

# Poll the log until the "Full URL:" line appears
while ! grep -q "Full URL:" "$LOGFILE" 2>/dev/null; do
    sleep 0.2
done

sleep 0.2  # let the URL line itself get flushed to the file
URL=$(grep -A1 "Full URL:" "$LOGFILE" | tail -1 | xargs)

echo "Opening: $URL"
open "$URL"

# Wait for stax2aws to finish (i.e. until you complete auth in the browser)
wait "$PID"
