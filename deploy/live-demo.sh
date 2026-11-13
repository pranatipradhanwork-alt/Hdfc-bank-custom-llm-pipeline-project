#!/bin/bash
# Live demo from this Mac: starts the app (on the Mac's GPU) and shares it on the account's fixed ngrok address.
#
#   ./deploy/live-demo.sh        # Ctrl+C stops both the app and the public link
#
# One-time setup: a free ngrok account, then `ngrok config add-authtoken <token>` (see the README).
set -e
cd "$(dirname "$0")/.."

PORT=7860
command -v ngrok >/dev/null || { echo "ngrok is not installed: brew install ngrok"; exit 1; }
ngrok config check >/dev/null 2>&1 || { echo "ngrok has no authtoken yet: ngrok config add-authtoken <token>"; exit 1; }
[ -f frontend/dist/index.html ] || { echo "Web UI not built: cd frontend && npm run build"; exit 1; }

source hdfc_env/bin/activate
set -a && source .env && set +a

echo "Starting the app (about 30 seconds)..."
uvicorn server:app --port "$PORT" > live-demo.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

until curl -sf "http://localhost:$PORT/v1/health" >/dev/null; do
    kill -0 $SERVER_PID 2>/dev/null || { echo "The app failed to start, see live-demo.log"; exit 1; }
    sleep 3
done
echo "App is up. Opening the public link (the https address on the 'Forwarding' line)..."

# ngrok uses the account's fixed dev domain, so the link is the same every time
ngrok http "$PORT"
