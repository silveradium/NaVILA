#!/bin/bash
# One-command demo: start the NaVILA server (if not already running), wait for it, run the Isaac Sim runner, and stop
# the server again on exit (only if this script started it). Arguments go to run_navila_go2.py, e.g.:
#   ./navila_go2/scripts/demo.sh --scene hospital --episode 3
#   ./navila_go2/scripts/demo.sh --scene hospital --episodes all --headless --record
cd "$(dirname "$0")/../.."
mkdir -p output
started=0
if ! curl -sf http://127.0.0.1:8000/health >/dev/null; then
  echo "[demo] starting NaVILA server (log: output/navila_server.log)"
  ./navila_go2/scripts/start_server.sh 4bit 8000 > output/navila_server.log 2>&1 &
  server_pid=$!
  started=1
  trap '[ $started = 1 ] && kill $server_pid 2>/dev/null' EXIT
  until curl -sf http://127.0.0.1:8000/health >/dev/null; do
    kill -0 $server_pid 2>/dev/null || { echo "[demo] server died, see output/navila_server.log"; exit 1; }
    sleep 2
  done
fi
echo "[demo] NaVILA server ready: $(curl -s http://127.0.0.1:8000/health)"
./navila_go2/scripts/run_sim.sh "$@"
