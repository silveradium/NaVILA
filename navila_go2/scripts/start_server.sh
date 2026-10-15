#!/bin/bash
# Start the NaVILA high-level policy server (conda env `navila`).
#   ./navila_go2/scripts/start_server.sh [4bit|8bit|fp16] [port]
#
# The navila env relies on packages in ~/.local, but ~/.local also holds a newer torch (2.2.2) that breaks
# torchvision in this env. Putting the env's own site-packages first makes its torch 2.3.0 win while the
# remaining packages still resolve from ~/.local. ROS's PYTHONPATH entries are dropped.
cd "$(dirname "$0")/../.."
PRECISION=${1:-4bit}
PORT=${2:-8000}
source ~/miniconda3/etc/profile.d/conda.sh && conda activate navila
export PYTHONPATH="$CONDA_PREFIX/lib/python3.10/site-packages"
exec python navila_go2/server/navila_server.py --precision "$PRECISION" --port "$PORT"
