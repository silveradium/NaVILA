#!/bin/bash
# Camera-profile experiment: the same hospital episodes with different robot camera optics.
#   ./navila_go2/scripts/cam_experiment.sh [episodes] [profiles...]
# Needs the NaVILA server running (navila_go2/scripts/start_server.sh).
cd "$(dirname "$0")/../.."
EPISODES=${1:-0,1,2}
shift
PROFILES=${*:-habitat go2 mast}
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
for p in $PROFILES; do
  python -u navila_go2/sim/run_navila_go2.py --scene hospital --episodes "$EPISODES" --cam_profile "$p" --headless \
    --record --max_actions 40 --out "output/sim_runs/cam_$p" > "output/run_cam_$p.log" 2>&1
done
