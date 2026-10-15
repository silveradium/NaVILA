#!/bin/bash
# Run every scripted episode of the scouted scenes headless, with dashboard videos and a results table per scene.
#   ./navila_go2/scripts/benchmark.sh [extra run_navila_go2.py args, e.g. --policy isaaclab_flat --cam_profile go2_crop]
# Needs the NaVILA server (navila_go2/scripts/start_server.sh). Results: output/sim_runs/benchmark/<run>/results.json
cd "$(dirname "$0")/../.."
for scene in hospital office; do
  ./navila_go2/scripts/run_sim.sh --scene "$scene" --episodes all --headless --record --out output/sim_runs/benchmark \
    "$@" > "output/benchmark_$scene.log" 2>&1
  grep "\[navila-go2\]   " "output/benchmark_$scene.log"
done
