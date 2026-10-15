#!/bin/bash
# Run NaVILA on the simulated Go2 (conda env `env_isaaclab`). Start navila_go2/scripts/start_server.sh first.
#   ./navila_go2/scripts/run_sim.sh --scene hospital --episode 3                 # GUI, scripted episode
#   ./navila_go2/scripts/run_sim.sh --scene hospital                             # GUI, type instructions live
#   ./navila_go2/scripts/run_sim.sh --scene hospital --episodes all --headless --record   # benchmark + videos
# Low-level policy: the ported legged-loco go2_base (newest checkpoint) by default; --policy isaaclab_flat for the
# stock Isaac Lab flat policy, --checkpoint <model_N.pt> to pick a specific checkpoint.
cd "$(dirname "$0")/../.."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
exec python -u navila_go2/sim/run_navila_go2.py "$@"
