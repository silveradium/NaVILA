# leggedloco: legged-loco for Isaac Lab 2.x

A port of `omni.isaac.leggedloco` (legged-loco, MIT License © 2024 the legged-loco/NaVILA authors, see
`../../LICENSE`) from Isaac Sim 4.1 / Isaac Lab 1.1 to **Isaac Sim 5.1 / Isaac Lab 2.3 / rsl-rl-lib 5**.
The original extension next to this folder is unchanged.

Only the Go2 configurations (the robot NaVILA runs on) are ported. Gym ids are the same as in the original:
`go2_base`, `go2_base_play`, `go2_vision`, `go2_vision_play`. Importing `leggedloco` registers them.

| Original | Port |
|---|---|
| `omni.isaac.lab*`, `omni.isaac.lab_tasks*` | `isaaclab*`, `isaaclab_tasks*`, RSL-RL cfgs from `isaaclab_rl` |
| `RslRlVecEnvHistoryWrapper(history_length=9)` + legged-loco's rsl_rl fork | observation group `proprio` with `history_length=9`, `obs_groups={"actor": ["policy", "proprio"], "critic": ["critic"]}` (actor input is the same 450 dims) |
| `math_utils.quat_rotate_inverse` | `math_utils.quat_apply_inverse` |
| `RayCasterCfg(attach_yaw_only=True/False)` | `ray_alignment="yaw"/"base"` |
| `sim.disable_contact_processing` | removed (no longer exists) |
| go2_vision flat patches keyed by sub-terrain name | keyed `init_pos` for every sub-terrain (what `reset_root_state_from_terrain` reads) |

Train and use:
```bash
python legged-loco-main/scripts/train_lab2.py --task go2_base --headless --max_iterations 2000
python navila_go2/sim/run_navila_go2.py --policy legged_loco ...   # newest checkpoint in logs/rsl_rl/go2_base
```
The run shipped here (`logs/rsl_rl/go2_base/2026-10-07_18-51-04_navila`) was stopped at iteration 1000 with mean
reward 41.7 and full-length episodes (~40 min on an RTX 4080 SUPER). Continue it with `--resume <model_1000.pt>`.
