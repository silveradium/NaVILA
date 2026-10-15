# NaVILA on a Unitree Go2: Isaac Sim integration

This package wires **NaVILA** (the high-level vision-language-action model) to a **legged locomotion policy**
(legged-loco, the low level from the same research), running on a simulated Unitree Go2 in **Isaac Sim 5.1 /
Isaac Lab 2.3**. The same high- and mid-level code also drives the real robot through `go2_sdk` (see the
real-robot section below).

```
 Process A, conda env "navila" (py3.10)          Process B, conda env "env_isaaclab" (Isaac Sim 5.1)
 ┌──────────────────────────────────────┐       ┌──────────────────────────────────────────────────────────┐
 │ server/navila_server.py              │ HTTP  │ sim/run_navila_go2.py                                    │
 │ NaVILA llama3-8b-8f, 4-bit           │◄─────►│  robot camera ──► FrameHistory (1 frame / 25 cm / 15°)   │
 │ 8 frames + instruction ──►           │ JPEG  │  "turn left 30 degrees" ──► common/mid_level.py          │
 │ "The next action is turn left 30°"   │       │     closed loop: 0.5 m/s, π/6 rad/s (NaVILA paper)       │
 │ ~0.75 s per decision                 │       │  (vx, vy, ωz) ──► RL locomotion policy @50 Hz            │
 └──────────────────────────────────────┘       │     joint targets ──► PD actuators @200 Hz physics       │
                                                └──────────────────────────────────────────────────────────┘
```

## Quick start

```bash
# all in one: starts the server if needed, then Isaac Sim with the control panel
./navila_go2/scripts/demo.sh --scene hospital --episode 3

# or in two terminals
./navila_go2/scripts/start_server.sh                      # terminal 1 (navila env)
./navila_go2/scripts/run_sim.sh --scene hospital          # terminal 2: GUI, type instructions in the panel
./navila_go2/scripts/run_sim.sh --scene hospital --episodes all --headless --record   # benchmark + videos
```

In the GUI, the **NaVILA Go2** panel shows the robot camera (NaVILA's input), an instruction box with
*Go / Stop / Reset robot* buttons, the current state and command, and NaVILA's decision log. The viewport follows
the robot with a chase camera, and the walked path is drawn in red.

With `--record`, each episode produces a 1080p dashboard video in `output/sim_runs/<run>/<episode>.mp4`: chase view,
robot camera, a bird's-eye floor plan with the ceiling cut away, and the instruction with the decision log. Each run
also writes `decisions.jsonl` (one line per NaVILA call: pose, text, action, latency) and `results.json` with
navigation error, success (the episode's own radius), and success@3m (the VLN-CE convention).

## Files

| Path | What it is |
|---|---|
| `common/mid_level.py` | Parses NaVILA text (same regexes as `evaluation/vlnce_baselines/navila_trainer.py`). `MidLevelController` turns an action into velocity commands, closed loop on odometry, with a persistent heading reference (Habitat-like exact headings, no drift). Pure Python. |
| `common/client.py` | `FrameHistory`, NaVILA frame sampling (identical to `sample_and_pad_images`), and a blocking/async HTTP client. |
| `server/navila_server.py` | FastAPI server around `scripts/inference/quick_infer.py`'s loader. Includes the lm_head last-token fix and OOM retry (see VRAM below). |
| `sim/run_navila_go2.py` | Isaac Sim runner: scene, Go2, camera profiles, low-level policy, async NaVILA calls, UI panel, recording, benchmark mode. |
| `sim/scenes.py` | Scenes with spawn poses and language episodes (hospital: 5, office: 2). |
| `sim/scout_scene.py` | Renders labeled floor plans and robot-height views, for writing new episodes. |
| `real/navila_go2_node.py` | ROS 2 node for the real Go2 via go2_sdk. **Untested on hardware**, dry-run by default. |
| `../legged-loco-main/isaaclab_exts/leggedloco/` | legged-loco Go2 policies (`go2_base`, `go2_vision`) **ported to Isaac Lab 2.x**. |
| `../legged-loco-main/scripts/train_lab2.py` | Training script for the port. |

## Low-level policy

| `--policy` | Policy |
|---|---|
| `legged_loco` (default) | **legged-loco `go2_base`** (NaVILA's low level), ported and trained here. Uses the newest checkpoint in `legged-loco-main/logs/rsl_rl/go2_base/` (override with `--checkpoint`). No base-velocity input; IMU roll/pitch/yaw; 9-step proprioceptive history; 20 ms actuator delay; rough terrain. The sim-to-real candidate. |
| `isaaclab_flat` | Isaac Lab's stock flat-ground Go2 policy, already trained on this machine. Uses base linear velocity (not measurable on a real robot) and **barely turns in place**: 23° of a commanded 90° in 11 s. |

Test the mid- and low-level layers without NaVILA by replaying fixed outputs:
```bash
./navila_go2/scripts/run_sim.sh --scene hospital --spawn 0,0,0 --instruction test --headless \
  --scripted "The next action is move forward 100 cm.;The next action is turn left 90 degrees.;The next action is stop."
```

Train the port (4096 envs, ~2.4 s/iteration on an RTX 4080 Super):
```bash
conda activate env_isaaclab
python legged-loco-main/scripts/train_lab2.py --task go2_base --headless --max_iterations 2000
```
What the port changed: namespaces `omni.isaac.lab*` → `isaaclab*`; `quat_rotate_inverse` → `quat_apply_inverse`;
`attach_yaw_only` → `ray_alignment`; legged-loco's `RslRlVecEnvHistoryWrapper(history_length=9)` became Isaac Lab's
native observation history (group `proprio`) plus rsl-rl 5 `obs_groups={"actor": ["policy", "proprio"]}`. The
actor input is the same 450 dims as the original. `go2_vision`'s flat-patch key was fixed so every sub-terrain
provides `init_pos`.

## Results

NaVILA 4-bit, legged-loco `go2_base` (iteration 1000), `habitat` camera, closed-loop mid level.
Videos and logs are in `output/sim_runs/benchmark/`.

| Scene | Ep | Instruction (abridged) | Outcome | Decisions | Path | NE |
|---|---|---|---|---|---|---|
| hospital | 0 | toward the green chairs, stop in front | stopped early | 3 | 1.5 m | 2.15 m ✓@3m |
| hospital | 1 | down the hall past reception, stop at vending machine | stopped | 12 | 9.2 m | 2.59 m ✓@3m |
| hospital | 2 | turn around, go to the green-chair waiting area | **success** | 8 | 10.1 m | 1.49 m |
| hospital | 3 | past reception, turn left into hall, stop at stairs | ✗ spun in place (62 left turns) | 80 | 7.9 m | 13.7 m |
| hospital | 4 | toward stairs, left into the long hallway, stop at the bed at its end | **success** | 40 | 33.4 m | 1.14 m |
| office | 0 | through the glass doors, stop in the hallway | **success** | 11 | 8.0 m | 2.09 m |
| office | 1 | turn left, through the glass doors | ✗ walked west | 44 | 44.5 m | 16.4 m |

**5 of 7 episodes ended within 3 m of the goal** (VLN-CE success threshold). The long-horizon hospital episode 4
also succeeded live in the GUI (40 decisions, NE 1.62 m). Both failures are high-level decision errors, not
execution errors: the scripted locomotion test shows the mid- and low-level stack lands forward moves within
1–3 cm and turns within 2–4°.

## Camera: what NaVILA sees

NaVILA was trained on 512×512 Habitat frames with a 90° FOV (plus YouTube videos), and it resizes every input
to a square. Three camera profiles were measured on hospital episodes 0–2 (4-bit NaVILA, flat low-level policy):

| `--cam_profile` | Camera | SR@3m | Mean NE |
|---|---|---|---|
| **`habitat`** (default) | head, 512×512, 90° | **1.00** | **2.15 m** |
| `go2` | real Go2 front camera, 16:9, 73° (squashed to a square) | 0.67 | 4.54 m |
| `mast` | 512×512, 90°, ~1.2 m high (Habitat's agent height) | 0.67 | 3.55 m |

So the FOV and aspect ratio matter more than height. For the real robot, prefer a square crop
(`--image_mode crop`, sim profile `go2_crop`) or a wider-FOV camera over squashing the full 16:9 frame.

## Real Go2 (go2_sdk), prepared but untested

`real/navila_go2_node.py` subscribes to `camera/image_raw` and `odom`, and publishes `cmd_vel`. go2_sdk's twist_mux
gives the joystick priority 10 over navigation's 5, so the remote always overrides NaVILA. On the real robot the
low level is the Go2's built-in sport-mode controller. Deploying the trained legged-loco policy there needs a
low-level (joint) control bridge, which is not done yet.
```bash
source /opt/ros/humble/setup.bash
python3 navila_go2/real/navila_go2_node.py --instruction "Walk to the door and stop."            # dry run
python3 navila_go2/real/navila_go2_node.py --instruction "Walk to the door and stop." --enable_motion
```

## Constraints and known limitations

- **VRAM (16 GB)**: NaVILA 4-bit peaks at about 7.2 GB, Isaac Sim with three cameras takes about 6 GB, and the
  desktop about 0.6 GB. The server releases cached blocks after each call, slices the prefill logits to the last
  token (that alone saves ~1.3 GB versus stock transformers 4.37), and retries once on OOM. Do not train while the
  demo runs.
- **Scenes**: NaVILA's own benchmark uses Matterport3D (licensed, not available here). NVIDIA's Hospital and Office
  scenes are used instead. Episode goals were placed from scouted floor plans, and they only score runs: NaVILA
  never sees them.
- **4-bit** quantization is slightly less accurate than fp16 (fp16 does not fit next to Isaac Sim).
- PhysX scene queries return nothing under Isaac Lab's GPU pipeline, so the scout's floor/wall probe prints `None`.
  The colliders themselves work (2161 colliders in the hospital).
- `go2_vision` (LiDAR) uses Isaac Lab's single-mesh ray caster. In the indoor scenes it would only see the hidden
  ground plane. Making it see the furniture needs `MultiMeshRayCaster` or a merged scene mesh.

## Troubleshooting

- `RuntimeError: operator torchvision::nms does not exist` in the navila env: a torch 2.2.2 in `~/.local` shadows
  the env's torch 2.3.0. `start_server.sh` puts the env's site-packages first, so use the script.
- `CUDA out of memory` on the server: close other GPU apps (Chrome alone uses ~0.5 GB), or run the sim without
  `--record` (two fewer cameras).
