"""NaVILA vision-language navigation on a simulated Unitree Go2 in Isaac Sim / Isaac Lab.

    NaVILA (VLA, high level, ~1 Hz)  --"turn left 30 degrees"-->  MidLevelController  --(vx, vy, wz)-->
    legged locomotion RL policy (50 Hz)  --joint targets-->  PD actuators (200 Hz physics)

Start the NaVILA server first (conda env `navila`), then run this with the Isaac Lab python (conda env `env_isaaclab`):

  python navila_go2/sim/run_navila_go2.py --scene hospital --episode 0
  python navila_go2/sim/run_navila_go2.py --scene hospital --instruction "Walk past the reception desk and stop at the benches."
  python navila_go2/sim/run_navila_go2.py --scene hospital --episode 1 --headless --record   # fully automatic run + mp4

In the GUI a "NaVILA Go2" panel shows the robot's camera, lets you type new instructions and shows NaVILA's decisions.
"""

import argparse
import os
import os.path as osp
import sys

from isaaclab.app import AppLauncher

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
LEGGED_LOCO_EXT = osp.join(REPO_ROOT, "legged-loco-main", "isaaclab_exts", "leggedloco")
LEGGED_LOCO_LOGS = osp.join(REPO_ROOT, "legged-loco-main", "logs", "rsl_rl", "go2_base")
POLICIES = {
    # legged-loco go2_base ported to Isaac Lab 2.x (train: legged-loco-main/scripts/train_lab2.py); newest checkpoint
    "legged_loco": ("go2_base_play", None),
    # Isaac Lab's stock flat-ground Go2 policy trained on this machine (weak at turning in place)
    "isaaclab_flat": (
        "Isaac-Velocity-Flat-Unitree-Go2-Play-v0",
        osp.expanduser("~/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-06-22_12-26-52/model_1499.pt"),
    ),
}


def latest_checkpoint(log_root: str) -> str | None:
    """Newest model_<N>.pt of the newest run under an rsl_rl experiment directory."""
    import glob
    import re

    for run in sorted(glob.glob(osp.join(log_root, "*")), reverse=True):
        ckpts = glob.glob(osp.join(run, "model_*.pt"))
        if ckpts:
            return max(ckpts, key=lambda p: int(re.search(r"model_(\d+)\.pt$", p).group(1)))
    return None

parser = argparse.ArgumentParser(description="NaVILA + legged locomotion on a simulated Go2.")
parser.add_argument("--scene", default="hospital", help="Scene name from navila_go2/sim/scenes.py.")
parser.add_argument("--episode", type=int, default=None, help="Episode index of the scene (instruction + goal).")
parser.add_argument("--episodes", default=None, help='Benchmark several episodes in one run: "0,2,3" or "all".')
parser.add_argument("--instruction", default=None, help="Free-form instruction (overrides --episode's text).")
parser.add_argument("--spawn", default=None, help='Override spawn "x,y,yaw_deg".')
parser.add_argument("--server", default="http://127.0.0.1:8000", help="NaVILA server URL.")
parser.add_argument("--scripted", default=None,
                    help='Replace NaVILA by fixed outputs to test the mid/low level, e.g. "The next action is move '
                         'forward 100 cm.;The next action is turn left 90 degrees.;The next action is stop."')
parser.add_argument("--policy", default="legged_loco", choices=list(POLICIES),
                    help="Low-level locomotion policy (legged_loco falls back to isaaclab_flat if not trained yet).")
parser.add_argument("--task", default=None, help="Override the Isaac Lab task (env/agent cfg) of --policy.")
parser.add_argument("--checkpoint", default=None, help="Override the rsl_rl checkpoint (model_*.pt) of --policy.")
parser.add_argument("--max_actions", type=int, default=80, help="Max NaVILA decisions per episode.")
parser.add_argument("--open_loop", action="store_true", help="Paper-style timed execution instead of closed loop.")
parser.add_argument("--lin_speed", type=float, default=0.5)
parser.add_argument("--ang_speed_deg", type=float, default=30.0)
parser.add_argument("--snap", action="store_true", help="Snap NaVILA values to the Habitat grid (25cm / 15deg).")
parser.add_argument("--cam_profile", default="habitat", choices=["habitat", "go2", "go2_crop", "mast"],
                    help="habitat: 512x512 90deg on the head (NaVILA training optics, best in our tests); go2: the real "
                         "Go2 front camera (16:9, 73deg, go2_sdk calibration); go2_crop: its center square (45deg); "
                         "mast: 512x512 90deg at ~1.2 m like Habitat's agent.")
parser.add_argument("--cam_pos", default=None, help="Override the profile's camera offset in the base frame (m).")
parser.add_argument("--cam_pitch_deg", type=float, default=0.0, help="Ego camera pitch, positive looks down.")
parser.add_argument("--cam_hfov_deg", type=float, default=None, help="Override the profile's horizontal FOV.")
parser.add_argument("--pause_while_thinking", action="store_true", help="Freeze physics during NaVILA inference.")
parser.add_argument("--record", action="store_true", help="Write an mp4 (chase view + ego view + overlay).")
parser.add_argument("--no_trace", action="store_true", help="Do not draw the traveled path.")
parser.add_argument("--show_goal", action="store_true", help="Show the episode goal region (NaVILA may see it!).")
parser.add_argument("--real_time", action="store_true", help="Throttle the loop to real time (GUI default).")
parser.add_argument("--out", default=osp.join(REPO_ROOT, "output", "sim_runs"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
_task, _ckpt = POLICIES[args.policy]
if args.policy == "legged_loco" and not (args.checkpoint or latest_checkpoint(LEGGED_LOCO_LOGS)):
    print("[navila-go2] no trained legged-loco go2_base checkpoint found, using the isaaclab_flat policy")
    args.policy = "isaaclab_flat"
    _task, _ckpt = POLICIES["isaaclab_flat"]
args.task = args.task or _task
args.checkpoint = args.checkpoint or _ckpt or latest_checkpoint(LEGGED_LOCO_LOGS)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import textwrap  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab.envs.mdp as base_mdp  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab_tasks  # noqa: E402, F401
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

if not args.headless:
    import carb

    # Editor gizmos (light / camera / audio icons) are otherwise drawn into the robot camera images NaVILA sees.
    for _key in ("/app/viewport/show/lights", "/app/viewport/show/camera", "/app/viewport/show/audio"):
        carb.settings.get_settings().set(_key, False)

sys.path.insert(0, REPO_ROOT)
if osp.isdir(LEGGED_LOCO_EXT):
    sys.path.insert(0, LEGGED_LOCO_EXT)
    import leggedloco  # noqa: E402, F401  (registers the ported legged-loco tasks)

from concurrent.futures import Future  # noqa: E402

from navila_go2.common.client import FrameHistory, NaVilaClient  # noqa: E402
from navila_go2.common.mid_level import STOP, MidLevelConfig, MidLevelController, parse_navila_output  # noqa: E402
from navila_go2.sim.scenes import SCENES, Episode  # noqa: E402


CAM_PROFILES = {
    # pos in the base frame (base is ~0.30 m above the floor when standing)
    "habitat": dict(pos=(0.34, 0.0, 0.08), hfov=90.0, width=512, height=512),
    "go2": dict(pos=(0.32, 0.0, 0.04), hfov=73.0, width=640, height=360),  # fx=864 @1280x720 (go2_sdk calibration)
    "go2_crop": dict(pos=(0.32, 0.0, 0.04), hfov=45.2, width=512, height=512),  # its center square (real node: crop)
    "mast": dict(pos=(0.0, 0.0, 0.90), hfov=90.0, width=512, height=512),
}
CAM = dict(CAM_PROFILES[args.cam_profile])
if args.cam_pos:
    CAM["pos"] = tuple(float(v) for v in args.cam_pos.split(","))
if args.cam_hfov_deg:
    CAM["hfov"] = args.cam_hfov_deg


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def configure_env(env_cfg, scene, spawn):
    """Turn a locomotion training env cfg into a single-robot navigation env in an indoor scene."""
    x, y, z, yaw_deg = spawn
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 1.0e6
    env_cfg.viewer.eye = (x - 3.0, y, z + 3.0)
    env_cfg.viewer.lookat = (x, y, z)

    # Physics ground plane at z=0 (hidden later; backs up the scene's own floor collider), plus the indoor scene as
    # a static asset. Ray casters of the policy (critic height scan) keep pointing at the plane.
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.scene.terrain.visual_material = None
    env_cfg.scene.indoor = AssetBaseCfg(prim_path="/World/Indoor", spawn=sim_utils.UsdFileCfg(usd_path=scene.usd_path))
    if getattr(env_cfg.curriculum, "terrain_levels", None) is not None:
        env_cfg.curriculum.terrain_levels = None

    # Robot start pose.
    env_cfg.scene.robot.init_state.pos = (x, y, z + 0.42)
    env_cfg.scene.robot.init_state.rot = yaw_quat(math.radians(yaw_deg))
    # Reset exactly to the robot's default root state (= the spawn pose). Replaces e.g. go2_vision's
    # reset_root_state_from_terrain, which needs terrain-generator flat patches that an indoor scene does not have.
    env_cfg.events.reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform, mode="reset", params={"pose_range": {}, "velocity_range": {}}
    )
    # training-time domain randomization off: deterministic, comparable runs across low-level policies
    for name in ("push_robot", "base_external_force_torque", "add_base_mass", "base_com", "actuator_gains",
                 "physics_material"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)

    # The velocity command is written by the mid-level controller every step: never resample it.
    cmd = env_cfg.commands.base_velocity
    cmd.resampling_time_range = (1.0e9, 1.0e9)
    cmd.heading_command = False
    cmd.rel_standing_envs = 0.0
    cmd.rel_heading_envs = 0.0
    cmd.ranges.heading = None
    cmd.ranges.lin_vel_x = cmd.ranges.lin_vel_y = cmd.ranges.ang_vel_z = (0.0, 0.0)
    cmd.debug_vis = False  # the arrows would show up in the ego camera

    # No episode resets: the navigation loop decides when an episode ends.
    for name in list(vars(env_cfg.terminations)):
        if not name.startswith("_"):
            setattr(env_cfg.terminations, name, None)

    # NaVILA's eyes: RGB camera on the robot.
    px, py, pz = CAM["pos"]
    pitch = math.radians(args.cam_pitch_deg)
    focal = 12.0
    h_ap = 2 * focal * math.tan(math.radians(CAM["hfov"]) / 2)
    env_cfg.scene.ego_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/ego_cam",
        update_period=0.0,
        height=CAM["height"],
        width=CAM["width"],
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal,
            horizontal_aperture=h_ap,
            vertical_aperture=h_ap * CAM["height"] / CAM["width"],  # square pixels
            clipping_range=(0.05, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(px, py, pz), rot=(math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0), convention="world"
        ),
    )
    if args.record:
        env_cfg.scene.chase_cam = CameraCfg(
            prim_path="/World/ChaseCam",
            update_period=0.0,
            height=540,
            width=960,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, horizontal_aperture=24.0, clipping_range=(0.05, 200.0)),
        )
        # Bird's-eye floor plan: TOPCAM_HEIGHT above the floor, the near clip plane slices off everything higher
        # than TOPCAM_CUT (ceilings), so the path through rooms and corridors is visible. ~16 x 9 m field of view.
        env_cfg.scene.top_cam = CameraCfg(
            prim_path="/World/TopCam",
            update_period=0.0,
            height=540,
            width=960,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                horizontal_aperture=24.0 * 16.0 / TOPCAM_HEIGHT,
                clipping_range=(TOPCAM_HEIGHT - TOPCAM_CUT, TOPCAM_HEIGHT + 5.0),
            ),
        )
    return env_cfg


TOPCAM_HEIGHT = 20.0
TOPCAM_CUT = 2.1
# camera looking straight down with image-up = +Y and image-right = +X (q = q_yaw(90deg) * q_pitch(90deg))
TOPCAM_QUAT = (0.5, -0.5, 0.5, 0.5)


def ensure_static_colliders(root_path: str) -> int:
    """Indoor assets usually ship with colliders; add triangle-mesh colliders if this one does not."""
    stage = omni.usd.get_context().get_stage()
    meshes, with_col = [], 0
    # environment assets are mostly instanced: colliders live inside the instance prototypes
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            with_col += 1
        elif prim.IsA(UsdGeom.Mesh) and not prim.IsInstanceProxy():
            meshes.append(prim)
    if with_col == 0:
        for prim in meshes:
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("none")
        print(f"[navila-go2] scene had no colliders: added triangle-mesh colliders to {len(meshes)} meshes")
    return with_col


def hide_ground_plane():
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/ground")
    if prim.IsValid():
        UsdGeom.Imageable(prim).MakeInvisible()


def letterbox(rgb: np.ndarray, size: int) -> np.ndarray:
    """Fit an image into a size x size black square, keeping its aspect ratio."""
    h, w = rgb.shape[:2]
    s = size / max(h, w)
    small = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), np.uint8)
    y0, x0 = (size - small.shape[0]) // 2, (size - small.shape[1]) // 2
    out[y0 : y0 + small.shape[0], x0 : x0 + small.shape[1]] = small
    return out


class NavPanel:
    """omni.ui control panel: ego view, instruction box, NaVILA decision log."""

    def __init__(self, instruction: str):
        import omni.ui as ui

        self.ui = ui
        self.go_requested, self.stop_requested, self.reset_requested = None, False, False
        self.window = ui.Window("NaVILA Go2", width=440, height=900)
        self.provider = ui.ByteImageProvider()
        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label("Robot camera (NaVILA input)", height=18)
                ui.ImageWithProvider(self.provider, width=420, height=420)
                ui.Label("Instruction", height=18)
                self.instr = ui.StringField(multiline=True, height=90)
                self.instr.model.set_value(instruction or "")
                with ui.HStack(height=32, spacing=6):
                    ui.Button("Go", clicked_fn=self._go)
                    ui.Button("Stop", clicked_fn=self._stop)
                    ui.Button("Reset robot", clicked_fn=self._reset)
                self.status = ui.Label("", height=60, word_wrap=True)
                with ui.ScrollingFrame(height=260):
                    self.log = ui.Label("", word_wrap=True, alignment=ui.Alignment.LEFT_TOP)
        self._lines = []

    def _go(self):
        self.go_requested = self.instr.model.get_value_as_string().strip()

    def _stop(self):
        self.stop_requested = True

    def _reset(self):
        self.reset_requested = True

    def set_image(self, rgb: np.ndarray):
        small = letterbox(rgb, 256)
        rgba = np.concatenate([small, np.full(small.shape[:2] + (1,), 255, np.uint8)], axis=-1)
        if hasattr(self.provider, "set_data_array"):
            self.provider.set_data_array(rgba, [256, 256])
        else:
            self.provider.set_bytes_data(rgba.flatten().tolist(), [256, 256])

    def set_status(self, text: str):
        self.status.text = text

    def add_log(self, line: str):
        self._lines = (self._lines + [line])[-40:]
        self.log.text = "\n".join(reversed(self._lines))


class ScriptedClient:
    """Stands in for NaVilaClient: replays fixed NaVILA outputs (no server, no GPU needed for the VLA)."""

    def __init__(self, texts: list[str]):
        self.texts = texts
        self.i = 0

    def wait_ready(self) -> dict:
        return {"scripted": self.texts}

    def act_async(self, instruction, history) -> Future:
        text = self.texts[min(self.i, len(self.texts) - 1)]
        self.i += 1
        fut = Future()
        fut.set_result((parse_navila_output(text), {"text": text, "latency_s": 0.0}))
        return fut


class Recorder:
    """1920x1080 dashboard: chase view | robot camera (NaVILA input) / bird's-eye floor plan | instruction + log."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, path: str, fps: int = 25):
        self.path = path
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))

    def _label(self, img, text, color=(0, 255, 255)):
        cv2.rectangle(img, (0, 0), (12 + 11 * len(text), 30), (0, 0, 0), -1)
        cv2.putText(img, text, (6, 21), self.FONT, 0.6, color, 2, cv2.LINE_AA)

    def write(self, chase_rgb, ego_rgb, top_rgb, instruction, status, log_lines):
        chase = chase_rgb[:, :, ::-1].copy()
        ego = np.zeros((540, 960, 3), np.uint8)
        ego[:, 210:750] = letterbox(ego_rgb, 540)[:, :, ::-1]
        top = top_rgb[:, :, ::-1].copy()
        cv2.drawMarker(top, (480, 270), (0, 0, 255), cv2.MARKER_TRIANGLE_UP, 18, 2)  # robot is always centered
        text = np.full((540, 960, 3), 24, np.uint8)
        y = 40
        for line in textwrap.wrap(f"Instruction: {instruction}", 70)[:5]:
            cv2.putText(text, line, (20, y), self.FONT, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            y += 32
        y += 10
        cv2.putText(text, status, (20, y), self.FONT, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        y += 40
        for line in log_lines[-(480 - y) // 28:]:
            cv2.putText(text, line, (20, y), self.FONT, 0.6, (190, 230, 190), 1, cv2.LINE_AA)
            y += 28
        self._label(chase, "Isaac Sim - Unitree Go2")
        self._label(ego, "NaVILA input: robot camera")
        self._label(top, "Floor plan (ceiling cut away), red = path")
        frame = np.concatenate([np.concatenate([chase, ego], axis=1), np.concatenate([top, text], axis=1)], axis=0)
        self.writer.write(frame)

    def close(self):
        self.writer.release()
        print(f"[navila-go2] video: {self.path}")


def build_queue(scene) -> list[tuple[str, str, Episode | None, tuple]]:
    """(label, instruction, episode-or-None, spawn) for every episode to run."""
    custom_spawn = None
    if args.spawn:
        sx, sy, syaw = (float(v) for v in args.spawn.split(","))
        custom_spawn = (sx, sy, scene.floor_z or 0.0, syaw)
    if args.episodes:
        ids = range(len(scene.episodes)) if args.episodes == "all" else [int(i) for i in args.episodes.split(",")]
    elif args.episode is not None:
        ids = [args.episode]
    else:
        ids = []
    queue = []
    for i in ids:
        ep = scene.episodes[i]
        spawn = custom_spawn or ep.spawn or scene.spawn
        if args.instruction:  # same start/goal, different wording
            queue.append((f"ep{i}_custom", args.instruction, ep, spawn))
        else:
            queue.append((f"ep{i}", ep.instruction, ep, spawn))
    if not queue and args.instruction:
        queue.append(("custom", args.instruction, None, custom_spawn or scene.spawn))
    return queue


def main():
    scene = SCENES[args.scene]
    queue = build_queue(scene)
    if args.headless and not queue:
        raise SystemExit("--headless needs --instruction, --episode or --episodes")
    first_spawn = queue[0][3] if queue else scene.spawn

    # ---------------- environment + low-level policy ----------------
    env_cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=1)
    env_cfg = configure_env(env_cfg, scene, first_spawn)
    env = gym.make(args.task, cfg=env_cfg)
    hide_ground_plane()
    n_col = ensure_static_colliders("/World/Indoor")
    print(f"[navila-go2] scene '{scene.name}': {n_col} collision prims")

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device=venv.unwrapped.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=venv.unwrapped.device)
    print(f"[navila-go2] low-level policy: {args.checkpoint}")

    uenv = venv.unwrapped
    robot = uenv.scene["robot"]
    ego_cam = uenv.scene["ego_cam"]
    chase_cam = uenv.scene["chase_cam"] if args.record else None
    top_cam = uenv.scene["top_cam"] if args.record else None
    cmd_term = uenv.command_manager.get_term("base_velocity")
    dt = uenv.step_dt

    # ---------------- high level ----------------
    if args.scripted:
        client = ScriptedClient([t.strip() for t in args.scripted.split(";") if t.strip()])
    else:
        client = NaVilaClient(args.server)
        print(f"[navila-go2] waiting for NaVILA server at {args.server} ...")
    print(f"[navila-go2] NaVILA server: {client.wait_ready()}")
    ctrl = MidLevelController(
        MidLevelConfig(lin_speed=args.lin_speed, ang_speed=math.radians(args.ang_speed_deg), closed_loop=not args.open_loop)
    )
    history = FrameHistory()

    # ---------------- visuals / outputs ----------------
    run_dir = osp.join(args.out, time.strftime("%Y%m%d-%H%M%S") + f"_{scene.name}")
    os.makedirs(run_dir, exist_ok=True)
    log_f = open(osp.join(run_dir, "decisions.jsonl"), "w")
    panel = None if args.headless else NavPanel(queue[0][1] if queue else "")
    recorder = None
    results = []
    trace_markers = None
    if not args.no_trace:
        trace_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/NavTrace",
                markers={
                    "dot": sim_utils.SphereCfg(
                        radius=0.04, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1))
                    )
                },
            )
        )
    goal_marker = None
    if args.show_goal:
        goal_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/NavGoal",
                markers={
                    "goal": sim_utils.CylinderCfg(
                        radius=1.0,
                        height=0.02,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.2), opacity=0.4),
                    )
                },
            )
        )

    # ---------------- state ----------------
    state = {"mode": "idle", "instruction": None, "future": None, "n_actions": 0, "t0": 0.0, "path_len": 0.0,
             "last_text": "", "last_action": "", "start_xy": None, "result": None, "episode": None, "label": None}
    trace_pts = []

    def pose():
        p = robot.data.root_pos_w[0]
        return float(p[0]), float(p[1]), float(robot.data.heading_w[0])

    def ego_rgb():
        return ego_cam.data.output["rgb"][0, ..., :3].cpu().numpy()

    def settle(seconds=1.0):
        nonlocal obs
        cmd_term.vel_command_b[:] = 0.0
        for _ in range(int(seconds / dt)):
            with torch.inference_mode():
                obs, _, _, _ = venv.step(policy(obs))
            sim_time[0] += dt

    def teleport(spawn):
        """Move the robot to a spawn pose through the env's own reset event (default root state)."""
        nonlocal obs
        x, y, z, yaw_deg = spawn
        # manager buffers were (lazily) created by steps under inference mode, so reset under it as well
        with torch.inference_mode():
            root = robot.data.default_root_state
            root[0, :3] = torch.tensor([x, y, z + 0.42], device=uenv.device) - uenv.scene.env_origins[0]
            root[0, 3:7] = torch.tensor(yaw_quat(math.radians(yaw_deg)), device=uenv.device)
            obs, _ = venv.reset()
        settle()

    def start_episode(label, text, ep: Episode | None, spawn=None):
        nonlocal recorder
        ctrl.active = False
        if spawn is not None:
            teleport(spawn)
        history.reset()
        trace_pts.clear()
        ctrl.reset_heading(pose()[2])
        if isinstance(client, ScriptedClient):
            client.i = 0
        state.update(mode="thinking", instruction=text, n_actions=0, t0=sim_time[0], path_len=0.0, result=None,
                     start_xy=pose()[:2], episode=ep, future=None, label=label, done_steps=0, last_text="", log=[])
        if goal_marker is not None and ep is not None:
            goal_marker.visualize(torch.tensor([[*ep.goal_xy, (scene.floor_z or 0.0) + 0.01]]),
                                  scales=torch.tensor([[ep.success_radius, ep.success_radius, 1.0]]))
        if args.record:
            recorder = Recorder(osp.join(run_dir, f"{label}.mp4"))
        print(f"[navila-go2] new episode [{label}]: {text!r}")
        if panel:
            panel.add_log(f"NEW [{label}]: {text}")
        query_navila()

    def finish_episode(reason):
        x, y, _ = pose()
        res = {"label": state["label"], "instruction": state["instruction"], "reason": reason,
               "num_actions": state["n_actions"], "sim_time_s": round(sim_time[0] - state["t0"], 2),
               "path_length_m": round(state["path_len"], 2), "final_xy": [round(x, 2), round(y, 2)]}
        ep = state["episode"]
        if ep is not None:
            ne = math.dist((x, y), ep.goal_xy)
            stopped = reason == "stop"
            res.update(goal_xy=list(ep.goal_xy), navigation_error_m=round(ne, 2),
                       success=bool(stopped and ne <= ep.success_radius),  # episode's own (strict) radius
                       success_3m=bool(stopped and ne <= 3.0))  # VLN-CE convention
        state.update(mode="done", result=res, done_steps=0)
        state["log"].append(f"DONE ({reason}): {res['path_length_m']} m walked"
                            + (f", {res['navigation_error_m']} m from goal, success@3m={res['success_3m']}"
                               if ep is not None else ""))
        results.append(res)
        print(f"[navila-go2] episode finished: {json.dumps(res)}")
        log_f.write(json.dumps({"episode_result": res}) + "\n")
        log_f.flush()
        if panel:
            panel.add_log(f"DONE ({reason}): " + (f"NE={res.get('navigation_error_m')} m success={res.get('success')}"
                                                  if ep else f"{res['path_length_m']} m in {res['num_actions']} actions"))

    def query_navila(add_frame=True):
        # one frame per decision (+ one per completed 25 cm / 15 deg unit while executing); retries reuse the history
        if add_frame:
            history.add(ego_rgb())
        state["future"] = client.act_async(state["instruction"], history)
        state["mode"] = "thinking"

    sim_time = [0.0]
    obs = venv.get_observations()
    settle()  # let the robot get on its feet before the first decision

    if queue:
        label, text, ep, spawn = queue.pop(0)
        start_episode(label, text, ep, spawn)

    step = 0
    last_xy = pose()[:2]
    hb_wall, hb_sim = time.time(), sim_time[0]
    while simulation_app.is_running():
        t_wall = time.time()
        if t_wall - hb_wall > 15.0:  # heartbeat: sim speed + free VRAM (the GPU is shared with NaVILA)
            free, total = torch.cuda.mem_get_info()
            print(f"[navila-go2] heartbeat: sim {sim_time[0]:.1f}s, {(sim_time[0] - hb_sim) / (t_wall - hb_wall):.2f}x "
                  f"real time, mode={state['mode']}, GPU free {free / 2**30:.1f}/{total / 2**30:.1f} GiB")
            hb_wall, hb_sim = t_wall, sim_time[0]

        # ---- UI events ----
        if panel:
            if panel.reset_requested:
                panel.reset_requested = False
                ctrl.active = False
                teleport(first_spawn)
                state["mode"] = "idle"
                panel.add_log("robot reset")
            if panel.stop_requested:
                panel.stop_requested = False
                ctrl.active = False
                if state["mode"] in ("thinking", "executing"):
                    finish_episode("user_stop")
            if panel.go_requested:
                text, panel.go_requested = panel.go_requested, None
                if state["mode"] in ("thinking", "executing"):
                    finish_episode("user_new_instruction")
                # an instruction typed in the panel continues from wherever the robot is
                match = next((e for e in scene.episodes if e.instruction.strip() == text), None)
                start_episode(f"ui{len(results) + 1}", text, match, None)

        # ---- high level: consume NaVILA decisions ----
        if state["mode"] == "thinking" and state["future"] is None and time.time() >= state.get("retry_at", 0.0):
            query_navila(add_frame=False)
        if state["mode"] == "thinking" and state["future"] is not None and state["future"].done():
            fut, state["future"] = state["future"], None
            try:
                action, out = fut.result()
            except Exception as e:  # server hiccup: retry the same request after a short back-off
                print(f"[navila-go2] NaVILA request failed: {e}; retrying in 1 s")
                state["retry_at"] = time.time() + 1.0
                action = None
            if action is not None:
                state["n_actions"] += 1
                state["last_text"], state["last_action"] = out["text"], str(action)
                x, y, yaw = pose()
                rec = {"i": state["n_actions"], "sim_t": round(sim_time[0], 2), "pose": [round(x, 3), round(y, 3),
                       round(math.degrees(yaw), 1)], "text": out["text"], "action": str(action),
                       "latency_s": round(out["latency_s"], 3), "frames_in_history": len(history)}
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                print(f"[navila-go2] #{rec['i']:02d} t={rec['sim_t']:6.1f}s  {out['text']!r} -> {action}  "
                      f"({out['latency_s']:.2f}s)")
                state["log"].append(f"#{rec['i']:02d}  {out['text']}")
                if panel:
                    panel.add_log(f"#{rec['i']:02d} {action}   ({out['latency_s']:.1f}s)")
                if action.kind == STOP:
                    finish_episode("stop")
                elif state["n_actions"] >= args.max_actions:
                    finish_episode("max_actions")
                else:
                    if args.snap:
                        from navila_go2.common.mid_level import parse_navila_output

                        action = parse_navila_output(out["text"], snap=True)
                    ctrl.start(action, pose())
                    state["mode"] = "executing"
                    state["units_recorded"] = 0

        # ---- mid level: action -> velocity command ----
        if state["mode"] == "executing":
            cmd = ctrl.step(pose(), dt)
            # record a history frame at the start of every 25 cm / 15 deg unit (NaVILA's training granularity)
            while state["units_recorded"] < ctrl.units_done and ctrl.active:
                history.add(ego_rgb())
                state["units_recorded"] += 1
            if not ctrl.active:
                if ctrl.timed_out:
                    print("[navila-go2] action timed out (blocked?)")
                query_navila()
        else:
            cmd = (0.0, 0.0, 0.0)

        if args.pause_while_thinking and state["mode"] == "thinking":
            uenv.sim.render()
            continue

        # ---- low level: locomotion policy ----
        cmd_term.vel_command_b[0] = torch.tensor(cmd, device=uenv.device)
        with torch.inference_mode():
            obs, _, dones, _ = venv.step(policy(obs))
            if hasattr(policy, "reset"):
                policy.reset(dones)
        sim_time[0] += dt
        step += 1

        # ---- bookkeeping / visuals ----
        x, y, yaw = pose()
        state["path_len"] += math.dist((x, y), last_xy) if state["mode"] in ("thinking", "executing") else 0.0
        last_xy = (x, y)
        grav_z = float(robot.data.projected_gravity_b[0, 2])
        if grav_z > -0.5 and state["mode"] in ("thinking", "executing"):
            print("[navila-go2] robot fell over")
            finish_episode("fell")
        if trace_markers is not None and state["mode"] in ("thinking", "executing", "done"):
            if not trace_pts or math.dist(trace_pts[-1][:2], (x, y)) > 0.1:
                trace_pts.append((x, y, (scene.floor_z or 0.0) + 0.03))
                trace_markers.visualize(torch.tensor(trace_pts[-3000:]))

        if not args.headless:
            # chase view in the GUI viewport (close and high, so it stays inside ~3 m wide corridors)
            eye = (x - 1.2 * math.cos(yaw), y - 1.2 * math.sin(yaw), (scene.floor_z or 0.0) + 1.5)
            uenv.sim.set_camera_view(eye=eye, target=(x + 1.0 * math.cos(yaw), y + 1.0 * math.sin(yaw), 0.3))
            if step % 5 == 0:
                panel.set_image(ego_rgb())
                panel.set_status(
                    f"[{state['mode'].upper()}]  actions: {state['n_actions']}  cmd: vx={cmd[0]:.2f} wz={cmd[2]:.2f}\n"
                    f"NaVILA: {state['last_text']}"
                )
        if recorder is not None and step % 2 == 0 and state["mode"] != "idle":
            floor = scene.floor_z or 0.0
            eye = torch.tensor([[x - 2.2 * math.cos(yaw), y - 2.2 * math.sin(yaw), floor + 1.9]], device=uenv.device)
            tgt = torch.tensor([[x + 1.0 * math.cos(yaw), y + 1.0 * math.sin(yaw), floor + 0.3]], device=uenv.device)
            chase_cam.set_world_poses_from_view(eye, tgt)
            top_cam.set_world_poses(torch.tensor([[x, y, floor + TOPCAM_HEIGHT]], device=uenv.device),
                                    torch.tensor([TOPCAM_QUAT], device=uenv.device), convention="world")
            wz_deg = round(math.degrees(cmd[2])) or 0  # int, no "-0"
            status = (f"[{state['mode']}]  decision #{state['n_actions']}  cmd vx={cmd[0]:.2f} m/s "
                      f"wz={wz_deg:+d} deg/s  t={sim_time[0] - state['t0']:.1f}s")
            recorder.write(chase_cam.data.output["rgb"][0, ..., :3].cpu().numpy(), ego_rgb(),
                           top_cam.data.output["rgb"][0, ..., :3].cpu().numpy(), state["instruction"], status,
                           state["log"])

        if state["mode"] == "done" and state["done_steps"] >= 0:
            # 2 s of extra footage, then the next queued episode (or exit when headless)
            state["done_steps"] += 1
            if state["done_steps"] > int(2.0 / dt):
                state["done_steps"] = -1
                if recorder:
                    recorder.close()
                    recorder = None
                if queue:
                    label, text, ep, spawn = queue.pop(0)
                    start_episode(label, text, ep, spawn)
                elif args.headless:
                    break
        if args.real_time or not args.headless:
            time.sleep(max(0.0, dt - (time.time() - t_wall)))

    if recorder:
        recorder.close()
    log_f.close()
    write_summary(run_dir, results)
    env.close()


def write_summary(run_dir: str, results: list[dict]):
    with open(osp.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    scored = [r for r in results if "navigation_error_m" in r]
    print(f"[navila-go2] ===== {len(results)} episode(s), outputs in {run_dir} =====")
    for r in results:
        print(f"[navila-go2]   {r['label']:>10s}  {r['reason']:>12s}  actions={r['num_actions']:3d}  "
              f"path={r['path_length_m']:5.1f} m  NE={r.get('navigation_error_m', float('nan')):5.2f} m  "
              f"success={r.get('success', '-')}  success@3m={r.get('success_3m', '-')}")
    if scored:
        n = len(scored)
        print(f"[navila-go2]   SR={sum(r['success'] for r in scored) / n:.2f}  "
              f"SR@3m={sum(r['success_3m'] for r in scored) / n:.2f}  "
              f"NE={sum(r['navigation_error_m'] for r in scored) / n:.2f} m")


if __name__ == "__main__":
    main()
    simulation_app.close()
