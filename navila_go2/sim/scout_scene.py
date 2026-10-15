"""Render a labeled top-down map (and optional egocentric views) of an indoor scene.

Used to pick robot spawn poses, goal regions and write language instructions for a scene.
Run with the Isaac Lab python (conda env `env_isaaclab`):

  python navila_go2/sim/scout_scene.py --scene hospital
  python navila_go2/sim/scout_scene.py --scene hospital --views "2,3,0;5,-1,90" --view_height 0.4
"""

import argparse
import os
import os.path as osp
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--scene", default="hospital")
parser.add_argument("--usd", default=None, help="Override the scene USD path.")
parser.add_argument("--out", default=None, help="Output directory (default: output/scout/<scene>).")
parser.add_argument("--map_res", type=int, default=1600)
parser.add_argument("--cut_height", type=float, default=2.0, help="Clip geometry above floor+cut_height (ceilings).")
parser.add_argument("--grid", type=float, default=2.0, help="Grid spacing in meters on the map.")
parser.add_argument("--views", default="", help='Ego views "x,y,yaw_deg;x,y,yaw_deg" (4 headings if yaw omitted).')
parser.add_argument("--view_height", type=float, default=0.4, help="Ego camera height above the floor.")
parser.add_argument("--region", default="", help='Map only "xmin,xmax,ymin,ymax" instead of the whole bbox.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

import math  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..", "..")))
from navila_go2.sim.scenes import SCENES  # noqa: E402


def quat_from_yaw_pitch(yaw, pitch):
    """World-convention camera orientation (forward +X, up +Z): yaw about Z, then pitch about Y."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    # q = q_yaw * q_pitch
    return (cy * cp, -sy * sp, cy * sp, sy * cp)


def render(sim, cam, steps=6):
    for _ in range(steps):
        sim.step()
        cam.update(sim.get_physics_dt())
    return cam.data.output["rgb"][0, ..., :3].cpu().numpy()


def raycast_floor(x, y, z_from, max_dist=30.0):
    from omni.physx import get_physx_scene_query_interface

    hit = get_physx_scene_query_interface().raycast_closest((x, y, z_from), (0.0, 0.0, -1.0), max_dist)
    return hit["position"][2] if hit["hit"] else None


def probe_walls(x, y, z, max_dist=20.0):
    """Horizontal PhysX rays in 8 directions: shows whether walls/furniture have colliders."""
    from omni.physx import get_physx_scene_query_interface

    out = {}
    for deg in range(0, 360, 45):
        d = (math.cos(math.radians(deg)), math.sin(math.radians(deg)), 0.0)
        hit = get_physx_scene_query_interface().raycast_closest((x, y, z), d, max_dist)
        out[deg] = round(hit["distance"], 2) if hit["hit"] else None
    return out


def main():
    spec = SCENES.get(args.scene)
    usd_path = args.usd or spec.usd_path
    out_dir = args.out or osp.join("output", "scout", args.scene)
    os.makedirs(out_dir, exist_ok=True)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device or "cuda:0"))
    scene_cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
    scene_cfg.func("/World/Scene", scene_cfg)
    light = sim_utils.DomeLightCfg(intensity=800.0, color=(1.0, 1.0, 1.0))
    light.func("/World/ScoutLight", light)

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath("/World/Scene")
    rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(root).ComputeAlignedRange()
    lo, hi = rng.GetMin(), rng.GetMax()
    n_mesh, col_paths = 0, []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh):
            n_mesh += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            col_paths.append(str(prim.GetPath()))
    print(f"[scout] bbox min={tuple(round(v, 2) for v in lo)} max={tuple(round(v, 2) for v in hi)}")
    print(f"[scout] meshes={n_mesh} collision prims={len(col_paths)}")
    for p in col_paths[:: max(1, len(col_paths) // 12)][:12]:
        print(f"[scout]   collider: {p}")
    if args.region:
        x0, x1, y0, y1 = (float(v) for v in args.region.split(","))
        lo, hi = type(lo)(x0, y0, lo[2]), type(hi)(x1, y1, hi[2])

    # Top-down camera: narrow FOV from far away ~ orthographic. Near clip removes the ceiling.
    fov = math.radians(20.0)
    span = max(hi[0] - lo[0], hi[1] - lo[1]) * 1.05
    cam_z_rel = span / (2 * math.tan(fov / 2))
    focal = 24.0
    aperture = 2 * focal * math.tan(fov / 2)
    map_cam = Camera(
        CameraCfg(
            prim_path="/World/MapCam",
            height=args.map_res,
            width=args.map_res,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal, horizontal_aperture=aperture, clipping_range=(0.1, 1.0e5)
            ),
        )
    )
    ego_cam = Camera(
        CameraCfg(
            prim_path="/World/EgoCam",
            height=512,
            width=512,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, horizontal_aperture=24.0, clipping_range=(0.05, 1.0e4)),
        )
    )
    sim.reset()

    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    floor = raycast_floor(cx, cy, hi[2] - 0.05)
    print(f"[scout] floor z at bbox center (from above): {floor}")
    floor_z = spec.floor_z if spec.floor_z is not None else (floor if floor is not None else lo[2])

    cam_z = floor_z + cam_z_rel
    near = cam_z - (floor_z + args.cut_height)
    map_cam._sensor_prims[0].GetClippingRangeAttr().Set((max(near, 0.1), cam_z - floor_z + 5.0))
    pos = torch.tensor([[cx, cy, cam_z]], device=sim.device, dtype=torch.float32)
    rot = torch.tensor([quat_from_yaw_pitch(math.pi / 2, math.pi / 2)], device=sim.device, dtype=torch.float32)
    map_cam.set_world_poses(pos, rot, convention="world")
    img = render(sim, map_cam, steps=20)
    img = np.ascontiguousarray(img[:, :, ::-1])  # RGB -> BGR for cv2

    # With yaw=90 then pitch down: image up = +Y, image right = +X.
    m_per_px = span / args.map_res

    def to_px(x, y):
        return int(args.map_res / 2 + (x - cx) / m_per_px), int(args.map_res / 2 - (y - cy) / m_per_px)

    g = args.grid
    for gx in np.arange(math.floor(lo[0] / g) * g, hi[0] + g, g):
        u, _ = to_px(gx, cy)
        cv2.line(img, (u, 0), (u, args.map_res), (0, 200, 255), 1)
        cv2.putText(img, f"{gx:g}", (u + 2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    for gy in np.arange(math.floor(lo[1] / g) * g, hi[1] + g, g):
        _, v = to_px(cx, gy)
        cv2.line(img, (0, v), (args.map_res, v), (0, 200, 255), 1)
        cv2.putText(img, f"{gy:g}", (2, v - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    for ep in spec.episodes:
        cv2.circle(img, to_px(*ep.goal_xy), int(ep.success_radius / m_per_px), (0, 255, 0), 2)
    for sx, sy, syaw in [spec.spawn[:2] + (spec.spawn[3],)] + [e.spawn[:2] + (e.spawn[3],) for e in spec.episodes if e.spawn]:
        u, v = to_px(sx, sy)
        cv2.circle(img, (u, v), 6, (255, 0, 0), -1)
        r = math.radians(syaw)
        cv2.arrowedLine(img, (u, v), to_px(sx + 0.8 * math.cos(r), sy + 0.8 * math.sin(r)), (255, 0, 0), 2)
    cv2.imwrite(osp.join(out_dir, "topdown.png"), img)
    print(f"[scout] wrote {osp.join(out_dir, 'topdown.png')}  ({m_per_px * 100:.1f} cm/px, floor_z={floor_z:.3f})")

    # Egocentric views
    for k, item in enumerate(v for v in args.views.split(";") if v.strip()):
        vals = [float(t) for t in item.split(",")]
        x, y = vals[0], vals[1]
        yaws = [vals[2]] if len(vals) > 2 else [0, 90, 180, 270]
        fz = raycast_floor(x, y, floor_z + 1.5)
        print(f"[scout] view {k}: ({x}, {y}) floor hit z={fz}  wall distances: {probe_walls(x, y, floor_z + 0.3)}")
        tiles = []
        for yaw in yaws:
            p = torch.tensor([[x, y, floor_z + args.view_height]], device=sim.device, dtype=torch.float32)
            q = torch.tensor([quat_from_yaw_pitch(math.radians(yaw), 0.0)], device=sim.device, dtype=torch.float32)
            ego_cam.set_world_poses(p, q, convention="world")
            tile = np.ascontiguousarray(render(sim, ego_cam)[:, :, ::-1])
            cv2.putText(tile, f"({x:g},{y:g}) yaw={yaw:g}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            tiles.append(tile)
        cv2.imwrite(osp.join(out_dir, f"ego_{k}_{x:g}_{y:g}.jpg"), np.concatenate(tiles, axis=1))
    # app.close() can hang for minutes with active render products; nothing left to save.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
