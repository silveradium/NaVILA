"""Observation terms of legged-loco (omni.isaac.leggedloco), ported to Isaac Lab 2.x."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def base_rpy(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root roll, pitch, yaw (what the real Go2 IMU reports; used instead of projected gravity)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    root_quat_w = asset.data.root_quat_w

    qw = root_quat_w[:, 0]
    qx = root_quat_w[:, 1]
    qy = root_quat_w[:, 2]
    qz = root_quat_w[:, 3]

    roll = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    sinp = 2 * (qw * qy - qz * qx)
    pitch = torch.where(torch.abs(sinp) < 1, torch.asin(sinp), torch.sign(sinp) * (torch.pi / 2))
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

    return torch.stack((roll, pitch, yaw), dim=1)


# 2.5D voxel map around the robot built from the head LiDAR (same grid as the original)
voxel_size_xy = 0.06
range_x = [-0.8, 0.2 + 1e-9]
range_y = [-0.8, 0.8 + 1e-9]
range_z = [0.0, 5.0]


def height_map_lidar(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height map from the Go2 head LiDAR (Unitree L1, mounted upside down and tilted), in the sensor frame.

    Port of ``omni.isaac.leggedloco.leggedloco.mdp.observations.height_map_lidar``; the only API change is
    ``quat_rotate_inverse`` -> ``quat_apply_inverse``.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    hit_vec = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    hit_vec[torch.isinf(hit_vec)] = 0.0
    hit_vec[torch.isnan(hit_vec)] = 0.0

    hit_vec_shape = hit_vec.shape
    hit_vec = hit_vec.view(-1, hit_vec.shape[-1])
    robot_base_quat_w = env.scene["robot"].data.root_quat_w
    sensor_quat_default = (
        torch.tensor([-0.131, 0.0, -0.991, 0.0], device=robot_base_quat_w.device)
        .unsqueeze(0)
        .repeat(hit_vec_shape[0], 1)
    )
    sensor_quat_w = math_utils.quat_mul(robot_base_quat_w, sensor_quat_default)
    quat_w_dup = (sensor_quat_w.unsqueeze(1).repeat(1, hit_vec_shape[1], 1)).view(-1, sensor_quat_w.shape[-1])
    hit_vec_lidar_frame = math_utils.quat_apply_inverse(quat_w_dup, hit_vec)
    hit_vec_lidar_frame = hit_vec_lidar_frame.view(hit_vec_shape[0], hit_vec_shape[1], hit_vec_lidar_frame.shape[-1])

    num_envs = hit_vec_lidar_frame.shape[0]
    x_bins = torch.arange(range_x[0], range_x[1], voxel_size_xy, device=hit_vec_lidar_frame.device)
    y_bins = torch.arange(range_y[0], range_y[1], voxel_size_xy, device=hit_vec_lidar_frame.device)

    x = hit_vec_lidar_frame[..., 0]
    y = hit_vec_lidar_frame[..., 1]
    z = hit_vec_lidar_frame[..., 2]

    valid_indices = (
        (x > range_x[0]) & (x <= range_x[1]) & (y > range_y[0]) & (y <= range_y[1]) & (z >= range_z[0]) & (z <= range_z[1])
    )

    x_filtered = x[valid_indices]
    y_filtered = y[valid_indices]
    z_filtered = z[valid_indices]

    x_indices = torch.bucketize(x_filtered, x_bins) - 1
    y_indices = torch.bucketize(y_filtered, y_bins) - 1

    env_indices = torch.arange(num_envs, device=hit_vec_lidar_frame.device).unsqueeze(1).expand_as(valid_indices)
    flat_env_indices = env_indices[valid_indices]

    map_2_5D = torch.full((num_envs, len(x_bins), len(y_bins)), float("inf"), device=hit_vec_lidar_frame.device)
    linear_indices = flat_env_indices * len(x_bins) * len(y_bins) + x_indices * len(y_bins) + y_indices

    map_2_5D = map_2_5D.view(-1).scatter_reduce_(0, linear_indices, z_filtered, reduce="amin") - offset
    map_2_5D = torch.where(map_2_5D < 0.05, torch.tensor(0.0, device=map_2_5D.device), map_2_5D)
    map_2_5D = torch.where(torch.isinf(map_2_5D), torch.tensor(0.0, device=map_2_5D.device), map_2_5D)

    map_2_5D = map_2_5D.view(num_envs, len(x_bins), len(y_bins))
    return F.max_pool2d(map_2_5D, kernel_size=3, stride=1, padding=1).view(num_envs, -1)
