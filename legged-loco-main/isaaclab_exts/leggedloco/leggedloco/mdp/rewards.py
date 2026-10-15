"""Reward terms of legged-loco (omni.isaac.leggedloco ... rewards/objnav_rewards.py), ported to Isaac Lab 2.x."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def collision_penalty(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet hitting vertical surfaces."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts_norm = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2].norm(dim=-1)
    vertical_contacts = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    return torch.any(contacts_norm > 5 * vertical_contacts, dim=1)


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output."""
    return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)


def power_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize power consumption of the robot."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.applied_torque) * torch.abs(asset.data.joint_vel), dim=1)


def stand_still_penalty(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint deviation from the default pose when the velocity command is small."""
    asset: RigidObject = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    rew = torch.sum(torch.abs(angle), dim=1)
    small_commands = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=-1) < 0.1
    return rew * small_commands
