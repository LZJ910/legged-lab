# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if asset_cfg.body_ids == slice(None) or (isinstance(asset_cfg.body_ids, list) and len(asset_cfg.body_ids) != 1):
        raise ValueError("body_ids must be a single body id")
    # compute the error
    lin_vel_b = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids, :], asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    ).squeeze(1)
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if asset_cfg.body_ids == slice(None) or (isinstance(asset_cfg.body_ids, list) and len(asset_cfg.body_ids) != 1):
        raise ValueError("body_ids must be a single body id")
    # compute the error
    ang_vel_b = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids, :], asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :]
    ).squeeze(1)
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.body_ids == slice(None) or (isinstance(asset_cfg.body_ids, list) and len(asset_cfg.body_ids) != 1):
        raise ValueError("body_ids must be a single body id")
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.body_quat_w[:, asset_cfg.body_ids, :]),
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :],
    ).squeeze(1)
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.body_ids == slice(None) or (isinstance(asset_cfg.body_ids, list) and len(asset_cfg.body_ids) != 1):
        raise ValueError("body_ids must be a single body id")
    ang_vel = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - ang_vel[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def body_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids, :], asset.data.GRAVITY_VEC_W.unsqueeze(1)
    ).mean(dim=1)
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def body_lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    body_lin_vel_b = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids, :], asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    ).mean(dim=1)
    return torch.square(body_lin_vel_b[:, 2])


def body_ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    body_ang_vel_b = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids, :], asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :]
    ).mean(dim=1)
    return torch.sum(torch.square(body_ang_vel_b[:, :2]), dim=1)


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


class joint_acc_diff_l2(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.prev_joint_vel = self.asset.data.joint_vel[:, self.asset_cfg.joint_ids].clone()

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        joint_vel = self.asset.data.joint_vel[:, self.asset_cfg.joint_ids]
        joint_acc = (joint_vel - self.prev_joint_vel) / env.step_dt
        self.prev_joint_vel = joint_vel.clone()
        return torch.sum(torch.square(joint_acc), dim=1)
