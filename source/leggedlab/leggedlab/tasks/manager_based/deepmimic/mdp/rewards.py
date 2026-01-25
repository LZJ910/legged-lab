# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from isaaclab.managers import SceneEntityCfg

from .commands import MotionTrackingCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def tracking_joint_pos_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    joint_indices = command.get_joint_indices(asset_cfg.joint_ids)
    joint_pos_error = command.joint_pos_error[:, joint_indices]
    joint_pos_error = torch.square(joint_pos_error).mean(dim=-1)
    reward = torch.exp(-joint_pos_error / std**2)
    return reward


def tracking_joint_vel_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    joint_indices = command.get_joint_indices(asset_cfg.joint_ids)
    joint_vel_error = command.joint_vel_error[:, joint_indices]
    joint_vel_error = torch.square(joint_vel_error).mean(dim=-1)
    reward = torch.exp(-joint_vel_error / std**2)
    return reward


def tracking_body_pos_w_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    body_indices = command.get_body_indices(asset_cfg.body_ids)
    body_pos_w_error = command.body_pos_w_error[:, body_indices]
    body_pos_w_error = torch.square(body_pos_w_error).sum(dim=-1).mean(dim=-1)
    reward = torch.exp(-body_pos_w_error / std**2)
    return reward


def tracking_body_pos_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    body_indices = command.get_body_indices(asset_cfg.body_ids)
    body_pos_error = command.body_pos_base_yaw_align_error[:, body_indices]
    body_pos_error = torch.square(body_pos_error).sum(dim=-1).mean(dim=-1)
    reward = torch.exp(-body_pos_error / std**2)
    return reward


def tracking_body_quat_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    body_indices = command.get_body_indices(asset_cfg.body_ids)
    body_quat_error_magnitude = command.body_quat_error_magnitude[:, body_indices]
    reward = torch.exp(-body_quat_error_magnitude / std**2)
    return reward


def tracking_body_vel_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    body_indices = command.get_body_indices(asset_cfg.body_ids)
    body_lin_vel_error = command.body_lin_vel_error[:, body_indices]
    body_lin_vel_error = torch.square(body_lin_vel_error).mean(dim=-1).mean(dim=-1)
    reward = torch.exp(-body_lin_vel_error / std**2)
    return reward


def tracking_body_ang_vel_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    body_indices = command.get_body_indices(asset_cfg.body_ids)
    body_ang_vel_error = command.body_ang_vel_error[:, body_indices]
    body_ang_vel_error = torch.square(body_ang_vel_error).mean(dim=-1).mean(dim=-1)
    reward = torch.exp(-body_ang_vel_error / std**2)
    return reward


def tracking_key_points_w_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    body_indices = command.get_body_indices(asset_cfg.body_ids)
    key_points_w_error = command.key_points_w_error[:, body_indices]
    key_points_w_error = torch.square(key_points_w_error).sum(dim=-1).mean(dim=-1).mean(dim=-1)
    reward = torch.exp(-key_points_w_error / std**2)
    return reward


def tracking_key_points_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    body_indices = command.get_body_indices(asset_cfg.body_ids)
    key_points_base_yaw_align_error = command.key_points_base_yaw_align_error[:, body_indices]
    key_points_base_yaw_align_error = (
        torch.square(key_points_base_yaw_align_error).sum(dim=-1).mean(dim=-1).mean(dim=-1)
    )
    reward = torch.exp(-key_points_base_yaw_align_error / std**2)
    return reward


def tracking_body_height_l2(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    body_indices = command.get_body_indices(asset_cfg.body_ids)
    body_height_error = command.body_height_error[:, body_indices]
    reward = torch.square(body_height_error).sum(dim=-1)
    return reward
