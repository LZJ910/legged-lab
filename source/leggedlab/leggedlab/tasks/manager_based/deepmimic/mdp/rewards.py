# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
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
    reward = torch.exp(-body_pos_w_error / std**2)
    return reward


def tracking_body_pos_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # Convert the target body positions to the asset's root link frame
    body_pos_xyz = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_link_quat_w.unsqueeze(1)),
        asset.data.body_link_pos_w[:, asset_cfg.body_ids] - asset.data.root_link_pos_w.unsqueeze(1),
    )
    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    body_indices = command.get_body_indices(asset_cfg.body_ids)
    body_pos_xyz_target = command.tracking_body_pos[:, body_indices]
    body_pos_error = torch.square(body_pos_xyz_target - body_pos_xyz).mean(dim=-1).mean(dim=-1)
    reward = torch.exp(-body_pos_error / std**2)
    return reward


def tracking_body_quat_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    body_indices = command.get_body_indices(asset_cfg.body_ids)
    body_quat_target = command.tracking_body_quat[:, body_indices]
    quat_error_magnitude = math_utils.quat_error_magnitude(body_quat_target, body_quat).mean(dim=-1).mean(dim=-1)
    reward = torch.exp(-quat_error_magnitude / std**2)
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
    asset: Articulation = env.scene[asset_cfg.name]
    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    body_link_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    keypoints_b = command.cfg.side_length * torch.eye(3, device=asset.device, dtype=torch.float)
    keypoints_b = keypoints_b.unsqueeze(0).expand(body_link_pos_w.shape[0], -1, -1, -1)
    keypoints_w = math_utils.quat_apply(
        asset.data.body_link_quat_w[:, asset_cfg.body_ids].unsqueeze(2).expand(-1, -1, keypoints_b.shape[2], -1),
        keypoints_b.expand(-1, len(asset_cfg.body_ids), -1, -1),
    ) + body_link_pos_w.unsqueeze(2)

    body_indices = command.get_body_indices(asset_cfg.body_ids)
    key_points_w_target = command.tracking_key_points_w[:, body_indices]
    body_pos_error = torch.square(key_points_w_target - keypoints_w).sum(dim=-2).mean(dim=-1).mean(dim=-1)
    reward = torch.exp(-body_pos_error / std**2)
    return reward


def tracking_key_points_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    body_link_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    keypoints_b = command.cfg.side_length * torch.eye(3, device=asset.device, dtype=torch.float)
    keypoints_b = keypoints_b.unsqueeze(0).expand(body_link_pos_w.shape[0], -1, -1, -1)
    keypoints_w = math_utils.quat_apply(
        asset.data.body_link_quat_w[:, asset_cfg.body_ids].unsqueeze(2).expand(-1, -1, keypoints_b.shape[2], -1),
        keypoints_b.expand(-1, len(asset_cfg.body_ids), -1, -1),
    ) + body_link_pos_w.unsqueeze(2)
    keypoints = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_link_quat_w.unsqueeze(1).unsqueeze(1)).expand(
            -1, keypoints_w.shape[1], keypoints_w.shape[2], -1
        ),
        keypoints_w - asset.data.root_link_pos_w.unsqueeze(1).unsqueeze(1),
    )

    body_indices = command.get_body_indices(asset_cfg.body_ids)
    key_points_target = command.tracking_key_points[:, body_indices]
    body_pos_error = torch.square(key_points_target - keypoints).sum(dim=-2).mean(dim=-1).mean(dim=-1)
    reward = torch.exp(-body_pos_error / std**2)
    return reward


def tracking_body_height_l2(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    body_indices = command.get_body_indices(asset_cfg.body_ids)
    body_height_error = command.body_height_error[:, body_indices]
    reward = torch.square(body_height_error).sum(dim=-1)
    return reward
