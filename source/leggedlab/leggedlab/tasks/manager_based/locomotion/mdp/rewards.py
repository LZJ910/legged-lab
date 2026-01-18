# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_height_up(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        base_height = asset.data.root_pos_w[:, 2] - sensor.data.ray_hits_w[..., 2].mean(dim=-1)
    else:
        base_height = asset.data.root_link_pos_w[:, 2]
    # Replace NaNs with the base_height
    base_height = torch.nan_to_num(base_height, nan=target_height, posinf=target_height, neginf=target_height)
    height_diff = torch.clamp(target_height - base_height, min=0) / target_height
    reward = 1 - height_diff
    return reward


def body_orientation_up(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # For other bodies or multiple bodies, compute manually
    # Transform gravity vector to body frame for each specified body
    if asset_cfg.body_ids == slice(None):
        num_bodies = asset.num_bodies
    elif isinstance(asset_cfg.body_ids, list):
        num_bodies = len(asset_cfg.body_ids)
    else:
        raise ValueError("body_ids must be a list or a slice")
    projected_gravity = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids, :],
        asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, num_bodies, -1),
    )
    reward = torch.square(1 - projected_gravity[:, :, 2]).mean(dim=1) / 4.0
    return reward
