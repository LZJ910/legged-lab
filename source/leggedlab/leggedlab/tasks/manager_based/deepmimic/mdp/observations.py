# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .commands import MotionTrackingCommand


def key_points_pos_b(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_link_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]

    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    keypoints_b = command.cfg.side_length * torch.eye(3, device=asset.device, dtype=torch.float)
    keypoints_b = keypoints_b.unsqueeze(0).expand(body_link_pos_w.shape[0], -1, -1, -1)
    keypoints_w = math_utils.quat_apply(
        asset.data.body_link_quat_w[:, asset_cfg.body_ids].unsqueeze(2).expand(-1, -1, keypoints_b.shape[2], -1),
        keypoints_b.expand(-1, body_link_pos_w.shape[1], -1, -1),
    ) + body_link_pos_w.unsqueeze(2)
    keypoints = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_link_quat_w.unsqueeze(1).unsqueeze(1)).expand(
            -1, keypoints_w.shape[1], keypoints_w.shape[2], -1
        ),
        keypoints_w - asset.data.root_link_pos_w.unsqueeze(1).unsqueeze(1),
    )
    return keypoints.flatten(1)


def body_repr_6d(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    quat_wxyz = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].clone()
    rot_matrix = math_utils.matrix_from_quat(quat_wxyz)
    repr_6d: torch.Tensor = rot_matrix[..., :2]
    return repr_6d.flatten(1)
