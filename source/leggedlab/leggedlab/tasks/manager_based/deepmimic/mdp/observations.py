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
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.sensors import Imu

from .commands import MotionTrackingCommand


def imu_repr_6d(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    asset: Imu = env.scene[asset_cfg.name]
    quat_wxyz = asset.data.quat_w.clone()
    rot_matrix = math_utils.matrix_from_quat(quat_wxyz)
    repr_6d: torch.Tensor = rot_matrix[..., :2]
    return repr_6d.flatten(1)


def tracking_errors(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    return torch.cat(
        [
            command.key_points_w_error.flatten(1),
            command.body_lin_vel_error[:, command.root_link_ids].flatten(1),
            command.body_ang_vel_error[:, command.root_link_ids].flatten(1),
            command.joint_pos_error.flatten(1),
            command.joint_vel_error.flatten(1),
        ],
        dim=-1,
    )
