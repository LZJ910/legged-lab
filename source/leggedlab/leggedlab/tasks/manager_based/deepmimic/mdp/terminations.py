# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
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


def root_pos_err_termination(
    env: ManagerBasedRLEnv,
    command_name: str,
    probability: float = 0.005,
    threshold: float = 0.25,
) -> torch.Tensor:
    random_values = torch.rand(env.num_envs, device=env.device)
    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    root_height_error = command.root_height_error.squeeze(1)
    return (torch.abs(root_height_error) > threshold) & (random_values < probability)


def root_quat_error_magnitude_termination(
    env: ManagerBasedRLEnv,
    command_name: str,
    probability: float = 0.005,
    threshold: float = 1.57,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    random_values = torch.rand(env.num_envs, device=env.device)
    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    # Get target body orientation from dataset
    target_quat = command.tracking_body_quat[:, command.root_link_ids].squeeze(1)
    quat_error_magnitude = math_utils.quat_error_magnitude(target_quat, asset.data.root_link_quat_w)
    return (quat_error_magnitude > threshold) & (random_values < probability)
