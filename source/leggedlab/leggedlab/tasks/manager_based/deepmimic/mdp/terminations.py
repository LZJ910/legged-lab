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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .commands import MotionTrackingCommand


def root_height_err_termination(
    env: ManagerBasedRLEnv,
    command_name: str,
    probability: float = 0.005,
    threshold: float = 0.25,
) -> torch.Tensor:
    random_values = torch.rand(env.num_envs, device=env.device)
    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    height_error = command.body_height_error[..., command.root_link_ids].squeeze(1)
    return (torch.abs(height_error) > threshold) & (random_values < probability)


def root_quat_error_magnitude_termination(
    env: ManagerBasedRLEnv,
    command_name: str,
    probability: float = 0.005,
    threshold: float = 1.57,
) -> torch.Tensor:
    random_values = torch.rand(env.num_envs, device=env.device)
    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    quat_error_magnitude = command.body_quat_error_magnitude[:, command.root_link_ids].squeeze(1)
    return (quat_error_magnitude > threshold) & (random_values < probability)
