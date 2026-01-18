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

import isaaclab.envs.mdp as mdp
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class illegal_contact_with_immune(ManagerTermBase):
    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.immune_probability = cast(float, cfg.params["immune_probability"])
        self.immune_ids = torch.rand(env.num_envs, device=env.device) < self.immune_probability

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        immune_probability: float,
        resample_interval: int,
        sensor_cfg: SceneEntityCfg,
    ):
        # resample the immune ids if the interval is reached
        env_ids = env.episode_length_buf * resample_interval == 0
        self.immune_ids[env_ids] = torch.rand(env_ids.sum(), device=env.device) < self.immune_probability
        is_contact = mdp.illegal_contact(env, threshold, sensor_cfg)
        is_contact = is_contact & ~self.immune_ids
        return is_contact


def robot_power_out_of_bounds(
    env: ManagerBasedRLEnv,
    power_limit: float,
    warmup_steps: int = 10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    power = asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    mask = env.episode_length_buf > warmup_steps
    return (power.clip(min=0.0).sum(dim=-1) > power_limit) & mask
