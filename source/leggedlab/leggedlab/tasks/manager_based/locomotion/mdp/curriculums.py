# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import RewardManager, SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import CurriculumTermCfg
from isaaclab.terrains import TerrainImporter

from leggedlab.tasks.manager_based.locomotion.mdp.commands.velocity_command import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def normalize_terrain_name(name: str) -> str:
    pattern_number = r"_(\d+)$"
    normalized = re.sub(pattern_number, "", name)
    return normalized


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    move_up_threshold: float,
    move_down_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor | dict[str, torch.Tensor]:
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    reward: RewardManager = env.reward_manager
    lin_track_reward_sum = reward._episode_sums["track_lin_vel_xy_exp"][env_ids] / env.max_episode_length_s
    lin_track_reward_weight = reward.get_term_cfg("track_lin_vel_xy_exp").weight

    ang_track_reward_sum = reward._episode_sums["track_ang_vel_z_exp"][env_ids] / env.max_episode_length_s
    ang_track_reward_weight = reward.get_term_cfg("track_ang_vel_z_exp").weight
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = (
        (distance > terrain.cfg.terrain_generator.size[0] / 2)
        & (lin_track_reward_sum > lin_track_reward_weight * move_up_threshold)
        & (ang_track_reward_sum > ang_track_reward_weight * 0.7)
    )
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = (lin_track_reward_sum < lin_track_reward_weight * move_down_threshold) | (
        ang_track_reward_sum < ang_track_reward_weight * move_down_threshold
    )
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)

    # update the terrain level based on the proportion of the terrain
    result = {"average": torch.mean(terrain.terrain_levels.float())}
    # get the terrain level based on the proportion of the terrain
    terrain_gen_cfg = terrain.cfg.terrain_generator
    terrain_names = list(terrain_gen_cfg.sub_terrains.keys())
    # get the proportion of the terrain
    proportions = [terrain_gen_cfg.sub_terrains[name].proportion for name in terrain_names]
    total_proportion = sum(proportions)
    proportions = [proportion / total_proportion for proportion in proportions]

    terrain_groups = {}
    current_col = 0
    for i, terrain_name in enumerate(terrain_names):
        if i == len(terrain_names) - 1:
            width = terrain_gen_cfg.num_cols - current_col
        else:
            width = int(proportions[i] * terrain_gen_cfg.num_cols + 1e-4)
        normalize_name = normalize_terrain_name(terrain_name)
        if normalize_name not in terrain_groups:
            terrain_groups[normalize_name] = []
        terrain_groups[normalize_name].extend(range(current_col, current_col + width))
        current_col += width

    for group_name, col_indices in terrain_groups.items():
        mask = torch.zeros(len(terrain.terrain_types), dtype=torch.bool, device=env.device)
        for col_index in col_indices:
            mask |= terrain.terrain_types == col_index
        if mask.any():
            group_level = terrain.terrain_levels[mask].float()
            result[group_name] = torch.mean(group_level)
    return result


class command_levels_vel(ManagerTermBase):
    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.env = env
        delta_vel = cfg.params["delta_vel"]
        self.deltas = (torch.tensor(delta_vel).unsqueeze(-1) * torch.tensor([-1, 1])).to(device=env.device)
        # allowed_terrain_names
        self.allowed_terrain_names: list[str] = cfg.params.get("allowed_terrain_names", ["random", "plane"])
        self.allowed_terrain_cols = []
        self.stat_env_ids = torch.arange(env.num_envs, device=env.device)

        terrain: TerrainImporter = env.scene.terrain
        if terrain is not None and terrain.cfg.terrain_generator.curriculum:
            self._compute_terrain_groups(terrain)

    def _compute_terrain_groups(self, terrain: TerrainImporter):
        terrain_gen_cfg = terrain.cfg.terrain_generator
        terrain_names = list(terrain_gen_cfg.sub_terrains.keys())
        proportions = [terrain_gen_cfg.sub_terrains[name].proportion for name in terrain_names]
        total_proportion = sum(proportions)
        proportions = [proportion / total_proportion for proportion in proportions]

        terrain_groups = {}
        current_col = 0
        for i, terrain_name in enumerate(terrain_names):
            if i == len(terrain_names) - 1:
                width = terrain_gen_cfg.num_cols - current_col
            else:
                width = int(proportions[i] * terrain_gen_cfg.num_cols + 1e-4)
            normalize_name = normalize_terrain_name(terrain_name)
            if normalize_name not in terrain_groups:
                terrain_groups[normalize_name] = []
            terrain_groups[normalize_name].extend(range(current_col, current_col + width))
            current_col += width

        for terrain_name in self.allowed_terrain_names:
            if terrain_name in terrain_groups:
                self.allowed_terrain_cols.extend(terrain_groups[terrain_name])

        if len(self.allowed_terrain_cols) > 0:
            all_terrain_mask = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
            for col_idx in self.allowed_terrain_cols:
                all_terrain_mask |= terrain.terrain_types == col_idx
            self.stat_env_ids = torch.where(all_terrain_mask)[0]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        delta_vel: list[float],
        max_vel_range: list[tuple[float, float]],
        lin_vel_reward_threshold: float = 0.8,
        ang_vel_reward_threshold: float = 0.7,
        allowed_terrain_names: list[str] = ["random", "plane"],
    ) -> dict[str, torch.Tensor]:
        command: UniformVelocityCommand = self.env.command_manager.get_term("base_velocity")
        reward: RewardManager = self.env.reward_manager
        lin_track_reward_sum = reward._episode_sums["track_lin_vel_xy_exp"][env_ids] / self.env.max_episode_length_s
        lin_track_reward_weight = reward.get_term_cfg("track_lin_vel_xy_exp").weight

        ang_track_reward_sum = reward._episode_sums["track_ang_vel_z_exp"][env_ids] / self.env.max_episode_length_s
        ang_track_reward_weight = reward.get_term_cfg("track_ang_vel_z_exp").weight

        mask = (lin_track_reward_sum.mean() > lin_track_reward_weight * lin_vel_reward_threshold) & (
            ang_track_reward_sum.mean() > ang_track_reward_weight * ang_vel_reward_threshold
        )
        vel_update_ids = env_ids[mask]

        if len(self.allowed_terrain_cols) > 0 and vel_update_ids.numel() > 0:
            terrain: TerrainImporter = self.env.scene.terrain
            terrain_type_mask = torch.zeros(len(vel_update_ids), dtype=torch.bool, device=self.env.device)
            for env_idx, env_id in enumerate(vel_update_ids):
                env_terrain_type = terrain.terrain_types[env_id]
                if env_terrain_type in self.allowed_terrain_cols:
                    terrain_type_mask[env_idx] = True
            vel_update_ids = vel_update_ids[terrain_type_mask]

        if vel_update_ids.numel() > 0:
            command.lin_vel_x_ranges[vel_update_ids] = torch.clamp(
                command.lin_vel_x_ranges[vel_update_ids] + self.deltas[0], max_vel_range[0][0], max_vel_range[0][1]
            )
            command.lin_vel_y_ranges[vel_update_ids] = torch.clamp(
                command.lin_vel_y_ranges[vel_update_ids] + self.deltas[1], max_vel_range[1][0], max_vel_range[1][1]
            )
            command.ang_vel_z_ranges[vel_update_ids] = torch.clamp(
                command.ang_vel_z_ranges[vel_update_ids] + self.deltas[2], max_vel_range[2][0], max_vel_range[2][1]
            )

        result = {
            "avg_vel_lin_x": torch.mean(command.lin_vel_x_ranges[self.stat_env_ids, 1]),
            "avg_vel_lin_y": torch.mean(command.lin_vel_y_ranges[self.stat_env_ids, 1]),
            "avg_vel_ang_z": torch.mean(command.ang_vel_z_ranges[self.stat_env_ids, 1]),
        }

        return result
