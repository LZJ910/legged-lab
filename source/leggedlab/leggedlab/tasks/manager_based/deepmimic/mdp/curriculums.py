# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch

from isaaclab.managers import RewardManager, SceneEntityCfg

from .commands import MotionTrackingCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_sampling_weights(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    reward: RewardManager = env.reward_manager
    tracking_key_points_exp_reward_sum = (
        reward._episode_sums["tracking_key_points_exp"][env_ids] / env.max_episode_length_s
    )
    tracking_key_points_exp_reward_weight = reward.get_term_cfg("tracking_key_points_exp").weight

    tracking_key_points_w_exp_reward_sum = (
        reward._episode_sums["tracking_key_points_w_exp"][env_ids] / env.max_episode_length_s
    )
    tracking_key_points_w_exp_reward_weight = reward.get_term_cfg("tracking_key_points_w_exp").weight

    episode_length = env.episode_length_buf[env_ids]

    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    current_weights = command.weights.clone()
    index_offset = command.current_index.clone()[env_ids]
    dataset_length = command.dataset_length

    mask_success = (tracking_key_points_exp_reward_sum > tracking_key_points_exp_reward_weight * 0.8) & (
        tracking_key_points_w_exp_reward_sum > tracking_key_points_w_exp_reward_weight * 0.9
    )
    mask_failure = (tracking_key_points_exp_reward_sum < tracking_key_points_exp_reward_weight * 0.8) | (
        tracking_key_points_w_exp_reward_sum < tracking_key_points_w_exp_reward_weight * 0.9
    )

    if mask_success.any():
        offsets = index_offset[mask_success]
        lengths = episode_length[mask_success] // 2
        if lengths.numel() > 0:
            max_len = torch.max(lengths)
            steps = torch.arange(max_len, device=env.device)
            all_indices = offsets.unsqueeze(1) + steps.unsqueeze(0)
            valid_mask = steps.unsqueeze(0) < lengths.unsqueeze(1)
            indices_to_update = all_indices[valid_mask]
            indices_to_update %= dataset_length
            current_weights[indices_to_update] -= 0.05

    if mask_success.any():
        successful_lengths = episode_length[mask_success]

        mask_full_length = successful_lengths >= env.max_episode_length
        mask_early = successful_lengths < env.max_episode_length

        all_successful_offsets = index_offset[mask_success]
        offsets_full = all_successful_offsets[mask_full_length]
        lengths_full = successful_lengths[mask_full_length]

        if lengths_full.numel() > 0:
            max_l = torch.max(lengths_full)
            steps = torch.arange(max_l, device=env.device)
            all_indices = offsets_full.unsqueeze(1) + steps.unsqueeze(0)
            valid_mask = steps.unsqueeze(0) < lengths_full.unsqueeze(1)
            indices_to_update = all_indices[valid_mask]
            indices_to_update %= dataset_length
            current_weights[indices_to_update] -= 0.05

        offsets_early = all_successful_offsets[mask_early]
        lengths_early = successful_lengths[mask_early]

        if lengths_early.numel() > 0:
            max_l = torch.max(lengths_early)
            steps = torch.arange(max_l, device=env.device)

            all_indices = offsets_early.unsqueeze(1) + steps.unsqueeze(0)
            valid_mask = steps.unsqueeze(0) < lengths_early.unsqueeze(1)

            lengths_part1 = (lengths_early // 3).unsqueeze(1)
            lengths_part2 = ((lengths_early * 2) // 3).unsqueeze(1)

            part1_mask = (steps.unsqueeze(0) < lengths_part1) & valid_mask
            indices_part1 = all_indices[part1_mask]
            indices_part1 %= dataset_length
            current_weights[indices_part1] -= 0.05

            part3_mask = (steps.unsqueeze(0) >= lengths_part2) & valid_mask
            indices_part3 = all_indices[part3_mask]
            indices_part3 %= dataset_length
            current_weights[indices_part3] += 0.1

    if mask_failure.any():
        offsets = index_offset[mask_failure]
        lengths = episode_length[mask_failure]
        if lengths.numel() > 0:
            max_len = torch.max(lengths)
            steps = torch.arange(max_len, device=env.device)
            all_indices = offsets.unsqueeze(1) + steps.unsqueeze(0)
            valid_mask = steps.unsqueeze(0) < lengths.unsqueeze(1)
            indices_to_update = all_indices[valid_mask]
            indices_to_update %= dataset_length
            current_weights[indices_to_update] += 0.1

    current_weights = torch.clamp(current_weights, min=0.05, max=1.0)

    command.update_sampling_weights(current_weights)

    return torch.mean(current_weights)
