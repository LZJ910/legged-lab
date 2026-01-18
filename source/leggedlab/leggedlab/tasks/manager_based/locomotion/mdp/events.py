# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def apply_external_force_torque_assist(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    mass_scale: float = 0.0,
    warmup_steps: int = 0,
    duration: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    sampled in world frame and then transformed to body frame before being applied. The forces and torques are
    scaled down over time (assistive force that gradually decreases during training). The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # sample random forces and torques
    size = (len(env_ids), num_bodies, 3)

    torques_w = torch.zeros(size, device=asset.device)
    forces_w = torch.zeros(size, device=asset.device)
    forces_w[:, :, 2] = asset.data.default_mass.sum(dim=-1).unsqueeze(1) * 9.81 * mass_scale

    step_counter = max(0, env.common_step_counter - warmup_steps)
    scale = max(0.0, 1.0 - step_counter / (duration + 1e-5))

    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces_w * scale,
        torques=torques_w * scale,
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids,
        is_global=True,
    )
