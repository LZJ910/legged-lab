# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeG1RoughEnvCfg

from leggedlab.terrains import RANDOM_TERRAINS_CFG  # isort: skip


@configclass
class UnitreeG1FlatEnvCfg(UnitreeG1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator = RANDOM_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.height_scanner = None  # type: ignore
        # observation
        self.observations.privileged.height_scan = None  # type: ignore
        # curriculum
        self.curriculum.terrain_levels = None  # type: ignore
        self.curriculum.command_levels.params["max_vel_range"] = [(-1.0, 3.0), (-0.5, 0.5), (-3.0, 3.0)]


@configclass
class UnitreeG1FlatEnvCfg_PLAY(UnitreeG1FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5

        # disable randomization for play
        self.observations.proprioception.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None  # type: ignore

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
