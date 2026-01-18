# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

RANDOM_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=50.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.25,
        ),
        "random_rough_1": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(-0.01, 0.01),
            noise_step=0.02,
            border_width=0.25,
        ),
        "random_rough_2": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(-0.015, 0.015),
            noise_step=0.02,
            border_width=0.25,
        ),
        "random_rough_3": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(-0.02, 0.02),
            noise_step=0.02,
            border_width=0.25,
        ),
    },
)
