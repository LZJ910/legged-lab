# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import leggedlab.tasks.manager_based.locomotion.mdp as mdp

from .flat_env_cfg import UnitreeG1FlatEnvCfg


@configclass
class UnitreeG1GetupEnvCfg(UnitreeG1FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # termination
        self.terminations.base_contact.params["immune_probability"] = 1.0

        # remove command levels
        self.curriculum.command_levels = None  # type: ignore

        # event
        self.events.reset_base.params["pose_range"]["z"] = (-0.65, -0.55)
        self.events.reset_base.params["pose_range"]["roll"] = (math.pi / 2, math.pi / 2)
        self.events.reset_base.params["pose_range"]["pitch"] = (-math.pi, math.pi)
        self.events.reset_base.params["pose_range"]["yaw"] = (-math.pi, math.pi)
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={"position_range": (-1.5, 1.5), "velocity_range": (-1.5, 1.5)},
        )
        self.events.base_external_force = EventTerm(
            func=mdp.apply_external_force_torque_assist,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "mass_scale": 0.6,
                "warmup_steps": 400 * 24,
                "duration": 1400.0 * 24,
            },
        )

        # rewards
        self.rewards.track_lin_vel_xy_exp = None  # type: ignore
        self.rewards.track_ang_vel_z_exp = None  # type: ignore
        self.rewards.undesired_contacts = None  # type: ignore
        self.rewards.body_lin_vel_z_l2 = None  # type: ignore
        self.rewards.body_ang_vel_xy_l2 = None  # type: ignore
        self.rewards.body_orientation_l2 = None  # type: ignore
        self.rewards.feet_air_time = None  # type: ignore
        self.rewards.feet_slide.weight = -1.0
        self.rewards.feet_stumble = None  # type: ignore

        self.rewards.base_height_up = RewTerm(
            func=mdp.base_height_up,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), "target_height": 0.79},
            weight=2.0,
        )
        self.rewards.body_orientation_up = RewTerm(
            func=mdp.body_orientation_up,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link", "pelvis"])},
            weight=1.0,
        )
        self.terminations.robot_power_out_of_bounds = DoneTerm(
            func=mdp.robot_power_out_of_bounds, params={"power_limit": 1000.0}
        )

        # episode length
        self.episode_length_s = 5.0

        # amp dataset path for getup
        self.amp_dataset_path = [
            "source/leggedlab/leggedlab/dataset/sliced_g1_lanfan1_50fps_amp/fall*1_subject1*.npy",
            "source/leggedlab/leggedlab/dataset/sliced_g1_lanfan1_50fps_amp/stance_1s.npy",
        ]


@configclass
class UnitreeG1EnvCfg_PLAY(UnitreeG1GetupEnvCfg):
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
