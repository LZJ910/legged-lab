# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from isaaclab.utils import configclass

from leggedlab.assets.robots.unitree import UNITREE_G1_29DOF_ACTION_SCALE, UNITREE_G1_29DOF_CFG
from leggedlab.tasks.manager_based.deepmimic.base_env_cfg import DeepMicicEnvCfg


@configclass
class UnitreeG1MimicEnvCfg(DeepMicicEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore
        self.scene.imu.prim_path = "{ENV_REGEX_NS}/Robot/pelvis"
        # action
        self.actions.joint_pos.scale = UNITREE_G1_29DOF_ACTION_SCALE
        self.actions.joint_pos.clip = {".*": [-100.0, 100]}  # type: ignore
        # event
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.scale_link_mass.params["asset_cfg"].body_names = "(?!.*torso_link.*).*"
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = "torso_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "torso_link"
        # termination

        # curriculum

        # rewards
        self.rewards.tracking_body_lin_vel.params["asset_cfg"].body_names = [
            "pelvis",
            "torso_link",
            ".*_knee_link",
            ".*_ankle_roll_link",
            ".*_elbow_link",
            ".*_wrist_yaw_link",
        ]
        self.rewards.tracking_body_ang_vel.params["asset_cfg"].body_names = "pelvis"
        self.rewards.tracking_key_points_w_exp.params["asset_cfg"].body_names = "pelvis"
        self.rewards.tracking_key_points_exp.params["asset_cfg"].body_names = [
            "torso_link",
            ".*_knee_link",
            ".*_ankle_roll_link",
            ".*_elbow_link",
            ".*_wrist_yaw_link",
        ]
        self.rewards.feet_slide.params["sensor_cfg"].body_names = ".*ankle_roll.*"
        self.rewards.feet_slide.params["asset_cfg"].body_names = ".*ankle_roll.*"
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = ".*ankle_roll.*"
        self.rewards.feet_height_l2.params["asset_cfg"].body_names = ".*ankle_roll.*"
        # observation
        self.observations.proprioception.body_repr_6d.params["asset_cfg"].body_names = "pelvis"
        self.observations.privileged.body_repr_6d.params["asset_cfg"].body_names = "pelvis"
        self.observations.privileged.base_lin_vel.params["asset_cfg"].body_names = "pelvis"
        self.observations.privileged.feet_lin_vel.params["asset_cfg"].body_names = ".*ankle_roll.*"
        self.observations.privileged.feet_contact_force.params["sensor_cfg"].body_names = ".*ankle_roll.*"
        self.observations.privileged.base_mass_rel.params["asset_cfg"].body_names = "torso_link"
        self.observations.privileged.rigid_body_material.params["asset_cfg"].body_names = ".*ankle_roll.*"
        self.observations.privileged.base_com.params["asset_cfg"].body_names = "torso_link"
        self.observations.privileged.push_force.params["asset_cfg"].body_names = "torso_link"
        self.observations.privileged.push_torque.params["asset_cfg"].body_names = "torso_link"
        self.observations.privileged.contact_information.params["sensor_cfg"].body_names = [
            "pelvis",
            "torso_link",
            "waist_yaw_link",
            "waist_roll_link",
            "left_hip_pitch_link",
            "left_hip_roll_link",
            "left_hip_yaw_link",
            "left_knee_link",
            "left_shoulder_pitch_link",
            "left_shoulder_roll_link",
            "left_shoulder_yaw_link",
            "left_elbow_link",
            "left_wrist_roll_link",
            "left_wrist_pitch_link",
            "left_wrist_yaw_link",
            "right_hip_pitch_link",
            "right_hip_roll_link",
            "right_hip_yaw_link",
            "right_knee_link",
            "right_shoulder_pitch_link",
            "right_shoulder_roll_link",
            "right_shoulder_yaw_link",
            "right_elbow_link",
            "right_wrist_roll_link",
            "right_wrist_pitch_link",
            "right_wrist_yaw_link",
        ]
        self.observations.privileged.key_points_pos_b.params["asset_cfg"].body_names = [
            "pelvis",
            "torso_link",
            ".*_knee_link",
            ".*_ankle_roll_link",
            ".*_elbow_link",
            ".*_wrist_yaw_link",
        ]
        self.observations.privileged.body_link_lin_vel_b.params["asset_cfg"].body_names = [
            "torso_link",
            ".*_knee_link",
            ".*_ankle_roll_link",
            ".*_elbow_link",
            ".*_wrist_yaw_link",
        ]
        # command
        self.commands.motion_tracking.dataset_path = [
            "source/leggedlab/leggedlab/dataset/g1_lanfan1_50fps_tracking_py/dance1_subject2.npz"
        ]
        self.commands.motion_tracking.root_link_name = "pelvis"
        self.commands.motion_tracking.tracking_body_names = [
            "torso_link",
            ".*_knee_link",
            ".*_ankle_roll_link",
            ".*_elbow_link",
            ".*_wrist_yaw_link",
        ]


@configclass
class UnitreeG1MimicEnvCfg_PLAY(UnitreeG1MimicEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5

        self.curriculum.terrain_levels = None  # type: ignore
        self.curriculum.command_levels = None  # type: ignore

        # disable randomization for play
        self.observations.proprioception.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None  # type: ignore
