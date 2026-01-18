# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import math

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from leggedlab.actuators import DelayedImplicitActuatorCfg
from leggedlab.assets import ISAAC_ASSET_DIR
from leggedlab.assets.robots.utils import (
    compute_action_scale_from_articulation,
    create_motor_parameters,
)

# 电机基本规格
MOTOR_SPECS = {
    "7520_14": {
        "armature": 0.010177520,
        "effort_limit": 75,
        "velocity_limit": 32,
        "friction": 0.0,
        "dynamic_friction": 0.0,
        "viscous_friction": 0.0,
    },
    "7520_16": {
        "armature": 0.013,
        "effort_limit": 88,
        "velocity_limit": 23,
        "friction": 0.0,
        "dynamic_friction": 0.0,
        "viscous_friction": 0.0,
    },
    "7520_22": {
        "armature": 0.025101925,
        "effort_limit": 120,
        "velocity_limit": 20,
        "friction": 0.0,
        "dynamic_friction": 0.0,
        "viscous_friction": 0.0,
    },
    "5020": {
        "armature": 0.003609725,
        "effort_limit": 25,
        "velocity_limit": 37,
        "friction": 0.0,
        "dynamic_friction": 0.0,
        "viscous_friction": 0.0,
    },
    "5020_parallel": {
        "armature": 0.003609725 * 2.0,
        "effort_limit": 25 * 2.0,
        "velocity_limit": 37.0 * 1.0,
        "friction": 0.0 * 2.0,
        "dynamic_friction": 0.0 * 2.0,
        "viscous_friction": 0.0 * 2.0,
    },
    "4010": {
        "armature": 0.00425,
        "effort_limit": 5,
        "velocity_limit": 22,
        "friction": 0.0,
        "dynamic_friction": 0.0,
        "viscous_friction": 0.0,
    },
}

# 电机控制参数
NATURE_FREQ = 10.0 * 2 * math.pi
DAMPING_RATIO = 2.0
MIN_DELAY = 0
MAX_DELAY = 4

# 生成完整的电机参数（包括自动计算的 stiffness 和 damping）
MotorParameter = create_motor_parameters(MOTOR_SPECS, NATURE_FREQ, DAMPING_RATIO)

UNITREE_G1_29DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/robots/Unitree/g1_29dof.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=10000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.80),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.18,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.18,
            "right_shoulder_pitch_joint": 0.35,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "7520_14": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
            ],
            effort_limit_sim=MotorParameter["7520_14"]["effort_limit"],
            velocity_limit_sim=MotorParameter["7520_14"]["velocity_limit"],
            stiffness=MotorParameter["7520_14"]["stiffness"],
            damping=MotorParameter["7520_14"]["damping"],
            friction=MotorParameter["7520_14"]["friction"],
            dynamic_friction=MotorParameter["7520_14"]["dynamic_friction"],
            viscous_friction=MotorParameter["7520_14"]["viscous_friction"],
        ),
        "7520_16": DelayedImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_pitch_joint"],
            effort_limit_sim=MotorParameter["7520_16"]["effort_limit"],
            velocity_limit_sim=MotorParameter["7520_16"]["velocity_limit"],
            stiffness=MotorParameter["7520_16"]["stiffness"],
            damping=MotorParameter["7520_16"]["damping"],
            friction=MotorParameter["7520_16"]["friction"],
            dynamic_friction=MotorParameter["7520_16"]["dynamic_friction"],
            viscous_friction=MotorParameter["7520_16"]["viscous_friction"],
        ),
        "7520_22": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_roll_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=MotorParameter["7520_22"]["effort_limit"],
            velocity_limit_sim=MotorParameter["7520_22"]["velocity_limit"],
            stiffness=MotorParameter["7520_22"]["stiffness"],
            damping=MotorParameter["7520_22"]["damping"],
            friction=MotorParameter["7520_22"]["friction"],
            dynamic_friction=MotorParameter["7520_22"]["dynamic_friction"],
            viscous_friction=MotorParameter["7520_22"]["viscous_friction"],
        ),
        "5020": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim=MotorParameter["5020"]["effort_limit"],
            velocity_limit_sim=MotorParameter["5020"]["velocity_limit"],
            stiffness=MotorParameter["5020"]["stiffness"],
            damping=MotorParameter["5020"]["damping"],
            armature=MotorParameter["5020"]["armature"],
            friction=MotorParameter["5020"]["friction"],
            dynamic_friction=MotorParameter["5020"]["dynamic_friction"],
            viscous_friction=MotorParameter["5020"]["viscous_friction"],
            min_delay=MIN_DELAY,
            max_delay=MAX_DELAY,
        ),
        "5020_parallel": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "waist_pitch_joint",
                "waist_roll_joint",
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim=MotorParameter["5020_parallel"]["effort_limit"],
            velocity_limit_sim=MotorParameter["5020_parallel"]["velocity_limit"],
            stiffness=MotorParameter["5020_parallel"]["stiffness"],
            damping=MotorParameter["5020_parallel"]["damping"],
            armature=MotorParameter["5020_parallel"]["armature"],
            friction=MotorParameter["5020_parallel"]["friction"],
            dynamic_friction=MotorParameter["5020_parallel"]["dynamic_friction"],
            viscous_friction=MotorParameter["5020_parallel"]["viscous_friction"],
            min_delay=MIN_DELAY,
            max_delay=MAX_DELAY,
        ),
        "4010": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_yaw_joint",
                ".*_wrist_pitch_joint",
            ],
            effort_limit_sim=MotorParameter["4010"]["effort_limit"],
            velocity_limit_sim=MotorParameter["4010"]["velocity_limit"],
            stiffness=MotorParameter["4010"]["stiffness"],
            damping=MotorParameter["4010"]["damping"],
            friction=MotorParameter["4010"]["friction"],
            dynamic_friction=MotorParameter["4010"]["dynamic_friction"],
            viscous_friction=MotorParameter["4010"]["viscous_friction"],
        ),
    },
)
UNITREE_G1_29DOF_ACTION_SCALE = compute_action_scale_from_articulation(UNITREE_G1_29DOF_CFG, scale_factor=0.25)
