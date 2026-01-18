# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


def create_motor_impedance(armature, nature_freq, damping_ratio):
    """根据电机转子惯量和期望动态特性计算阻抗参数。

    基于二阶系统动力学模型：
    - 刚度 (stiffness) = J * ω_n^2
    - 阻尼 (damping) = 2 * ζ * J * ω_n

    Args:
        armature: 电机转子惯量 (kg·m²)
        nature_freq: 自然频率 ω_n (rad/s)
        damping_ratio: 阻尼比 ζ (无量纲)

    Returns:
        tuple: (stiffness, damping) 刚度和阻尼系数
    """
    stiffness = armature * nature_freq**2
    damping = 2 * damping_ratio * armature * nature_freq
    return stiffness, damping


def create_motor_parameters(motor_specs, nature_freq, damping_ratio):
    """创建电机参数字典并计算阻抗参数。

    Args:
        motor_specs: 字典，包含电机基本参数 (armature, effort_limit,
                     velocity_limit, friction, dynamic_friction,
                     viscous_friction)
        nature_freq: 自然频率
        damping_ratio: 阻尼比

    Returns:
        dict: 完整的电机参数字典，包含计算好的 stiffness 和 damping
    """
    params = {}
    for motor_name, specs in motor_specs.items():
        stiffness, damping = create_motor_impedance(specs["armature"], nature_freq, damping_ratio)
        params[motor_name] = {
            "armature": specs["armature"],
            "effort_limit": specs["effort_limit"],
            "velocity_limit": specs["velocity_limit"],
            "friction": specs.get("friction", 0.0),
            "dynamic_friction": specs.get("dynamic_friction", 0.0),
            "viscous_friction": specs.get("viscous_friction", 0.0),
            "stiffness": stiffness,
            "damping": damping,
        }
    return params


def compute_action_scale_from_articulation(articulation_cfg, scale_factor=0.25):
    """根据关节配置计算动作缩放比例。

    Args:
        articulation_cfg: ArticulationCfg 对象，包含 actuators 配置
        scale_factor: 缩放系数，默认为 0.25

    Returns:
        dict: 关节名称到动作缩放比例的映射字典
    """
    action_scale = {}
    for actuator in articulation_cfg.actuators.values():
        effort = actuator.effort_limit_sim
        stiffness = actuator.stiffness
        joint_names = actuator.joint_names_expr

        # 如果不是字典，则为每个关节名称创建字典
        if not isinstance(effort, dict):
            effort = {name: effort for name in joint_names}
        if not isinstance(stiffness, dict):
            stiffness = {name: stiffness for name in joint_names}

        # 计算每个关节的动作缩放比例
        for name in joint_names:
            if name in effort and name in stiffness and stiffness[name]:
                action_scale[name] = scale_factor * effort[name] / stiffness[name]

    return action_scale
