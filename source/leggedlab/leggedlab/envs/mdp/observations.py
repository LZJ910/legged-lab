# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from leggedlab.actuators import DelayedImplicitActuator

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import ContactSensor, Imu

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class rigid_body_mass(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if self.asset_cfg.body_ids == slice(None):
            self.body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int, device="cpu")
        else:
            self.body_ids = torch.tensor(self.asset_cfg.body_ids, dtype=torch.int, device="cpu")

        self.bady_mass = self.asset.root_physx_view.get_masses()[:, self.body_ids].to(env.device)
        self.count = 0

    def __call__(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        if self.count < 5:
            self.count += 1
            self.bady_mass = self.asset.root_physx_view.get_masses()[:, self.body_ids].to(env.device)
        return self.bady_mass


class rigid_body_material(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if self.asset_cfg.body_ids == slice(None):
            self.body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int, device="cpu")
        else:
            self.body_ids = torch.tensor(self.asset_cfg.body_ids, dtype=torch.int, device="cpu")

        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(None):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)
        self.idxs = []
        for body_id in self.body_ids:
            idx = sum(self.num_shapes_per_body[:body_id])
            self.idxs.append(idx)

        materials = self.asset.root_physx_view.get_material_properties()
        self.materials = materials[:, self.idxs].reshape(env.num_envs, -1).to(env.device)

        self.count = 0

    def __call__(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        if self.count < 5:
            self.count += 1
            materials = self.asset.root_physx_view.get_material_properties()
            self.materials = materials[:, self.idxs].reshape(env.num_envs, -1).to(env.device)
        return self.materials


class rigid_body_com(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if self.asset_cfg.body_ids == slice(None):
            self.body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int, device="cpu")
        else:
            self.body_ids = torch.tensor(self.asset_cfg.body_ids, dtype=torch.int, device="cpu")

        self.coms = self.asset.root_physx_view.get_coms()[:, self.body_ids, :3].to(env.device).squeeze(1)
        self.count = 0

    def __call__(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        if self.count < 5:
            self.count += 1
            self.coms = self.asset.root_physx_view.get_coms()[:, self.body_ids, :3].to(env.device).squeeze(1)
        return self.coms


def action_delay(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    time_lags = []
    for actuator_name in asset.actuators:
        actuators = cast(DelayedImplicitActuator, asset.actuators[actuator_name])
        time_lags.append(actuators.positions_delay_buffer.time_lags)
    time_lags = torch.stack(time_lags, dim=1).float().to(env.device)
    return time_lags


def joint_accs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint accelerations of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their accelerations returned.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


class joint_acc_diff(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.prev_joint_vel = self.asset.data.joint_vel[:, self.asset_cfg.joint_ids].clone()

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        joint_vel = self.asset.data.joint_vel[:, self.asset_cfg.joint_ids]
        joint_acc = (joint_vel - self.prev_joint_vel) / env.step_dt
        self.prev_joint_vel = joint_vel.clone()
        return joint_acc


def base_link_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b


def base_link_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_ang_vel_b


def body_link_lin_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    body_lin_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    root_link_quat_w = math_utils.yaw_quat(asset.data.root_link_quat_w).unsqueeze(1)
    root_link_quat_w = root_link_quat_w.expand(-1, body_lin_vel_w.shape[1], -1).contiguous()
    body_lin_vel_b = math_utils.quat_apply_yaw(root_link_quat_w, body_lin_vel_w)
    return body_lin_vel_b.flatten(1)


def body_link_ang_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    body_ang_vel_w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :]
    root_link_quat_w = math_utils.yaw_quat(asset.data.root_link_quat_w).unsqueeze(1)
    root_link_quat_w = root_link_quat_w.expand(-1, body_ang_vel_w.shape[1], -1).contiguous()
    body_ang_vel_b = math_utils.quat_apply_yaw(root_link_quat_w, body_ang_vel_w)
    return body_ang_vel_b.flatten(1)


def body_contact_force_b(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    contact_sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])
    contact_force_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]

    asset: Articulation = env.scene[asset_cfg.name]
    root_link_quat_w = math_utils.yaw_quat(asset.data.root_link_quat_w).unsqueeze(1)
    root_link_quat_w = root_link_quat_w.expand(-1, contact_force_w.shape[1], -1)

    contact_force_b = math_utils.quat_apply_inverse(root_link_quat_w, contact_force_w)
    return contact_force_b.flatten(1)


def body_contact_information(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    contact_sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])
    data = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact_information = torch.sum(torch.square(data), dim=-1) > 0
    return contact_information.float()


def body_composed_force_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    if hasattr(asset, "permanent_wrench_composer"):
        composed_force = asset.permanent_wrench_composer.composed_force_as_torch[:, asset_cfg.body_ids, :]
    else:
        composed_force = asset._external_force_b[:, asset_cfg.body_ids, :]
    return composed_force.flatten(1)


def body_composed_torque_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    if hasattr(asset, "permanent_wrench_composer"):
        composed_torque = asset.permanent_wrench_composer.composed_torque_as_torch[:, asset_cfg.body_ids, :]
    else:
        composed_torque = asset._external_torque_b[:, asset_cfg.body_ids, :]
    return composed_torque.flatten(1)


def imu_lin_vel_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Imu = env.scene[asset_cfg.name]
    # return the angular velocity
    return asset.data.lin_vel_b
