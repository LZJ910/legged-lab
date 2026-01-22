# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The leggedlab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import glob
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import MotionTrackingCommandCfg


class ReferenceState:
    """Container for reference motion state data."""

    def __init__(self, num_envs: int, num_bodies: int, num_joints: int, device: str):
        """Initialize reference state tensors.

        Args:
            num_envs: Number of environments.
            num_bodies: Number of body parts.
            num_joints: Number of joints.
            device: Device to create tensors on.
        """
        self.body_pos_w = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float32)
        self.body_pos_base_yaw_align = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float32)
        self.body_quat = torch.zeros((num_envs, num_bodies, 4), device=device, dtype=torch.float32)
        self.body_vel = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float32)
        self.body_ang_vel = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float32)
        self.joint_pos = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        self.joint_vel = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        self.key_points_w = torch.zeros((num_envs, num_bodies, 3, 3), device=device, dtype=torch.float32)
        self.key_points_base_yaw_align = torch.zeros((num_envs, num_bodies, 3, 3), device=device, dtype=torch.float32)


class TrackingErrors:
    """Container for pre-computed tracking errors."""

    def __init__(self, num_envs: int, num_bodies: int, num_joints: int, device: str):
        """Initialize tracking error tensors.

        Args:
            num_envs: Number of environments.
            num_bodies: Number of body parts.
            num_joints: Number of joints.
            device: Device to create tensors on.
        """
        self.body_pos_w = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float32)
        self.body_pos_base_yaw_align = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float32)
        self.body_height = torch.zeros((num_envs, num_bodies), device=device, dtype=torch.float32)
        self.body_quat_magnitude = torch.zeros((num_envs, num_bodies), device=device, dtype=torch.float32)
        self.body_lin_vel = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float32)
        self.body_ang_vel = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float32)
        self.joint_pos = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        self.joint_vel = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        self.key_points_w = torch.zeros((num_envs, num_bodies, 3, 3), device=device, dtype=torch.float32)
        self.key_points_base_yaw_align = torch.zeros((num_envs, num_bodies, 3, 3), device=device, dtype=torch.float32)


class MotionTrackingCommand(CommandTerm):
    cfg: MotionTrackingCommandCfg

    def __init__(self, cfg: MotionTrackingCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # load the dataset
        self._load_dataset()

        # setup sampling weights
        self._setup_sampling_weights()

        # initialize the current index for each environment
        self.current_index = self._sample_indices(env.num_envs) % self.dataset_length

        # Initialize reference state container (updated in _update_command)
        self.ref = ReferenceState(env.num_envs, len(self.body_names), len(self.joint_names), env.device)

        # Initialize tracking errors container (updated in _compute_tracking_errors)
        self.errors = TrackingErrors(env.num_envs, len(self.body_names), len(self.joint_names), env.device)

    """
    Properties
    """

    def _compute_tracking_errors(self):
        """Compute all tracking errors at once to avoid redundant calculations."""
        body_link_pos_w = self.robot.data.body_link_pos_w[:, self.robot_body_indices]
        body_link_quat_w = self.robot.data.body_link_quat_w[:, self.robot_body_indices]
        body_link_lin_vel_w = self.robot.data.body_link_lin_vel_w[:, self.robot_body_indices]
        body_link_ang_vel_w = self.robot.data.body_link_ang_vel_w[:, self.robot_body_indices]
        joint_pos = self.robot.data.joint_pos[:, self.robot_joint_indices]
        joint_vel = self.robot.data.joint_vel[:, self.robot_joint_indices]

        # Body position error (world frame)
        if self.cfg.use_world_frame:
            self.errors.body_pos_w = self.ref.body_pos_w - body_link_pos_w
        else:
            offset_pos_w = (self.ref.body_pos_w - body_link_pos_w)[:, self.root_link_ids]
            offset_pos_w[..., 2] = 0
            self.errors.body_pos_w = self.ref.body_pos_w - offset_pos_w - body_link_pos_w

        # Body position error (base yaw-aligned frame)
        body_pos_base_yaw_align = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(body_link_quat_w[:, self.root_link_ids]).expand(-1, body_link_pos_w.shape[1], -1),
            body_link_pos_w - body_link_pos_w[:, self.root_link_ids],
        )
        self.errors.body_pos_base_yaw_align = self.ref.body_pos_base_yaw_align - body_pos_base_yaw_align

        # Body height error
        self.errors.body_height = self.errors.body_pos_w[..., 2]

        # Body orientation error magnitude
        self.errors.body_quat_magnitude = math_utils.quat_error_magnitude(self.ref.body_quat, body_link_quat_w)

        # Body linear velocity error (robot body frame)
        body_lin_vel_b = math_utils.quat_apply_inverse(body_link_quat_w, body_link_lin_vel_w)
        self.errors.body_lin_vel = self.ref.body_vel - body_lin_vel_b

        # Body angular velocity error (robot body frame)
        body_ang_vel_b = math_utils.quat_apply_inverse(body_link_quat_w, body_link_ang_vel_w)
        self.errors.body_ang_vel = self.ref.body_ang_vel - body_ang_vel_b

        # Joint position error
        self.errors.joint_pos = self.ref.joint_pos - joint_pos

        # Joint velocity error
        self.errors.joint_vel = self.ref.joint_vel - joint_vel

        # Key points error (world frame)
        key_points_local = self.cfg.side_length * torch.eye(
            3, device=body_link_pos_w.device, dtype=body_link_pos_w.dtype
        )
        key_points_local = (
            key_points_local.unsqueeze(0)
            .unsqueeze(0)
            .expand(body_link_pos_w.shape[0], len(self.robot_body_indices), -1, -1)
        )
        keypoints_w = math_utils.quat_apply(
            body_link_quat_w.unsqueeze(2).expand(-1, -1, key_points_local.shape[2], -1), key_points_local
        ) + body_link_pos_w.unsqueeze(2)
        if self.cfg.use_world_frame:
            self.errors.key_points_w = self.ref.key_points_w - keypoints_w
        else:
            offset_pos_w = (self.ref.body_pos_w - body_link_pos_w)[:, self.root_link_ids]
            offset_pos_w[..., 2] = 0
            self.errors.key_points_w = self.ref.key_points_w - offset_pos_w.unsqueeze(1) - keypoints_w

        # Key points error (base yaw-aligned frame)
        key_points_base_yaw_align = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(body_link_quat_w[:, self.root_link_ids].unsqueeze(1)).expand(
                -1, keypoints_w.shape[1], keypoints_w.shape[2], -1
            ),
            keypoints_w - body_link_pos_w[:, self.root_link_ids].unsqueeze(1),
        )
        self.errors.key_points_base_yaw_align = self.ref.key_points_base_yaw_align - key_points_base_yaw_align

    @property
    def command(self) -> torch.Tensor:
        indices = self.current_index
        return torch.cat(
            [
                self.root_rot_6d[:, self.root_link_ids][indices].flatten(start_dim=1),
                self.body_linear_velocities[:, self.root_link_ids][indices].flatten(start_dim=1),
                self.body_angular_velocities[:, self.root_link_ids][indices].flatten(start_dim=1),
                self.joint_pos[indices].flatten(start_dim=1),
                self.joint_vel[indices].flatten(start_dim=1),
            ],
            dim=-1,
        )

    @property
    def body_pos_w_error(self) -> torch.Tensor:
        return self.errors.body_pos_w

    @property
    def body_pos_base_yaw_align_error(self) -> torch.Tensor:
        return self.errors.body_pos_base_yaw_align

    @property
    def body_height_error(self) -> torch.Tensor:
        return self.errors.body_height

    @property
    def body_quat_error_magnitude(self) -> torch.Tensor:
        return self.errors.body_quat_magnitude

    @property
    def body_lin_vel_error(self) -> torch.Tensor:
        return self.errors.body_lin_vel

    @property
    def body_ang_vel_error(self) -> torch.Tensor:
        return self.errors.body_ang_vel

    @property
    def joint_pos_error(self) -> torch.Tensor:
        return self.errors.joint_pos

    @property
    def joint_vel_error(self) -> torch.Tensor:
        return self.errors.joint_vel

    @property
    def key_points_w_error(self) -> torch.Tensor:
        return self.errors.key_points_w

    @property
    def key_points_base_yaw_align_error(self) -> torch.Tensor:
        return self.errors.key_points_base_yaw_align

    def get_joint_indices(self, joint_ids: list[int]) -> list[int]:
        idx_list = []
        for jid in joint_ids:
            matches = (self.robot_joint_indices == jid).nonzero(as_tuple=True)[0]
            if len(matches) == 0:
                raise ValueError(f"Joint {jid} not found in the robot")
            idx_list.append(matches.item())
        return idx_list

    def get_body_indices(self, body_ids: list[int]) -> list[int]:
        idx_list = []
        for bid in body_ids:
            matches = (self.robot_body_indices == bid).nonzero(as_tuple=True)[0]
            if len(matches) == 0:
                raise ValueError(f"Joint {bid} not found in the robot")
            idx_list.append(matches.item())
        return idx_list

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        self.metrics["body_pos_w_error"] = self.body_pos_w_error.abs().norm(dim=-1).mean(dim=-1)
        self.metrics["body_pos_base_yaw_align_error"] = (
            self.body_pos_base_yaw_align_error.abs().norm(dim=-1).mean(dim=-1)
        )
        self.metrics["body_height_error"] = self.body_height_error.abs().mean(dim=-1)
        self.metrics["body_quat_error_magnitude"] = self.body_quat_error_magnitude.abs().mean(dim=-1)
        self.metrics["body_lin_vel_error"] = self.body_lin_vel_error.abs().norm(dim=-1).mean(dim=-1)
        self.metrics["body_ang_vel_error"] = self.body_ang_vel_error.abs().norm(dim=-1).mean(dim=-1)
        self.metrics["joint_pos_error"] = self.joint_pos_error.abs().mean(dim=-1)
        self.metrics["joint_vel_error"] = self.joint_vel_error.abs().mean(dim=-1)
        self.metrics["key_points_w_error"] = self.key_points_w_error.abs().sum(dim=2).norm(dim=-1).mean(dim=-1)
        self.metrics["key_points_base_yaw_align_error"] = (
            self.key_points_base_yaw_align_error.abs().sum(dim=2).norm(dim=-1).mean(dim=-1)
        )

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample command indices and replay initial state."""
        self.current_index[env_ids] = (
            self._sample_indices(
                len(env_ids) if isinstance(env_ids, (list, tuple, torch.Tensor)) else self._env.num_envs
            )
            % self.dataset_length
        )
        self._update_ref_state(env_ids)
        self._data_replay(env_ids=env_ids)
        self._compute_tracking_errors()

    def _update_command(self):
        """Update command every step and optionally replay dataset."""
        self.current_index += 1
        self.current_index %= self.dataset_length
        # Update reference state for all environments
        self._update_ref_state()
        # Optionally replay dataset to physics simulation every step
        if self.cfg.replay_dataset:
            self._data_replay()
        # Compute tracking errors after updating reference state
        self._compute_tracking_errors()

    def _update_ref_state(self, env_ids: Sequence[int] | None = None):
        """Update reference state from dataset based on current_index.

        Args:
            env_ids: Environment IDs to update. If None, update all environments.
        """
        if env_ids is None:
            env_ids = slice(None)

        # Get current index from dataset
        index = self.current_index[env_ids] % self.dataset_length

        # Update body orientations, velocities, joint states
        self.ref.body_pos_base_yaw_align[env_ids] = self.body_pos_base_yaw_align[index]
        self.ref.body_quat[env_ids] = self.body_quat_wxyz[index]
        self.ref.body_vel[env_ids] = self.body_linear_velocities[index]
        self.ref.body_ang_vel[env_ids] = self.body_angular_velocities[index]
        self.ref.joint_pos[env_ids] = self.joint_pos[index]
        self.ref.joint_vel[env_ids] = self.joint_vel[index]
        self.ref.key_points_base_yaw_align[env_ids] = self.key_points_base_yaw_align[index]
        # Update world frame positions based on configuration
        self.ref.body_pos_w[env_ids] = self.body_pos_w[index] + self._env.scene.env_origins[env_ids].unsqueeze(1)
        self.ref.key_points_w[env_ids] = self.keypoints_w[index] + self._env.scene.env_origins[env_ids].unsqueeze(
            1
        ).unsqueeze(1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "tracking_body_frame_visualizer"):
                self.tracking_body_frame_visualizer = VisualizationMarkers(self.cfg.tracking_body_frame_visualizer)
                self.key_points_visualizer = VisualizationMarkers(self.cfg.key_points_visualizer)
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.tracking_body_frame_visualizer.set_visibility(True)
            self.key_points_visualizer.set_visibility(False)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "tracking_body_frame_visualizer"):
                self.tracking_body_frame_visualizer.set_visibility(False)
                self.key_points_visualizer.set_visibility(False)
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return

        # Visualize tracking body frames using reference state
        if self.cfg.use_world_frame:
            body_positions = self.ref.body_pos_w[:, self.tracking_body_ids, :]
        else:
            body_positions = self.ref.body_pos_w[:, self.tracking_body_ids, :].clone()
            offset_pos_w_xy = (self.ref.body_pos_w - self.robot.data.body_link_pos_w[:, self.robot_body_indices])[
                :, self.root_link_ids, :2
            ]
            body_positions[..., :2] -= offset_pos_w_xy

        body_orientations = self.ref.body_quat[:, self.tracking_body_ids, :]
        self.tracking_body_frame_visualizer.visualize(
            translations=body_positions.reshape(-1, 3),
            orientations=body_orientations.reshape(-1, 4),
        )

        # Visualize velocity
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 1.0
        # -- resolve the scales and quaternions
        body_velocities = self.ref.body_vel[:, self.root_link_ids, :2].squeeze(1)
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(body_velocities)
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_link_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    def _load_dataset(self):
        # load the dataset from the path
        if isinstance(self.cfg.dataset_path, str):
            self.cfg.dataset_path = [self.cfg.dataset_path]
        # find all files in the dataset path
        self.cfg.dataset_path = find_all_files(self.cfg.dataset_path, exclude_prefix=self.cfg.exclude_prefix)

        body_pos_xyz_w_list = []
        body_quat_wxyz_list = []
        body_linear_velocities_list = []
        body_angular_velocities_list = []
        joint_pos_list = []
        joint_vel_list = []

        for dataset_path in self.cfg.dataset_path:
            # load the dataset
            dataset = np.load(dataset_path)

            # get the robot data
            if not hasattr(self, "dof_names") and not hasattr(self, "body_names"):
                # Convert bytes to string if necessary
                body_names_raw = dataset["body_names"].tolist()
                self.body_names = [name.decode("utf-8") if isinstance(name, bytes) else name for name in body_names_raw]
                robot_body_indices, _ = self.robot.find_bodies(self.body_names, preserve_order=True)
                self.robot_body_indices = torch.tensor(robot_body_indices, device=self._env.device)

                # Convert bytes to string if necessary
                joint_names_raw = dataset["dof_names"].tolist()
                self.joint_names = [
                    name.decode("utf-8") if isinstance(name, bytes) else name for name in joint_names_raw
                ]
                robot_joint_indices, _ = self.robot.find_joints(self.joint_names, preserve_order=True)
                self.robot_joint_indices = torch.tensor(robot_joint_indices, device=self._env.device)

                ids, _ = self.robot.find_bodies(self.cfg.root_link_name, preserve_order=True)
                self.root_link_ids = self.get_body_indices(ids)

                # get the tracking body names and ids
                if self.cfg.tracking_body_names is not None:
                    ids, _ = self.robot.find_bodies(self.cfg.tracking_body_names, preserve_order=True)
                    self.tracking_body_ids = self.get_body_indices(ids)
                else:
                    self.tracking_body_ids, _ = self.robot.find_bodies(".*")

            body_pos_xyz_w_list.append(
                torch.from_numpy(dataset["body_positions"]).to(self._env.device).to(torch.float32)
            )
            body_quat_wxyz_list.append(
                torch.from_numpy(dataset["body_rotations_wxyz"]).to(self._env.device).to(torch.float32)
            )
            body_linear_velocities_list.append(
                torch.from_numpy(dataset["body_linear_velocities"]).to(self._env.device).to(torch.float32)
            )
            body_angular_velocities_list.append(
                torch.from_numpy(dataset["body_angular_velocities"]).to(self._env.device).to(torch.float32)
            )
            joint_pos_list.append(torch.from_numpy(dataset["dof_positions"]).to(self._env.device).to(torch.float32))
            joint_vel_list.append(torch.from_numpy(dataset["dof_velocities"]).to(self._env.device).to(torch.float32))

        self.body_pos_w = torch.cat(body_pos_xyz_w_list, dim=0)
        self.body_quat_wxyz = torch.cat(body_quat_wxyz_list, dim=0)
        self.body_linear_velocities = torch.cat(body_linear_velocities_list, dim=0)
        self.body_angular_velocities = torch.cat(body_angular_velocities_list, dim=0)
        self.joint_pos = torch.cat(joint_pos_list, dim=0)
        self.joint_vel = torch.cat(joint_vel_list, dim=0)

        self.dataset_length = self.body_pos_w.shape[0]

        self.body_pos_base_yaw_align = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self.body_quat_wxyz[:, self.root_link_ids, :]).expand(-1, self.body_pos_w.shape[1], -1),
            self.body_pos_w - self.body_pos_w[:, self.root_link_ids, :],
        )
        key_points_local = self.cfg.side_length * torch.eye(
            3, device=self.body_pos_w.device, dtype=self.body_pos_w.dtype
        )
        key_points_local = (
            key_points_local.unsqueeze(0)
            .unsqueeze(0)
            .expand(self.body_pos_w.shape[0], self.body_quat_wxyz.shape[1], -1, -1)
        )
        self.keypoints_w = math_utils.quat_apply(
            self.body_quat_wxyz.unsqueeze(2).expand(-1, -1, key_points_local.shape[2], -1), key_points_local
        ) + self.body_pos_w.unsqueeze(2)
        self.key_points_base_yaw_align = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self.body_quat_wxyz[:, self.root_link_ids, :].unsqueeze(2)).expand(
                -1, self.body_pos_w.shape[1], self.body_pos_w.shape[2], -1
            ),
            self.keypoints_w - self.body_pos_w[:, self.root_link_ids, :].unsqueeze(2),
        )
        body_rot_matrix = math_utils.matrix_from_quat(self.body_quat_wxyz)
        root_rot_6d: torch.Tensor = body_rot_matrix[..., :, :2]
        self.root_rot_6d = root_rot_6d.reshape(*self.body_quat_wxyz.shape[:-1], 6)

    def _data_replay(self, env_ids: Sequence[int] | None = None):
        """Replay dataset to physics simulation using reference state."""
        env_idx = slice(None) if env_ids is None else env_ids

        # Get data from reference state
        positions_w = self.ref.body_pos_w[env_idx, self.root_link_ids, :].reshape(-1, 3)
        orientations = self.ref.body_quat[env_idx, self.root_link_ids, :].reshape(-1, 4)
        velocities_b = self.ref.body_vel[env_idx, self.root_link_ids, :].reshape(-1, 3)
        angle_velocities_b = self.ref.body_ang_vel[env_idx, self.root_link_ids, :].reshape(-1, 3)
        joint_pos = self.ref.joint_pos[env_idx]
        joint_vel = self.ref.joint_vel[env_idx]

        # Convert the velocities and angle_velocities to the format expected by the robot
        velocities_w = math_utils.quat_apply(orientations, velocities_b)
        angle_velocities_w = math_utils.quat_apply(orientations, angle_velocities_b)

        # Set into the physics simulation
        root_link_state = torch.cat([positions_w, orientations, velocities_w, angle_velocities_w], dim=-1)
        self.robot.write_root_link_state_to_sim(root_link_state, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=self.robot_joint_indices, env_ids=env_ids)

    def _setup_sampling_weights(self):
        """Setup sampling weights for the dataset."""
        weights = torch.ones(self.dataset_length, device=self._env.device, dtype=torch.float32)
        # Use the update method to set the weights
        self.update_sampling_weights(weights)

    def _sample_indices(self, num_samples: int) -> torch.Tensor:
        """Sample indices from the dataset using the configured sampling strategy."""
        # Always use weighted sampling (weights default to uniform if not specified)
        indices = torch.multinomial(self.sampling_probabilities, num_samples, replacement=True)
        return indices

    def update_sampling_weights(self, new_weights: list[float] | torch.Tensor):
        """
        Update the sampling weights during runtime.

        Args:
            new_weights: New weights for sampling. Should have the same length as dataset_length.
        """
        if isinstance(new_weights, list):
            if len(new_weights) != self.dataset_length:
                raise ValueError(
                    f"Length of new_weights ({len(new_weights)}) must match dataset length ({self.dataset_length})"
                )
            weights = torch.tensor(new_weights, device=self._env.device, dtype=torch.float32)
        else:
            if new_weights.shape[0] != self.dataset_length:
                raise ValueError(
                    f"Length of new_weights ({new_weights.shape[0]}) must match dataset length ({self.dataset_length})"
                )
            weights = new_weights.to(device=self._env.device, dtype=torch.float32)

        # Ensure weights are positive
        self.weights = torch.clamp(weights, min=1e-8)
        # Normalize to sum to 1
        self.sampling_probabilities = weights / weights.sum()

    def get_current_weights(self) -> torch.Tensor:
        """Get the current sampling weights (probabilities)."""
        return self.weights.clone()


def find_all_files(path_list: list[str], exclude_prefix: str = None) -> list[str]:
    found_files = set()
    for item in path_list:
        matching_paths = glob.glob(item)
        for path in matching_paths:
            if os.path.isfile(path):
                if exclude_prefix is None or not os.path.basename(path).startswith(exclude_prefix):
                    found_files.add(os.path.abspath(path))
    return list(found_files)
