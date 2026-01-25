# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    FRAME_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
    RAY_CASTER_MARKER_CFG,
)
from isaaclab.utils import configclass

from .motion_tracking_command import MotionTrackingCommand


@configclass
class MotionTrackingCommandCfg(CommandTermCfg):
    """Configuration for the motion tracking command generator."""

    class_type: type = MotionTrackingCommand

    asset_name: str = MISSING

    dataset_path: list[str] | str = MISSING

    exclude_prefix: str | None = None

    root_link_name: str = MISSING

    tracking_body_names: list[str] | None = None

    replay_dataset: bool = False

    use_world_frame: bool = False

    side_length: float = 0.1

    # Sampling mode configuration
    random_sampling: bool = True
    """If True, use random sampling. If False, all environments start from index 0."""

    # Adaptive sampling configuration
    adaptive_sampling_enabled: bool = True
    """Whether to enable adaptive sampling based on tracking errors."""

    success_mean_error_threshold: float = 0.05
    """Success threshold for mean error in meters. Default is 5cm."""

    success_weight_decrease: float = 0.1
    """Amount to decrease sampling weight for successful trajectories."""

    failure_weight_increase: float = 0.1
    """Amount to increase sampling weight for failed trajectories."""

    weight_clamp_min: float = 0.05
    """Minimum sampling weight value."""

    weight_clamp_max: float = 1.0
    """Maximum sampling weight value."""

    tracking_body_frame_visualizer: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/tracking_body_frame"
    )
    key_points_visualizer: VisualizationMarkersCfg = RAY_CASTER_MARKER_CFG.replace(
        prim_path="/Visuals/Command/key_points"
    )
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    tracking_body_frame_visualizer.markers["frame"].scale = (0.05, 0.05, 0.05)
