# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.envs.mdp import UniformVelocityCommandCfg as VelocityCommandCfg
from isaaclab.utils import configclass

from .velocity_command import UniformVelocityCommand


@configclass
class UniformVelocityCommandCfg(VelocityCommandCfg):
    """Configuration for the normal velocity command generator."""

    class_type: type = UniformVelocityCommand
