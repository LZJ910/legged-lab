# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import leggedlab.tasks.manager_based.deepmimic.mdp as mdp

##
# Pre-defined configs
##
from leggedlab.terrains import RANDOM_TERRAINS_CFG  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=RANDOM_TERRAINS_CFG,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING  # type: ignore
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base")
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion_tracking = mdp.MotionTrackingCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e9, 1e9),
        dataset_path=".*",
        root_link_name="pelvis",
        tracking_body_names=[".*"],
        debug_vis=True,
        replay_dataset=False,
        use_world_frame=False,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class CommandCfg(ObsGroup):
        """Observations for command group."""

        # observation terms (order preserved)
        motion_tracking_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion_tracking"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ProprioceptionCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.imu_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), clip=(-100, 100)
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-100, 100))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), clip=(-100, 100))
        actions = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        body_repr_6d = ObsTerm(func=mdp.body_repr_6d, params={"asset_cfg": SceneEntityCfg("robot", body_names="base")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.imu_ang_vel)
        projected_gravity = ObsTerm(func=mdp.imu_projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, clip=(-100, 100))
        actions = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        body_repr_6d = ObsTerm(func=mdp.body_repr_6d, params={"asset_cfg": SceneEntityCfg("robot", body_names="base")})
        base_lin_vel = ObsTerm(
            func=mdp.body_link_lin_vel_b, params={"asset_cfg": SceneEntityCfg("robot", body_names="base")}
        )
        joint_effort = ObsTerm(func=mdp.joint_effort)
        joint_accs = ObsTerm(func=mdp.joint_acc_diff)  # type: ignore
        feet_lin_vel = ObsTerm(
            func=mdp.body_link_lin_vel_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*")},
        )
        feet_contact_force = ObsTerm(
            func=mdp.body_contact_force_b,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*")},
        )
        base_mass_rel = ObsTerm(
            func=mdp.rigid_body_mass,  # type: ignore
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        rigid_body_material = ObsTerm(
            func=mdp.rigid_body_material,  # type: ignore
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*")},
        )
        base_com = ObsTerm(
            func=mdp.rigid_body_com,  # type: ignore
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        action_delay_legs = ObsTerm(func=mdp.action_delay)
        push_force = ObsTerm(
            func=mdp.body_composed_force_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        push_torque = ObsTerm(
            func=mdp.body_composed_torque_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        contact_information = ObsTerm(
            func=mdp.body_contact_information,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
            },
        )
        key_points_pos_b = ObsTerm(
            func=mdp.key_points_pos_b,
            params={
                "command_name": "motion_tracking",
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            },
        )
        body_link_lin_vel_b = ObsTerm(
            func=mdp.body_link_lin_vel_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    command: CommandCfg = CommandCfg()
    proprioception: ProprioceptionCfg = ProprioceptionCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    scale_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "pos_distribution_params": (-0.02, 0.02),
            "operation": "add",
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.7, 0.7), "y": (-0.7, 0.7)},
            "velocity_range": {},
        },
    )

    # # interval
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque_stochastic,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": {
                "x": (-1000.0, 1000.0),
                "y": (-1000.0, 1000.0),
                "z": (-500.0, 500.0),
            },  # force = mass * dv / dt
            "torque_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0.0, 0.0)},
            "probability": 0.002,  # Expect step = 1 / probability
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # tracking_body_pos_w_exp = RewTerm(
    #     func=mdp.tracking_body_pos_w_exp,
    #     weight=1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "command_name": "motion_tracking",
    #         "std": math.sqrt(0.005),
    #     },
    # )

    # tracking_body_pos_exp = RewTerm(
    #     func=mdp.tracking_body_pos_exp,
    #     weight=1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "command_name": "motion_tracking",
    #         "std": math.sqrt(0.005),
    #     },
    # )

    tracking_joint_pos = RewTerm(
        func=mdp.tracking_joint_pos_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "command_name": "motion_tracking",
            "std": math.sqrt(0.1),
        },
    )
    tracking_joint_vel = RewTerm(
        func=mdp.tracking_joint_vel_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "command_name": "motion_tracking",
            "std": math.sqrt(5.0),
        },
    )
    tracking_body_lin_vel = RewTerm(
        func=mdp.tracking_body_vel_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "command_name": "motion_tracking",
            "std": math.sqrt(0.25),
        },
    )
    tracking_body_ang_vel = RewTerm(
        func=mdp.tracking_body_ang_vel_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "command_name": "motion_tracking",
            "std": math.sqrt(0.5),
        },
    )
    tracking_key_points_w_exp = RewTerm(
        func=mdp.tracking_key_points_w_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "command_name": "motion_tracking",
            "std": math.sqrt(0.01),
        },
    )
    tracking_key_points_exp = RewTerm(
        func=mdp.tracking_key_points_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "command_name": "motion_tracking",
            "std": math.sqrt(0.01),
        },
    )
    # -- penalties
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_diff_l2, weight=-2.5e-7)  # type: ignore
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*")},
    )
    feet_height_l2 = RewTerm(
        func=mdp.tracking_body_height_l2,
        weight=-10.0,
        params={"command_name": "motion_tracking", "asset_cfg": SceneEntityCfg("robot", body_names=".*")},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    root_pos_err_termination = DoneTerm(
        func=mdp.root_pos_err_termination,
        params={
            "command_name": "motion_tracking",
            "threshold": 0.25,
            "probability": 0.005,
        },
    )
    root_quat_error_magnitude_termination = DoneTerm(
        func=mdp.root_quat_error_magnitude_termination,
        params={
            "command_name": "motion_tracking",
            "threshold": math.pi / 2,
            "probability": 0.005,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    command_sampling_weights = CurrTerm(
        func=mdp.command_sampling_weights,  # type: ignore
        params={"command_name": "motion_tracking"},
    )


##
# Environment configuration
##


@configclass
class DeepMicicEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
