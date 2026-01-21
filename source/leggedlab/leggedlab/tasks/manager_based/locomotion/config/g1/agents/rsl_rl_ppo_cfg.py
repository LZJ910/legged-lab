# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from rsl_rl.isaaclab_rl import (
    AmpCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)

from isaaclab.utils import configclass


@configclass
class UnitreeG1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = "unitree_g1_rough"
    obs_groups = {
        "policy": ["command", "proprioception"],
        "critic": ["command", "privileged"],
    }
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        rnn_type="gru",
        rnn_hidden_dim=512,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    amp = AmpCfg(
        obs_group="amp",
        num_frames=2,
        obs_normalization=True,
        amp_reward_weight=1.0,
        amp_lambda=10.0,
        lr_scale=0.5,
        num_learning_epochs=1,
        num_mini_batches=10,
        noise_scale=[0.2] * 3 + [0.05] * 3 + [0.01] * 29 + [1.5] * 29,
        data_loader_func="cfg.load_amp_data",
    )


@configclass
class UnitreeG1FlatPPORunnerCfg(UnitreeG1RoughPPORunnerCfg):
    def __post_init__(self):
        self.experiment_name = "unitree_g1_flat"
        self.policy = RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            actor_obs_normalization=True,
            critic_obs_normalization=True,
        )


@configclass
class UnitreeG1GetupPPORunnerCfg(UnitreeG1RoughPPORunnerCfg):
    def __post_init__(self):
        self.experiment_name = "unitree_g1_getup"
        self.max_iterations = 5000
        self.save_interval = 500
        self.amp.amp_reward_weight = 2.0
        self.policy.init_noise_std = 0.5
