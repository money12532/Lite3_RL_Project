# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DeeproboticsLite3RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24 #每个环境采样多少步
    max_iterations = 10000 #最大训练轮数
    save_interval = 100   #保存模型间隔每100 iteration保存一次
    experiment_name = "deeprobotics_lite3_rough"
    empirical_normalization = False
    clip_actions = 5.0        #clip_actions 是对 Actor 输出的机器人动作进行范围限制，防止策略网络输出过大的控制指令导致机器人仿真不稳定
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5, #初始探索噪声
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],  #Actor网络三层网络
        critic_hidden_dims=[512, 256, 128],  #Critic网络三层网络
        activation="elu",  #elu激活函数
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, #Value loss权重
        use_clipped_value_loss=True, 
        clip_param=0.2,  #clip参数
        # 较小的熵奖励既保留探索能力，又避免 log_std 增长到
        # 之前长时间训练中出现的极端数值。
        entropy_coef=0.001,#熵系数
        num_learning_epochs=5,#PPO更新次数，一次采样的数据，。重复训练5遍
        num_mini_batches=4, #minibatch数量 采集一批数据分成四批来训练
        learning_rate=3.0e-4,#学习率
        schedule="adaptive",#动态调整学习率
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,#防止梯度爆炸
    )


@configclass
class DeeproboticsLite3FlatPPORunnerCfg(DeeproboticsLite3RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # RTX 3090 可以利用大量并行环境采集大批次数据，因此 24 步 rollout
        # 能提高训练速度，同时仍可为 PPO 提供足够的样本。
        self.num_steps_per_env = 24   #一次24多少步
        self.max_iterations = 5000  #整个循环执行5000次
        self.experiment_name = "deeprobotics_lite3_flat"
