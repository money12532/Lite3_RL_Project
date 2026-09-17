"""PPO configuration dedicated to the Lite3 stair task."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import DeeproboticsLite3FlatPPORunnerCfg


@configclass
class DeeproboticsLite3StairPPORunnerCfg(DeeproboticsLite3FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "deeprobotics_lite3_stairs"
        self.num_steps_per_env = 24
        self.max_iterations = 15000
        self.save_interval = 100
        # Fine-tune the validated flat policy conservatively. A smaller step
        # avoids erasing its natural gait when the terrain distribution changes.
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 0.002
