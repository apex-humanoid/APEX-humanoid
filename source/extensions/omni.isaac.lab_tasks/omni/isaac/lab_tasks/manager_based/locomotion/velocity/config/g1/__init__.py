"""Register G1 locomotion skills with Gymnasium."""

import gymnasium as gym

gym.register(
    id="Isaac-ClimbUp-G1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.g1.climbup_env_cfg:G1ClimbUpEnvCfg",
        "rsl_rl_cfg_entry_point": "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1ClimbUpPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Down-G1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.g1.down_env_cfg:G1DownEnvCfg",
        "rsl_rl_cfg_entry_point": "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1DownPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-LyDown-G1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.g1.lyingdown_env_cfg:G1LyingDownEnvCfg",
        "rsl_rl_cfg_entry_point": "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1LyDownPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-StandUp-G1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.g1.standup_env_cfg:G1StandUpEnvCfg",
        "rsl_rl_cfg_entry_point": "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1StandUpPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Velocity-Crawl",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.g1.g1_crawl_cfg:G1CrawlEnvCfg",
        "rsl_rl_cfg_entry_point": "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1CrawlPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-G1-Velocity-Walk",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.g1.g1_walk_cfg:G1WalkEnvCfg",
        "rsl_rl_cfg_entry_point": "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.g1.agents.walk_ppo_cfg:G1WalkPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-G1-Velocity-Crawl-Stage1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.g1.g1_crawl_stages_cfg:G1CrawlStage1EnvCfg",
        "rsl_rl_cfg_entry_point": "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1CrawlPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-G1-Velocity-Crawl-Stage2",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.g1.g1_crawl_stages_cfg:G1CrawlStage2EnvCfg",
        "rsl_rl_cfg_entry_point": "omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1CrawlPPORunnerCfg",
    },
)
