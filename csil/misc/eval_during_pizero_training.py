from csil import (
    run_vla_eval,
    get_config,
    get_args_parsed,
    mimicgen_env_creation,
    robomimic_state_obs_parse_fn,
    robomimic_env_creation,
    default_gym_env_creation,
)
import os
from openpi.training import config as pizero_config_module
from openpi.policies import policy_config
from datetime import datetime
import wandb
import time

last_evaluated_steps = 3400
pizero_param_path = "/home/scherer/openpi/checkpoints/pi0_nutassemblysquare_ph/pi0_nutassemblysquare_ph/"
num_eval_episodes = 25
pizero_config = "pi0_nutassemblysquare_ph"
checkpoint_step_size = 200

args = get_args_parsed()
cur_date = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
config = get_config(args, cur_date)

wandb.init(
    entity="robot-learning-rt2", project="csil-for-vlas", name="openpi training eval"
)

while last_evaluated_steps < 5000:
    if os.path.exists(
        os.path.join(
            pizero_param_path, str(last_evaluated_steps + checkpoint_step_size)
        )
    ):

        pizero_config = pizero_config_module.get_config(pizero_config)
        checkpoint_dir = os.path.join(
            pizero_param_path, str(last_evaluated_steps + checkpoint_step_size)
        )

        config.vla.policy = policy_config.create_trained_policy(
            pizero_config, checkpoint_dir
        )
        # for env_name in ["Lift", "PickPlaceCan", "NutAssemblySquare", "ThreePieceAssembly_D0"]:
        for env_name in ["NutAssemblySquare"] * 5:
            config.environment.name = env_name

            if env_name in ["ThreePieceAssembly_D0"]:
                config.environment.max_episode_length = 800
                config.environment.env_type = "mimicgen"
                config.environment.env_creation_fn = mimicgen_env_creation
                config.environment.observation_parse_fn = robomimic_state_obs_parse_fn
                config.environment.task_prompt = {
                    "ThreePieceAssembly_D2": "assemble the three pieces"
                }[env_name]
            elif env_name in ["Lift", "PickPlaceCan", "NutAssemblySquare"]:
                config.environment.max_episode_length = 400
                config.environment.task_prompt = {
                    "Lift": "lift cube",
                    "PickPlaceCan": "put can in field with can",
                    "NutAssemblySquare": "put ring on stick",
                }[env_name]
                config.environment.env_type = "robomimic"
                config.environment.env_creation_fn = robomimic_env_creation
                config.environment.observation_parse_fn = robomimic_state_obs_parse_fn
            else:
                config.environment.env_type = "normal"
                config.environment.env_creation_fn = default_gym_env_creation
                config.environment.observation_parse_fn = lambda x: x

            # Create a trained policy.
            success_rate = run_vla_eval(
                config,
                render_video=True,
                num_eval_episodes=25,
                video_name_prefix=f"vla_eval_{last_evaluated_steps}",
                executed_actions_per_chunk=1,
            )
            wandb.log(
                {
                    "eval_success_rate": success_rate,
                    "pizero_train_steps": last_evaluated_steps,
                }
            )
            last_evaluated_steps += checkpoint_step_size

    time.sleep(30)
