from csil import run_vla_eval, get_config, get_args_parsed, robomimic_state_obs_parse_fn, robomimic_env_creation, default_gym_env_creation, mimicgen_env_creation
import os
from openpi.training import config as pizero_config_module
from openpi.policies import policy_config
from datetime import datetime
import wandb
import time
import mimicgen

last_evaluated_steps = 8000
pizero_param_path = "/home/stud_scherer/openpi/checkpoints/pi0_mimicgen_robomimic_ph/balanced_mimicgen_lora"
num_eval_episodes = 25
pizero_config_name = "pi0_mimicgen_robomimic_ph"
checkpoint_step_size = -500

args = get_args_parsed()
cur_date = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
config = get_config(args, cur_date)

wandb.init(entity="robot-learning-rt2", project="csil-for-vlas", name=f"openpi training eval - {cur_date}")

initial = False

# while True:
for env_name in ["ThreePieceAssembly_D0", "NutAssemblySquare", "PickPlaceCan"]:
    config.environment.name = env_name
    if env_name in ["ThreePieceAssembly_D0"]:
        config.environment.max_episode_length = 800
        config.environment.env_type = "mimicgen"
        config.environment.env_creation_fn = mimicgen_env_creation
        config.environment.observation_parse_fn = robomimic_state_obs_parse_fn
        config.environment.task_prompt = "assemble the three pieces"
    elif env_name in ["Lift", "PickPlaceCan", "NutAssemblySquare"]:
        config.environment.max_episode_length = 400
        config.environment.task_prompt = {"Lift": "lift cube", "PickPlaceCan": "put can in field with can", "NutAssemblySquare": "put ring on stick"}[env_name]
        config.environment.env_type = "robomimic"
        config.environment.env_creation_fn = robomimic_env_creation
        config.environment.observation_parse_fn = robomimic_state_obs_parse_fn
    else:
        config.environment.env_type = "normal"
        config.environment.env_creation_fn = default_gym_env_creation
        config.environment.observation_parse_fn = lambda x: x
    for to_be_evaluated_checkpoint_step in [29999]:
        # to_be_evaluated_checkpoint_step = last_evaluated_steps - checkpoint_step_size
        to_be_evaluated_checkpoint = os.path.join(pizero_param_path, str(to_be_evaluated_checkpoint_step))
        if os.path.exists(to_be_evaluated_checkpoint):
            pizero_config = pizero_config_module.get_config(pizero_config_name)
            checkpoint_dir = to_be_evaluated_checkpoint
            print(f"Loading newest checkpoint {to_be_evaluated_checkpoint}")
            # Create a trained policy.
            config.vla.policy = policy_config.create_trained_policy(pizero_config, checkpoint_dir)
            print(env_name)

            success_rate = run_vla_eval(config, render_video=True, num_eval_episodes=num_eval_episodes, video_name_prefix=f"{env_name}_eval_{to_be_evaluated_checkpoint_step}", executed_actions_per_chunk=4)
            wandb.log({f"evaluation/{env_name}_eval_success_rate": success_rate, "evaluation/pizero_train_steps": to_be_evaluated_checkpoint_step})
        # last_evaluated_steps -= checkpoint_step_size
    # print(f"Waiting for next checkpoint {to_be_evaluated_checkpoint_step}")

