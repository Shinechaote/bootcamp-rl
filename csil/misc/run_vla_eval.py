import numpy as np
import tensorflow_probability.substrates.jax as tfp
from datetime import datetime
import wandb
from ml_collections import config_dict

from csil import get_args_parsed, get_config, run_vla_eval

tfd = tfp.distributions


np.set_printoptions(precision=3, suppress=True)
args = get_args_parsed()

cur_date = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
config = get_config(args, cur_date)

tags = ["thesis experiment", "vla_base", config.environment.name]
if args.newer_model_lora:
    tags.append("new_model_lora")
elif args.newer_model_full:
    tags.append("new_model_full")
elif args.old_model:
    tags.append("old_model")
else:
    tags.append("new_model")

if args.num_action_chunk_actions != 1:
    tags.append(f"{args.num_action_chunk_actions} step action chunk")

wandb.init(
    entity="robot-learning-rt2",
    project="csil-for-vlas",
    name=f"VLA Base Performance - {config.environment.name} - {cur_date}",
    tags=tags,
)

if wandb.run is not None:
    wandb.run.log_code(".")
    config_for_wandb = {}
    for key in config.keys():
        if type(config[key]) == config_dict.ConfigDict:
            config_for_wandb[key] = dict(config[key])

    wandb.config.update(config_for_wandb)

for i in range(5):
    run_vla_eval(
        config,
        num_eval_episodes=25,
        executed_actions_per_chunk=args.num_action_chunk_actions,
        render_video=True,
    )
