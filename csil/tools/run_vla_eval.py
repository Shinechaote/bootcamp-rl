from ml_collections import config_dict
import wandb
import orbax.checkpoint as ocp
import jax.numpy as jnp
import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from datetime import datetime
from regression import train_model

from csil import get_args_parsed, get_config, load_data, run_vla_eval
from networks import StationaryHeteroskedasticNormalTanhDistribution
from utils import calculate_dataset_statistics, disable_tf_from_using_gpus, visualize_action_dims

tfd = tfp.distributions


np.set_printoptions(precision=3, suppress=True)
args = get_args_parsed()

cur_date = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
config = get_config(args, cur_date)


for i in range(5):
    run_vla_eval(config, num_eval_episodes=25, executed_actions_per_chunk=1)
