from ml_collections import config_dict
import wandb
import orbax.checkpoint as ocp
import jax.numpy as jnp
import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from datetime import datetime
from regression import train_model

from csil import get_args_parsed, get_config, load_data
from networks import StationaryHeteroskedasticNormalTanhDistribution
from utils import (
    calculate_dataset_statistics,
    disable_tf_from_using_gpus,
    visualize_action_dims,
)

tfd = tfp.distributions


np.set_printoptions(precision=3, suppress=True)
args = get_args_parsed()

cur_date = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
config = get_config(args, cur_date)

disable_tf_from_using_gpus()

demonstrations, action_dataset_statistics, residual_action_dataset_statistics = (
    load_data(config, args)
)
s, a, next_s, next_a, rewards, terminations, discounts, vla_a, next_state_vla_a = (
    demonstrations
)

tags = [config.environment.name, "partial embeddings"]
if config.algorithm.use_vla:
    if config.algorithm.use_vla_action_for_hetstat:
        tags.append("residual")
    else:
        tags.append("action head")

    if args.newer_model_lora:
        tags.append("new_model_lora")
    elif args.newer_model_full:
        tags.append("new_model_full")
    elif args.old_model:
        tags.append("old_model")
    else:
        tags.append("new_model")
else:
    tags.append("states only")

if not args.debug:
    wandb.init(
        entity="robot-learning-rt2",
        project="csil-for-vlas",
        name=f"{config.algorithm.num_embeds_index} Partial Embeddings, Stochastic Performance over training steps - {config.environment.name} - {cur_date}",
        tags=tags,
    )

if wandb.run is not None:
    wandb.run.log_code(".")
    config_for_wandb = {}
    for key in config.keys():
        if type(config[key]) == config_dict.ConfigDict:
            config_for_wandb[key] = dict(config[key])

    wandb.config.update(config_for_wandb)

env = config.environment.env_creation_fn(config)

sampled_obs = np.expand_dims(np.squeeze(config.environment.sampled_obs), 0)
sampled_action = np.expand_dims(np.squeeze(config.environment.sampled_action), 0)

print("Obs shape:", sampled_obs.shape, "Action shape", sampled_action.shape)

nr_hidden_units = config.algorithm.nr_hidden_units
bottleneck_size = config.algorithm.bottleneck_size

# For sanity checking whether residual code works with just state-based csil
simulated_vla_network = None
simulated_vla_params = None


# visualize_action_dims(a)

if not config.algorithm.use_vla and config.algorithm.use_vla_action_for_hetstat:
    simulated_vla_network = StationaryHeteroskedasticNormalTanhDistribution(
        sampled_action.shape[1],
        layers=[nr_hidden_units, nr_hidden_units, bottleneck_size],
        feature_dimension=nr_hidden_units,
        layer_norm_mlp=config.algorithm.layer_norm_policy,
        prior_var=0.75,
        activation=jax.nn.elu,
        min_var=config.algorithm.min_var,
        stationary_activation=config.algorithm.stationary_activation_function,
        is_residual=False,
    )
    simulated_vla_params = simulated_vla_network.init(
        jax.random.PRNGKey(0), sampled_obs
    )

policy_network = StationaryHeteroskedasticNormalTanhDistribution(
    sampled_action.shape[1],
    layers=[nr_hidden_units, nr_hidden_units, bottleneck_size],
    feature_dimension=nr_hidden_units,
    layer_norm_mlp=config.algorithm.layer_norm_policy,
    prior_var=0.75,
    activation=jax.nn.elu,
    min_var=config.algorithm.min_var,
    faithful_distributions=True,
    stationary_activation=config.algorithm.stationary_activation_function,
    is_residual=config.algorithm.use_vla_action_for_hetstat,
    residual_action_dataset_statistics=residual_action_dataset_statistics,
)
if config.algorithm.use_vla_action_for_hetstat:
    policy_params = policy_network.init(
        jax.random.PRNGKey(0), sampled_obs, sampled_action
    )
else:
    policy_params = policy_network.init(jax.random.PRNGKey(0), sampled_obs)


if config.checkpoints.reload_checkpoint:
    print(f"Reloading checkpoint from {config.checkpoints.checkpoint_dir}")
    policy_params = config.checkpoints.manager.restore(step=200000)["params"]

wandb_prefix = "pretrainer_policy"
last_policy_params, best_policy_params, eval_replay_buffer_data = train_model(
    policy_network,
    policy_params,
    s,
    a,
    vla_a,
    env,
    config,
    action_dataset_statistics,
    n_iters=config.pretraining.policy_pretrain_steps,
    learning_rate=config.pretraining.policy_learning_rate,
    batch_size=config.algorithm.batch_size,
    target_entropy_factor=config.pretraining.target_entropy_factor,
    seed=config.general.seed,
    return_data_for_replay_buffer=False,
    wandb_prefix=wandb_prefix,
    stochastic_eval=True,
)
