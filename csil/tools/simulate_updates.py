import flax.linen as nn
import os
import shutil
import json
import logging
from collections import deque
import tree
import numpy as np
import jax
from jax.lax import stop_gradient
import jax.numpy as jnp
import flax
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import orbax
import optax
import wandb
from ml_collections import config_dict
from tqdm import tqdm
import dataclasses
from typing import NamedTuple
import enum
import cv2
import mediapy as media
from gymnasium.utils.save_video import save_video
from csil import (
    get_config,
    make_hetstat_mlp,
    prior_policy_log_likelihood,
    get_args_parsed,
)
from networks import (
    StationaryHeteroskedasticNormalTanhDistribution,
    DoubleMLP,
    CriticMLP,
    RewardNetwork,
    RewardNetworkForResidual,
    triangle_activation,
    periodic_relu_activation,
)
from sac import (
    EntropyCoefficient,
    ConstantEntropyCoefficient,
    ReplayBuffer,
    RLTrainState,
)
from argparse import ArgumentParser
from datetime import datetime


args = get_args_parsed()

cur_date = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
config = get_config(args, cur_date)

sampled_obs = np.expand_dims(config.environment.sampled_obs, 0)
sampled_action = np.expand_dims(config.environment.sampled_action, 0)

print("Obs shape:", sampled_obs.shape, "Action shape", sampled_action.shape)

nr_hidden_units = config.algorithm.nr_hidden_units
bottleneck_size = config.algorithm.bottleneck_size

if config.algorithm.use_octo_action_for_hetstat:
    policy_network = make_hetstat_mlp(
        sampled_action.shape[1],
        [nr_hidden_units, nr_hidden_units, bottleneck_size, nr_hidden_units],
        prior_variance=0.75,
        use_skip_connection=True,
        min_var=config.algorithm.min_var,
        stationary_activation=config.algorithm.stationary_activation_function,
    )
    policy_params = policy_network.init(
        jax.random.PRNGKey(0), sampled_obs, sampled_action
    )
else:
    policy_network = make_hetstat_mlp(
        sampled_action.shape[1],
        [nr_hidden_units, nr_hidden_units, bottleneck_size, nr_hidden_units],
        prior_variance=0.75,
        min_var=config.algorithm.min_var,
        stationary_activation=config.algorithm.stationary_activation_function,
    )
    policy_params = policy_network.init(jax.random.PRNGKey(0), sampled_obs)


critic_network = CriticMLP([nr_hidden_units, nr_hidden_units, 1])

prior = prior_policy_log_likelihood(sampled_action)
csil_alpha = config.algorithm.reward_scale_factor / config.environment.num_actions

if config.algorithm.use_octo_action_for_hetstat:
    reward_network = RewardNetworkForResidual(
        policy_network,
        prior,
        csil_alpha,
        config.algorithm.negative_reward,
        config.environment.num_actions,
        config.algorithm.max_reward,
    )
else:
    reward_network = RewardNetwork(
        policy_network,
        prior,
        csil_alpha,
        config.algorithm.negative_reward,
        config.environment.num_actions,
        config.algorithm.max_reward,
    )

entropy_coefficient = ConstantEntropyCoefficient(config.algorithm.entropy_coefficient)

# TODO Load policy, critic and reward params here

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
checkpoint_manager = orbax.checkpoint.CheckpointManager(
    os.path.join(args.run_dir, "checkpoints"), orbax_checkpointer, options
)

# Need to specify which step needs to be loaded
step = checkpoint_manager.latest_step()  # step = 4
checkpoint = checkpoint_manager.restore(step)

policy_params = checkpoint["policy_params"]
critic_params = checkpoint["critic_params"]
reward_params = checkpoint["reward_params"]
if "rgn_key" in checkpoint:
    key = checkpoint["rng_key"]
else:
    key = jax.random.PRNGKey(0)


policy_state = RLTrainState.create(
    apply_fn=policy_network.apply,
    params=policy_params,
    target_params=jax.tree_util.tree_map(lambda x: x.copy(), policy_params),
    tx=optax.inject_hyperparams(optax.adam)(
        learning_rate=config.algorithm.policy_learning_rate
    ),
)

critic_state = RLTrainState.create(
    apply_fn=critic_network.apply,
    params=critic_params,
    target_params=jax.tree_util.tree_map(lambda x: x.copy(), critic_params),
    tx=optax.inject_hyperparams(optax.adam)(
        learning_rate=config.algorithm.critic_learning_rate
    ),
)

entropy_state = TrainState.create(
    apply_fn=entropy_coefficient.apply,
    params=entropy_coefficient.init(jax.random.PRNGKey(config.general.seed)),
    tx=optax.inject_hyperparams(optax.adam)(
        learning_rate=config.algorithm.policy_learning_rate
    ),
)

# Reward Gradient Norm Clipping
if config.algorithm.reward_grad_norm_clip_threshold != -1:
    reward_state = TrainState.create(
        apply_fn=reward_network.apply,
        params=reward_params,
        tx=optax.chain(
            optax.clip_by_global_norm(config.algorithm.reward_grad_norm_clip_threshold),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=config.algorithm.reward_learning_rate
            ),
        ),
    )
else:
    reward_state = TrainState.create(
        apply_fn=reward_network.apply,
        params=reward_params,
        tx=optax.inject_hyperparams(optax.adam)(
            learning_rate=config.algorithm.reward_learning_rate
        ),
    )

reward_state = TrainState.create(
    apply_fn=reward_network.apply,
    params=reward_params,
    tx=optax.chain(
        optax.clip_by_global_norm(75),
        optax.inject_hyperparams(optax.adam)(
            learning_rate=config.algorithm.reward_learning_rate
        ),
    ),
)

# Load replay buffer
all_replay_buffers = np.load(
    os.path.join(args.run_dir, "replay_buffer.npy"), allow_pickle=True
).item()
online_replay_buffer = ReplayBuffer.load_from_numpy(
    all_replay_buffers["online_replay_buffer"]
)
demo_replay_buffer = ReplayBuffer.load_from_numpy(
    all_replay_buffers["demo_replay_buffer"]
)


def reward_grad_fn(config):
    def expert_loss_fn(
        reward_fn,
        reward_params,
        demonstration_obs,
        demonstration_actions,
        demonstration_next_obs,
        demonstration_rewards,
        demonstration_terminations,
        demonstration_discount,
        demonstration_octo_actions,
        demonstration_octo_next_state_action,
    ):
        reward_fn = jax.vmap(reward_fn, in_axes=(None, 0, 0, None), out_axes=0)

        expert_reward = reward_fn(
            reward_params,
            demonstration_obs,
            demonstration_actions,
            demonstration_discount,
        )

        csil_alpha = (
            config.algorithm.reward_scale_factor / config.environment.num_actions
        )

        expert_log_ratio = (
            expert_reward / csil_alpha
        ) + config.environment.num_actions * config.algorithm.max_reward

        expert_loss = -expert_log_ratio

        metrics = {}
        metrics["expert_metrics/expert_log_ratio_min"] = expert_log_ratio.min()
        metrics["expert_metrics/expert_log_ratio"] = expert_log_ratio.mean()
        metrics["expert_metrics/expert_log_ratio_max"] = expert_log_ratio.max()
        metrics["expert_metrics/expert_loss_min"] = expert_loss.min()
        metrics["expert_metrics/expert_loss"] = expert_loss.mean()
        metrics["expert_metrics/expert_loss_max"] = expert_loss.max()
        metrics["expert_metrics/expert_reward_min"] = expert_reward.min()
        metrics["expert_metrics/expert_reward"] = expert_reward.mean()
        metrics["expert_metrics/expert_reward_max"] = expert_reward.max()

        return expert_loss.mean(), metrics

    def kl_loss_fn(
        reward_fn,
        reward_params,
        online_obs,
        online_actions,
        online_next_obs,
        online_rewards,
        online_terminations,
        online_discount,
        online_octo_actions,
        online_octo_next_state_action,
    ):
        reward_fn = jax.vmap(reward_fn, in_axes=(None, 0, 0, None), out_axes=0)
        online_reward = reward_fn(
            reward_params,
            online_obs,
            online_actions,
            online_discount,
        )

        csil_alpha = (
            config.algorithm.reward_scale_factor / config.environment.num_actions
        )

        # In some cases the online reward can go as low as -50
        # even when the mean is positive. To be robust to these outliers, we just
        # clip the negative online rewards, as the role of this term is to
        # regularize the large positive values.
        # This KL estimator is motivated in http://joschu.net/blog/kl-approx.html
        online_log_ratio = (
            online_reward / csil_alpha
            + config.environment.num_actions * config.algorithm.max_reward
        )

        # the estimator is log r + 1/r - 1, and the reward is beta log r
        # the estimator is logr.exp() + 1 - r and the reward is beta * logr for the positive and beta * (log r - num_actions * max_reward) for the negative case
        # policy_kl_est = jnp.exp(-online_log_ratio + jax.lax.stop_gradient(online_log_ratio -jnp.clip(online_log_ratio, a_min=-5))) + online_log_ratio - 1.

        # policy_kl_est = jnp.exp(-online_log_ratio + jax.lax.stop_gradient(online_log_ratio -jnp.clip(online_log_ratio, a_min=-5))) + online_log_ratio + 1.

        # Changed sign
        policy_kl_est = (
            jnp.exp(
                online_log_ratio
                - jax.lax.stop_gradient(
                    online_log_ratio - jnp.clip(online_log_ratio, a_max=5)
                )
            )
            - online_log_ratio
            - 1.0
        )

        # policy_kl_est = jnp.exp(jnp.minimum(online_log_ratio, 5)) + 1. - online_log_ratio
        # safe_online_log_ratio = jnp.maximum(online_log_ratio, -5.)
        # policy_kl_est = jnp.maximum(jnp.exp(-safe_online_log_ratio) + 1. + online_log_ratio, 0)

        kl_loss = policy_kl_est

        metrics = {}
        metrics["kl_metrics/online_log_ratio_min"] = online_log_ratio.min()
        metrics["kl_metrics/online_log_ratio"] = online_log_ratio.mean()
        metrics["kl_metrics/online_log_ratio_max"] = online_log_ratio.max()
        metrics["kl_metrics/kl_loss_min"] = kl_loss.min()
        metrics["kl_metrics/kl_loss"] = kl_loss.mean()
        metrics["kl_metrics/kl_loss_max"] = kl_loss.max()
        metrics["online_metrics/online_reward_min"] = online_reward.min()
        metrics["online_metrics/online_reward"] = online_reward.mean()
        metrics["online_metrics/online_reward_max"] = online_reward.max()

        return kl_loss.mean(), metrics

    expert_grad_loss_fn = jax.value_and_grad(expert_loss_fn, argnums=(1), has_aux=True)

    kl_grad_loss_fn = jax.value_and_grad(kl_loss_fn, argnums=(1), has_aux=True)

    return expert_grad_loss_fn, kl_grad_loss_fn


@jax.jit
def clip_gradients_by_global_norm(grads, max_norm):
    g_norm = optax.global_norm(grads)
    # Compute the scaling factor
    trigger = g_norm > max_norm
    scale = jnp.where(trigger, max_norm / (g_norm + 1e-6), 1.0)
    return jax.tree_map(lambda g: g * scale, grads)


@jax.jit
def update(
    policy_state: RLTrainState,
    critic_state: RLTrainState,
    entropy_state: TrainState,
    reward_state: TrainState,
    batched_online_sample,
    batched_demonstration_sample,
    key: jax.random.PRNGKey,
):
    if config.algorithm.use_octo_action_for_hetstat:

        def policy_fn(params, x, y):
            return policy_network.apply(params, x, y)

    else:

        def policy_fn(params, x):
            return policy_network.apply(params, x)

    def critic_fn(params, x, y):
        return critic_network.apply(params, x, y)

    def entropy_fn(params):
        return entropy_coefficient.apply(params)

    if reward_network is not None:
        if config.algorithm.use_octo_action_for_hetstat:

            def reward_fn(params, x, octo_action, y, discount):
                return reward_network.apply(
                    params,
                    jnp.expand_dims(x, 0),
                    jnp.expand_dims(octo_action, 0),
                    jnp.expand_dims(y, 0),
                )[0]

        else:

            def reward_fn(params, x, y, discount):
                return reward_network.apply(
                    params, jnp.expand_dims(x, 0), jnp.expand_dims(y, 0)
                )[0]

    key, critic_key, policy_key, entropy_key = jax.random.split(key, 4)

    metrics = {}

    (expert_loss, expert_metrics), expert_gradient = config.expert_loss_fn(
        reward_fn, reward_state.params, *batched_demonstration_sample
    )

    (kl_loss, kl_metrics), kl_gradient = config.kl_loss_fn(
        reward_fn, reward_state.params, *batched_online_sample
    )

    # expert_gradient = clip_gradients_by_global_norm(expert_gradient, 75)

    # total_gradient = kl_gradient

    total_gradient = jax.tree_util.tree_map(
        lambda x, y: x + y, expert_gradient, kl_gradient
    )

    metrics.update(expert_metrics)
    metrics.update(kl_metrics)

    metrics["expert_gradient"] = optax.global_norm(expert_gradient)
    metrics["kl_gradient"] = optax.global_norm(kl_gradient)
    metrics["total_gradient"] = optax.global_norm(total_gradient)

    metrics["reward_loss"] = expert_loss + kl_loss

    reward_state = reward_state.apply_gradients(grads=total_gradient)

    # online_states = batched_online_sample[0]
    # octo_action = batched_online_sample[6]
    # if reward_network is not None:
    #     (entropy_loss, (entropy_metrics)), entropy_gradients = config.beta_grad_loss_fn(
    #         policy_fn, policy_state.params, policy_state.target_params, entropy_fn, entropy_state.params, config.algorithm.target_entropy, online_states, octo_action, entropy_key)
    # else:
    #     entropy_keys = jax.random.split(entropy_key, config.algorithm.batch_size)
    #     (entropy_loss, (entropy_metrics)), entropy_gradients = config.beta_grad_loss_fn(
    #         policy_fn, policy_state.params, entropy_fn, entropy_state.params, config.algorithm.target_entropy, online_states, entropy_keys)

    # entropy_state = entropy_state.apply_gradients(
    #     grads=entropy_gradients)

    # if reward_network is not None:
    #     (critic_loss, (critic_metrics)), (critic_gradients, reward_gradients) = config.critic_grad_loss_fn(
    #         policy_fn, policy_state.params, policy_state.target_params, critic_fn, critic_state.params, critic_state.target_params, entropy_fn, entropy_state.params, config.algorithm.target_entropy,
    #         reward_fn, reward_state.params, *batched_online_sample, *batched_demonstration_sample, critic_key)
    # else:
    #     critic_keys = jax.random.split(critic_key, config.algorithm.batch_size)
    #     (critic_loss, (critic_metrics)), critic_gradients = config.critic_grad_loss_fn(
    #         policy_fn, policy_state.params, critic_fn, critic_state.params, critic_state.target_params, entropy_fn, entropy_state.params, *batched_online_sample[:5], critic_keys)

    # # critic_state = critic_state.apply_gradients(grads=critic_gradients)

    # if reward_network is not None:
    #     reward_state = reward_state.apply_gradients(
    #         grads=reward_gradients)

    # # Update targets (policy_target_params is just the bc policy params so they do not get updated)
    # critic_state = critic_state.replace(target_params=optax.incremental_update(
    #     critic_state.params, critic_state.target_params, config.algorithm.tau))

    # if reward_network is not None:
    #     # Only used if we dont use the bc_policy as a prior (not sure when this is the case)
    #     prior_fn = None

    #     (policy_loss, (policy_metrics)), policy_gradients = config.policy_grad_loss_fn(
    #         policy_fn, policy_state.params, policy_state.target_params, critic_fn, critic_state.params, entropy_fn, entropy_state.params, config.algorithm.target_entropy,
    #         prior_fn, reward_fn, reward_state.params, *batched_online_sample, *batched_demonstration_sample, policy_key)
    # else:
    #     policy_keys = jax.random.split(policy_key, config.algorithm.batch_size)
    #     (policy_loss, (policy_metrics)), policy_gradients = config.policy_grad_loss_fn(
    #         policy_fn, policy_state.params, critic_fn, critic_state.params, entropy_fn, entropy_state.params, *batched_online_sample[:5], policy_keys)

    # policy_state = policy_state.apply_gradients(grads=policy_gradients)

    # metrics.update(critic_metrics)
    # metrics.update(policy_metrics)
    # metrics.update(entropy_metrics)

    # metrics["lr/learning_rate"] = policy_state.opt_state.hyperparams["learning_rate"]
    # metrics["gradients/policy_grad_norm"] = optax.global_norm(
    #     policy_gradients)
    # if reward_network is not None:
    #     metrics["gradients/reward_grad_norm"] = optax.global_norm(
    #         reward_gradients)
    # metrics["gradients/critic_grad_norm"] = optax.global_norm(
    #     critic_gradients)
    # metrics["gradients/entropy_grad_norm"] = optax.global_norm(
    #     entropy_gradients)

    return policy_state, critic_state, entropy_state, reward_state, metrics, key


wandb.init(
    entity="robot-learning-rt2",
    project="csil-for-vlas",
    name=f"clipped total_grad to 75, simulated replay buffer - {config.environment.name} - {cur_date}",
    tags=["simulated replay buffer"],
)
if wandb.run is None:
    print("WANDB NOT ACTIVATED")
size = 10000
iteration_length = all_replay_buffers["online_replay_buffer"]["states"].shape[0] - size

for i in tqdm(range(iteration_length)):
    online_replay_buffer.size = size + i
    # online_replay_buffer.size = 150000

    batched_online_sample = online_replay_buffer.sample(config.algorithm.batch_size)
    batched_demonstration_sample = demo_replay_buffer.sample(
        config.algorithm.batch_size
    )

    config.expert_loss_fn, config.kl_loss_fn = reward_grad_fn(config)

    policy_state, critic_state, entropy_state, reward_state, metrics, key = update(
        policy_state,
        critic_state,
        entropy_state,
        reward_state,
        batched_online_sample,
        batched_demonstration_sample,
        key,
    )
    metrics["csil/learning_steps"] = i
    if wandb.run is not None:
        wandb.log(metrics)
