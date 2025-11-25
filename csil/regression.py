"""
Core regression library with model definitions. 
Takes the soft-actor critic tanh transform from
https://github.com/deepmind/acme/jax/networks/distributional.py.
"""

from typing import NamedTuple
import jax.numpy as jnp
import jax
import numpy as np
import optax
import wandb
from tqdm import tqdm
import tensorflow_probability.substrates.jax as tfp
from collections import deque
from optax._src.base import Params
from csil.sac import ItemBuffer
import mediapy as media

tfd = tfp.distributions
tfb = tfp.bijectors


class State(NamedTuple):
    opt_state: optax.OptState
    params: jnp.array


@jax.jit
def random_crop(image, key, crop_size=84, padding=4):
    key_h, key_w = jax.random.split(key)

    image = jnp.pad(
        image, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode="edge"
    )
    B, H, W, C = image.shape

    # Random top-left corner per image
    max_x = H - crop_size
    max_y = W - crop_size
    x = jax.random.randint(key_h, (B,), 0, max_x + 1)
    y = jax.random.randint(key_w, (B,), 0, max_y + 1)

    # Use vmap to gather crops efficiently
    def crop_one(img, x0, y0):
        return jax.lax.dynamic_slice(img, (x0, y0, 0), (crop_size, crop_size, C))

    return jax.vmap(crop_one)(image, x, y)


def pretrain_policy(network, params, demo_buffer_state, config):
    key = jax.random.PRNGKey(config.general.seed)

    print(
        f"Stochastic Eval during Policy Pretraining: {config.pretraining.stochastic_eval}"
    )

    wandb_prefix = "policy_pretraining"

    opt = optax.adam(config.pretraining.policy_learning_rate)
    opt_state = opt.init(params)
    state = State(opt_state, params)
    eval_batches = config.pretraining.evaluation_at_steps

    num_actions = np.prod(config.environment.sampled_action.shape)

    def loss_fn(params, batch, key):
        key, subkey = jax.random.split(key, 2)

        dist, cut_dist, _ = network.apply(params, batch.obs)
        predicted_actions = dist.sample(seed=subkey)
        mse = ((batch.actions - dist.mode()) ** 2).mean()
        nllh = -cut_dist.log_prob(batch.actions).mean()
        entropy = -cut_dist.log_prob(predicted_actions).mean()

        loss = mse + nllh

        metrics = {
            f"{wandb_prefix}/expert_mse": mse,
            f"{wandb_prefix}/expert_entropy": entropy,
            f"{wandb_prefix}/expert_nllh": nllh,
            f"{wandb_prefix}/loss": loss,
            f"{wandb_prefix}/embedding_noise_alpha": (
                config.algorithm.embedding_noise_alpha
                if config.algorithm.use_embedding_noise
                else 0.0
            ),
        }

        return loss, metrics

    def step(state, batch, key):
        params = state.params

        (loss_value, metrics), policy_grad = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch, key
        )
        policy_updates, opt_state = opt.update(policy_grad, state.opt_state)
        params = optax.apply_updates(state.params, policy_updates)

        return (
            State(
                opt_state,
                params,
            ),
            metrics,
        )

    step = jax.jit(step)

    history = 50
    ent_threshold = -config.pretraining.target_entropy_factor * num_actions

    entropy_history = deque(maxlen=history)

    max_success_rate = 0
    max_success_rate_params = jax.tree_util.tree_map(lambda x: x.copy(), state.params)

    buffer = ItemBuffer()

    for batch_num in tqdm(range(config.pretraining.policy_pretrain_steps)):
        key, subkey = jax.random.split(key)
        batch = buffer.sample(
            demo_buffer_state, subkey, shape=(config.algorithm.batch_size,)
        )
        batch = jax.tree.map(lambda x: x.copy(), batch)

        if config.algorithm.use_vla and config.algorithm.use_embedding_noise:
            if batch_num == 0:
                print(
                    f"Using Embedding Noise with alpha {config.algorithm.embedding_noise_alpha}"
                )
            key, rand_key = jax.random.split(key)
            embedding_shape = batch.obs.embeddings.shape
            noise = (
                config.algorithm.embedding_noise_alpha / embedding_shape[1]
            ) * jax.random.uniform(
                rand_key,
                (embedding_shape[1], embedding_shape[2]),
                minval=-1.0,
                maxval=1.0,
            )
            batch.obs.embeddings = batch.obs.embeddings + noise

        key, _ = jax.random.split(key)

        if config.algorithm.image_based_csil:
            key, gaussian_key = jax.random.split(key)
            batch.obs.image = random_crop(batch.obs.image, gaussian_key)

        state, metrics = step(state, batch, key)

        if wandb.run is not None:
            wandb.log(metrics)
        entropy_history.append(metrics[f"{wandb_prefix}/expert_entropy"])

        if batch_num in eval_batches:
            success_rate = config.environment.run_eval_fn(
                config,
                network,
                state.params,
                render_video=config.algorithm.record_eval_episodes,
                num_eval_episodes=25,
                video_name_prefix=f"before_rl_{batch_num}_",
                stochastic_eval=config.pretraining.stochastic_eval,
            )

            if success_rate > max_success_rate:
                max_success_rate = success_rate
                max_success_rate_params = jax.tree_util.tree_map(
                    lambda x: x.copy(), state.params
                )
            print(
                f"Success Rate ({batch_num}): {success_rate} | Max Success rate ({max_success_rate})"
            )

        if batch_num >= history and all(m < ent_threshold for m in entropy_history):
            break
    print(metrics)

    if config.pretraining.eval_frequency <= 0:
        max_success_rate_params = jax.tree_util.tree_map(
            lambda x: x.copy(), state.params
        )

    return state.params, max_success_rate_params


class TrainingState(NamedTuple):
    params: Params
    target_params: Params
    opt_state: optax.OptState


def critic_pretraining(
    critic_network,
    critic_params,
    demo_buffer_state,
    config,
):
    discount_factor = config.algorithm.gamma
    tau = config.pretraining.tau
    num_steps = config.pretraining.critic_pretrain_steps
    batch_size = config.algorithm.batch_size
    learning_rate = config.pretraining.critic_learning_rate

    key = jax.random.PRNGKey(config.general.seed)
    optimiser = optax.adam(learning_rate)

    initial_opt_state = optimiser.init(critic_params)

    target_params = jax.tree.map(lambda x: x.copy(), critic_params)

    critic_state = TrainingState(critic_params, target_params, initial_opt_state)

    def loss(params, target_params, batch, key):
        """."""
        key, key_embedding_noise, key_next_embedding_noise = jax.random.split(key, 3)

        reward = batch.rewards

        next_v_target = critic_network.apply(
            target_params, batch.next_obs, batch.next_actions
        )

        next_v_target = next_v_target.min(axis=-1)
        next_v_target = jax.lax.stop_gradient(next_v_target)

        max_value = 0.0 if config.algorithm.negative_reward else config.algorithm.max_reward / (1.0 - 0.99)
        next_v_target = jnp.clip(next_v_target, a_max=max_value)

        q_sarsa_target = jnp.expand_dims(
            reward + (1 - batch.terminations) * discount_factor * next_v_target, -1
        )

        q = critic_network.apply(params, batch.obs, batch.actions)
        sarsa_loss = ((q_sarsa_target - q) ** 2).mean()

        def nonbatch_critic(s, a):
            batch_s = jax.tree.map(lambda f: jnp.expand_dims(f, 0), s)
            return critic_network.apply(params, batch_s, a[None, ...])[0]

        dqda = jax.vmap(jax.jacfwd(nonbatch_critic, argnums=1), in_axes=(0))
        grads = dqda(batch.obs, batch.actions)
        # Sum over actions, average over the rest.
        grad_norm = jnp.sqrt((grads**2).sum(axis=-1).mean())

        loss = sarsa_loss + grad_norm

        metrics = {
            "pretrainer_critic/loss": loss,
            "pretrainer_critic/sarsa_loss": sarsa_loss,
            "pretrainer_critic/grad_norm": grad_norm,
            "pretrainer_critic/q_mean": q.mean(),
            "pretrainer_critic/q_max": q.max(),
            "pretrainer_critic/q_sarsa_target_mean": q_sarsa_target.mean(),
            "pretrainer_critic/q_sarsa_target_max": q_sarsa_target.max(),
            "pretrainer_critic/tau": tau,
            "pretrainer_critic/learning_rate": learning_rate,
            "pretrainer_critic/reward": reward.mean(),
        }

        return loss, metrics

    @jax.jit
    def step(state, batch, key):
        values, grads = jax.value_and_grad(loss, has_aux=True)(
            state.params, state.target_params, batch, key
        )
        _, metrics = values
        updates, opt_state = optimiser.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        target_params = jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau, state.target_params, params
        )
        return TrainingState(params, target_params, opt_state), metrics

    buffer = ItemBuffer()

    for batch_num in tqdm(range(num_steps)):
        key, sample_key, step_key = jax.random.split(key, 3)
        batch = buffer.sample(demo_buffer_state, sample_key, shape=(batch_size,))
        batch.rewards = jnp.squeeze(batch.rewards)
        batch.terminations = jnp.squeeze(batch.terminations)

        critic_state, metrics = step(critic_state, batch, step_key)

        if wandb.run is not None:
            metrics["pretrainer_critic/learning_steps"] = batch_num
            wandb.log(metrics)

    print(metrics)
    return critic_state.params
