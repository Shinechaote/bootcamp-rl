# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Soft imitation learning types, configurations and hyperparameters.

Useful resources for these implementations:
    IQ-Learn: https://arxiv.org/abs/2106.12142
              https://github.com/Div99/IQ-Learn
    P^2IL:    https://arxiv.org/abs/2209.10968
              https://github.com/lviano/P2IL
"""
import abc
import dataclasses
from typing import Any, Dict, Tuple
import jax.numpy as jnp
import jax
from jax.lax import stop_gradient
from csil.utils import concatenate_samples, expand_obs_dim


@dataclasses.dataclass
class SoftImitationConfig(abc.ABC):
    """Abstact base class for soft imitation learning."""

    @abc.abstractmethod
    def critic_loss_factory(self):
        """Define the critic loss based on the algorithm."""

    @abc.abstractmethod
    def policy_loss_factory(self):
        """Define the critic loss based on the algorithm."""


@dataclasses.dataclass
class BasicSACConfig(SoftImitationConfig):

    gamma: float
    monitor_bc_metrics: bool = False

    def critic_loss_factory(
        self, policy_network, critic_network, entropy_coefficient_network
    ):
        """Define the critic loss based on the algorithm."""

        def critic_loss_fn(
            policy_params,
            critic_params,
            critic_target_params,
            entropy_coefficient_params,
            online_samples,
            key,
            demo_samples=None,
        ):

            # Critic loss
            dist = stop_gradient(
                policy_network.apply(policy_params, expand_obs_dim(online_samples.obs))[0]
            )
            next_action = dist.sample(seed=key)[0]
            next_log_prob = dist.log_prob(next_action)

            beta_with_grad = entropy_coefficient_network.apply(
                entropy_coefficient_params
            )
            beta = stop_gradient(beta_with_grad)

            next_q_target = critic_network.apply(
                critic_target_params, expand_obs_dim(online_samples.next_obs), jnp.expand_dims(next_action, 0)
            )
            min_next_q_target = jnp.min(next_q_target)

            # # Use knowledge to bound rogue Q values.
            # max_value = 1.0
            # min_next_q_target = jnp.clip(min_next_q_target, a_max=max_value)

            y = online_samples.rewards + self.gamma * (
                1 - online_samples.terminations
            ) * (min_next_q_target - beta * next_log_prob)

            q = critic_network.apply(
                critic_params, expand_obs_dim(online_samples.obs), jnp.expand_dims(online_samples.actions, 0)
            )
            q_loss = (q - y) ** 2

            # For uniform signature
            metrics = {
                "loss/q_loss": q_loss,
                "q_value/entropy_bonus": -(beta * next_log_prob).mean(),
            }

            # During training, expert actions should become / stay high likelihood.
            if self.monitor_bc_metrics:
                expert_action_dist, _, expert_min_variance = policy_network.apply(
                    policy_params, expand_obs_dim(demo_samples.obs)
                )
                expert_action_sample = expert_action_dist.sample(seed=key)
                expert_ent_approx = -expert_action_dist.log_prob(
                    expert_action_sample
                ).mean()
                expert_llhs = expert_action_dist.log_prob(demo_samples.actions)
                expert_se = (expert_action_dist.mode() - demo_samples.actions) ** 2
                expert_min_variance = jnp.min(expert_action_dist.variance())

                online_action_dist, _, online_min_variance = policy_network.apply(
                    policy_params, expand_obs_dim(online_samples.obs)
                )
                online_action_sample = online_action_dist.sample(seed=key)
                online_ent_approx = -online_action_dist.log_prob(
                    online_action_sample
                ).mean()
                online_llh = online_action_dist.log_prob(online_samples.actions).mean()
                online_se = (online_action_dist.mode() - online_samples.actions) ** 2

                metrics.update(
                    {
                        "monitor_bc_metrics/expert_llh_mean": expert_llhs.mean(),
                        "monitor_bc_metrics/expert_llh_max": expert_llhs.max(),
                        "monitor_bc_metrics/expert_llh_min": expert_llhs.min(),
                        "monitor_bc_metrics/expert_mse": expert_se.mean(),
                        "monitor_bc_metrics/online_llh": online_llh,
                        "monitor_bc_metrics/online_mse": online_se.mean(),
                        "monitor_bc_metrics/expert_ent": expert_ent_approx,
                        "monitor_bc_metrics/online_ent": online_ent_approx,
                        "monitor_bc_metrics/expert_min_variance": expert_min_variance.min(),
                        "monitor_bc_metrics/online_min_variance": online_min_variance.min(),
                    }
                )

            return q_loss, metrics

        def safe_mean(x):
            return jnp.mean(x) if x is not None else x

        critic_vmap_loss_fn = jax.vmap(
            critic_loss_fn,
            in_axes=(
                None,
                None,
                None,
                None,
                0,
                0,
                0,
            ),
            out_axes=0,
        )
        mean_critic_vmapped_loss_fn = lambda *a, **k: jax.tree.map(
            safe_mean, critic_vmap_loss_fn(*a, **k)
        )
        critic_grad_loss_fn = jax.value_and_grad(
            mean_critic_vmapped_loss_fn, argnums=1, has_aux=True
        )

        return critic_grad_loss_fn

    def policy_loss_factory(
        self, policy_network, critic_network, entropy_coefficient_network
    ):
        def policy_loss_fn(
            policy_params,
            critic_params,
            entropy_coefficient_params,
            online_samples,
            key: jax.random.PRNGKey,
        ):
            # Policy loss
            dist = policy_network.apply(policy_params, expand_obs_dim(online_samples.obs))[0]
            current_action = dist.sample(seed=key)[0]
            current_log_prob = dist.log_prob(current_action)

            beta_with_grad = entropy_coefficient_network.apply(
                entropy_coefficient_params
            )
            beta = stop_gradient(beta_with_grad)

            q = critic_network.apply(
                stop_gradient(critic_params), expand_obs_dim(online_samples.obs), jnp.expand_dims(current_action, 0)
            )
            min_q = jnp.min(q)

            policy_loss = beta * current_log_prob - min_q

            metrics = {
                "loss/policy_loss": policy_loss,
                "entropy/beta": beta,
                "q_value/q_value": min_q,
            }

            return policy_loss, metrics

        def safe_mean(x):
            return jnp.mean(x) if x is not None else x

        policy_vmap_loss_fn = jax.vmap(
            policy_loss_fn,
            in_axes=(None, None, None, 0, 0),
            out_axes=0,
        )
        mean_policy_vmapped_loss_fn = lambda *a, **k: jax.tree.map(
            safe_mean, policy_vmap_loss_fn(*a, **k)
        )
        policy_grad_loss_fn = jax.value_and_grad(
            mean_policy_vmapped_loss_fn, argnums=(0), has_aux=True
        )

        return policy_grad_loss_fn

    def beta_loss_factory(self, policy_network, entropy_coefficient_network):
        def beta_loss_fn(
            policy_params,
            entropy_coefficient_params,
            target_entropy,
            online_samples,
            key,
        ):
            dist = policy_network.apply(policy_params, expand_obs_dim(online_samples.obs))[0]
            current_action = dist.sample(seed=key)[0]
            current_log_prob = dist.log_prob(current_action)
            entropy = stop_gradient(-current_log_prob)

            beta_with_grad = entropy_coefficient_network.apply(
                entropy_coefficient_params
            )
            entropy_loss = beta_with_grad * (entropy - target_entropy)

            metrics = {
                "loss/entropy_loss": entropy_loss,
                "entropy/entropy": entropy,
            }

            return entropy_loss, metrics

        def safe_mean(x):
            return jnp.mean(x) if x is not None else x

        beta_vmap_loss_fn = jax.vmap(
            beta_loss_fn, in_axes=(None, None, None, 0, 0), out_axes=0
        )
        mean_beta_vmapped_loss_fn = lambda *a, **k: jax.tree.map(
            safe_mean, beta_vmap_loss_fn(*a, **k)
        )
        beta_grad_loss_fn = jax.value_and_grad(
            mean_beta_vmapped_loss_fn, argnums=(1), has_aux=True
        )

        return beta_grad_loss_fn

