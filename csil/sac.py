import os
from flax.training.train_state import TrainState
from flax.core import FrozenDict
from collections import deque
import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import optax
import wandb
from tqdm import tqdm
from enum import Enum
import mediapy as media
from networks import EntropyCoefficient, ConstantEntropyCoefficient
from csil.environment_wrappers import (
    Sample,
    RobomimicEncodedImage,
    RobomimicStateOnly,
    RobomimicPizero,
    RobomimicNormalizedImage,
    NormalGymWrapper
)
from dataclasses import asdict
from csil.utils import ItemBuffer, normalize_observation, expand_obs_dim


# SAC Implementation from Nico Bohlinger
# https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/algorithms/sac/flax


class RLTrainState(TrainState):
    target_params: FrozenDict


class Algorithm(Enum):
    SAC = 0
    CSIL = 1


class SAC:
    def __init__(
        self,
        config,
        policy_network,
        policy_params,
        critic_network,
        critic_params,
        image_encoder_params=None,
        dataset_statistics=None,
        simulated_vla_network=None,
        simulated_vla_params=None,
    ):
        self.config = config
        self.env = config.environment.env_creation_fn(
            config,
            render=False,
            offcamera_render=config.algorithm.save_training_recordings
            or config.algorithm.use_vla
            or config.algorithm.image_based_csil,
        )

        if config.algorithm.is_normal_env:
            self.env = NormalGymWrapper(self.env)
        elif config.algorithm.joint_space_obs:
            self.env = RobomimicStateOnly(self.env)
        elif config.algorithm.image_based_csil and config.algorithm.freeze_embeddings:
            self.env = RobomimicEncodedImage(
                self.env,
                config.algorithm.image_encoder_network,
                image_encoder_params,
                eef_state=config.algorithm.concat_joint_state,
                object_state=config.algorithm.concat_object_state,
            )
        elif config.algorithm.image_based_csil:
            self.env = RobomimicNormalizedImage(
                self.env,
                eef_state=config.algorithm.concat_joint_state,
                object_state=config.algorithm.concat_object_state,
            )
        elif config.algorithm.use_vla:
            self.env = RobomimicPizero(
                self.env,
                config.vla.policy,
                config.environment.task_prompt,
                config.algorithm.action_dataset_statistics,
                config.algorithm.action_normalization,
                object_state=config.algorithm.concat_object_state,
            )
        elif (
            config.algorithm.use_linear_residual_combination
            or config.algorithm.use_vla_action_for_hetstat
        ):
            self.env = RobomimicStateOnly(
                self.env, simulated_vla_network, simulated_vla_params
            )
        else:
            raise NotImplementedError(
                "The selected environment wrapper option is not defined"
            )

        self.image_encoder_params = image_encoder_params

        print("Max Episode Length:", config.environment.max_episode_length)
        self.total_timesteps = config.algorithm.total_timesteps
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.target_entropy = config.algorithm.target_entropy
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.start_with_evaluation = config.algorithm.start_with_evaluation
        self.record_eval_episodes = config.algorithm.record_eval_episodes
        self.num_recordings_per_eval = config.algorithm.num_recordings_per_eval

        self.use_vla = config.algorithm.use_vla
        self.algorithm = Algorithm.SAC

        self.key = jax.random.PRNGKey(config.general.seed)
        self.key, policy_key, critic_key, reward_key, entropy_coefficient_key = (
            jax.random.split(self.key, 5)
        )

        self.policy_network = policy_network
        self.critic_network = critic_network

        self.max_eval_success_rate = 0.0
        self.max_eval_params = jax.tree_util.tree_map(lambda x: x.copy(), policy_params)

        self.saving_frequency = config.checkpoints.saving_frequency

        if self.algorithm == Algorithm.SAC:
            self.entropy_coefficient_network = EntropyCoefficient(
                config.algorithm.entropy_coefficient
            )
        else:
            self.entropy_coefficient_network = ConstantEntropyCoefficient(
                config.algorithm.entropy_coefficient
            )

        self.q_learning_rate = config.algorithm.critic_learning_rate
        self.policy_learning_rate = config.algorithm.policy_learning_rate
        self.entropy_learning_rate = config.algorithm.policy_learning_rate

        self.critic_grad_loss_fn = config.algorithm.loss_config.critic_loss_factory(
            self.policy_network,
            self.critic_network,
            self.entropy_coefficient_network,
        )
        self.policy_grad_loss_fn = config.algorithm.loss_config.policy_loss_factory(
            self.policy_network,
            self.critic_network,
            self.entropy_coefficient_network,
        )
        self.beta_grad_loss_fn = config.algorithm.loss_config.beta_loss_factory(
            self.policy_network, self.entropy_coefficient_network
        )

        self.policy_state = RLTrainState.create(
            apply_fn=self.policy_network.apply,
            params=policy_params,
            target_params=jax.tree_util.tree_map(lambda x: x.copy(), policy_params),
            tx=optax.inject_hyperparams(optax.adam)(
                learning_rate=self.policy_learning_rate
            ),
        )

        self.bc_critic_params = jax.tree_util.tree_map(
            lambda x: x.copy(), critic_params
        )

        self.critic_state = RLTrainState.create(
            apply_fn=self.critic_network.apply,
            params=critic_params,
            target_params=jax.tree_util.tree_map(lambda x: x.copy(), critic_params),
            tx=optax.inject_hyperparams(optax.adam)(learning_rate=self.q_learning_rate),
        )

        self.entropy_coefficient_state = TrainState.create(
            apply_fn=self.entropy_coefficient_network.apply,
            params=self.entropy_coefficient_network.init(entropy_coefficient_key),
            tx=optax.inject_hyperparams(optax.adam)(
                learning_rate=self.entropy_learning_rate
            ),
        )

    def save_replay_buffers(self, online_buffer_state, demo_buffer_state=None):
        fields = {"online": asdict(online_buffer_state)}
        if demo_buffer_state is not None:
            fields["demo"] = demo_buffer_state
        np.save(
            os.path.join(self.config.general.run_dir, "replay_buffer.npy"), fields, allow_pickle=True
        )

    def train(self, demo_buffer_state=None):
        @jax.jit
        def get_processed_action(action):
            return jnp.clip(
                action, self.config.environment.as_low, self.config.environment.as_high
            )

        @jax.jit
        def get_random_action(key):
            action = jax.random.uniform(
                key,
                (1, self.config.environment.sampled_action.shape[0]),
                minval=-0.999,
                maxval=0.999,
            )
            if self.config.algorithm.action_dataset_statistics is not None:
                processed_action = self.config.algorithm.action_unnormalization(
                    action, *self.config.algorithm.action_dataset_statistics
                )
            else:
                processed_action = action

            processed_action = get_processed_action(processed_action)[0]

            return processed_action, action

        @jax.jit
        def get_q_value(critic_state: TrainState, obs, action: np.ndarray):
            return self.critic_network.apply(
                critic_state.params, expand_obs_dim(obs), action
            )

        @jax.jit
        def get_action(
            policy_params: dict, obs, key: jax.random.PRNGKey
        ):
            dist = self.policy_network.apply(policy_params, expand_obs_dim(obs))[0]
            key, subkey = jax.random.split(key)
            action = dist.sample(seed=subkey)

            if self.config.algorithm.action_dataset_statistics is not None:
                processed_action = self.config.algorithm.action_unnormalization(
                    action, *self.config.algorithm.action_dataset_statistics
                )
            else:
                processed_action = action

            if self.config.algorithm.use_linear_residual_combination:
                if self.config.algorithm.action_dataset_statistics is not None:
                    pizero_action = self.config.algorithm.action_unnormalization(
                        obs.vla_action, *self.config.algorithm.action_dataset_statistics
                    )
                processed_action = (processed_action + pizero_action) / 2.0

            processed_action = get_processed_action(processed_action)[0]

            return processed_action, action, key

        @jax.jit
        def update(
            policy_state: RLTrainState,
            critic_state: RLTrainState,
            entropy_state: TrainState,
            batched_online_sample,
            batched_demonstration_sample,
            key: jax.random.PRNGKey,
        ):
            key, critic_key, policy_key, entropy_key = jax.random.split(key, 4)

            metrics = {}

            entropy_keys = jax.random.split(
                entropy_key, self.config.algorithm.batch_size 
            )
            (entropy_loss, (entropy_metrics)), entropy_gradients = (
                self.beta_grad_loss_fn(
                    policy_state.params,
                    entropy_state.params,
                    self.target_entropy,
                    batched_online_sample,
                    entropy_keys,
                )
            )

            entropy_state = entropy_state.apply_gradients(grads=entropy_gradients)

            for i in range(self.config.algorithm.critic_and_reward_utd):
                critic_keys = jax.random.split(
                    critic_key, self.config.algorithm.batch_size
                )
                (critic_loss, (critic_metrics)), critic_gradients = (
                    self.critic_grad_loss_fn(
                        policy_state.params,
                        critic_state.params,
                        critic_state.target_params,
                        entropy_state.params,
                        batched_online_sample,
                        critic_keys,
                        batched_demonstration_sample,
                    )
                )

                critic_state = critic_state.apply_gradients(grads=critic_gradients)

                # Update targets (policy_target_params is just the bc policy params so they do not get updated)
                critic_state = critic_state.replace(
                    target_params=optax.incremental_update(
                        critic_state.params,
                        critic_state.target_params,
                        self.config.algorithm.tau,
                    )
                )

                if self.config.algorithm.image_based_csil and not self.config.algorithm.freeze_embeddings:
                    policy_state.params["params"]["ResNetTorso_0"] = jax.tree.map(
                        lambda x: x.copy(), critic_state.params["params"]["ResNetTorso_0"]
                    )

            policy_keys = jax.random.split(
                policy_key, self.config.algorithm.batch_size 
            )
            (policy_loss, (policy_metrics)), policy_gradients = (
                self.policy_grad_loss_fn(
                    policy_state.params,
                    critic_state.params,
                    entropy_state.params,
                    batched_online_sample,
                    policy_keys,
                )
            )

            policy_state = policy_state.apply_gradients(grads=policy_gradients)

            metrics.update(critic_metrics)
            metrics.update(policy_metrics)
            metrics.update(entropy_metrics)

            metrics["csil/learning_rate"] = policy_state.opt_state.hyperparams[
                "learning_rate"
            ]
            metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
            metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)
            metrics["gradients/entropy_grad_norm"] = optax.global_norm(
                entropy_gradients
            )

            return policy_state, critic_state, entropy_state, metrics, key

        buffer = ItemBuffer()
        online_buffer_state = buffer.init(
            self.config.algorithm.buffer_size, self.config.general.example_item
        )

        global_step = 0
        if self.start_with_evaluation:
            steps_since_evaluation = self.evaluation_frequency + 1
        else:
            steps_since_evaluation = 0
        steps_since_critic_reset = 0
        nr_updates = 0
        nr_episodes = 0
        metrics = {}
        last_eval_metrics = None
        scores = deque(maxlen=100)
        scores.append(0)
        successes = deque(maxlen=100)
        successes.append(0)
        last_episode_score = 0
        demo_samples_added = 0
        max_succ_episode_length = 0

        progress_bar = tqdm(
            range(self.total_timesteps + self.learning_starts), initial=global_step
        )

        while global_step < self.total_timesteps + self.learning_starts:
            images = []

            obs, _ = self.env.reset()
            if (
                self.config.algorithm.save_training_recordings
                and nr_episodes > 0
                and nr_episodes % self.config.algorithm.training_recording_frequency
                == 0
            ):
                images.append(self.env.render())

            episode_score = 0

            if self.config.algorithm.normalize_observations:
                obs = normalize_observation(
                    obs, self.config.algorithm.obs_dataset_statistics
                )

            for i in range(self.config.environment.max_episode_length):
                if (
                    self.algorithm == Algorithm.SAC
                    and global_step < self.learning_starts
                ):
                    self.key, subkey = jax.random.split(self.key)
                    processed_action, action = get_random_action(subkey)
                else:
                    processed_action, action, self.key = get_action(
                        self.policy_state.params,
                        obs,
                        self.key,
                    )

                next_obs, reward, terminated, truncated, info = self.env.step(
                    processed_action
                )
                if (
                    self.config.algorithm.save_training_recordings
                    and nr_episodes > 0
                    and nr_episodes % self.config.algorithm.training_recording_frequency
                    == 0
                ):
                    images.append(self.env.render())

                if self.config.algorithm.normalize_observations:
                    next_obs = normalize_observation(
                        next_obs, self.config.algorithm.obs_dataset_statistics
                    )

                # We do not need next_actions for our loss configs, only for pre-training
                cur_sample = Sample(
                    obs=obs,
                    actions=action,
                    next_obs=next_obs,
                    next_actions=None,
                    rewards=np.array([reward]),
                    terminations=np.array([terminated]),
                )
                online_buffer_state = buffer.add(online_buffer_state, cur_sample)
                obs = next_obs

                episode_score += reward

                done = terminated | truncated

                should_optimize = (
                    global_step > self.learning_starts and global_step > self.batch_size
                )

                # Optimizing - Prepare batches
                if should_optimize:
                    self.key, online_key, expert_key = jax.random.split(self.key, 3)
                    online_batch = buffer.sample(
                        online_buffer_state, online_key, shape=(self.batch_size,)
                    )
                    online_batch.terminations = jnp.squeeze(online_batch.terminations)
                    online_batch.rewards = jnp.squeeze(online_batch.rewards)

                    demonstration_batch = None
                    if demo_buffer_state is not None:
                        demonstration_batch = buffer.sample(
                            demo_buffer_state, expert_key, shape=(self.batch_size,)
                        )
                        demonstration_batch.terminations = jnp.squeeze(
                            demonstration_batch.terminations
                        )
                        demonstration_batch.rewards = jnp.squeeze(
                            demonstration_batch.rewards
                        )

                    # Optimizing - Q-functions, policy and entropy coefficient
                    (
                        upd_policy_state,
                        upd_critic_state,
                        upd_entropy_coefficient_state,
                        metrics,
                        self.key,
                    ) = update(
                        self.policy_state,
                        self.critic_state,
                        self.entropy_coefficient_state,
                        online_batch,
                        demonstration_batch,
                        self.key,
                    )

                    if (
                        global_step
                        >= self.learning_starts
                        + self.config.algorithm.critic_warmup_steps
                        or self.config.algorithm.critic_warmup_steps < 0
                    ):
                        self.policy_state = upd_policy_state
                    self.critic_state = upd_critic_state
                    self.entropy_coefficient_state = upd_entropy_coefficient_state

                    metrics["episode_statistics/average_reward_100"] = np.mean(scores)
                    metrics["episode_statistics/success_rate_training"] = np.mean(
                        successes
                    )
                    metrics["episode_statistics/last_episode_score"] = (
                        last_episode_score
                    )
                    metrics["episode_statistics/max_successful_episode_length"] = (
                        max_succ_episode_length
                    )
                    metrics["episode_statistics/total_episodes"] = nr_episodes

                    metrics["q_value/env_reward"] = reward

                    if self.algorithm == Algorithm.SAC:
                        metrics["sacfd/demo_samples_added"] = demo_samples_added
                        metrics["sacfd/online_rewards_mean"] = (
                            online_batch.rewards.mean()
                        )
                        metrics["sacfd/online_rewards_std"] = online_batch.rewards.std()
                        if demo_buffer_state is not None:
                            metrics["sacfd/demo_rewards_mean"] = (
                                demonstration_batch.rewards.mean()
                            )
                            metrics["sacfd/demo_rewards_std"] = (
                                demonstration_batch.rewards.std()
                            )

                    metrics["csil/terminated_mean"] = online_batch.terminations.mean()
                    metrics["general/learning_steps"] = (
                        global_step - self.learning_starts
                    )

                    if last_eval_metrics is not None:
                        metrics.update(last_eval_metrics)
                        last_eval_metrics = None

                    nr_updates += 1
                    if wandb.run is not None:
                        wandb.log(metrics)

                global_step += 1
                if should_optimize:
                    steps_since_evaluation += 1
                    steps_since_critic_reset += 1
                progress_bar.update(1)

                if (
                    global_step % self.saving_frequency == 0
                    and self.saving_frequency > 0
                    and global_step > 0
                ):
                    checkpoint = {
                        "policy_params": self.policy_state.params,
                        "policy_target_params": self.policy_state.target_params,
                        "critic_params": self.critic_state.params,
                        "critic_target_params": self.critic_state.target_params,
                        "entropy_coefficient_params": self.entropy_coefficient_state.params,
                        "rng_key": self.key,
                    }

                    self.config.checkpoints.manager.save(
                        (global_step // self.saving_frequency) + 1,
                        args=ocp.args.Composite(checkpoint=checkpoint),
                    )
                    self.config.checkpoints.manager.wait_until_finished()

                    if self.config.general.save_replay_buffers:
                        self.save_replay_buffers(online_buffer_state, demo_buffer_state)

                if done or global_step >= self.total_timesteps + self.learning_starts:
                    break

            # Episode > 0 is a hotfix because the first recording is always broken
            if (
                self.config.algorithm.save_training_recordings
                and nr_episodes > 0
                and nr_episodes % self.config.algorithm.training_recording_frequency
                == 0
            ):
                video_path = os.path.join(
                    self.config.general.run_dir,
                    f"rendered_video/training_episode_{(global_step//1000)*1000}.mp4",
                )

                if not os.path.exists(os.path.join(self.config.general.run_dir, "rendered_video")):
                    os.makedirs(os.path.join(self.config.general.run_dir, "rendered_video"))

                media.write_video(
                    video_path, images, fps=self.config.environment.render_fps
                )

                # Sometimes the rendering breaks down during an episode
                if os.path.getsize(video_path) / (1024**2) > 5:
                    os.remove(video_path)
                images = []

            nr_episodes += 1
            scores.append(episode_score)
            last_episode_score = episode_score
            successes.append(1.0 if terminated else 0.0)
            if terminated and i > max_succ_episode_length:
                max_succ_episode_length = i

            while (
                demo_samples_added / (global_step + demo_samples_added)
                < self.config.algorithm.demo_to_online_ratio
                and self.algorithm == Algorithm.SAC
                and demo_buffer_state is not None
            ):
                demos = buffer.sample(
                    demo_buffer_state,
                    jax.random.PRNGKey(0),
                    shape=(demo_buffer_state.size,),
                )
                demo_end_indeces = np.where(
                    np.diff(demos.terminations, append=0) == -1
                )[0]
                selected_episode = np.random.choice(demo_end_indeces[:-1].shape[0])

                # https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/lfd/lfd_adder.py#L52
                episode = jax.tree.map(
                    lambda b: b[
                        demo_end_indeces[selected_episode]
                        + 1 : demo_end_indeces[selected_episode + 1]
                        + 1
                    ],
                    demo_buffer_state.buffer,
                )
                buffer.add_batch(online_buffer_state, episode)
                demo_samples_added += episode.obs.shape[0]

            # Evaluation
            if (
                steps_since_evaluation >= self.evaluation_frequency
                and self.evaluation_frequency > 0
                and (self.start_with_evaluation or should_optimize)
            ):
                # Only one evaluation at the start for if eval frequency is lower than warmup phase
                self.start_with_evaluation = False
                steps_since_evaluation = 0
                learning_steps = global_step - self.learning_starts
                if learning_steps < 0:
                    learning_steps = -1
                global_step += 1

                success_rate = self.config.environment.run_eval_fn(
                    self.config,
                    self.policy_network,
                    self.policy_state.params,
                    image_encoder_params=self.image_encoder_params,
                    render_video=self.record_eval_episodes,
                    num_eval_episodes=self.config.algorithm.evaluation_episodes,
                    video_name_prefix=f"during_rl_{(global_step//1000)*1000}",
                    policy_target_params=self.policy_state.target_params,
                    critic_network=self.critic_network,
                    critic_params=self.critic_state.params,
                    learning_steps=learning_steps,
                    num_episodes=nr_episodes,
                )
                if success_rate > self.max_eval_success_rate:
                    self.max_eval_success_rate = success_rate
                    self.max_eval_params = jax.tree_util.tree_map(
                        lambda x: x.copy(), self.policy_state.params
                    )

        if self.saving_frequency > 0:
            # checkpoint = {"policy_params": self.policy_state.params, "policy_target_params": self.policy_state.target_params, "critic_params": self.critic_state.params, "critic_target_params": self.critic_state.target_params,
            #               "entropy_coefficient_params": self.entropy_coefficient_state.params, "reward_params": self.reward_state.params, "rng_key": self.key}

            # self.config.checkpoints.manager.save((global_step//self.saving_frequency) + 1, args=ocp.args.Composite(checkpoint=checkpoint))
            # self.config.checkpoints.manager.wait_until_finished()

            if self.config.general.save_replay_buffers:
                self.save_replay_buffers(online_buffer_state, demo_buffer_state)

        if self.evaluation_frequency > 0:
            learning_steps = global_step - self.learning_starts
            if learning_steps < 0:
                learning_steps = -1
            global_step += 1

            success_rate = self.config.environment.run_eval_fn(
                self.config,
                self.policy_network,
                self.policy_state.params,
                image_encoder_params=self.image_encoder_params,
                render_video=self.record_eval_episodes,
                num_eval_episodes=self.config.algorithm.evaluation_episodes,
                video_name_prefix=f"during_rl_{(global_step//1000)*1000}",
                policy_target_params=self.policy_state.target_params,
                critic_network=self.critic_network,
                critic_params=self.critic_state.params,
                learning_steps=learning_steps,
                num_episodes=nr_episodes,
            )

        return (
            self.policy_state.params,
            self.max_eval_params,
            self.policy_state.target_params,
            self.critic_state.params,
            nr_episodes,
        )
