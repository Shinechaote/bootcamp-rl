import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import mediapy as media
import wandb
import os
import matplotlib.pyplot as plt
from csil.environment_wrappers import (
    RobomimicPizero,
    RobomimicNormalizedImage,
    RobomimicEncodedImage,
    RobomimicStateOnly,
    NormalGymWrapper,
    get_joint_states,
)
from csil.utils import expand_obs_dim, normalize_observation
from openpi_client import image_tools


def get_processed_action(config, action):
    return jnp.clip(action, config.environment.as_low, config.environment.as_high)


def run_normal_eval(
    config,
    policy_network,
    policy_params,
    image_encoder_params=None,
    render_video=False,
    num_eval_episodes=5,
    video_name_prefix="",
    policy_target_params=None,
    critic_network=None,
    critic_params=None,
    learning_steps=None,
    num_episodes=None,
    final_eval=False,
    stochastic_eval=False,
):
    @jax.jit
    def get_action_deterministic(params, obs):
        dist = policy_network.apply(params, expand_obs_dim(obs))[0]
        return dist.mode().squeeze()

    @jax.jit
    def get_action(params, obs, key):
        dist = policy_network.apply(params, expand_obs_dim(obs))[0]
        key, subkey = jax.random.split(key)
        action = dist.sample(seed=subkey)
        entropy = -dist.log_prob(action)
        return action, entropy, key

    if config.algorithm.use_ibrl:
        assert critic_network is not None
        assert policy_target_params is not None

        @jax.jit
        def get_q_value(critic_params, obs, action):
            return critic_network.apply(critic_params, expand_obs_dim(obs), action)

    env = config.environment.env_creation_fn(
        config,
        render=False,
        offcamera_render=config.algorithm.record_eval_episodes
        or config.algorithm.use_vla
        or config.algorithm.image_based_csil,
    )
    if config.algorithm.is_normal_env:
        env = NormalGymWrapper(env)
    elif config.algorithm.joint_space_obs:
        env = RobomimicStateOnly(env)
    elif config.algorithm.image_based_csil and config.algorithm.freeze_embeddings:
        env = RobomimicEncodedImage(
            env,
            config.algorithm.image_encoder_network,
            image_encoder_params,
            eef_state=True,
            object_state=config.algorithm.concat_object_state,
        )
    elif config.algorithm.image_based_csil:
        env = RobomimicNormalizedImage(
            env, eef_state=True, object_state=config.algorithm.concat_object_state
        )
    elif config.algorithm.use_vla:
        env = RobomimicPizero(
            env,
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
        env = RobomimicStateOnly(
            env, config.vla.simulated_vla_network, config.vla.simulated_vla_params
        )
    elif config.environment.name == "HalfCheetah-v4":
        pass
    else:
        raise NotImplementedError(
            "The selected environment wrapper option is not defined"
        )

    num_success = 0
    episode_rewards = np.zeros((num_eval_episodes))
    last_successful = False

    key = jax.random.PRNGKey(0)
    entropies = []

    for eval_episode_ind in (eval_pbar := tqdm(range(num_eval_episodes))):
        eval_pbar.set_description(
            f"Num successes: {num_success} | Last successful: {last_successful}"
        )
        images = []
        obs, _ = env.reset()

        total_reward = 0

        if config.algorithm.use_vla and config.algorithm.use_vla_action_for_hetstat:
            base_actions = np.zeros((config.environment.max_episode_length, config.environment.sampled_action.shape[0]))
            complete_actions = np.zeros((config.environment.max_episode_length, config.environment.sampled_action.shape[0]))
            residual_actions = np.zeros((config.environment.max_episode_length, config.environment.sampled_action.shape[0]))

        for i in tqdm(range(config.environment.max_episode_length), leave=False):

            if config.algorithm.normalize_observations:
                obs = normalize_observation(
                    obs, config.algorithm.obs_dataset_statistics
                )

            if stochastic_eval:
                action, entropy, key = get_action(policy_params, obs, key)
                entropies.append(entropy)
            else:
                action = get_action_deterministic(policy_params, obs)

            if config.algorithm.use_ibrl:
                if stochastic_eval:
                    bc_action, bc_entropy, key = get_action(
                        policy_target_params, obs, key
                    )
                else:
                    bc_action = get_action_deterministic(policy_target_params, obs)

                q_policy = get_q_value(critic_params, obs, jnp.expand_dims(action, 0))
                q_bc = get_q_value(critic_params, obs, jnp.expand_dims(bc_action, 0))

                if q_bc > q_policy:
                    action = bc_action

            action = config.algorithm.action_unnormalization(
                action, *config.algorithm.action_dataset_statistics
            )
            if config.algorithm.use_linear_residual_combination:
                action = (obs.vla_action + action) / 2.0

            if config.algorithm.use_vla and config.algorithm.use_vla_action_for_hetstat:
                residual_actions[i] = action - obs.vla_action
                base_actions[i] = obs.vla_action
                complete_actions[i] = action

            action = get_processed_action(config, action)
            action = action.squeeze()

            obs, reward, terminated, truncated, info = env.step(action)
            images.append(env.render())

            total_reward += reward
            if terminated or truncated:
                break

        episode_rewards[eval_episode_ind] = total_reward
        if terminated:
            num_success += 1
        last_successful = terminated
        eval_pbar.set_description(
            f"Num successes: {num_success} | Last successful: {last_successful}"
        )

        if render_video and eval_episode_ind < config.algorithm.num_recordings_per_eval:
            # Plot base action, residual action and total action and log the plots
            if (
                config.algorithm.use_vla
                and config.algorithm.use_vla_action_for_hetstat
                and config.algorithm.plot_actions
            ):
                num_steps_in_episode = i

                actions_fig, actions_axes = plt.subplots(
                    7, 1, figsize=(10, 14), sharex=True
                )

                for i in range(7):
                    actions_axes[i].plot(
                        complete_actions[:num_steps_in_episode, i],
                        label="Action + Residual",
                        color="orange",
                    )
                    actions_axes[i].plot(
                        base_actions[:num_steps_in_episode, i],
                        label="Base Action",
                        color="blue",
                    )
                    actions_axes[i].set_ylabel(f"Dim {i+1}")
                    actions_axes[i].set_ylim([-2, 2])
                    actions_axes[i].grid(True)
                    actions_axes[i].legend(loc="upper right", fontsize="small")

                actions_axes[-1].set_xlabel("Index")
                actions_fig.tight_layout()

                residual_fig, residual_axes = plt.subplots(
                    7, 1, figsize=(10, 14), sharex=True
                )

                for i in range(7):
                    residual_axes[i].plot(
                        residual_actions[:num_steps_in_episode, i],
                        label="Residual",
                        color="orange",
                    )
                    residual_axes[i].set_ylabel(f"Dim {i+1}")
                    residual_axes[i].set_ylim([-2, 2])
                    residual_axes[i].grid(True)
                    residual_axes[i].legend(loc="upper right", fontsize="small")

                residual_axes[-1].set_xlabel("Index")
                residual_fig.tight_layout()

                if wandb.run is not None:
                    wandb.log(
                        {
                            "evaluation/actions": wandb.Image(actions_fig),
                            "evaluation/residual_actions": wandb.Image(residual_fig),
                        }
                    )

            if eval_episode_ind < config.algorithm.num_recordings_per_eval:
                video_path = os.path.join(
                    config.general.run_dir,
                    f"rendered_video/{video_name_prefix}_{eval_episode_ind}.mp4",
                )
                if not os.path.exists(
                    os.path.join(config.general.run_dir, "rendered_video")
                ):
                    os.makedirs(os.path.join(config.general.run_dir, "rendered_video"))

                media.write_video(video_path, images, fps=config.environment.render_fps)

                # Sometimes the rendering breaks down during an episode
                if os.path.getsize(video_path) / (1024**2) > 5:
                    os.remove(video_path)

    print("Successful episodes:", num_success)
    if stochastic_eval:
        print(f"Eval entropy: {np.array(entropies).mean()}")
    if wandb.run is not None:
        if final_eval:
            metrics = {
                "evaluation/final_eval_success_rate": float(num_success)
                / num_eval_episodes
            }
        else:
            metrics = {
                "evaluation/success_rate": float(num_success) / num_eval_episodes
            }
        if learning_steps is not None:
            metrics["general/learning_steps"] = learning_steps
        if num_episodes is not None:
            metrics["q_value/total_episodes"] = num_episodes
        if config.algorithm.is_normal_env:
            metrics["evaluation/eval_mean_reward"] = episode_rewards.mean()
        if stochastic_eval:
            metrics = {
                "evaluation/stochastic_success_rate": float(num_success)
                / num_eval_episodes
            }
            metrics["evaluation/eval_entropy"] = np.array(entropies).mean()
        wandb.log(metrics)

    return num_success / float(num_eval_episodes)


def run_vla_eval(
    config,
    render_video=False,
    num_eval_episodes=5,
    video_name_prefix="",
    executed_actions_per_chunk=4,
    action_dataset_statistics=None,
):
    # We dont use a wrapper here because we need the raw inputs
    env = config.environment.env_creation_fn(
        config, render=False, offcamera_render=True
    )
    env = RobomimicPizero(
        env,
        config.vla.policy,
        config.environment.task_prompt,
        config.algorithm.action_dataset_statistics,
        config.algorithm.action_normalization,
        object_state=config.algorithm.concat_object_state,
    )

    num_success = 0
    last_successful = False

    for eval_episode_ind in (eval_pbar := tqdm(range(num_eval_episodes))):
        eval_pbar.set_description(
            f"Num successes: {num_success} | Last successful: {last_successful}"
        )
        images = []
        obs, _ = env.reset()
        # images.append(env.sim.render(
        #         height=224, width=224, camera_name="agentview"
        #     )[::-1])
        images.append(env.render())

        total_reward = 0
        current_chunk_index = executed_actions_per_chunk
        # cur_chunk = None

        for i in tqdm(range(config.environment.max_episode_length), leave=False):

            # if current_chunk_index >= executed_actions_per_chunk:
            #     if config.algorithm.debug:
            #         return np.zeros((50, 7)), np.ones((1024,))

            #     joint_states = get_joint_states(obs)

            #     obs["agentview"] = env.sim.render(
            #         height=224, width=224, camera_name="agentview"
            #     )[::-1]
            #     obs["robot0_eye_in_hand"] = env.sim.render(
            #         height=224, width=224, camera_name="robot0_eye_in_hand"
            #     )[::-1]

            #     pizero_observation = {
            #         "observation/image": image_tools.convert_to_uint8(
            #             image_tools.resize_with_pad(obs["agentview"], 224, 224)
            #         ),
            #         "observation/wrist_image": image_tools.convert_to_uint8(
            #             image_tools.resize_with_pad(obs["robot0_eye_in_hand"], 224, 224)
            #         ),
            #         "observation/state": joint_states,
            #         "prompt": config.environment.task_prompt,
            #     }
            #     response = config.vla.policy.infer(pizero_observation)
            #     cur_chunk = response["actions"]

            #     current_chunk_index = 0
            # action = cur_chunk[current_chunk_index]
            action = config.algorithm.action_unnormalization(obs.vla_action, *config.algorithm.action_dataset_statistics)
            action = get_processed_action(config, action)
            current_chunk_index += 1

            obs, reward, terminated, truncated, info = env.step(action)
            # obs, reward, done, info = env.step(action)
            # terminated = reward >= 1
            # truncated = i == config.environment.max_episode_length - 1
            # images.append(env.sim.render(
            #         height=224, width=224, camera_name="agentview"
            #     )[::-1])
            images.append(env.render())

            total_reward += reward

            if terminated or truncated:
                break

        if terminated:
            num_success += 1
        last_successful = terminated
        eval_pbar.set_description(
            f"Num successes: {num_success} | Last successful: {last_successful}"
        )

        if render_video and eval_episode_ind < config.algorithm.num_recordings_per_eval:
            video_path = os.path.join(
                config.general.run_dir,
                f"rendered_video/{video_name_prefix}_{eval_episode_ind}.mp4",
            )

            if not os.path.exists(
                os.path.join(config.general.run_dir, "rendered_video")
            ):
                os.makedirs(os.path.join(config.general.run_dir, "rendered_video"))
            media.write_video(video_path, images, fps=config.environment.render_fps)

            # Sometimes the rendering breaks down during an episode and just produces noise which leads to very large but unusable videos so we delete them
            if os.path.getsize(video_path) / (1024**2) > 5:
                os.remove(video_path)

    print("Successful episodes:", num_success)
    if wandb.run is not None:
        wandb.log(
            {"evaluation/vla_eval_success_rate": float(num_success) / num_eval_episodes}
        )

    return num_success
