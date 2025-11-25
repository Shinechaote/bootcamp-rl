import h5py
import numpy as np
from tqdm import tqdm
from csil.utils import (
    calculate_action_statistics,
    calculate_observation_statistics,
    get_statistics,
    dataclass_from_dict,
    normalize_residual,
    unnormalize_residual,
    normalize_observation,
    ItemBuffer,
    ItemBufferState
)
import os
import jax.numpy as jnp
from enum import Enum
from csil.environment_wrappers import Sample, Observation
from dataclasses import asdict
import jax
from csil.networks import ResNetTorso


class NormalizationTypes(Enum):
    LINEAR = 0
    MEAN_AND_VARIANCE = 1


def load_hdf5_data(config):
    # assert config.pretraining.num_demonstrations <= 200
    ds = h5py.File(config.pretraining.dataset_filename)["data"]

    state_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

    if config.algorithm.concat_object_state:
        state_keys.append("object")

    num_samples = 0

    for i in range(config.pretraining.num_demonstrations):
        cur_episode = ds[f"demo_{i}"]
        num_samples += cur_episode["obs"]["object"].shape[0]

    print("Num Samples:", num_samples)

    cur_sample = 0
    buffer = ItemBuffer()
    demo_buffer_state = buffer.init(num_samples, config.general.example_item)

    for i in tqdm(range(config.pretraining.num_demonstrations)):
        cur_episode = ds[f"demo_{i}"]
        episode_length = cur_episode["actions"].shape[0]
        last_sample = None
        cur_sample = None
        for step in range(episode_length):
            s = np.concatenate(
                [cur_episode["obs"][key][step] for key in state_keys], axis=0
            )
            a = cur_episode["actions"][step]

            # WARNING THIS ASSUMES THAT ONLY THE LAST STATE HAS A REWARD
            # We do this because robomimic's datasets have multiple states with termination=True
            reward = 1 if step == episode_length - 1 else 0
            termination = step == episode_length - 1
            reward = np.array([reward])
            termination = np.array([termination])

            if config.algorithm.use_vla:
                state_keys = ["robot0_eef_pos", "robot0_eef_quat"]
                state = np.concatenate([[cur_episode["obs"][key][step]] for key in state_keys], axis=1)[0]

                gripper = int(
                    (cur_episode["obs"]["robot0_gripper_qpos"][step][0] - cur_episode["obs"]["robot0_gripper_qpos"][step][1]) < 0.05
                )
                state = np.concatenate([state, [gripper]])
                
                s = state

                pizero_obs = {
                    "observation/image": cur_episode["obs"]["agentview_image"][step],
                    "observation/wrist_image": cur_episode["obs"][
                        "robot0_eye_in_hand_image"
                    ][step],
                    "observation/state": state,
                    "prompt": config.environment.task_prompt
                }

                response = config.vla.policy.infer(pizero_obs)
                action_chunk = response["actions"]
                embeddings = response["embeddings"][:1024]

                vla_a = action_chunk[0]
                embeddings = embeddings[:1024]
            else:
                vla_a = None
                embeddings = None

            if config.algorithm.image_based_csil:
                image_keys = ["agentview_image", "robot0_eye_in_hand_image"]
                image = np.concatenate(
                    [cur_episode["obs"][key][step] for key in image_keys], axis=-1
                )
                image = (image - 127.5) / 255.0
            else:
                image = None

            obs = Observation(
                state=s, embeddings=embeddings, image=image, vla_action=vla_a
            )
            cur_sample = Sample(
                obs=obs,
                actions=a,
                next_obs=None,
                next_actions=None,
                rewards=reward,
                terminations=termination,
            )

            if last_sample is not None:
                last_sample.next_obs = obs
                last_sample.next_actions = a

                demo_buffer_state = buffer.add(demo_buffer_state, last_sample)

            last_sample = cur_sample
            cur_sample = None

        # This maybe messes with some metrics e.g. in the critic loss but it should not affect the loss itself because we need to discard next_obs in terminating states anyway
        last_sample.next_obs = jax.tree.map(lambda x: np.zeros_like(x), last_sample.obs)
        last_sample.next_actions = a.copy()
        demo_buffer_state = buffer.add(demo_buffer_state, last_sample)

    assert (
        sum(demo_buffer_state.buffer.terminations)
        == config.pretraining.num_demonstrations
    )

    return demo_buffer_state


def load_data(config, args):
    # For normal gym envs
    print("Loading Data")
    if config.general.algorithm != "sac":
        if config.algorithm.use_vla:
            vla_data_path = os.path.join(
                config.general.root_dir_path,
                "preprocessed_datasets_with_vla_embeds",
                f"{config.environment.name}_{config.vla.vla_type}_{config.pretraining.num_demonstrations}_dataset_with_pizero_actions_double.npy",
            )
        else:
            vla_data_path = ""
        if config.algorithm.use_vla and os.path.exists(vla_data_path):
            print("WARNING: LOADING PREPROCESSED DATASET FROM ", vla_data_path)
            loaded_dataset = np.load(vla_data_path, allow_pickle=True).item()
            demo_buffer_state = dataclass_from_dict(ItemBufferState, loaded_dataset)
            demo_buffer_state.buffer = dataclass_from_dict(Sample, demo_buffer_state.buffer)
            demo_buffer_state.item = dataclass_from_dict(Sample, demo_buffer_state.item)
            demo_buffer_state.buffer.obs = dataclass_from_dict(Observation, demo_buffer_state.buffer.obs)
            demo_buffer_state.buffer.next_obs = dataclass_from_dict(Observation, demo_buffer_state.buffer.next_obs)

            indeces_of_termination = np.where(
                np.diff(demo_buffer_state.buffer.terminations, append=0) == -1
            )[0]
            samples_for_demos_index = (
                indeces_of_termination[
                    min(
                        config.pretraining.num_demonstrations - 1,
                        indeces_of_termination.shape[0],
                    )
                ]
                + 1
            )
            print("Num Samples:", samples_for_demos_index)
            demo_buffer_state.buffer = jax.tree.map(
                lambda x: x[:samples_for_demos_index], demo_buffer_state.buffer
            )
            demo_buffer_state.n_items = jax.tree.leaves(demo_buffer_state.buffer)[0].shape[0] - 1
            demo_buffer_state.index = jax.tree.leaves(demo_buffer_state.buffer)[0].shape[0] - 1
            assert demo_buffer_state.n_items == samples_for_demos_index - 1
            assert demo_buffer_state.index == samples_for_demos_index - 1
        else:
            demo_buffer_state = load_hdf5_data(config)

            if config.algorithm.use_vla and not args.debug:
                if not os.path.exists(
                    os.path.join(
                        config.general.root_dir_path,
                        "preprocessed_datasets_with_vla_embeds",
                    )
                ):
                    os.mkdir(
                        os.path.join(
                            config.general.root_dir_path,
                            "preprocessed_datasets_with_vla_embeds",
                        )
                    )
                np.save(vla_data_path, asdict(demo_buffer_state), allow_pickle=True)

        action_dataset_statistics = calculate_action_statistics(demo_buffer_state)
        demo_buffer_state.buffer.actions = config.algorithm.action_normalization(
            demo_buffer_state.buffer.actions, *action_dataset_statistics
        )
        demo_buffer_state.buffer.next_actions = config.algorithm.action_normalization(
            demo_buffer_state.buffer.next_actions, *action_dataset_statistics
        )
        
        # from csil.utils import visualize_action_dims
        # visualize_action_dims(demo_buffer_state.buffer.obs.vla_action)
        # visualize_action_dims(demo_buffer_state.buffer.next_obs.vla_action)

        if config.algorithm.use_vla:
            demo_buffer_state.buffer.obs.vla_action = config.algorithm.action_normalization(
                demo_buffer_state.buffer.obs.vla_action, *action_dataset_statistics
            )
            demo_buffer_state.buffer.next_obs.vla_action = config.algorithm.action_normalization(
                demo_buffer_state.buffer.next_obs.vla_action, *action_dataset_statistics
            )

        print("ENVIRONMENT ACTION MIN ACTION", config.environment.as_low)
        print("ENVIRONMENT ACTION MAX ACTION", config.environment.as_high)
        print("ACTION DATASET_STATISTICS:", action_dataset_statistics)
        config.algorithm.action_dataset_statistics = action_dataset_statistics

        # TODO This does not work with vla_actions because they are clipped at 0.999 because of our arctanh term
        if config.algorithm.normalize_observations:
            obs_dataset_statistics = calculate_observation_statistics(demo_buffer_state)
            config.algorithm.obs_dataset_statistics = obs_dataset_statistics
            demo_buffer_state.buffer.obs = normalize_observation(
                demo_buffer_state.buffer.obs, obs_dataset_statistics
            )
            demo_buffer_state.buffer.next_obs = normalize_observation(
                demo_buffer_state.buffer.next_obs, obs_dataset_statistics
            )

        if demo_buffer_state.size < config.algorithm.batch_size:
            repeat_factor = (config.algorithm.batch_size // demo_buffer_state.size) + 1
            demo_buffer_state.buffer = jax.tree.map(
                lambda b, i: np.repeat(b, repeat_factor, axis=0),
                demo_buffer_state.buffer,
                demo_buffer_state.item,
            )

        # if (
        #     config.algorithm.use_vla
        #     and config.algorithm.use_vla_action_for_hetstat
        #     and not config.algorithm.use_linear_residual_combination
        # ):
        #     residual_actions = jnp.arctanh(
        #         jnp.clip(demo_buffer_state.buffer.actions, -0.999, 0.999)
        #     ) - jnp.arctanh(
        #         jnp.clip(demo_buffer_state.buffer.obs.vla_action, -0.999, 0.999)
        #     )
        #     residual_action_dataset_statistics = get_statistics(
        #         residual_actions
        #     )

        #     normalized_residual_actions = normalize_residual(
        #         residual_actions, *residual_action_dataset_statistics
        #     )
        #     demo_buffer_state.buffer.actions = np.tanh(
        #         unnormalize_residual(
        #             normalized_residual_actions, *residual_action_dataset_statistics
        #         )
        #         + np.arctanh(jnp.clip(demo_buffer_state.buffer.obs.vla_action, -0.999, 0.999))
        #     )

        #     next_state_residual_actions = jnp.arctanh(
        #         jnp.clip(demo_buffer_state.buffer.next_actions, -0.999, 0.999)
        #     ) - jnp.arctanh(
        #         jnp.clip(demo_buffer_state.buffer.next_obs.vla_action, -0.999, 0.999)
        #     )
        #     next_state_normalized_residual_actions = normalize_residual(
        #         next_state_residual_actions, *residual_action_dataset_statistics
        #     )
        #     demo_buffer_state.buffer.next_actions = np.tanh(
        #         unnormalize_residual(
        #             next_state_normalized_residual_actions,
        #             *residual_action_dataset_statistics,
        #         )
        #         + np.arctanh(
        #             jnp.clip(
        #                 demo_buffer_state.buffer.next_obs.vla_action, -0.999, 0.999
        #             )
        #         )
        #     )

        #     config.algorithm.residual_action_dataset_statistics = (
        #         residual_action_dataset_statistics
        #     )

        #     print(
        #         "RESIDUAL ACTION DATASET STATISTICS:",
        #         residual_action_dataset_statistics,
        #     )
        #     print("residual action max:", normalized_residual_actions.max(axis=0))
        #     print("residual action min:", normalized_residual_actions.min(axis=0))
        #     print("residual action mean:", normalized_residual_actions.mean(axis=0))
        #     print("residual action std:", normalized_residual_actions.std(axis=0))
        # else:
        #     config.algorithm.residual_action_dataset_statistics = None
        config.algorithm.residual_action_dataset_statistics = None

        return demo_buffer_state


def calculate_simulated_vla_actions(
    simulated_vla_network, simulated_vla_params, demo_buffer_state, config
):
    @jax.jit
    def get_simulated_vla_action(params, obs):
        dist = simulated_vla_network.apply(params, obs)[0]
        obs.vla_action = dist.mode().squeeze()
        return obs

    print("Calculate simulated vla action")
    demo_buffer_state.buffer.obs = get_simulated_vla_action(
        simulated_vla_params, demo_buffer_state.buffer.obs
    )
    demo_buffer_state.buffer.next_obs = get_simulated_vla_action(
        simulated_vla_params, demo_buffer_state.buffer.next_obs
    )

    demo_buffer_state.buffer.obs.vla_action = config.algorithm.action_unnormalization(
        demo_buffer_state.buffer.obs.vla_action,
        *config.algorithm.action_dataset_statistics,
    )
    demo_buffer_state.buffer.next_obs.vla_action = (
        config.algorithm.action_unnormalization(
            demo_buffer_state.buffer.next_obs.vla_action,
            *config.algorithm.action_dataset_statistics,
        )
    )

    print("Num Terminations:", demo_buffer_state.buffer.terminations.sum())
    print("action max:", config.algorithm.action_dataset_statistics[0])
    print("action min:", config.algorithm.action_dataset_statistics[1])
    print("action mean:", config.algorithm.action_dataset_statistics[2])
    print("action std:", config.algorithm.action_dataset_statistics[3])

    if config.algorithm.normalize_observations:
        print("Observation Dataset Statistics")
        for key, value in asdict(config.algorithm.obs_dataset_statistics).items():
            print(key)
            print(value)
        print("----------------------------------------")

    return demo_buffer_state


@jax.jit
def image_obs_encoder(params, obs):
    encoder = ResNetTorso()
    return jnp.squeeze(encoder.apply(params, jnp.expand_dims(obs, 0)))


def encode_images(encoder_params, demo_buffer_state, config):
    print("Encoding demonstration images")
    vmapped_obs_encoder = jax.vmap(image_obs_encoder, in_axes=(None, 0))
    config.algorithm.vmapped_obs_encoder = vmapped_obs_encoder
    demo_buffer_state.buffer.obs.embeddings = np.zeros(
        (demo_buffer_state.size, config.algorithm.embedding_dim)
    )
    demo_buffer_state.buffer.next_obs.embeddings = np.zeros(
        (demo_buffer_state.size, config.algorithm.embedding_dim)
    )

    for i in tqdm(range((demo_buffer_state.size // config.algorithm.batch_size) + 1)):
        demo_buffer_state.buffer.obs.embeddings[
            i * config.algorithm.batch_size : (i + 1) * config.algorithm.batch_size
        ] = vmapped_obs_encoder(
            encoder_params,
            demo_buffer_state.buffer.obs.image[
                i * config.algorithm.batch_size : (i + 1) * config.algorithm.batch_size
            ].reshape((-1,) + demo_buffer_state.item.obs.image.shape),
        )
        demo_buffer_state.buffer.next_obs.embeddings[
            i * config.algorithm.batch_size : (i + 1) * config.algorithm.batch_size
        ] = vmapped_obs_encoder(
            encoder_params,
            demo_buffer_state.buffer.next_obs.image[
                i * config.algorithm.batch_size : (i + 1) * config.algorithm.batch_size
            ].reshape((-1,) + demo_buffer_state.item.obs.image.shape),
        )

    demo_buffer_state.item.obs.image = None
    demo_buffer_state.item.next_obs.image = None
    demo_buffer_state.buffer.obs.image = None
    demo_buffer_state.buffer.next_obs.image = None

    config.environment.sampled_obs.embeddings = np.ones(
        (config.algorithm.embedding_dim,)
    )

    return demo_buffer_state
