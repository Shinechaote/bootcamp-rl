import numpy as np
import tensorflow_probability.substrates.jax as tfp
from datetime import datetime

from csil import get_args_parsed, get_config, load_data
from regression import train_model
from networks import StationaryHeteroskedasticNormalTanhDistribution
from utils import disable_tf_from_using_gpus
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp


tfd = tfp.distributions


np.set_printoptions(precision=3, suppress=True)
args = get_args_parsed()

cur_date = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
config = get_config(args, cur_date)

disable_tf_from_using_gpus()

demonstrations, action_dataset_statistics, residual_action_dataset_statistics = (
    load_data(config, args)
)
(
    expert_s,
    expert_a,
    expert_next_s,
    expert_next_a,
    expert_rewards,
    expert_terminations,
    expert_discounts,
    expert_vla_a,
    expert_next_state_vla_a,
    _,
    _,
) = demonstrations

if config.algorithm.use_vla:
    checkpoint_dir = "/home/scherer/openpi/csil/vla_square/checkpoints/"
    replay_buffer_path = "vla_square/replay_buffer.npy"
else:
    checkpoint_dir = "/home/scherer/openpi/csil/state-only-square/checkpoints/"
    replay_buffer_path = "state-only-square/replay_buffer.npy"

online_demos = np.load(replay_buffer_path, allow_pickle=True).item()[
    "online_replay_buffer"
]
online_s = online_demos["states"][:, 0, :]
online_vla_a = online_demos["vla_actions"][:, 0, :]
online_ends_of_episodes = online_demos["ends_of_episode"]


# Run of the replay buffer: https://wandb.ai/robot-learning-rt2/csil-for-vlas/runs/pzp4o6kp/workspace?nw=nwuserchrfesche
sampled_action = np.expand_dims(np.squeeze(config.environment.sampled_action), 0)
nr_hidden_units = config.algorithm.nr_hidden_units
bottleneck_size = config.algorithm.bottleneck_size

# Define the checkpoint manager
options = ocp.CheckpointManagerOptions()
manager = ocp.CheckpointManager(directory=checkpoint_dir, options=options)

print(f"Reloading checkpoint from {checkpoint_dir}")
policy_params = manager.restore(step=1)["params"]

policy_network = StationaryHeteroskedasticNormalTanhDistribution(
    sampled_action.shape[1],
    layers=[nr_hidden_units, nr_hidden_units, bottleneck_size],
    feature_dimension=nr_hidden_units,
    layer_norm_mlp=config.algorithm.layer_norm_policy,
    prior_var=0.75,
    activation=jax.nn.elu,
    min_var=config.algorithm.min_var,
    stationary_activation=config.algorithm.stationary_activation_function,
    is_residual=config.algorithm.use_vla_action_for_hetstat
    and not config.algorithm.use_linear_residual_combination,
    residual_action_dataset_statistics=residual_action_dataset_statistics,
    residual_action_scaling=config.algorithm.residual_action_scaling,
)

sampled_obs = np.expand_dims(np.squeeze(config.environment.sampled_obs), 0)
sampled_action = np.expand_dims(np.squeeze(config.environment.sampled_action), 0)

print("Obs shape:", sampled_obs.shape, "Action shape", sampled_action.shape)

# policy_params = policy_network.init(
#     jax.random.PRNGKey(0), sampled_obs, sampled_action)

# s, a, next_s, next_a, rewards, terminations, discounts, vla_a, next_state_vla_a = demonstrations

# env = config.environment.env_creation_fn(config)

# policy_params, _, _ = train_model(
#     policy_network, policy_params, s, a, vla_a, env, config,
#     action_dataset_statistics, n_iters=config.pretraining.policy_pretrain_steps,
#     learning_rate=config.pretraining.policy_learning_rate, batch_size=config.algorithm.batch_size,
#     target_entropy_factor=config.pretraining.target_entropy_factor, seed=config.general.seed, return_data_for_replay_buffer=False, stochastic_eval=True
#     )


@jax.jit
def get_embeds(policy_params, obs, vla_actions):
    dist, cut_dist, embeds = policy_network.apply(
        policy_params, jnp.expand_dims(obs, 0), jnp.expand_dims(vla_actions, 0)
    )
    return embeds


vm_get_embeds = jax.vmap(get_embeds, in_axes=(None, 0, 0))

online_embeds = vm_get_embeds(policy_params, online_s, online_vla_a)
expert_embeds = vm_get_embeds(policy_params, expert_s, expert_vla_a)

expert_start_indeces = np.concatenate([[0], np.where(expert_terminations)[0] + 1])
online_start_indeces = np.concatenate([[0], np.where(online_ends_of_episodes)[0] + 1])

print("Num online episodes:", online_start_indeces.shape)

expert_to_expert = []
expert_to_online = []
online_to_expert = []
online_to_online = []

ls = jnp.diag(jnp.exp(policy_params["params"]["log_lengthscales"]))

for i in tqdm(range(2000)):
    expert_index = np.random.randint(0, expert_embeds.shape[0])
    online_index = np.random.randint(0, online_embeds.shape[0])

    expert_sample = np.expand_dims(expert_embeds[expert_index], 0)
    online_sample = np.expand_dims(online_embeds[online_index], 0)

    expert_to_expert_diffs = np.delete(expert_embeds, expert_index, axis=0) - np.repeat(
        expert_sample, expert_embeds.shape[0] - 1, axis=0
    )
    expert_to_online_diffs = np.delete(expert_embeds, expert_index, axis=0) - np.repeat(
        online_sample, expert_embeds.shape[0] - 1, axis=0
    )
    online_to_expert_diffs = np.delete(online_embeds, online_index, axis=0) - np.repeat(
        expert_sample, online_embeds.shape[0] - 1, axis=0
    )
    online_to_online_diffs = np.delete(online_embeds, online_index, axis=0) - np.repeat(
        online_sample, online_embeds.shape[0] - 1, axis=0
    )

    expert_to_expert.append(np.min(np.linalg.norm(expert_to_expert_diffs, axis=-1)))
    expert_to_online.append(np.min(np.linalg.norm(expert_to_online_diffs, axis=-1)))
    online_to_expert.append(np.min(np.linalg.norm(online_to_expert_diffs, axis=-1)))
    online_to_online.append(np.min(np.linalg.norm(online_to_online_diffs, axis=-1)))

print(expert_to_expert[0].shape)

if config.algorithm.use_vla:
    print("Comparing VLA Values")
else:
    print("Comparing state-only Values")
print("Expert to expert:", np.array(expert_to_expert).mean())
print("Expert to online:", np.array(expert_to_online).mean())
print("online to expert:", np.array(online_to_expert).mean())
print("online to online:", np.array(online_to_online).mean())
