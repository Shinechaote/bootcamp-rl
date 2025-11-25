import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from dataclasses import dataclass
import dataclasses
from typing import Sequence, Any
from jax import Array
from environment_wrappers import ObservationStats

Shape = Sequence[int]


@jax.jit
def normalize_residual(action, max_action, min_action, mean_action, action_std):
    return jnp.clip(
        normalize_for_unit_gaussian(
            action, max_action, min_action, mean_action, action_std
        ),
        -3,
        3,
    )
    # return normalize_for_unit_gaussian(action, max_action, min_action, mean_action, action_std), -3, 3)


@jax.jit
def unnormalize_residual(action, max_action, min_action, mean_action, action_std):
    return unnormalize_for_unit_gaussian(
        action, max_action, min_action, mean_action, action_std
    )


@jax.jit
def normalize(action, max_action, min_action, mean_action, action_std):
    pos_and_rot = jnp.clip(
        normalize_for_unit_gaussian(
            action[..., :-1],
            max_action[:-1],
            min_action[:-1],
            mean_action[:-1],
            action_std[:-1],
        )
        / 3,
        -1,
        1,
    )
    gripper = linear_normalize(
        action[..., -1:],
        max_action[-1:],
        min_action[-1:],
        mean_action[-1:],
        action_std[-1:],
    )
    return jnp.concatenate([pos_and_rot, gripper], axis=-1)


@jax.jit
def unnormalize(action, max_action, min_action, mean_action, action_std):
    pos_and_rot = unnormalize_for_unit_gaussian(
        action[..., :-1] * 3,
        max_action[:-1],
        min_action[:-1],
        mean_action[:-1],
        action_std[:-1],
    )
    gripper = linear_unnormalize(
        action[..., -1:],
        max_action[-1:],
        min_action[-1:],
        mean_action[-1:],
        action_std[-1:],
    )
    return jnp.concatenate([pos_and_rot, gripper], axis=-1)


@jax.jit
def normalize_for_unit_gaussian(
    action, max_action, min_action, mean_action, action_std
):
    # Normalize for Mean
    return (action - mean_action) / (action_std + 1e-8)


@jax.jit
def unnormalize_for_unit_gaussian(
    action, max_action, min_action, mean_action, action_std
):
    # Unnormalize from normalization for mean to be 0 and std to be 1
    return action * (action_std + 1e-8) + mean_action


@jax.jit
def linear_normalize(action, max_action, min_action, mean_action, action_std):
    return 2 * (action - min_action) / (max_action - min_action + 1e-8) - 1


@jax.jit
def linear_unnormalize(action, max_action, min_action, mean_action, action_std):
    return (action + 1) * 0.5 * (max_action - min_action + 1e-8) + min_action


def dataclass_from_dict(klass, d):
    if isinstance(d, list):
        (inner,) = klass.__args__
        return [dataclass_from_dict(inner, i) for i in d]

    try:
        fieldtypes = {f.name: f.type for f in dataclasses.fields(klass)}
        return klass(**{f: dataclass_from_dict(fieldtypes[f], d[f]) for f in d})
    except:
        return d  # Not a dataclass field


def concatenate_samples(s1, s2):
    return jax.tree.map(lambda x1, x2: jnp.concatenate([x1, x2], axis=0), s1, s2)


def visualize_action_dims(data):
    # Create subplots: 7 columns, arranged nicely
    fig, axes = plt.subplots(1, 7, figsize=(20, 3), sharey=True)

    # Loop over the 7 columns
    for i in range(7):
        axes[i].hist(data[:, i], bins=30, color="skyblue", edgecolor="black")
        axes[i].set_title(f"Column {i+1}")

    plt.tight_layout()
    plt.show()


def resize_image(image: np.ndarray) -> np.ndarray:
    image = tf.image.resize(
        image,
        size=(256, 256),
        method="lanczos3",
        antialias=True,
    )
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
    return image


@jax.tree_util.register_dataclass
@dataclass
class ItemBufferState:
    item: Any
    index: Array | int
    n_items: Array | int
    buffer: Any

    @property
    def size(self):
        return jax.tree.leaves(self.buffer)[0].shape[0]


class ItemBuffer:
    def init(self, size: int, item: Any) -> ItemBufferState:
        buffer = jax.tree.map(
            lambda i: np.zeros((size, i.ravel().shape[0]), dtype=i.dtype), item
        )
        idx = 0
        n_items = 0
        return ItemBufferState(item=item, index=idx, n_items=n_items, buffer=buffer)

    def update_array(self, buffer, idx, item):
        buffer[idx] = item
        return buffer

    def add(self, buffer_state, item):
        idx = buffer_state.index
        item_flat = jax.tree.map(lambda i: i.ravel(), item)
        buffer = jax.tree.map(
            lambda b, i_flat: self.update_array(b, idx, i_flat),
            buffer_state.buffer,
            item_flat,
        )
        idx = (idx + 1) % buffer_state.size
        n_items = jnp.minimum(buffer_state.n_items + 1, buffer_state.size)
        return ItemBufferState(
            item=buffer_state.item,
            index=idx,
            n_items=n_items,
            buffer=buffer,
        )

    def add_batch(self, buffer_state, items):
        buffer_state, _ = jax.lax.scan(
            lambda bs, item: (self.add(bs, item), None), buffer_state, items
        )
        return buffer_state

    def sample(self, buffer_state, key, shape: Shape = ()):
        idxs = jax.random.randint(key, shape, 0, buffer_state.n_items)
        batch = jax.tree.map(
            lambda b, i: np.reshape(b[idxs], shape + i.shape),
            buffer_state.buffer,
            buffer_state.item,
        )
        return batch

    def get_entire_buffer(self, buffer_state):
        return self.sample(
            buffer_state, jax.random.PRNGKey(0), shape=(buffer_state.size,)
        )

    def can_sample(self, buffer_state, batch_size):
        return buffer_state.n_items >= batch_size


def get_statistics(arr):
    max_ds = arr.max(axis=0)
    min_ds = arr.min(axis=0)
    mean_ds = arr.mean(axis=0)
    ds_std = arr.std(axis=0)
    dataset_statistics = (max_ds, min_ds, mean_ds, ds_std)
    return dataset_statistics


def calculate_action_statistics(replay_buffer_state):
    action = replay_buffer_state.buffer.actions
    return get_statistics(action)


def calculate_observation_statistics(replay_buffer_state):
    obs = ItemBuffer().get_entire_buffer(replay_buffer_state).obs

    statistics = ObservationStats()
    statistics = jax.tree.map(
        lambda batch: None if batch is None else get_statistics(batch), obs
    )

    return statistics


def normalize_observation(obs, statistics):
    obs = jax.tree.map(
        lambda field, stats: (
            None if field is None else normalize_for_unit_gaussian(field, *stats)
        ),
        obs,
        statistics,
    )
    return obs


@jax.jit
def expand_obs_dim(obs):
    obs = jax.tree.map(
        lambda field: None if field is None else jnp.expand_dims(field, 0), obs
    )
    return obs


def disable_tf_from_using_gpus():
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], "GPU")
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != "GPU"
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
