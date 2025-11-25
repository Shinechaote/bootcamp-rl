from dataclasses import dataclass, asdict
from functools import partial
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

Shape = Sequence[int]


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

    def can_sample(self, buffer_state, batch_size):
        return buffer_state.n_items >= batch_size


@jax.tree_util.register_dataclass
@dataclass
class Sample:
    obs: jnp.ndarray
    actions: jnp.ndarray
    next_obs: jnp.ndarray
    next_actions: jnp.ndarray
    rewards: jnp.ndarray
    terminations: jnp.ndarray


replay_buffer = ItemBuffer()

obs_shape = (23,)
action_shape = (7,)

sample_item = Sample(
    np.zeros(obs_shape),
    np.zeros(action_shape),
    np.zeros(obs_shape),
    np.zeros(action_shape),
    np.ones((1,)),
    np.ones((1,)),
)

state = replay_buffer.init(10, sample_item)

for i in range(10):
    sample_item = Sample(
        np.ones(obs_shape),
        np.ones(action_shape),
        np.random.uniform(obs_shape),
        np.random.uniform(action_shape),
        np.random.uniform((1,)),
        np.random.uniform((1,)),
    )
    state = replay_buffer.add(state, sample_item)


repeat_factor = 2
state.buffer = jax.tree.map(
    lambda b, i: np.repeat(b, repeat_factor, axis=0),
    state.buffer,
    state.item,
)

print(state.buffer.obs.shape)
