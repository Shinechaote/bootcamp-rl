import numpy as np
import tensorflow_probability.substrates.jax as tfp
import matplotlib.pyplot as plt
import math
import jax.numpy as jnp
from networks import TanhTransformedDistribution
import jax
from tqdm import tqdm

tfd = tfp.distributions

num_actions = 7
num_samples = 10000

min_var = 1e-5
max_reward = -0.5 * np.log(min_var) + np.log(2.0)

# max_reward = np.log(np.tanh(1/np.sqrt(2 * np.pi * (min_var **2)))) + np.log(2.0)
max_reward = -0.5 * np.log((np.pi * min_var**2) / 2) + -np.log(1 - (0.999**2))

prior_var = 0.75
prior_stddev = np.sqrt(prior_var)


@jax.jit
def sampler(distribution, key):
    return distribution.log_prob(distribution.sample(seed=key))


fig, axes = plt.subplots(3, 3, figsize=(12, 4))  # (rows, columns)
for i, mean in enumerate([-1, 0, 1]):
    for j, var in enumerate([-1, min_var, 1]):
        loc = np.full((1, num_actions), mean, dtype=np.float32)
        var_arr = np.full((1, num_actions), var, dtype=np.float32)
        scale = math.sqrt(min_var) + prior_stddev * jnp.tanh(jnp.sqrt(var_arr))

        loc = [
            -1.6535828,
            -2.2499075,
            4.011861,
            -3.8217757,
            -3.8217757,
            -3.8217757,
            4.2806606,
        ]
        scale = [
            0.01541348,
            0.01209793,
            0.86414397,
            0.86918724,
            0.86918724,
            0.86918724,
            0.8691751,
        ]
        action = [[-0.93077747, -0.97782595, 0.99999985, -1.0, -1.0, -1.0, 0.99999998]]

        distribution = tfd.Normal(loc=loc, scale=scale)

        transformed_distribution = tfd.Independent(
            TanhTransformedDistribution(distribution),
            reinterpreted_batch_ndims=1,
        )
        # print(transformed_distribution.distribution._type_spec)
        # for name, value in transformed_distribution.distribution.distribution.parameters.items():
        #     print(f"{name}: {value}")

        # log_probs = []
        # key = jax.random.PRNGKey(0)
        # for i in tqdm(range(num_samples)):
        #     log_probs.append(sampler(transformed_distribution, key))
        #     key, _ = jax.random.split(key)
        # print(f"Mean: {mean} Var: {var} Rewards: {np.max(log_probs)}")

        # action = np.repeat(np.expand_dims(np.array(transformed_distribution.mode().squeeze()), 0), num_samples, axis=0)
        # action = np.full((num_samples, num_actions), mean, dtype=np.float32)
        # action[:, 0] = np.linspace(mean - var, mean + var, num_samples)
        action = np.repeat(
            np.array(
                [[-0.93077747, -0.97782595, 0.99999985, -1.0, -1.0, -1.0, 0.99999998]]
            ),
            num_samples,
            axis=0,
        )
        action[:, 0] = np.linspace(-1, 1, num_samples)

        policy_log_prob = transformed_distribution.log_prob(action)
        prior_log_prob = -num_actions * jnp.log(2.0)

        log_ratio = policy_log_prob - prior_log_prob
        rewards = (1 - 0.99) / num_actions * (log_ratio - num_actions * max_reward)
        print(f"Mean: {mean} Var: {var} Rewards: {np.max(rewards)}")

        # axes[i, j].plot(np.linspace(-1, 1, num_samples), rewards)
        # axes[i, j].plot(action[:, 0], rewards)
        plt.plot(action[:, 0], rewards)
        plt.ylim([0, 0.005])
        # axes[i, j].set_title(f"Mean: {mean} Var: {var}")
        # axes[i, j].ticklabel_format(style="plain")

        plt.show()
