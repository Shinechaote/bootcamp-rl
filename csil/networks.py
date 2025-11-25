import flax.linen as nn
import math
from typing import Any, Optional, Tuple, Callable, Sequence, Union

import jax.numpy as jnp
import jax
import numpy as np
from jax.nn import initializers
from flax.linen.normalization import _compute_stats, _normalize, _canonicalize_axes
from flax.linen.module import Module, compact, merge_param
from utils import unnormalize_residual
import tensorflow_probability.substrates.jax as tfp
import functools
import enum
from jax.lax import stop_gradient


PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any
Axes = Union[int, Sequence[int]]

tfd = tfp.distributions
tfb = tfp.bijectors


@jax.jit
def _triangle_activation(x: jnp.ndarray) -> jnp.ndarray:
    z = jnp.floor(x / jnp.pi + 0.5)
    return (x - jnp.pi * z) * (-1) ** z


@jax.jit
def triangle_activation(x: jnp.ndarray) -> jnp.ndarray:
    pdiv2sqrt2 = 1.1107207345
    return pdiv2sqrt2 * _triangle_activation(x)


@jax.jit
def periodic_relu_activation(x: jnp.ndarray) -> jnp.ndarray:
    pdiv4 = 0.785398163
    pdiv2 = 1.570796326
    return (_triangle_activation(x) + _triangle_activation(x + pdiv2)) * pdiv4


@jax.jit
def sin_cos_activation(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sin(x) + jnp.cos(x)


@jax.jit
def hard_sin(x: jnp.ndarray) -> jnp.ndarray:
    pdiv4 = 0.785398163  # π/4
    return periodic_relu_activation(x - pdiv4)


@jax.jit
def hard_cos(x: jnp.ndarray) -> jnp.ndarray:
    pdiv4 = 0.785398163  # π/4
    return periodic_relu_activation(x + pdiv4)


class TanhTransformedDistribution(tfd.TransformedDistribution):
    """Distribution followed by tanh."""

    def __init__(self, distribution, threshold=0.999, validate_args=False):
        """Initialize the distribution.

        Args:
          distribution: The distribution to transform.
          threshold: Clipping value of the action when computing the logprob.
          validate_args: Passed to super class.
        """
        super().__init__(
            distribution=distribution,
            bijector=tfp.bijectors.Tanh(),
            validate_args=validate_args,
        )
        # Computes the log of the average probability distribution outside the
        # clipping range, i.e. on the interval [-inf, -atanh(threshold)] for
        # log_prob_left and [atanh(threshold), inf] for log_prob_right.
        self._threshold = threshold
        inverse_threshold = self.bijector.inverse(threshold)
        # average(pdf) = p/epsilon
        # So log(average(pdf)) = log(p) - log(epsilon)
        log_epsilon = jnp.log(1.0 - threshold)
        # Those 2 values are differentiable w.r.t. model parameters, such that the
        # gradient is defined everywhere.
        self._log_prob_left = (
            self.distribution.log_cdf(-inverse_threshold) - log_epsilon
        )
        self._log_prob_right = (
            self.distribution.log_survival_function(inverse_threshold) - log_epsilon
        )

    def log_prob(self, event):
        # Without this clip there would be NaNs in the inner tf.where and that
        # causes issues for some reasons.
        event = jnp.clip(event, -self._threshold, self._threshold)
        # The inverse image of {threshold} is the interval [atanh(threshold), inf]
        # which has a probability of "log_prob_right" under the given distribution.
        return jnp.where(
            event <= -self._threshold,
            self._log_prob_left,
            jnp.where(
                event >= self._threshold, self._log_prob_right, super().log_prob(event)
            ),
        )

    def variance(self):
        deriv = 1.0 - self.mode() ** 2
        return self.distribution.variance() * deriv**2

    def mode(self):
        return self.bijector.forward(self.distribution.mode())

    def entropy(self, seed=None):
        # We return an estimation using a single sample of the log_det_jacobian.
        # We can still do some backpropagation with this estimate.
        return self.distribution.entropy() + self.bijector.forward_log_det_jacobian(
            self.distribution.sample(seed=seed), event_ndims=0
        )

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties


def prior_policy_log_likelihood(action_space) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """We assume a uniform hyper prior in a [-1, 1] action space."""
    # TODO action_space.shape is just temporary, dont know if this actually works like that
    num_actions = np.prod(action_space.shape, dtype=int)

    def prior_llh(x):
        return -num_actions * jnp.log(2.0)

    return jax.vmap(prior_llh)


gaussian_init = nn.initializers.normal(1.0)

class MLP(nn.Module):
    layer_sizes: tuple
    activate_final: bool = False
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.elu
    w_init: callable = nn.initializers.xavier_normal()
    layer_norm: bool = False
    batch_renorm: bool = False

    @nn.compact
    def __call__(self, x):
        layers = []
        for i, size in enumerate(self.layer_sizes[:-1]):  # Hidden layers

            layers.append(nn.Dense(size, kernel_init=self.w_init))
            if self.layer_norm and i == 0:
                layers.append(nn.LayerNorm())
                layers.append(jax.lax.tanh)
            else:
                layers.append(self.activation)  # Default activation function
        # Output layer
        layers.append(nn.Dense(self.layer_sizes[-1], kernel_init=self.w_init))
        if self.activate_final:
            layers.append(self.activation)
        model = nn.Sequential(tuple(layers))  # Use built-in Flax Sequential
        return model(x)


# Type aliases
MakeInnerOp = Callable[[], nn.Module]
NonLinearity = Callable[[jnp.ndarray], jnp.ndarray]


class ResidualBlock(nn.Module):
    """Residual block of operations, e.g. convolutional or MLP."""

    make_inner_op: MakeInnerOp
    non_linearity: NonLinearity = nn.relu
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        output = x

        # First layer in residual block
        if self.use_layer_norm:
            output = nn.LayerNorm(reduction_axes=(-3, -2, -1))(output)
        output = self.non_linearity(output)
        output = self.make_inner_op()(output)

        # Second layer in residual block
        if self.use_layer_norm:
            output = nn.LayerNorm(reduction_axes=(-3, -2, -1))(output)
        output = self.non_linearity(output)
        output = self.make_inner_op()(output)

        return x + output


class DownsamplingStrategy(enum.Enum):
    AVG_POOL = "avg_pool"
    CONV_MAX = "conv+max"  # Used in IMPALA
    LAYERNORM_RELU_CONV = "layernorm+relu+conv"  # Used in MuZero
    CONV = "conv"


def make_downsampling_layer(
    strategy: Union[str, DownsamplingStrategy], output_channels: int
) -> nn.Module:
    """Returns a module corresponding to the desired downsampling."""
    strategy = DownsamplingStrategy(strategy)

    if strategy is DownsamplingStrategy.AVG_POOL:
        return nn.avg_pool

    elif strategy is DownsamplingStrategy.CONV:
        return nn.Sequential(
            [
                nn.Conv(
                    features=output_channels,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    kernel_init=nn.initializers.truncated_normal(1e-2),
                )
            ]
        )

    elif strategy is DownsamplingStrategy.LAYERNORM_RELU_CONV:
        return nn.Sequential(
            [
                nn.LayerNorm(axis=(-3, -2, -1)),
                nn.relu,
                nn.Conv(
                    features=output_channels,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    kernel_init=nn.initializers.truncated_normal(1e-2),
                ),
            ]
        )

    elif strategy is DownsamplingStrategy.CONV_MAX:
        return nn.Sequential(
            [
                nn.Conv(features=output_channels, kernel_size=(3, 3), strides=(1, 1)),
                lambda x: nn.max_pool(
                    x, window_shape=(3, 3), strides=(2, 2), padding="SAME"
                ),
            ]
        )
    else:
        raise ValueError(f"Unrecognized downsampling strategy: {strategy}")


class ResNetTorso(nn.Module):
    """ResNetTorso for visual inputs, inspired by the IMPALA paper."""

    channels_per_group: Sequence[int] = (16, 32, 32)
    blocks_per_group: Sequence[int] = (2, 2, 2)
    downsampling_strategies: Sequence[DownsamplingStrategy] = (
        DownsamplingStrategy.CONV_MAX,
    ) * 3
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        output = inputs

        if len(self.channels_per_group) != len(self.blocks_per_group) or len(
            self.channels_per_group
        ) != len(self.downsampling_strategies):
            raise ValueError(
                f"Length mismatch: channels={self.channels_per_group}, "
                f"blocks={self.blocks_per_group}, strategies={self.downsampling_strategies}"
            )

        for i, (num_channels, num_blocks, strategy) in enumerate(
            zip(
                self.channels_per_group,
                self.blocks_per_group,
                self.downsampling_strategies,
            )
        ):
            output = make_downsampling_layer(strategy, num_channels)(output)

            for j in range(num_blocks):
                output = ResidualBlock(
                    make_inner_op=functools.partial(
                        nn.Conv, features=num_channels, kernel_size=(3, 3)
                    ),
                    use_layer_norm=self.use_layer_norm,
                )(output)

        output = output.reshape(output.shape[0], -1)
        output = MLP(
            (256, 48),
            activation=nn.elu,
            w_init=nn.initializers.orthogonal(),
            activate_final=False,
            layer_norm=True,
        )(output)
        output = nn.tanh(output)

        return output


class StationaryFeatures(nn.Module):
    """Stationary feature layer.

    Combines an MLP feature component (with bottleneck output) into a relatively
    wider feature layer that has periodic activation function. The from of the
    final weight distribution and periodic activation dictates the nature of the
    parametic stationary process.

    For more details see
    Periodic Activation Functions Induce Stationarity, Meronen et al.
    https://arxiv.org/abs/2110.13572
    """

    num_dimensions: int
    layers: Sequence[int]
    feature_dimension: int = 512
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.elu
    stationary_activation: Callable[[jnp.ndarray], jnp.ndarray] = sin_cos_activation
    stationary_init: nn.initializers.Initializer = gaussian_init

    def features(self, inputs: jnp.ndarray) -> jnp.ndarray:
        input_dimension = inputs.shape[-1]

        # While the theory says that these random weights should be fixed, it's
        # crucial in practice to let them be trained. The distribution does not
        # actually change much, so they still contribute to the stationary
        # behaviour, and letting them be trained alleviates potential underfitting.
        random_weights = self.param(
            "random_weights",
            self.stationary_init,
            (input_dimension, self.feature_dimension // 2),
        )

        log_lengthscales = self.param(
            "log_lengthscales",
            nn.initializers.constant(-5.0),
            (input_dimension),
        )

        ls = jnp.diag(jnp.exp(log_lengthscales))
        wx = inputs @ ls @ random_weights
        pdiv4 = 0.785398163  # π/4
        f = jnp.concatenate(
            (
                self.stationary_activation(wx + pdiv4),
                self.stationary_activation(wx - pdiv4),
            ),
            axis=-1,
        )
        return f / math.sqrt(self.feature_dimension)


class StationaryHeteroskedasticNormalTanhDistribution(StationaryFeatures):
    """Module that produces a stationary TanhTransformedDistribution."""

    """Initialization.

    Args:
      num_dimensions: Number of dimensions of the output distribution.
      layers: feature MLP architecture up to the feature layer.
      feature_dimension: size of feature layer.
      prior_variance: initial variance of the predictive.
      activation: Activation of MLP network.
      stationary_activation: Periodic activation of feature layer.
      stationary_init: Random initialization of last layer
      layer_norm_mlp: Use layer norm in the first MLP layer.
    """
    num_dimensions: int
    layers: Sequence[int]
    feature_dimension: int = 512
    prior_var: float = 0.75
    min_var: float = 1e-5
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.elu
    stationary_activation: Callable[[jnp.ndarray], jnp.ndarray] = sin_cos_activation
    stationary_init: nn.initializers.Initializer = gaussian_init
    faithful_distributions: bool = True
    is_residual: bool = False
    residual_action_dataset_statistics: Sequence[float] = None
    residual_action_scaling: Any = None
    use_resnet: bool = False
    use_embeddings: bool = False
    # Projecting the joint state/object state information to the same dim as the embeddings
    # Set to -1 to disable the projection
    embedding_dim: int = -1
    freeze_resnet: bool = False

    def setup(self):
        self.prior_stddev = np.sqrt(self.prior_var)

    @nn.compact
    def __call__(
        self, obs
    ) -> Union[tfd.Distribution, Tuple[tfd.Distribution, tfd.Distribution]]:
        if self.use_resnet:
            embeddings = ResNetTorso()(obs.image)
            if self.freeze_resnet:
                embeddings = stop_gradient(embeddings)
        else:
            embeddings = obs.embeddings

        if self.use_embeddings:
            state = obs.state

            inputs = jnp.concatenate([embeddings, state], axis=-1)
        else:
            inputs = obs.state

        mlp_features = MLP(
            tuple(self.layers),
            activation=self.activation,
            w_init=nn.initializers.orthogonal(),
            activate_final=True,
            layer_norm=True,
        )(inputs)

        features = self.features(mlp_features)
        if self.faithful_distributions:
            features_ = jax.lax.stop_gradient(features)
        else:
            features_ = features

        loc_weights = self.param(
            "loc_weights",
            nn.initializers.constant(0.0),
            (self.feature_dimension, self.num_dimensions),
        )

        # Parameterize the PSD matrix in lower triangular form as a 'raw' vector.
        # This minimizes the memory footprint to between 50-75% of the full matrix.
        n_sqrt = self.feature_dimension * (self.feature_dimension + 1) // 2
        scale_cross_weights_sqrt_raw = self.param(
            "scale_cross_weights_sqrt",
            nn.initializers.constant(0.0),
            (self.num_dimensions, n_sqrt),
        )
        # convert vector into a lower triagular matrix with exponentiated diagonal,
        # so a vector of zeros becomes the identity matrix.
        b = tfb.FillScaleTriL(diag_bijector=tfb.Exp(), diag_shift=None)
        scale_cross_weights_sqrt = jax.vmap(b.forward)(scale_cross_weights_sqrt_raw)
        loc = features @ loc_weights
        # Cholesky decompositon: A = LL^T where L is lower triangular
        # Variance is diagonal of x @ A @ x.T = x @ L @ L.T @ x.T
        # so first compute x @ L per output d

        var_sqrt = jnp.einsum("dij,bi->bdj", scale_cross_weights_sqrt, features_)
        var = jnp.einsum("bdi,bdi->bd", var_sqrt, var_sqrt)

        scale = math.sqrt(self.min_var) + self.prior_stddev * jnp.tanh(jnp.sqrt(var))

        # TODO Old Scale code (remember to also change max_reward for this)
        # scale = self.prior_stddev * jnp.tanh(jnp.sqrt(self.min_var + var))

        if self.is_residual:
            if self.residual_action_dataset_statistics is not None:
                loc = unnormalize_residual(
                    loc, *self.residual_action_dataset_statistics
                )

            if self.residual_action_scaling is not None:
                loc = loc * self.residual_action_scaling

            clipped_normalized_base_action = jnp.clip(obs.vla_action, -0.999, 0.999)
            loc = loc + jnp.arctanh(clipped_normalized_base_action)
            # loc = loc / 2.0

        distribution = tfd.Normal(loc=loc, scale=scale)

        transformed_distribution = tfd.Independent(
            TanhTransformedDistribution(distribution),
            reinterpreted_batch_ndims=1,
        )

        if self.faithful_distributions:
            cut_distribution = tfd.Normal(loc=jax.lax.stop_gradient(loc), scale=scale)
            cut_transformed_distribution = tfd.Independent(
                TanhTransformedDistribution(cut_distribution),
                reinterpreted_batch_ndims=1,
            )
            return transformed_distribution, cut_transformed_distribution, mlp_features
        else:
            return transformed_distribution, transformed_distribution, mlp_features


class CriticMLP(nn.Module):
    layers: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.elu
    w_init: nn.initializers.Initializer = nn.initializers.orthogonal()
    use_resnet: bool = False
    use_embeddings: bool = False
    embedding_dim: int = -1
    freeze_resnet: bool = False

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        if self.use_resnet:
            embeddings = ResNetTorso()(obs.image)
            if self.freeze_resnet:
                embeddings = stop_gradient(embeddings)
        else:
            embeddings = obs.embeddings

        if self.use_embeddings:
            state = obs.state

            obs_encoded = jnp.concatenate([embeddings, state], axis=-1)
        else:
            obs_encoded = obs.state

        network = MLP(
            self.layers, activation=self.activation, w_init=self.w_init, layer_norm=True
        )
        input_ = jnp.concatenate([obs_encoded, action], axis=-1)
        value = network(input_)
        return jnp.expand_dims(value, 0)


class DoubleMLP(nn.Module):
    layers: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.elu
    w_init: nn.initializers.Initializer = nn.initializers.orthogonal()
    use_resnet: bool = False
    use_embeddings: bool = False
    embedding_dim: int = -1

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        if self.use_resnet:
            embeddings = ResNetTorso()(obs.image)
        else:
            embeddings = obs.embeddings

        if self.use_embeddings:
            if self.embedding_dim != -1:
                pass
            else:
                state = obs.state

            obs_encoded = jnp.concatenate([embeddings, state], axis=-1)
        else:
            obs_encoded = obs.state

        network1 = MLP(
            self.layers, activation=self.activation, w_init=self.w_init, layer_norm=True
        )
        network2 = MLP(
            self.layers, activation=self.activation, w_init=self.w_init, layer_norm=True
        )
        input_ = jnp.concatenate([obs_encoded, action], axis=-1)
        value1 = network1(input_)
        value2 = network2(input_)
        return jnp.concatenate([value1, value2], axis=-1)


class EntropyCoefficient(nn.Module):
    init_ent_coef: float

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_alpha = self.param(
            "log_alpha", init_fn=lambda key: jnp.full((), jnp.log(self.init_ent_coef))
        )
        return jnp.exp(log_alpha)


class ConstantEntropyCoefficient(nn.Module):
    init_ent_coef: float

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        self.param("dummy_param", init_fn=lambda key: jnp.full((), self.init_ent_coef))
        return self.init_ent_coef
