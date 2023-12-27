"""Flax Maximal Update Parametrization (muP)

Reference: https://arxiv.org/abs/2203.03466

To train with μP there are two steps:
    1. Create a Flax module that subclasses `mup.Module`.
    2. Chain the μP Optax optimizer with your current optimizer.

For example,

```py
class Model(mup.Module):

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Dense(128 * self.width)(x)
        x = nn.relu(x)
        x = nn.Dense(128 * self.width)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

# Create your model specifying the target width with the width field.
model = Model(width=16)

# Initialize the model and optionally specify the base model width.
# It defaults to 1 but you can specify any integer value less than the
# target width above.
params = model.init(
    jax.random.PRNGKey(0),
    jnp.zeros((16)),
    base_width=1, # `base_width` defaults to 1
)

optim = optax.chain(
    optax.adam(learning_rate=1e-4),
    mup.scale_adam_by_mup(),
)
```
"""

import functools
import typing
from typing import Any, Callable, Mapping, Self, TypeVar

import chex
import jax
import optax
from flax import core, struct
from flax import linen as nn
from flax.linen import kw_only_dataclasses

T = TypeVar("T")


class MaximalUpdateParametrizationMetadata(
    struct.PyTreeNode, nn.meta.AxisMetadata[nn.meta.AxisMetadata[chex.Array] | chex.Array]
):
    """Maximal Update Parametrization axis metadata."""

    # The boxed value
    value: nn.meta.AxisMetadata[chex.Array] | chex.Array

    # The μP dimensions
    dims: tuple[float | None, ...] = struct.field(pytree_node=False)

    def unbox(self) -> nn.meta.AxisMetadata[chex.Array] | chex.Array:
        """Unbox the parameter or nested metadata."""
        return self.value

    def replace_boxed(self, value: nn.meta.AxisMetadata[chex.Array] | chex.Array) -> Self:
        """Replace the boxed value."""
        return self.replace(value=value)

    def add_axis(self, index: int, params: dict[Any, Any]) -> Self:
        """Add an axis, this is called through lifted transforms and will always be finite dimensional."""
        del params
        dims = list(self.dims)
        dims.insert(index, None)
        return self.replace(dims=tuple(dims))

    def remove_axis(self, index: int, params: dict[Any, Any]) -> Self:
        """Remove an axis, this is called through lifted transforms and will always be finite dimensional."""
        del params
        dims = list(self.dims)
        dims.pop(index)
        return self.replace(dims=tuple(dims))

    @property
    def ndims(self) -> int:
        """Number of infinite dimensions."""
        return len(list(filter(None, self.dims)))

    @property
    def width(self) -> float:
        """Width of the final transformation."""
        assert self.ndims <= 2, "Only supports a maximum of two infinite dimensions"
        return next(filter(None, reversed(self.dims)), 1.0)

    @property
    def is_vector_like(self) -> bool:
        """A parameter is vector-like if it has one infinite dimension."""
        return self.ndims == 1

    @property
    def is_matrix_like(self) -> bool:
        """A parameter is matrix-like if it has two infinite dimensions."""
        return self.ndims == 2

    @property
    def is_scalar_like(self) -> bool:
        """A parameter is scalar-like if it has zero infinite dimensions."""
        return self.ndims == 0

    @property
    def is_input_weight(self) -> bool:
        """A weight that maps from a finite dimension to an infinite dimension."""
        return self.is_vector_like and self.dims[-1] is not None

    @property
    def is_output_weight(self) -> bool:
        """A weight that maps from an infinite dimension to a finite dimension."""
        return self.is_vector_like and self.dims[-1] is None

    @property
    def is_hidden_weight(self) -> bool:
        """A weight that maps from an infinite dimension to an infinite dimension."""
        return self.is_matrix_like


tree_map_mupped = functools.partial(
    jax.tree_util.tree_map,
    is_leaf=lambda leaf: isinstance(leaf, MaximalUpdateParametrizationMetadata),
)


@typing.dataclass_transform(kw_only_default=True)
class Module(nn.Module):
    """Flax Maximal Update Parametrization (μP) module."""

    # Target width of the model.
    width: int = kw_only_dataclasses.field(default=1, kw_only=True)

    # Whether or not to allow the μP transform on this module
    allow_mup: bool = kw_only_dataclasses.field(default=True, kw_only=True)

    def init_with_output(
        self,
        *args,
        base_width: int = 1,
        allow_mup: bool = True,
        **kwargs,
    ) -> tuple[Any, core.FrozenDict[str, Mapping[str, Any]] | dict[str, Mapping[str, Any]]]:
        """Initialize the model with an optional `base_width`.

        The only argument that differs from `nn.Module.init_with_output` is `base_width`.
        If `base_width` != `self.width` then we'll perform initialization with the
        maximal update parametrization.

        Specifically, this follows Table 3 in Yang et al. 2022. What this means for Flax
        is as follows:
            1. We wrap each parameter with axis metadata specifying the implied
                infinite dimensional shape. See `MaximalUpdateParametrizationMetadata`.
            2. We rescale variable initialization for output weights in the network.
                Output weights are determined to be those that map from an infinite
                dimensional space to a finite dimensional space.

        To fully obtain a proper μP training loop you must also chain your optimizer
        with either `scale_adam_by_mup` or `scale_sgd_by_mup`.

        For other args see `flax.linen.module.Module.init_with_output`.

        Args:
            base_width: int, the base width of the network, defaults to 1.
            allow_mup: bool, should we use μP, defaults to True.

        Returns:
            tuple with the output and the initialized parameters.
        """

        if self.width == base_width or not (self.allow_mup and allow_mup):
            return super().init_with_output(*args, **kwargs)

        base_model = self.copy(parent=self.parent, width=base_width)

        # Evaluate the shape of the base model
        _, base_shapes = jax.eval_shape(
            functools.partial(super(Module, base_model).init_with_output, **kwargs), *args
        )
        # Evaluate the shape of the target model
        _, target_shapes = jax.eval_shape(
            functools.partial(super().init_with_output, **kwargs), *args
        )

        # Initialize the parameters of the target model.
        outputs, target_params = super().init_with_output(*args, **kwargs)

        # Now we'll wrap each parameter with the μP axis metadata
        def _wrap_and_maybe_rescale_param_with_mup(
            param: chex.Array,
            base_shape: jax.ShapeDtypeStruct,
            target_shape: jax.ShapeDtypeStruct,
        ) -> MaximalUpdateParametrizationMetadata:
            dims = ()
            for base, target in zip(base_shape.shape, target_shape.shape):
                dims += (target / base,) if target != base else (None,)
            mupped = MaximalUpdateParametrizationMetadata(value=param, dims=dims)

            # According to Table 3 we'll only rescale output weights to achieve
            # a variance of 1 / fan_in^2. We can simply multiply the weights by
            # 1 / sqrt(fan_in) to achieve this assuming we sample from a
            # scale invariant distribution which is typical in practice.
            if mupped.is_output_weight:
                mupped = mupped.replace_boxed(mupped.value / (mupped.width**0.5))
            return mupped

        return outputs, jax.tree_util.tree_map(
            _wrap_and_maybe_rescale_param_with_mup,
            target_params,
            base_shapes,
            target_shapes,
        )


def _scale_by_mup(
    func: Callable[[MaximalUpdateParametrizationMetadata], MaximalUpdateParametrizationMetadata],
) -> optax.GradientTransformation:
    def _maybe_scale_mupped_updates(
        maybe_mupped: MaximalUpdateParametrizationMetadata | chex.Array,
    ) -> MaximalUpdateParametrizationMetadata | chex.Array:
        if not isinstance(maybe_mupped, MaximalUpdateParametrizationMetadata):
            return maybe_mupped
        return func(maybe_mupped)

    def init_fn(params: optax.Params) -> optax.EmptyState:
        del params
        return optax.EmptyState()

    def update_fn(
        updates: optax.Updates,
        state: optax.OptState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, optax.OptState]:
        del params
        updates = tree_map_mupped(_maybe_scale_mupped_updates, updates)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def scale_adam_by_mup() -> optax.GradientTransformation:
    """Scale the gradient update of Adam-like optimizers according to μP."""

    def _scale_adam_like(
        mupped: MaximalUpdateParametrizationMetadata,
    ) -> MaximalUpdateParametrizationMetadata:
        if mupped.is_output_weight or mupped.is_hidden_weight:
            mupped = nn.meta.replace_boxed(mupped, nn.meta.unbox(mupped) / mupped.width)
        return mupped

    return _scale_by_mup(_scale_adam_like)


def scale_sgd_by_mup() -> optax.GradientTransformation:
    """Scale the gradient update of SGD-like optimizers according to μP."""

    def _scale_sgd_like(
        mupped: MaximalUpdateParametrizationMetadata,
    ) -> MaximalUpdateParametrizationMetadata:
        if mupped.is_vector_like and not mupped.is_output_weight:
            mupped = nn.meta.replace_boxed(mupped, nn.meta.unbox(mupped) * mupped.width)
        elif mupped.is_output_weight:
            mupped = nn.meta.replace_boxed(mupped, nn.meta.unbox(mupped) / mupped.width)
        return mupped

    return _scale_by_mup(_scale_sgd_like)
