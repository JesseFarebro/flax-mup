"""μP coordinate check."""

import argparse
import functools
import itertools
import pathlib
from typing import Iterator, Literal, TypedDict

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import seaborn.objects as so
import tqdm
from flax import linen as nn
from flax.training import train_state

import flax_mup as mup

LossType = Literal["mse", "ce"]


class ResultRecord(TypedDict):
    width: int
    mup: bool
    layer: int
    coord: float
    iteration: int
    seed: int


class TrainingBatch(TypedDict):
    images: chex.Array
    labels: chex.Array


def make_dataset(batch_size: int = 32) -> Iterator[TrainingBatch]:
    rng = np.random.default_rng()

    while True:
        images = rng.uniform(size=(batch_size, 28, 28, 1))
        labels = rng.integers(low=0, high=9, size=(batch_size, 1))
        yield TrainingBatch(images=images, labels=labels)


@jax.jit
@functools.partial(jax.vmap, in_axes=(None, 0))
def get_coordinates(state: train_state.TrainState, image: chex.Array):
    _, intermediates = state.apply_fn(
        state.params,
        image,
        capture_intermediates=True,
        mutable=["intermediates"],
    )
    intermediates = jax.tree_util.tree_map(
        lambda tensor: jnp.mean(jnp.abs(tensor.ravel())), intermediates
    )
    return jax.tree_util.tree_leaves(intermediates)


@functools.partial(jax.jit, static_argnames=("loss_type",), donate_argnames=("state",))
def train_step(
    state: train_state.TrainState, batch: TrainingBatch, *, loss_type: LossType
) -> train_state.TrainState:
    @jax.grad
    def grad_fn(params: optax.Params) -> chex.Array:
        outputs = jax.vmap(state.apply_fn, in_axes=(None, 0))(params, batch["images"])
        match loss_type:
            case "mse":
                return jnp.mean(optax.l2_loss(outputs, batch["labels"]))
            case "ce":
                return jnp.mean(
                    optax.softmax_cross_entropy_with_integer_labels(
                        outputs, jnp.squeeze(batch["labels"], axis=1)
                    )
                )

    grads = grad_fn(state.params)
    return state.apply_gradients(grads=grads)


def coordinate_check(
    model: nn.Module,
    *,
    width_multiplier: int = 1,
    with_mup: bool = False,
    seeds: int = 5,
    iterations: int = 3,
    loss_type: LossType = "ce",
) -> list[ResultRecord]:
    results = []
    dataset_iter = make_dataset()
    dummy_input = next(dataset_iter)["images"][0, ...]  # type: ignore
    model = model.copy(width=width_multiplier)

    for seed in range(seeds):
        key = jax.random.PRNGKey(seed)

        params = model.init(key, dummy_input, allow_mup=with_mup)
        optim = optax.chain(optax.adam(learning_rate=1e-2), mup.scale_adam_by_mup())
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optim,
        )
        batch = next(dataset_iter)

        for it in range(1, iterations + 1):
            coords = get_coordinates(state, batch["images"])  # type: ignore

            for layer, coord in enumerate(coords):
                results.append(
                    ResultRecord(
                        width=width_multiplier,
                        mup=with_mup,
                        layer=layer,
                        coord=jnp.mean(coord).item(),
                        iteration=it,
                        seed=seed,
                    )
                )

            state = train_step(state, batch, loss_type=loss_type)

    return results


def plot_records(records: list[ResultRecord], output_path: pathlib.Path):
    df = pd.DataFrame.from_records(records)
    (
        so.Plot(df, x="width", y="coord", color="layer")
        .layout(size=(15, 4))
        .facet(row="mup", col="iteration")
        .label(row="μP:", col="Iteration:", y=r"$\ell_1$")
        .add(so.Dot(), so.Agg())
        .add(so.Line(), so.Agg())
        .add(so.Band(), so.Est())
        .share(y=False)
        .scale(x="log2", y="symlog")  # type: ignore
        .save(output_path)
    )


class MLP(mup.Module):
    output_features: int

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        x = jnp.ravel(x)
        x = nn.Dense(features=16 * self.width)(x)
        x = nn.relu(x)
        x = nn.Dense(
            features=16 * self.width,
            kernel_init=nn.with_partitioning(
                nn.initializers.lecun_normal(),
                (None, None),
            ),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(features=16 * self.width)(x)
        x = nn.relu(x)

        DoubleWide = nn.vmap(
            nn.Dense,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            metadata_params={nn.meta.PARTITION_NAME: "model"},
            in_axes=None,  # type: ignore
            out_axes=0,
            axis_size=2,
        )
        x = DoubleWide(features=8 * self.width)(x)
        x = jnp.ravel(x)
        x = nn.relu(x)

        x = nn.Dense(features=self.output_features, kernel_init=nn.initializers.zeros)(x)
        return x


class CNN(mup.Module):
    output_features: int

    @nn.remat
    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        x = nn.Conv(features=4 * self.width, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=4 * self.width,
            kernel_size=(3, 3),
            strides=(2, 2),
            kernel_init=nn.with_partitioning(
                nn.initializers.lecun_normal(),
                (None, None),
            ),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(features=4 * self.width, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = jnp.ravel(x)

        x = nn.Dense(features=self.output_features, kernel_init=nn.initializers.zeros_init())(x)
        return x


def main(output_path: pathlib.Path):
    with_mups = [True, False]
    widths = list(reversed([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]))
    loss_types: list[LossType] = ["ce", "mse"]
    model_types = ["cnn", "mlp"]

    for loss_type, model_type in itertools.product(loss_types, model_types):
        if model_type == "mlp":
            model = MLP(output_features=10 if loss_type == "ce" else 1)
        elif model_type == "cnn":
            model = CNN(output_features=10 if loss_type == "ce" else 1)
        else:
            raise NotImplementedError

        records = []
        for with_mup, width in tqdm.tqdm(list(itertools.product(with_mups, widths))):
            print(f"Running loss_type={loss_type}, with_mup={with_mup}, width={width}")
            records.extend(
                coordinate_check(
                    model,
                    width_multiplier=width,
                    with_mup=with_mup,
                    loss_type=loss_type,
                )
            )

        plot_records(records, output_path / f"coord-checks-{model_type}-{loss_type}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str)

    args = parser.parse_args()
    main(pathlib.Path(args.output_path).resolve())
