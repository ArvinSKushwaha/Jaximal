import os.path

from typing import cast

import jax
import optax

from jax import numpy as np
from jaximal.core import Jaximal, Static, dedictify, dictify
from jaximal.io import load_file, save_file
from jaxtyping import Array, Float, PRNGKeyArray, Scalar


def test_core(tmp_path: str):
    class Linear(Jaximal):
        in_dim: Static[int]
        out_dim: Static[int]

        weight: Float[Array, '{self.out_dim} {self.in_dim}']
        bias: Float[Array, '{self.out_dim}']

        @staticmethod
        def init_state(in_dim: int, out_dim: int, key: PRNGKeyArray) -> 'Linear':
            w_key, b_key = jax.random.split(key)
            weight = (
                jax.random.normal(w_key, shape=(out_dim, in_dim))
                / (out_dim * in_dim) ** 0.5
            )
            bias = jax.random.normal(b_key, shape=(out_dim,)) / out_dim**0.5
            return Linear(in_dim, out_dim, weight, bias)

        def forward(
            self,
            x: Float[Array, '{self.in_dim}'],
        ) -> Float[Array, '{self.out_dim}']:
            return self.weight @ x + self.bias

    class MLP(Jaximal):
        in_dim: Static[int]
        out_dim: Static[int]
        hidden_dims: Static[list[int]]

        modules: list[Linear]

        @staticmethod
        def init_state(
            in_dim: int, out_dim: int, hidden_dims: list[int], key: PRNGKeyArray
        ) -> 'MLP':
            shapes = [in_dim, *hidden_dims, out_dim]
            keys = jax.random.split(key, len(shapes) - 1)

            modules = [
                Linear.init_state(shapes[i], shapes[i + 1], keys[i])
                for i in range(len(shapes) - 1)
            ]
            return MLP(in_dim, out_dim, hidden_dims, modules)

        def forward(
            self,
            x: Float[Array, '{self.in_dim}'],
        ) -> Float[Array, '{self.out_dim}']:
            tmp = x
            for i, module in enumerate(self.modules):
                tmp = module.forward(tmp)

                if i < len(self.modules) - 1:
                    tmp = jax.nn.swish(tmp)

            final = tmp
            return final

    key = jax.random.key(0)
    x_key, y_key, key = jax.random.split(key, 3)

    x_data = jax.random.normal(x_key, shape=(2, 3))
    y_data = jax.random.normal(y_key, shape=(2, 4))

    mlp_key, key = jax.random.split(key)

    optimizer = optax.chain(optax.adam(1e-1), optax.contrib.reduce_on_plateau())

    mlp = MLP.init_state(3, 4, [16, 16], mlp_key)
    opt_state: optax.OptState = optimizer.init(cast(optax.OptState, mlp))

    def loss(
        mlp: MLP,
        x_data: Float[Array, 'batch {mlp.in_dim}'],
        y_data: Float[Array, 'batch {mlp.out_dim}'],
    ) -> Float[Scalar, '']:
        return optax.l2_loss(jax.vmap(mlp.forward)(x_data), y_data).mean()

    loss_grad = jax.jit(jax.value_and_grad(loss))

    def update(
        i: int,
        a: tuple[MLP, optax.OptState],
    ) -> tuple[MLP, optax.OptState]:
        mlp, opt_state = a
        cost, grads = loss_grad(mlp, x_data, y_data)
        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            mlp,  # type: ignore
            value=cost,
        )

        mlp: MLP = optax.apply_updates(mlp, updates)  # type: ignore

        jax.debug.print('{} {}', i, cost)

        return mlp, opt_state

    mlp, opt_state = jax.lax.fori_loop(0, 500, update, (mlp, opt_state))
    save_file(os.path.join(tmp_path, 'test_mlp.safetensors'), *dictify(mlp))

    mlp_restored = dedictify(
        MLP,
        *load_file(os.path.join(tmp_path, 'test_mlp.safetensors')),
    )

    assert mlp == mlp_restored
    assert np.allclose(
        jax.vmap(mlp.forward)(x_data), jax.vmap(mlp_restored.forward)(x_data)
    )
