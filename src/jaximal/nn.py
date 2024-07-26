from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Callable, Self

import jax

from jax import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from jaximal.core import Jaximal, Static


class WeightInitialization(Enum):
    Zero = auto()
    RandomUniform = auto()
    RandomNormal = auto()
    GlorotUniform = auto()
    GlorotNormal = auto()
    HeUniform = auto()
    HeNormal = auto()

    def init(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        fan_in: int,
        fan_out: int,
        dtype: np.dtype = np.float32,
    ) -> Float[Array, '*']:
        match self:
            case WeightInitialization.Zero:
                return np.zeros(shape, dtype=dtype)
            case WeightInitialization.RandomUniform:
                return jax.random.uniform(
                    key,
                    shape,
                    dtype=dtype,
                    minval=-1.0,
                    maxval=1.0,
                )
            case WeightInitialization.RandomNormal:
                return jax.random.normal(key, shape, dtype=dtype)
            case WeightInitialization.GlorotUniform:
                scaling = (6 / (fan_in + fan_out)) ** 0.5
                return jax.random.uniform(
                    key,
                    shape,
                    dtype=dtype,
                    minval=-scaling,
                    maxval=scaling,
                )
            case WeightInitialization.GlorotNormal:
                scaling = (2 / (fan_in + fan_out)) ** 0.5
                return jax.random.normal(key, shape, dtype=dtype) * scaling
            case WeightInitialization.HeUniform:
                scaling = (6 / fan_in) ** 0.5
                return jax.random.uniform(
                    key,
                    shape,
                    dtype=dtype,
                    minval=-scaling,
                    maxval=scaling,
                )
            case WeightInitialization.HeNormal:
                scaling = (2 / fan_in) ** 0.5
                return jax.random.normal(key, shape, dtype=dtype) * scaling


class JaximalModule(Jaximal, ABC):
    @classmethod
    @abstractmethod
    def init_state(
        cls, *args: Any, **kwargs: Any
    ) -> Callable[[PRNGKeyArray], Self]: ...

    @abstractmethod
    def __call__(self, data: PyTree) -> PyTree: ...


class Activation(JaximalModule):
    func: Static[Callable[[Array], Array]]

    @classmethod
    def init_state(
        cls, func: Callable[[Array], Array]
    ) -> Callable[[PRNGKeyArray], Self]:
        return lambda key: cls(func)

    def __call__(self, data: PyTree) -> PyTree:
        return jax.tree.map(self.func, data)


class Linear(JaximalModule):
    in_dim: Static[int]
    out_dim: Static[int]

    weights: Float[Array, 'in_dim out_dim']
    biases: Float[Array, 'out_dim']

    @classmethod
    def init_state(
        cls,
        in_dim: int,
        out_dim: int,
        weight_initialization: WeightInitialization = WeightInitialization.GlorotUniform,
        bias_initialization: WeightInitialization = WeightInitialization.Zero,
    ) -> Callable[[PRNGKeyArray], Self]:
        def init(key: PRNGKeyArray) -> Self:
            w_key, b_key = jax.random.split(key)
            weights = weight_initialization.init(
                w_key, (in_dim, out_dim), in_dim, out_dim
            )
            biases = weight_initialization.init(b_key, (out_dim,), 1, out_dim)

            return cls(in_dim, out_dim, weights, biases)

        return init

    def __call__(self, data: PyTree) -> PyTree:
        return data @ self.weights + self.biases


class Sequential(JaximalModule):
    modules: list[JaximalModule]

    @classmethod
    def init_state(
        cls, partials: list[Callable[[PRNGKeyArray], JaximalModule]]
    ) -> Callable[[PRNGKeyArray], Self]:
        def init(key: PRNGKeyArray) -> Self:
            keys = jax.random.split(key, len(partials))

            modules = list(partial(key) for key, partial in zip(keys, partials))
            return cls(modules)

        return init

    def __call__(self, data: PyTree, *args: dict[str, Any]) -> PyTree:
        assert len(args) == len(self.modules), (
            'Expected `self.modules` and `args` to have the same length '
            f'but got {len(self.modules)} and {len(args)}, respectively.'
        )
        for kwargs, modules in zip(args, self.modules):
            data = modules(data, **kwargs)

        return data


__all__ = [
    'JaximalModule',
    'WeightInitialization',
    'Linear',
    'Sequential',
    'Activation',
]
