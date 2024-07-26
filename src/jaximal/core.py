from dataclasses import dataclass, fields
from typing import (
    TYPE_CHECKING,
    Annotated,
    Self,
    cast,
    dataclass_transform,
    get_origin,
)

import jax

from jaxtyping import AbstractArray, Array

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

type Static[T] = Annotated[T, 'jaximal::meta']


@dataclass_transform(eq_default=True, frozen_default=True)
class Jaximal:
    """
    The `Jaximal` class mimics the behavior of the `@dataclass` decorator and
    provides additional automatic `JAX` PyTree-flattening and
    PyTree-unflattening utilities. Additionally, the `dedictify` and `dictify`
    methods can be used to serialize and deserialize subclasses of `Jaximal`.

    To be a subclass of `Jaximal` and have its functionality work properly,
    there are some strict requirements.

    1. The script must not include a `from __futures__ import annotations`
    line. This is not yet supported and may never be.
    2. All types must be fully annotated.
    3. All non-static types must contain a PyTree of `Jaximal` modules or
    `JAX`-compatible types. This will likely be loosened in the future to
    support non-`Jaximal` JAX PyTrees. We also support `jaxtyping` types and
    recommend they be used in your code.
    4. All static types (in the `JAX` sense), must be annotated with
    `Jaximal.Static`. They must also all be able to be `JSON` serialized.
    5. The `__init__` function may not be manually defined. As an alternative,
    consider using a `staticmethod` to initialize your class in a custom
    manner.

    Here is an example of a `Jaximal` class.

    ```python
    class Linear(Jaximal):
        in_dim: Static[int]
        out_dim: Static[int]

        weight: Float[Array, '{self.out_dim} {self.in_dim}']
        bias: Float[Array, '{self.out_dim}']

        @staticmethod
        def init_state(in_dim: int, out_dim: int, key: PRNGKeyArray) -> 'Linear':
            w_key, b_key = jax.random.split(key)
            weight = jax.random.normal(w_key, shape=(out_dim, in_dim))
            bias = jax.random.normal(b_key, shape=(out_dim,))
            return Linear(in_dim, out_dim, weight, bias)

        def forward(
            self,
            x: Float[Array, '{self.in_dim}'],
        ) -> Float[Array, '{self.out_dim}']:
            return self.weight @ x + self.bias
    ```
    """

    def __init_subclass__(cls) -> None:
        # The `dataclass` decorator modifies `cls` itself, so we don't need to
        # worry in the later steps. Note: This only holds true when
        # `slots=False` in the `dataclass` decorator.
        dataclass(frozen=True, eq=False)(cls)

        cls_fields = fields(cast('DataclassInstance', cls))

        data_fields = []
        meta_fields = []

        for field in cls_fields:
            if get_origin(field.type) != Static:
                data_fields.append(field.name)
            if get_origin(field.type) == Static:
                meta_fields.append(field.name)

        def cls_eq(self: Self, other: object) -> bool:
            if type(other) != type(self):
                return False

            equal = True
            for meta in meta_fields:
                equal &= getattr(self, meta) == getattr(other, meta)

                if not equal:
                    return False

            for data in data_fields:
                if (ann := cls.__annotations__[data]) == Array or issubclass(
                    ann, AbstractArray
                ):
                    equal &= (getattr(self, data) == getattr(other, data)).all()
                else:
                    equal &= getattr(self, data) == getattr(other, data)

                if not equal:
                    return False

            return equal

        setattr(cls, '__eq__', cls_eq)

        jax.tree_util.register_dataclass(cls, data_fields, meta_fields)


__all__ = ['Jaximal', 'Static']
