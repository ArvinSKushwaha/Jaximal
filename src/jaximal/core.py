import typing

from dataclasses import make_dataclass
from json import dumps, loads
from typing import (
    Annotated,
    Any,
    Callable,
    Mapping,
    Self,
    Sequence,
    cast,
    dataclass_transform,
    get_origin,
)

import jax

from jaxtyping import AbstractArray, Array

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
        cls2 = make_dataclass(
            cls.__name__, list(cls.__annotations__), slots=True, frozen=True
        )

        setattr(cls, '__init__', cls2.__init__)
        setattr(cls, '__repr__', cls2.__repr__)
        setattr(cls, '__slots__', cls2.__slots__)
        setattr(cls, '__setattr__', cls2.__setattr__)
        setattr(cls, '__delattr__', cls2.__delattr__)
        setattr(cls, '__getattribute__', cls2.__getattribute__)

        data_fields = [
            key for key, typ in cls.__annotations__.items() if get_origin(typ) != Static
        ]
        meta_fields = [
            key for key, typ in cls.__annotations__.items() if get_origin(typ) == Static
        ]

        jax.tree_util.register_dataclass(cls, data_fields, meta_fields)

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

        cls.__eq__: Callable[[Self, object], bool] = cls_eq


def dictify(
    x: Any,
    prefix: str = '',
    typ: type | None = None,
) -> tuple[dict[str, Array], dict[str, str]]:
    """
    Given an object, a prefix, and optionally a type for the object, attempt to
    deconstruct the object into a `dict[str, jax.Array]` and a `dict[str, str]`
    where all keys have the given prefix.
    """

    typ = type(x) if typ is None else typ

    data: dict[str, Array] = {}
    metadata: dict[str, str] = {}

    if get_origin(typ) == Static:
        metadata |= {prefix.removesuffix('::'): dumps(x)}

    elif isinstance(x, Array):
        data |= {prefix.removesuffix('::'): x}

    elif issubclass(typ, Jaximal):
        for child_key, child_type in x.__annotations__.items():
            child_data, child_metadata = dictify(
                getattr(x, child_key), prefix + child_key + '::', typ=child_type
            )

            data |= child_data
            metadata |= child_metadata

    elif isinstance(x, Mapping):
        for child_key, child_elem in x.items():
            child_data, child_metadata = dictify(
                child_elem, prefix + str(child_key) + '::'
            )

            data |= child_data
            metadata |= child_metadata

    elif isinstance(x, Sequence):
        for child_idx, child_elem in enumerate(x):
            child_data, child_metadata = dictify(
                child_elem, prefix + str(child_idx) + '::'
            )

            data |= child_data
            metadata |= child_metadata

    else:
        raise TypeError(
            f'Unexpected type {typ} and prefix {prefix} recieved by `dictify`.'
        )

    return data, metadata


def dedictify[T](
    typ: type[T],
    data: dict[str, Array],
    metadata: dict[str, str],
    prefix: str = '',
) -> T:
    """
    Given a type, a `dict[str, jax.Array]`, a `dict[str, str]`, and a prefix
    for the dictionary keys, attempts to recreate an instance of the given
    type.
    """

    base_typ = get_origin(typ)
    if base_typ is None:
        base_typ = typ

    if get_origin(typ) == Static:
        return loads(metadata[prefix.removesuffix('::')])

    elif typ == Array or issubclass(base_typ, AbstractArray):
        return cast(T, data[prefix.removesuffix('::')])

    elif issubclass(base_typ, Jaximal):
        children = {}
        for child_key, child_type in typ.__annotations__.items():
            children[child_key] = dedictify(
                child_type, data, metadata, prefix + child_key + '::'
            )

        return typ(**children)

    elif issubclass(base_typ, Mapping):
        children = {}
        key_type, child_type = typing.get_args(typ)

        for keys in filter(lambda x: x.startswith(prefix), data):
            keys = keys[len(prefix) :]
            child_key = key_type(keys.split('::', 1)[0])
            child_prefix = prefix + str(child_key) + '::'

            if child_key in children:
                continue

            children[child_key] = dedictify(child_type, data, metadata, child_prefix)

        return cast(T, children)

    elif issubclass(base_typ, list):
        children = []
        (child_type,) = typing.get_args(typ)

        child_idx = 0
        while True:
            child_prefix = prefix + str(child_idx) + '::'
            try:
                next(filter(lambda x: x.startswith(child_prefix), data))
                next(filter(lambda x: x.startswith(child_prefix), metadata))
            except StopIteration:
                break
            children.append(dedictify(child_type, data, metadata, child_prefix))
            child_idx += 1

        return cast(T, children)

    raise TypeError(
        f'Unexpected type {typ} and prefix {prefix} recieved by `dedictify`.'
    )


__all__ = ['Jaximal', 'Static', 'dictify', 'dedictify']
