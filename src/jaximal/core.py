import typing

from dataclasses import make_dataclass
from json import dumps, loads
from typing import (
    Annotated,
    Any,
    Mapping,
    Self,
    Sequence,
    dataclass_transform,
    get_origin,
)

import jax

from .typing import AbstractArray, Array

type Static[T] = Annotated[T, 'jaximal::meta']


@dataclass_transform(eq_default=True, frozen_default=True)
class Jaximal:
    def __init_subclass__(cls) -> None:
        cls2 = make_dataclass(
            cls.__name__, list(cls.__annotations__), slots=True, frozen=True
        )
        cls.__init__ = cls2.__init__  # pyright: ignore
        cls.__repr__ = cls2.__repr__  # pyright: ignore
        cls.__slots__ = cls2.__slots__  # pyright: ignore
        cls.__setattr__ = cls2.__setattr__  # pyright: ignore
        cls.__delattr__ = cls2.__delattr__  # pyright: ignore
        cls.__getattribute__ = cls2.__getattribute__  # pyright: ignore

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

        cls.__eq__ = cls_eq


def dictify(
    x: Any,
    prefix: str = '',
    typ: type | None = None,
) -> tuple[dict[str, Array], dict[str, str]]:
    typ = type(x) if typ is None else typ

    data: dict[str, Array] = {}
    meta: dict[str, str] = {}

    if get_origin(typ) == Static:
        meta |= {prefix.removesuffix('::'): dumps(x)}

    elif isinstance(x, Array):
        data |= {prefix.removesuffix('::'): x}

    elif issubclass(typ, Jaximal):
        for child_key, child_type in x.__annotations__.items():
            child_data, child_meta = dictify(
                getattr(x, child_key), prefix + child_key + '::', typ=child_type
            )

            data |= child_data
            meta |= child_meta

    elif isinstance(x, Sequence):
        for child_idx, child_elem in enumerate(x):
            child_data, child_meta = dictify(child_elem, prefix + str(child_idx) + '::')

            data |= child_data
            meta |= child_meta

    elif isinstance(x, Mapping):
        for child_key, child_elem in x.items():
            child_data, child_meta = dictify(child_elem, prefix + str(child_key) + '::')

            data |= child_data
            meta |= child_meta

    else:
        raise TypeError(
            f'Unexpected type {typ} and prefix {prefix} recieved by `dictify`.'
        )

    return data, meta


def dedictify[T](
    typ: type[T],
    data: dict[str, Array],
    meta: dict[str, str],
    prefix: str = '',
) -> T:
    base_typ = get_origin(typ)
    if base_typ is None:
        base_typ = typ

    if get_origin(typ) == Static:
        return loads(meta[prefix.removesuffix('::')])

    elif typ == Array or issubclass(base_typ, AbstractArray):
        return data[prefix.removesuffix('::')]  # type: ignore

    elif issubclass(base_typ, Jaximal):
        children = {}
        for child_key, child_type in typ.__annotations__.items():
            children[child_key] = dedictify(
                child_type, data, meta, prefix + child_key + '::'
            )

        return typ(**children)

    elif issubclass(base_typ, list):
        children = []
        (child_type,) = typing.get_args(typ)

        child_idx = 0
        while True:
            child_prefix = prefix + str(child_idx) + '::'
            try:
                next(filter(lambda x: x.startswith(child_prefix), data))
                next(filter(lambda x: x.startswith(child_prefix), meta))
            except StopIteration:
                break
            children.append(dedictify(child_type, data, meta, child_prefix))
            child_idx += 1

        return children  # type: ignore

    elif issubclass(base_typ, Mapping):
        children = {}
        key_type, child_type = typing.get_args(typ)

        for keys in filter(lambda x: x.startswith(prefix), data):
            keys = keys[len(prefix) :]
            child_key = key_type(keys.split('::', 1)[0])
            child_prefix = prefix + str(child_key) + '::'

            if child_key in children:
                continue

            children[child_key] = dedictify(child_type, data, meta, child_prefix)

        return children  # type: ignore

    raise TypeError(
        f'Unexpected type {typ} and prefix {prefix} recieved by `dedictify`.'
    )


__all__ = ['Jaximal', 'Static', 'dictify']
