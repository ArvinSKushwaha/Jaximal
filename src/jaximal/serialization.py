import base64
import json
import pickle

from itertools import chain
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Sequence,
    cast,
    get_args,
    get_origin,
)

import jax

from jaxtyping import AbstractArray, Array

from jaximal.core import Jaximal, Static


class FnRegistry:
    functions: dict[Callable[..., Any], str] = {}
    inv_functions: dict[str, Callable[..., Any]] = {}

    def add(self, function: Callable[..., Any], name: str) -> None:
        if function in self.functions or name in self.inv_functions:
            raise ValueError(
                f'Function {function} with name {name} already in registry.'
            )

        self.functions[function] = name
        self.inv_functions[name] = function

    def lookup_name(self, name: str) -> Callable[..., Any] | None:
        return self.inv_functions.get(name)

    def lookup_function(self, function: Callable[..., Any]) -> str | None:
        return self.functions.get(function)


global_fn_registry: FnRegistry = FnRegistry()


global_fn_registry.add(jax.numpy.sin, 'jax.numpy.sin')
global_fn_registry.add(jax.numpy.cos, 'jax.numpy.cos')
global_fn_registry.add(jax.numpy.tan, 'jax.numpy.tan')
global_fn_registry.add(jax.numpy.log, 'jax.numpy.log')
global_fn_registry.add(jax.numpy.exp, 'jax.numpy.exp')
global_fn_registry.add(jax.numpy.tanh, 'jax.numpy.tanh')


class JSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> dict[str, Any] | None:
        if isinstance(o, Callable):
            if o_str := global_fn_registry.lookup_function(o):
                return {
                    'callable': True,
                    'jax_map': o_str,
                }

            else:
                return {
                    'callable': True,
                    'code': base64.b64encode(pickle.dumps(o)).decode('utf-8'),
                }


def json_object_hook(dct: Any) -> Any:
    if 'callable' in dct:
        if 'jax_map' in dct:
            return global_fn_registry.lookup_name(dct['jax_map'])
        elif 'code' in dct:
            return pickle.loads(base64.b64decode(dct['code']))
        else:
            return dct
    return dct


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
        metadata |= {prefix.removesuffix('::'): json.dumps(x, cls=JSONEncoder)}

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
        return json.loads(
            metadata[prefix.removesuffix('::')], object_hook=json_object_hook
        )

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
        key_type, child_type = get_args(typ)

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
        (child_type,) = get_args(typ)

        child_idx = 0
        while True:
            child_prefix = prefix + str(child_idx) + '::'
            try:
                next(
                    filter(
                        lambda x: x.startswith(child_prefix),
                        cast(Iterable[str], chain(data.keys(), metadata.keys())),
                    )
                )
            except StopIteration:
                break
            children.append(dedictify(child_type, data, metadata, child_prefix))
            child_idx += 1

        return cast(T, children)

    raise TypeError(
        f'Unexpected type {typ} and prefix {prefix} recieved by `dedictify`.'
    )


__all__ = [
    'dictify',
    'dedictify',
    'json_object_hook',
    'JSONEncoder',
    'global_fn_registry',
]
