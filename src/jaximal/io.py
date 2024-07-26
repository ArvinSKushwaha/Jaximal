import json
import struct

import safetensors.flax as safflax

from jaxtyping import Array

from jaximal.serialization import JSONEncoder, json_object_hook


def save_file(
    filename: str,
    data: dict[str, Array],
    metadata: dict[str, str],
) -> None:
    """
    Given a `dict[str, jax.Array]` called `data` and a dictionary `dict[str,
    str]` called `meta`, uses `safetensors.flax.save_file` to save both to
    the given `filename`.
    """

    if data:
        safflax.save_file(data, filename, metadata)
    else:
        with open(filename, 'wb') as f:
            metadata_ser = json.dumps(
                {'__metadata__': metadata, '__no_data__': True}, cls=JSONEncoder
            )
            f.write(struct.pack('<Q', len(metadata_ser)))
            f.write(metadata_ser.encode('utf-8'))


def load_file(filename: str) -> tuple[dict[str, Array], dict[str, str]]:
    """
    Uses `safetensors.flax.load_file` to load the `dict[str, jax.Array]` data
    from the given `filename` and then manually retrieves the `dict[str, str]`
    metadata from the file.
    """

    with open(filename, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        json_data = f.read(header_len)
        deser_data = json.loads(json_data, object_hook=json_object_hook)
        metadata = deser_data['__metadata__']

    data = {} if '__no_data__' in deser_data else safflax.load_file(filename)
    return data, metadata


def save(data: dict[str, Array], metadata: dict[str, str]) -> bytes:
    """
    Given a `dict[str, jax.Array]` called `data` and a dictionary `dict[str,
    str]` called `meta`, uses `safetensors.flax.save` to serialize both into
    `bytes`.
    """
    return safflax.save(data, metadata)


def load(raw_data: bytes) -> tuple[dict[str, Array], dict[str, str]]:
    """
    Uses `safetensors.flax.load` to load the `dict[str, jax.Array]` data from
    the given `bytes` and then manually retrieves the `dict[str, str]` metadata
    from the `bytes`.
    """
    data = safflax.load(raw_data)

    header_len = struct.unpack('<Q', raw_data[:8])[0]
    metadata = json.loads(raw_data[8 : 8 + header_len], object_hook=json_object_hook)[
        '__metadata__'
    ]

    return data, metadata


__all__ = ['save_file', 'save', 'load_file', 'load']
