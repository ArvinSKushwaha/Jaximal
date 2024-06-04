import json
import struct

from pathlib import Path

import safetensors.flax as safflax

from .typing import Array


def save_file(
    filename: Path | str,
    data: dict[str, Array],
    meta: dict[str, str],
) -> None:
    safflax.save_file(data, filename, meta)


def load_file(filename: Path | str) -> tuple[dict[str, Array], dict[str, str]]:
    data = safflax.load_file(filename)

    with open(filename, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        meta = json.loads(f.read(header_len))['__metadata__']

    return data, meta


def save(data: dict[str, Array], meta: dict[str, str]) -> bytes:
    return safflax.save(data, meta)


def load(raw_data: bytes) -> tuple[dict[str, Array], dict[str, str]]:
    data = safflax.load(raw_data)

    header_len = struct.unpack('<Q', raw_data[:8])[0]
    meta = json.loads(raw_data[8 : 8 + header_len])['__metadata__']

    return data, meta
