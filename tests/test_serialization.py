import os.path

from typing import Callable

import jax

from jax import numpy as np
from jaximal.core import Jaximal, Static
from jaximal.io import load_file, save_file
from jaximal.serialization import dedictify, dictify
from jaxtyping import Array


def activation_function(x: Array, data: Array) -> Array:
    return np.sin(x + data)


def test_serialization(tmp_path: str):
    class Activation(Jaximal):
        function: Static[Callable[[Array], Array]]

        def forward(self, x: Array) -> Array:
            return self.function(x)

    class ActivationWithData(Jaximal):
        function: Static[Callable[[Array, Array], Array]]
        data: Array

        def forward(self, x: Array) -> Array:
            return self.function(x, self.data)

    key = jax.random.key(0)
    x = jax.random.uniform(key, (1024,))

    activation = Activation(np.sin)
    save_file(os.path.join(tmp_path, 'test_mlp.safetensors'), *dictify(activation))

    activation_restored = dedictify(
        Activation,
        *load_file(os.path.join(tmp_path, 'test_mlp.safetensors')),
    )

    assert activation_restored == activation
    assert np.allclose(activation.forward(x), activation_restored.forward(x))

    activation_w_data = ActivationWithData(activation_function, np.array(1.0))
    save_file(
        os.path.join(tmp_path, 'test_mlp.safetensors'), *dictify(activation_w_data)
    )

    activation_w_data_restored = dedictify(
        ActivationWithData,
        *load_file(os.path.join(tmp_path, 'test_mlp.safetensors')),
    )

    assert activation_w_data_restored == activation_w_data
    assert np.allclose(
        activation_w_data.forward(x), activation_w_data_restored.forward(x)
    )


if __name__ == '__main__':
    test_serialization('.')
