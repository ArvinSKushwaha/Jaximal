from typing import Annotated, Any, dataclass_transform

from jax import Array

type Static[T] = Annotated[T, 'jaximal::meta']

@dataclass_transform(eq_default=True, frozen_default=True)
class Jaximal:
    def __init_subclass__(cls) -> None: ...

def dictify(
    x: Any, prefix: str = ..., typ: type | None = ...
) -> tuple[dict[str, Array], dict[str, str]]: ...
def dedictify[T](
    typ: type[T], data: dict[str, Array], meta: dict[str, str], prefix: str = ...
) -> T: ...

__all__ = ['Jaximal', 'Static', 'dictify', 'dedictify']
