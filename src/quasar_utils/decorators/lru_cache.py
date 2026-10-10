__all__ = ["lru_cache"]

from collections.abc import Callable
from functools import lru_cache as _lru_cache
from typing import TypeVar, final, overload

F = TypeVar("F", bound=Callable)

@overload
def lru_cache[F](
    func: F,
    *,
    maxsize: int | None = None,
) -> F: ...

@overload
def lru_cache[F](
    func: None = None,
    *,
    maxsize: int | None = None,
) -> Callable[[F], F]: ...

@final
def lru_cache[F](
    func: F | None = None,
    *,
    maxsize: int | None = None,
) -> F | Callable[[F], F]:
    """
    Simple wrapper for `functools.lru_cache` that preserves type annotations.
    """
    decorator = _lru_cache(maxsize=maxsize)
    return decorator if func is None else decorator(func)