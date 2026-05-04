__all__ = ['lru_cache']

from functools import lru_cache as _lru_cache
from typing import Callable, TypeVar, overload, Protocol

F = TypeVar('F', bound=Callable)

class LRUCached(Protocol[F]):    
    """
    Use '__wrapped__' to access the original (uncached) function.
    """
    __wrapped__: F
    def __call__(self, *args, **kwargs):
        ...

@overload
def lru_cache(
    func: F,
    *,
    maxsize: int | None = None,
) -> LRUCached[F]: ...

@overload
def lru_cache(
    func: None = None,
    *,
    maxsize: int | None = None,
) -> Callable[[F], LRUCached[F]]: ...

def lru_cache(
    func: F | None = None,
    *,
    maxsize: int | None = None,
) -> LRUCached[F] | Callable[[F], LRUCached[F]]:
    decorator = _lru_cache(maxsize=maxsize)
    return decorator if func is None else decorator(func)