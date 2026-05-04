__all__ = ['validated_lru_cache']

from functools import wraps
from typing import Callable, TypeVar, overload, Protocol

from pydantic import validate_call as pydantic_validate_call
from cachetools import LRUCache

F = TypeVar('F', bound=Callable)

class ValidatedAndLRUCached(Protocol[F]):
    """
    Attributes:
    - '_cache': the shared LRU cache.
    - '__unvalidated__': unvalidated but cached function.
    - '__uncached__': validated but uncached function.
    - '__wrapped__': original (unvalidated and uncached) function.
    """
    _cache: LRUCache
    __wrapped__: F
    __unvalidated__: F
    __uncached__: F

    def __call__(self, *args, **kwargs):
        ...

@overload
def validated_lru_cache(
    func: F,
) -> ValidatedAndLRUCached[F]: ...

@overload
def validated_lru_cache(
    func: None = None,
    *,
    maxsize: int = 256,
    validate_return: bool = False,
) -> Callable[[F], ValidatedAndLRUCached[F]]: ...

def validated_lru_cache(
    func: F | None = None,
    *,
    maxsize: int = 64,
    validate_return: bool = False,
) -> Callable[[F], ValidatedAndLRUCached[F]]:
    """
    Creates a decorator that applies both LRU caching and Pydantic validation to 
    a function.
    """
    validate_decorator = pydantic_validate_call(validate_return=validate_return)

    def decorator(f: F) -> ValidatedAndLRUCached[F]:
        nonlocal maxsize, validate_return

        cache = LRUCache(maxsize=maxsize)
        
        @wraps(f)
        def inner_func(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            if key not in cache:
                cache[key] = validate_decorator(f)(*args, **kwargs)
            return cache[key]
        
        def unvalidated_func(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            if key not in cache:
                cache[key] = f(*args, **kwargs)
            return cache[key]
        
        def uncached_func(*args, **kwargs):
            return validate_decorator(f)(*args, **kwargs)
        
        inner_func.__wrapped__ = f
        inner_func.__unvalidated__ = unvalidated_func
        inner_func.__uncached__ = uncached_func
        inner_func._cache = cache

        return inner_func

    return decorator if (func is None) else decorator(func)