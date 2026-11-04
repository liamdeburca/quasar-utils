__all__ = [
    "apply_info_to_method",
    "format_property",
    "lru_cache"
]

import functools
from collections.abc import Callable
from typing import TypeVar, final, overload

F = TypeVar("F", bound=Callable)
M = TypeVar("M", bound=Callable)

### lru_cache

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
    decorator = functools.lru_cache(maxsize=maxsize)
    return decorator if func is None else decorator(func)

### apply_info_to_method

@overload
def apply_info_to_method[M](
    method: None = None,
    *,
    subjects: tuple[str, ...] = (),
    start: int = 0,
    stop: int = 100,
    specific_kwargs: set[str] | None = None,
) -> Callable[[M], M]: ...


@overload
def apply_info_to_method[M](
    method: M,
    *,
    subjects: tuple[str, ...] = (),
    start: int = 0,
    stop: int = 100,
    specific_kwargs: set[str] | None = None,
) -> M: ...


def apply_info_to_method[M](
    method: M | None = None,
    *,
    subjects: tuple[str, ...] = (),
    start: int = 0,
    stop: int = 100,
    specific_kwargs: set[str] | None = None,
):
    def decorator(method: M):
        @functools.wraps(method)
        def wrapped_method(self: object, *args, **kwargs):
            nonlocal subjects, start, stop, specific_kwargs
            new_kwargs = kwargs.copy()
            all_keys = list(method.__kwdefaults__.keys())

            if len(all_keys) == 0:
                raise UserWarning

            for i in range(start, min(stop, len(all_keys))):
                key = all_keys[i]
                if kwargs.get(key) is not None:
                    continue
                if specific_kwargs is not None and (
                    key not in specific_kwargs
                ):
                    continue

                new_kwargs[key] = self.info.__getitem__(
                    key,
                    subjects=subjects,
                )

            return method(self, *args, **new_kwargs)
        return wrapped_method
    return decorator if (method is None) else decorator(method)

### format_property

@overload
def format_property[F](
    func: F,
    *,
    attr_name: str = "format_property",
) -> F: ...


@overload
def format_property[F](
    func: None = None,
    *,
    attr_name: str = "format_property",
) -> Callable[[F], F]: ...


def format_property[F](
    func: F | None = None,
    *,
    attr_name: str = "format_property",
) -> F | Callable[[F], F]:
    """
    Decorator to apply a formatting method to the result of a property.
    The method name is specified by `attr_name` and defaults to 
    "format_property".
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        def inner_func(self, *args, **kwargs):
            result = f(self, *args, **kwargs)
            return getattr(result, attr_name)(result)
        return inner_func
    return decorator if func is None else decorator(func)
