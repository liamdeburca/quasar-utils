__all__ = ["format_property"]

from collections.abc import Callable
from functools import wraps
from typing import Protocol, TypeVar, overload

F = TypeVar("F", bound=Callable)


class FormattedProperty(Protocol[F]):
    """
    Use '__wrapped__' to access the original (unformatted) function.
    """

    __wrapped__: F

    def __call__(self, *args, **kwargs): ...


@overload
def format_property(
    func: F,
    *,
    attr_name: str = "format_property",
) -> FormattedProperty[F]: ...


@overload
def format_property(
    func: None = None,
    *,
    attr_name: str = "format_property",
) -> Callable[[F], FormattedProperty[F]]: ...


def format_property(
    func: F | None = None,
    *,
    attr_name: str = "format_property",
) -> FormattedProperty[F] | Callable[[F], FormattedProperty[F]]:

    def decorator(f: F) -> FormattedProperty[F]:

        @wraps(f)
        def inner_func(self, *args, **kwargs):
            result = f(self, *args, **kwargs)
            return getattr(result, attr_name)(result)

        inner_func.__wrapped__ = f

        return inner_func

    return decorator if func is None else decorator(func)
