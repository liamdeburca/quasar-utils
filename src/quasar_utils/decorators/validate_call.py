__all__ = ["validate_call"]

from collections.abc import Callable
from typing import Protocol, TypeVar, final, overload

from pydantic import validate_call as pydantic_validate_call

F = TypeVar("F", bound=Callable)


class Validated(Protocol[F]):
    """
    Use '__wrapped__' to access the original (unvalidated) function.
    """
    __call__: F
    __wrapped__: F


@overload
def validate_call[F](
    func: F,
    *,
    validate_return: bool = False,
) -> Validated[F]: ...


@overload
def validate_call[F](
    func: None = None,
    *,
    validate_return: bool = False,
) -> Callable[[F], Validated[F]]: ...

@final
def validate_call[F](
    func: F | None = None,
    *,
    validate_return: bool = False,
) -> Validated[F] | Callable[[F], Validated[F]]:
    decorator = pydantic_validate_call(validate_return=validate_return)
    return decorator if func is None else decorator(func)
