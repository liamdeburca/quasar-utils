__all__ = ['validate_call']

from pydantic import validate_call as pydantic_validate_call
from typing import Callable, TypeVar, overload, Protocol

F = TypeVar('F', bound=Callable)

class Validated(Protocol[F]):    
    """
    Use '__wrapped__' to access the original (unvalidated) function.
    """
    __wrapped__: F
    def __call__(self, *args, **kwargs): ...

@overload
def validate_call(
    func: F,
    *,
    validate_return: bool = False,
) -> Validated[F]: ...

@overload
def validate_call(
    func: None = None,
    *,
    validate_return: bool = False,
) -> Callable[[F], Validated[F]]: ...

def validate_call(
    func: F | None = None,
    *,
    validate_return: bool = False,
) -> Validated[F] | Callable[[F], Validated[F]]:
    decorator = pydantic_validate_call(validate_return=validate_return)
    return decorator if func is None else decorator(func)