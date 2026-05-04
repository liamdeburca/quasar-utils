__all__ = ['validated_apply_info_to_method']

from functools import wraps
from pydantic import validate_call as pydantic_validate_call
from typing import Callable, TypeVar, overload, Protocol

from .apply_info_to_method import apply_info_to_method

M = TypeVar('M', bound=Callable)

class ValidatedAndApplied(Protocol[M]):    
    __wrapped__: M
    __unapplied__: M
    __unvalidated__: M
    def __call__(self, *args, **kwargs):
        ...

@overload
def validated_apply_info_to_method(
    func: M,
    *,
    subjects: tuple[str, ...],
    start: int = 0,
    stop: int = 100,
    specific_kwargs: set[str] | None = None,
    validate_return: bool = False,
) -> ValidatedAndApplied[M]: ...

@overload
def validated_apply_info_to_method(
    func: None = None,
    *,
    subjects: tuple[str, ...],
    start: int = 0,
    stop: int = 100,
    specific_kwargs: set[str] | None = None,
    validate_return: bool = False,
) -> Callable[[M], ValidatedAndApplied[M]]: ...

def validated_apply_info_to_method(
    func: M | None = None,
    *,
    subjects: tuple[str, ...],
    start: int = 0,
    stop: int = 100,
    specific_kwargs: set[str] | None = None,
    validate_return: bool = False,
) -> ValidatedAndApplied[M] | Callable[[M], ValidatedAndApplied[M]]:
    
    validate_decorator = pydantic_validate_call(
        validate_return=validate_return,
    )
    apply_decorator = apply_info_to_method(
        subjects=subjects,
        start=start,
        stop=stop,
        specific_kwargs=specific_kwargs,
    )
    
    def decorator(f: M) -> ValidatedAndApplied[M]:

        @wraps(f)
        def inner_func(*args, **kwargs):
            return validate_decorator(apply_decorator(f))(*args, **kwargs)
        
        inner_func.__unapplied__ = validate_decorator(f)
        inner_func.__unvalidated__ = apply_decorator(f)
        
        return inner_func
    
    return decorator if func is None else decorator(func)