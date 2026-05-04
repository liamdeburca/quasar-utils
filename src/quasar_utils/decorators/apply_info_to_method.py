__all__ = ['apply_info_to_method']

from functools import wraps
from typing import Callable, TypeVar, overload, Protocol

M = TypeVar('M', bound=Callable)

class Applied(Protocol[M]):    
    """
    Use '__wrapped__' to access the original method.
    """
    __wrapped__: M
    def __call__(self, *args, **kwargs):
        ...

@overload
def apply_info_to_method(
    method: None = None,
    *,
    subjects: tuple[str, ...] = (),
    start: int = 0,
    stop: int = 100,
    specific_kwargs: set[str] | None = None,
) -> Callable[[M], Applied[M]]: ...

@overload
def apply_info_to_method(
    method: M,
    *,
    subjects: tuple[str, ...] = (),
    start: int = 0,
    stop: int = 100,
    specific_kwargs: set[str] | None = None,
) -> Applied[M]: ...

def apply_info_to_method(
    method: M | None = None,
    *,
    subjects: tuple[str, ...] = (),
    start: int = 0,
    stop: int = 100,
    specific_kwargs: set[str] | None = None,
) -> Applied[M] | Callable[[M], Applied[M]]:
    def decorator(method: M) -> Applied[M]:

        @wraps(method)
        def wrapped_method(self, *args, **kwargs):
            nonlocal subjects, start, stop, specific_kwargs
            new_kwargs = kwargs.copy()
            all_keys = list(method.__kwdefaults__.keys())
            
            if len(all_keys) == 0:
                raise UserWarning
            
            for i in range(start, min(stop, len(all_keys))):
                key = all_keys[i]
                if kwargs.get(key) is not None:
                    continue
                if specific_kwargs is not None \
                    and (key not in specific_kwargs):
                    continue

                new_kwargs[key] = self.info.__getitem__(
                    key, 
                    subjects=subjects,
                )

            return method(self, *args, **new_kwargs)
        
        return wrapped_method
    
    return decorator if (method is None) else decorator(method)