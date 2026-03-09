from typing import Callable, Iterable, Any, Union, Optional
from functools import wraps
from .setup.info import Info

def apply_info_to_func(
    func: Callable, 
    info: Info,
    subjects: Iterable[str],
) -> Callable:

    @wraps(func)
    def new_func(*args, **kwargs):
        
        new_kwargs: dict[str, Any] = kwargs.copy()
        for key, val in kwargs.items():
            if val is not None:
                continue
            new_kwargs[key] = info.__getitem__(key, subjects=subjects)

        return func(*args, **new_kwargs)

    return new_func

def apply_info_to_method(
    *subjects: Union[str, tuple[str]], 
    start: int = 0,
    stop: int = 100,
    specific_kwargs: Optional[set[str]] = None,
):
    
    if isinstance(subjects, str):
        subjects = (subjects,)

    def inner_decorator(method: Callable) -> Callable:
        
        @wraps(method)
        def wrapped_method(self, *args, **kwargs):

            new_kwargs = kwargs.copy()
            all_keys = list(method.__kwdefaults__.keys())

            if len(all_keys) == 0:
                raise UserWarning(
                    "'apply_info_to_method' decorator can not be used on" \
                    "method with no keyword-only arguments!"
                )
            
            for i in range(start, min([stop, len(all_keys)])):
                key = all_keys[i]
                if kwargs.get(key) is not None:
                    continue
                if (specific_kwargs is not None) \
                    and (key not in specific_kwargs):
                    continue

                new_kwargs[key] = self.info.__getitem__(key, subjects=subjects)

            return method(self, *args, **new_kwargs)
        
        wrapped_method.raw = method
        
        return wrapped_method
    
    return inner_decorator