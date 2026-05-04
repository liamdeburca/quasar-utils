from pathlib import Path
from typing import Callable, Literal
from pandas import DataFrame
from cProfile import Profile
from pstats import Stats
from collections import defaultdict
from functools import wraps

_this_file: Path = Path(__file__)
path_to_module: Path = _this_file.parents[2]

print(f"{path_to_module=}")

type Metrics = Literal[
    "function", "filename", "line_number", 
    "ncalls", 
    "tottime", "cumtime", 
    "tottime_per_call", "cumtime_per_call",
]

ALL_METRICS: set[Metrics] = {
    "function", "filename", "line_number", 
    "ncalls", 
    "tottime", "cumtime", 
    "tottime_per_call", "cumtime_per_call",
}

def profile_function(metrics_to_include: set[Metrics] = ALL_METRICS):

    def inner_decorator(func: Callable):

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            with Profile() as profile:
                func_result = func(*args, **kwargs)

            stats = Stats(profile)
            results = defaultdict(list)
            for key, value in stats.stats.items():
                filename, line_num, func_name = key
                _, ncalls, tottime, cumtime, _ = value

                for metric in metrics_to_include:
                    match metric:
                        case 'function': val = func_name
                        case 'filename': val = filename
                        case 'line_number': val = line_num
                        case 'ncalls': val = ncalls
                        case 'tottime': val = tottime
                        case 'cumtime': val = cumtime
                        case 'tottime_per_call': val = tottime / ncalls
                        case 'cumtime_per_call': val = cumtime / ncalls

                    results[metric].append(val)

            profile_df = DataFrame(results)

            return func_result, profile_df
        
        # Provides access to original function
        wrapped_func.raw = func
        
        return wrapped_func
    
    return inner_decorator