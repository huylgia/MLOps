import timeit
from typing import Callable, Mapping, Any

def measure_execute_time(func: Callable, args: Mapping[str, Any], loop: int):
    result = timeit.timeit(lambda: func(**args), number=loop)

    return result/loop
