import functools
import time

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter_ns() 
        value = func(*args, **kwargs)
        end_time = time.perf_counter_ns()      
        run_time = end_time - start_time
        run_time = (end_time - start_time)/10e6
        print(f"[PERF] Finished {func.__name__!r} in {run_time:.4f} ms")
        return value
    return wrapper_timer