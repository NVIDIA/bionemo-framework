def _is_timing_disabled():
    """Check if timing/profiling is disabled via environment variable."""
    import os
    return os.getenv("DBG_TIMING_DISABLE") == "1"


# Keep as single line to be able to copy-paste it easily
def dbg(): import torch, inspect; f=inspect.currentframe().f_back; (torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank()!=0) or print(f"====== {f.f_code.co_filename}:{f.f_lineno} {f.f_code.co_name} cualloc={torch.cuda.memory_allocated()/1024/1024:.1f}MB")

def sz(t): return t.nelement() * t.element_size() // 1024 // 1024

def print_(*args, **kwargs):
    import torch
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        print(rank, *args, **kwargs)
    else:
        print(*args, **kwargs)

def print_gpu_memory():
    import torch
    if not torch.cuda.is_available():
        print_("CUDA is not available.")
        return

    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            free_mem, total_mem = torch.cuda.mem_get_info()
            print_(f"GPU {i}: Free Memory: {free_mem / 1e9:.2f} GB, Total Memory: {total_mem / 1e9:.2f} GB")


def check_cuda_launch_blocking():
    """Check that CUDA_LAUNCH_BLOCKING=1 is set, exit if not."""
    import os
    if os.getenv("CUDA_LAUNCH_BLOCKING") != "1":
        print("Error: CUDA_LAUNCH_BLOCKING=1 must be set for accurate profiling")
        exit(1)


def miniprof(fn, *args, **kwargs):
    from os import getenv
    enabled = getenv("MINIPROF", getenv("NIM_MINIPROF"))
    if not enabled:
        return

    if _is_timing_disabled():
        return

    check_cuda_launch_blocking()

    fn(*args, **kwargs) # warmup at given arguments

    import cProfile
    import pstats
    from pstats import SortKey
    with cProfile.Profile() as profile:
        for _ in range(3):
            fn(*args, **kwargs)
    stats = pstats.Stats(profile)
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(100)

    import torch
    from torch.profiler import profile, ProfilerActivity, record_function
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        #profile_memory=True
    ) as prof:
        with record_function("miniprof"):
            for _ in range(3):
                fn(*args, **kwargs)
            torch.cuda.synchronize()

    def p(s, d=""): torch.distributed.barrier(); print("rank", torch.distributed.get_rank(), d+"\n"+s, flush=True)

    p(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20), "cpu")
    p(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20), "cuda")
    p(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total", row_limit=40), "cpu")
    p(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=40), "cuda")

    if enabled == "pdb":
        p("pdb...")
        torch.distributed.breakpoint(rank=0)
    elif enabled == "exit":
        exit(0)


def _get_caller_frame_for_context():
    """Get the appropriate caller frame for context manager usage."""
    import inspect

    frame = inspect.currentframe()
    caller_frame = None

    # Try to find the right caller frame by going back through the stack
    current_frame = frame
    for _ in range(5):  # Look back up to 5 frames
        if current_frame and current_frame.f_back:
            current_frame = current_frame.f_back
            # Skip frames that are part of our timing infrastructure
            code_name = current_frame.f_code.co_name
            filename = current_frame.f_code.co_filename

            # Skip our own timing functions and the TimingWrapper
            if (code_name not in ['context_timing', 'timing', '__enter__', '__call__'] and
                not filename.endswith('dbg.py')):
                caller_frame = current_frame
                break
            # Also check if this frame looks like actual user code
            elif code_name not in ['context_timing', 'timing', '__enter__', '__call__']:
                if not caller_frame:  # Use as fallback
                    caller_frame = current_frame

    if not caller_frame:
        caller_frame = frame.f_back if frame else None

    return caller_frame


def _get_class_name_from_frame(frame):
    """Extract class name from a frame's local variables."""
    if not frame:
        return None

    try:
        # Look for 'self' or 'cls' in local variables
        if 'self' in frame.f_locals:
            self_obj = frame.f_locals['self']
            if hasattr(self_obj, '__class__'):
                return self_obj.__class__.__name__
        elif 'cls' in frame.f_locals:
            cls_obj = frame.f_locals['cls']
            if hasattr(cls_obj, '__name__'):
                return cls_obj.__name__
    except:
        pass

    return None


def _get_function_identifier_from_frame(frame):
    """Get function identifier (Class.function or just function) from a frame."""
    if not frame:
        return None

    # Create cache key from frame info
    cache_key = (frame.f_code.co_filename, frame.f_code.co_firstlineno, frame.f_code.co_name)

    if '_function_identifier_cache' not in globals():
        globals()['_function_identifier_cache'] = {}

    cache = globals()['_function_identifier_cache']

    # Check cache first
    if cache_key in cache:
        return cache[cache_key]

    try:
        func_name = frame.f_code.co_name
        class_name = _get_class_name_from_frame(frame)

        if class_name and func_name != '<module>':
            func_identifier = f"{class_name}.{func_name}"
        elif func_name != '<module>':
            func_identifier = func_name
        else:
            func_identifier = None

        cache[cache_key] = func_identifier
        return func_identifier
    except:
        cache[cache_key] = None
        return None


def _get_class_name_from_function(func, func_args):
    """Extract class name from function and its arguments."""
    import inspect

    # Create cache key from function
    cache_key = (id(func), func.__name__, func.__qualname__)

    if '_class_name_cache' not in globals():
        globals()['_class_name_cache'] = {}

    cache = globals()['_class_name_cache']

    if cache_key in cache:
        return cache[cache_key]

    class_name = None

    try:
        # First try to get class from qualname (works for methods defined in class)
        if '.' in func.__qualname__:
            parts = func.__qualname__.split('.')
            if len(parts) >= 2:
                class_name = parts[-2]

        # If that didn't work, check if first argument looks like self/cls
        if not class_name and func_args:
            first_arg = func_args[0]
            if hasattr(first_arg, '__class__'):
                # Check if this function is actually a method of the object's class
                if hasattr(first_arg.__class__, func.__name__):
                    method = getattr(first_arg.__class__, func.__name__)
                    # Verify it's the same function (unwrap if needed)
                    if hasattr(method, '__func__'):
                        if method.__func__ is func:
                            class_name = first_arg.__class__.__name__
                    elif method is func:
                        class_name = first_arg.__class__.__name__
                # Also handle case where first arg is a class (classmethod)
                elif inspect.isclass(first_arg):
                    if hasattr(first_arg, func.__name__):
                        class_name = first_arg.__name__
    except:
        pass

    cache[cache_key] = class_name
    return class_name


def _get_function_identifier_from_function(func, func_args):
    """Get function identifier (Class.function or just function) from function and args."""
    func_name = func.__name__
    class_name = _get_class_name_from_function(func, func_args)

    if class_name:
        return f"{class_name}.{func_name}"
    else:
        return func_name


from contextlib import contextmanager
@contextmanager
def context_timing(*args, tag_separator="-", n=100, skip=0):
    """
    Context manager that times scoped code execution time.

    Args:
        *args: Arguments to be converted to strings and concatenated with tag_separator
        tag_separator: String to use for joining tag components (default: "-")
        n: Number of recent execution times to keep (default: 100)
        skip: Number of initial executions to skip before printing stats (default: 3)
    """
    if _is_timing_disabled():
        yield
        return

    import time
    import statistics
    from collections import defaultdict

    check_cuda_launch_blocking()

    caller_frame = _get_caller_frame_for_context()
    func_identifier = _get_function_identifier_from_frame(caller_frame)

    tag_parts = [str(arg) for arg in args]

    if func_identifier:
        if tag_parts:
            # Format as <Class>.<Function>-<tags>
            tags_str = tag_separator.join(tag_parts)
            tag = tag_separator.join([func_identifier, tags_str])
        else:
            tag = func_identifier
    else:
        tag = tag_separator.join(tag_parts) if tag_parts else "unknown"

    if '_timing_stats' not in globals():
        globals()['_timing_stats'] = defaultdict(lambda: {'times': [], 'count': 0})

    stats = globals()['_timing_stats'][tag]

    start_time = time.perf_counter()

    try:
        yield
    finally:
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        stats['count'] += 1
        stats['times'].append(execution_time_ms)

        # Keep only the last n times
        if len(stats['times']) > n:
            stats['times'] = stats['times'][-n:]

        # Print statistics if we've passed the skip threshold
        if stats['count'] > skip and len(stats['times']) > 0:
            times = stats['times']
            mean_time = statistics.mean(times)
            max_time = max(times)
            min_time = min(times)
            stdev_time = statistics.stdev(times) if len(times) > 1 else 0.0

            print_(f"Timing[{tag}]: mean={mean_time:.2f}ms, max={max_time:.2f}ms, min={min_time:.2f}ms, stdev={stdev_time:.2f}ms (n={len(times)})")


def decorator_timing(*args, tag_separator="-", n=100, skip=0):
    """
    Function decorator that times function execution time.

    Args:
        *args: Arguments to be converted to strings and concatenated with tag_separator
        tag_separator: String to use for joining tag components (default: "-")
        n: Number of recent execution times to keep (default: 100)
        skip: Number of initial executions to skip before printing stats (default: 0)
    """
    def decorator(func):
        import functools
        @functools.wraps(func)
        def wrapper(*func_args, **func_kwargs):
            if _is_timing_disabled():
                return func(*func_args, **func_kwargs)

            check_cuda_launch_blocking()

            func_identifier = _get_function_identifier_from_function(func, func_args)

            if args:
                # If additional tags are present, format as <Class>.<Function>-tags
                tags_str = tag_separator.join(str(arg) for arg in args)
                tag = tag_separator.join([func_identifier, tags_str])
            else:
                tag = func_identifier

            with context_timing(tag, tag_separator=tag_separator, n=n, skip=skip):
                return func(*func_args, **func_kwargs)

        return wrapper
    return decorator


def timing(*args, **kwargs):
    """
    Backward-compatible timing function that can be used as both context manager and decorator.

    Usage:
        # As context manager:
        with timing("my_code"):
            # code here

        # As decorator:
        @timing("my_function")
        def my_function():
            # code here
    """
    # If called with a single callable argument and no other args/kwargs,
    # it's being used as a decorator without parentheses
    if len(args) == 1 and callable(args[0]) and not kwargs:
        func = args[0]
        return decorator_timing()(func)

    # If we have a callable in args along with other arguments,
    # it's likely being used incorrectly, so treat as context manager
    if any(callable(arg) for arg in args):
        return context_timing(*args, **kwargs)

    # Check if this is being used as a decorator factory or context manager
    # by looking at the call stack
    import inspect
    frame = inspect.currentframe()
    try:
        if frame and frame.f_back:
            # Get the code context to see how we're being called
            caller_frame = frame.f_back
            # This is a heuristic - if we're called and the result is immediately used
            # in a 'with' statement or assigned, it's likely a context manager
            # Otherwise, it's likely a decorator

            # For now, we'll default to returning a context manager
            # but allow decorator usage when the return value is called
            class TimingWrapper:
                def __init__(self, *args, **kwargs):
                    self.args = args
                    self.kwargs = kwargs

                def __enter__(self):
                    self.cm = context_timing(*self.args, **self.kwargs)
                    return self.cm.__enter__()

                def __exit__(self, *args):
                    return self.cm.__exit__(*args)

                def __call__(self, func):
                    return decorator_timing(*self.args, **self.kwargs)(func)

            return TimingWrapper(*args, **kwargs)
    finally:
        del frame

    # Fallback to context manager
    return context_timing(*args, **kwargs)
