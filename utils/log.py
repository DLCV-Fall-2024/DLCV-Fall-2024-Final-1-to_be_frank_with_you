import inspect
import time
from collections import defaultdict, deque

import numpy as np
import torch

try:
    import wandb

    __WANDB_AVAILABLE__ = True
except ImportError:
    __WANDB_AVAILABLE__ = False

__WANDB_ENABLED__ = False


def init_wandb(project_name, run_name, config, log_dir=None, local_rank=None, **kwargs):
    global __WANDB_ENABLED__
    if not __WANDB_AVAILABLE__ or (isinstance(local_rank, int) and local_rank != 0):
        return
    wandb.init(
        project=project_name, name=run_name, config=config, dir=str(log_dir), **kwargs
    )
    wandb.config.update(config)

    __WANDB_ENABLED__ = True
    print("========= Wandb initialized =========")


def init_logger(verbose=True, local_rank=None):
    if not __WANDB_ENABLED__:
        return print if verbose else lambda *args, **kwargs: None
    if (isinstance(local_rank, int) and local_rank != 0) or not verbose:
        return lambda *args, **kwargs: None
    return wandb.log


def pretty_print(elem, indent=0, after_key=False, max_depth=5, expand_tensors=False):
    prefix = "  " * indent
    if indent >= max_depth:
        print(prefix + "(exceeded max depth)")
        return
    if isinstance(elem, dict):
        if after_key:
            print("{")
        else:
            print(prefix + "{")
        for k, v in elem.items():
            print(prefix + f"  {k}: ", end="")
            pretty_print(v, indent + 1, after_key=True)
        print(prefix + "}")
    elif isinstance(elem, list):
        if after_key:
            print("[")
        else:
            print(prefix + "[")
        for v in elem:
            pretty_print(v, indent + 1)
        print(prefix + "]")
    else:
        printable_text = ""
        if isinstance(elem, (torch.Tensor, np.ndarray)) and not expand_tensors:
            printable_text = f"{type(elem)}: {elem.shape}, {elem.dtype}"
        else:
            printable_text = repr(elem)

        if after_key:
            print(printable_text)
        else:
            print(prefix + printable_text)


__PRINT_ONCE__ = set()


def print_once(*args, **kwargs):
    if args in __PRINT_ONCE__:
        return
    __PRINT_ONCE__.add(args)
    print(*args, **kwargs)


def get_caller_info(depth=2):
    """
    Get the file name and line number of the caller.
    """
    caller_stack = inspect.stack()
    caller_info = caller_stack[depth]
    return f"{caller_info.filename}:{caller_info.lineno}"


class Timer:
    def __init__(self, limit=0):
        """
        Timer class to track elapsed time.
        :param limit: Time limit in seconds. If set, `timesup` will return True when elapsed time exceeds the limit.
        """
        self.limit = limit
        self.start_time = None
        self.end_time = None
        self.start()

    def start(self):
        self.start_time = time.time()

    def restart(self, limit=None):
        self.start()
        self.end_time = None
        if limit is not None:
            self.limit = limit

    def reset(self):
        self.restart()

    def stop(self):
        self.end_time = time.time()

    def elapsed(self):
        if self.start_time is None:
            raise RuntimeError("Timer has not been started.")
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def timesup(self):
        if self.limit is None or self.limit <= 0:
            return False
        return self.elapsed() > self.limit

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class PerformanceMonitor:
    def __init__(self, debug=True, ema_decay=0.9, threshold=1.2, horizon_size=10):
        """
        Performance monitor class.

        :param ema_decay: Decay factor for EMA computation.
        :param threshold: Warning threshold (e.g., 1.2 means 20% over EMA triggers).
        :param horizon_size: Number of past execution times to maintain for analysis.
        """
        self.debug = debug
        self.ema_decay = ema_decay
        self.threshold = threshold
        self.horizon_size = horizon_size
        self.histories = defaultdict(
            lambda: deque(maxlen=self.horizon_size)
        )  # Separate history for each context.
        self.ema = defaultdict(lambda: None)  # Track EMA for each context key
        self.start_time = None  # To record when entering context.

        self.prev_stamp = None  # Store the previous time stamp
        self.stamps = defaultdict(
            lambda: deque(maxlen=self.horizon_size)
        )  # Store time stamps from `stamp`
        self.context_key = None
        self.params = defaultdict(lambda: deque(maxlen=10))

        self.logging_counter = defaultdict(lambda: 0)

    def set_params(self, **kwargs):
        """
        Set parameters that will be used for context tracking and reporting.
        """
        if not self.debug:
            return
        for key, value in kwargs.items():
            self.params[key].append(value)

    def log_warning(self, recent_time, key=None):
        """
        Logs the warning if execution time exceeds threshold.
        :param key: Unique key identifying the context.
        :param recent_time: The most recent execution time.
        """
        if not self.debug:
            return
        if key is None:
            key = self.context_key

        print(f"Performance warning triggered for context '{key}'!")
        print(f"Execution time: {recent_time:.4f} seconds exceeds EMA by 20%")
        print(f"Tracked params: {self.params}")
        print(f"Recent execution times: {list(self.histories[key])}")

        stamps = {k.split(":")[-1]: list(v) for k, v in self.stamps.items() if key in k}
        print(f"Time-stamps stored for context: {stamps}")

    def log_performance(self, log_per=10):
        """
        Logs the performance statistics for the context.
        :param key: Unique key identifying the context.
        """
        if not self.debug:
            return
        if self.context_key is None:
            raise RuntimeError(
                "Context key is not set. Call within an active context to log warning."
            )
        key = self.context_key
        self.logging_counter[key] += 1
        if self.logging_counter[key] % log_per != 0:
            return
        print(f"Performance statistics for context '{key}'")
        print(f"EMA: {self.ema[key]:.4f}")
        print(f"Recent execution times: {list(self.histories[key])}")
        stamps = {k.split(":")[-1]: list(v) for k, v in self.stamps.items() if key in k}
        print(f"Time-stamps stored for context: {stamps}")

    def stamp(self):
        """
        Function to capture the time from entering the context to the current time.
        This allows the monitoring to capture a "time-stamped" delay.
        These values are stored internally.
        """
        if not self.debug:
            return
        # Get the calling context (the function directly calling `stamp`)
        key = get_caller_info()

        if self.start_time is None:
            raise RuntimeError(
                "Context has not started. Call within an active context."
            )

        now = time.time()
        elapsed_time = now - self.start_time
        duration_to_prev = (
            now - self.prev_stamp if self.prev_stamp is not None else elapsed_time
        )
        self.prev_stamp = now
        self.stamps[f"{self.context_key}-{key}"].append(
            (elapsed_time, duration_to_prev)
        )  # Store time-stamp into the corresponding history
        return elapsed_time

    def __enter__(self):
        """
        Entering context. Record start time.
        """
        self.context_key = get_caller_info()

        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context. Evaluate execution time and determine if threshold has been exceeded.
        This tracks histories dynamically and identifies its calling context.
        """
        if not self.debug:
            self.context_key = None
            return
        if not self.start_time:
            raise RuntimeError("Context was improperly used without starting properly.")
        elapsed_time = time.time() - self.start_time
        # Determine the key uniquely for this context
        key = self.context_key

        # Update history
        history = self.histories[key]
        if len(history) >= self.horizon_size:
            history.pop()  # Maintain sliding window horizon
        history.append(elapsed_time)

        # Update EMA
        prev_ema = self.ema[key]
        if prev_ema is None:
            self.ema[key] = elapsed_time  # Set first EMA value directly
        else:
            self.ema[key] = (
                self.ema_decay * prev_ema + (1 - self.ema_decay) * elapsed_time
            )

        # Check if the latest execution time is over the EMA threshold
        if elapsed_time > self.threshold * self.ema[key]:
            self.log_warning(elapsed_time, key=key)

        self.context_key = None
