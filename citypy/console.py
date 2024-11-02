import sys
from contextlib import contextmanager
from functools import wraps
from inspect import Parameter, signature
from unittest.mock import patch

import geopandas
import momepy
import pandas
from rich.console import Console
from rich.status import Status
from tqdm import tqdm
from tqdm.rich import tqdm as tqdm_rich

console = Console(file=sys.stderr)


@contextmanager
def conditional_status(verbose: bool, message="Processing..."):
    if verbose:
        with patch("tqdm.auto.tqdm", new="tqdm.rich.tqdm") as _:
            with Status(message, console=console) as status:
                yield status
    else:
        # Yield a dummy context that does nothing
        class NoOpContext:
            def update(self, message: str):
                # No operation
                pass

        with patch("tqdm.auto.tqdm", new="tqdm.rich.tqdm") as _:
            yield NoOpContext()


def display_status_if_verbose(message=None):
    if callable(message):
        raise ValueError(
            "The 'message' argument must be a string, not a function. Did you forget to use parentheses?"
        )

    def decorator(func):
        # Check if 'verbose' is in the function signature
        sig = signature(func)
        if (
            "verbose" not in sig.parameters
            or sig.parameters["verbose"].default is Parameter.empty
        ):
            raise ValueError(
                f"Function '{func.__name__}' must have a 'verbose' keyword argument with a default value."
            )

        @wraps(func)
        def wrapper(*args, **kwargs):
            verbose = kwargs.get("verbose", True)
            _ = verbose or not verbose  # Dummy operation to utilize 'verbose'

            # Use the custom message if provided, otherwise default to function name
            func_name = func.__name__.replace("_", " ").capitalize()
            status_message = (
                message if message is not None else f"Running {func_name}..."
            )

            with conditional_status(verbose, message=status_message):
                result = func(*args, **kwargs)
                return result

        return wrapper

    return decorator
