"""Helper utilities for AURA platform"""

import asyncio
import time
import uuid
from datetime import datetime
from functools import wraps
from typing import Any, Callable, TypeVar

import structlog

logger = structlog.get_logger()

T = TypeVar("T")


def generate_id(prefix: str = "") -> str:
    """Generate a unique identifier"""
    unique_part = uuid.uuid4().hex[:12]
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    if prefix:
        return f"{prefix}_{timestamp}_{unique_part}"
    return f"{timestamp}_{unique_part}"


def retry_async(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """Decorator for retrying async functions with exponential backoff"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            "Retry attempt",
                            function=func.__name__,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            delay=current_delay,
                            error=str(e),
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            "All retry attempts failed",
                            function=func.__name__,
                            error=str(e),
                        )

            raise last_exception

        return wrapper

    return decorator


class Timer:
    """Context manager for timing code execution"""

    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
        logger.debug(
            "Timer completed",
            operation=self.name,
            elapsed_ms=round(self.elapsed_ms, 2),
        )

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *args):
        return self.__exit__(*args)


def chunk_list(lst: list, chunk_size: int) -> list:
    """Split a list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def safe_get(dictionary: dict, keys: str, default: Any = None) -> Any:
    """Safely get nested dictionary value using dot notation"""
    keys_list = keys.split(".")
    result = dictionary
    for key in keys_list:
        try:
            result = result[key]
        except (KeyError, TypeError):
            return default
    return result
