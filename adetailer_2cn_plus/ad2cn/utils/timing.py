"""Simple timing utilities."""
import time
from contextlib import contextmanager

@contextmanager
def Timer(operation_name: str):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"{operation_name}: {end_time - start_time:.3f}s")