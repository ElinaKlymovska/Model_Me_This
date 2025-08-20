"""
Timing utilities for profiling and performance measurement.
"""

import time
import functools
from typing import Optional, Dict, Any
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "operation", log: bool = True):
        """Initialize timer.
        
        Args:
            name: Name of the operation being timed
            log: Whether to log timing information
        """
        self.name = name
        self.log = log
        self.start_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        if self.log:
            logger.debug(f"Starting {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log results."""
        self.elapsed_time = time.time() - self.start_time
        
        if self.log:
            if exc_type is None:
                logger.debug(f"Completed {self.name} in {self.elapsed_time:.4f}s")
            else:
                logger.error(f"Failed {self.name} after {self.elapsed_time:.4f}s")
        
        return False  # Don't suppress exceptions
    
    @property
    def duration(self) -> float:
        """Get elapsed time in seconds."""
        if self.elapsed_time is None:
            raise RuntimeError("Timer not finished. Use as context manager.")
        return self.elapsed_time


class PerformanceProfiler:
    """Performance profiler for tracking multiple operations."""
    
    def __init__(self, name: str = "profiler"):
        """Initialize profiler.
        
        Args:
            name: Name of the profiler
        """
        self.name = name
        self.timings = {}
        self.start_times = {}
    
    def start(self, operation: str) -> None:
        """Start timing an operation.
        
        Args:
            operation: Name of the operation
        """
        self.start_times[operation] = time.time()
    
    def stop(self, operation: str) -> float:
        """Stop timing an operation.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Elapsed time in seconds
        """
        if operation not in self.start_times:
            raise ValueError(f"Operation '{operation}' was not started")
        
        elapsed = time.time() - self.start_times[operation]
        
        if operation not in self.timings:
            self.timings[operation] = []
        
        self.timings[operation].append(elapsed)
        del self.start_times[operation]
        
        return elapsed
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Dictionary with timing statistics
        """
        if operation not in self.timings or not self.timings[operation]:
            return {}
        
        times = self.timings[operation]
        
        return {
            'count': len(times),
            'total_time': sum(times),
            'mean_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_time': _calculate_std(times)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations.
        
        Returns:
            Dictionary mapping operation names to their statistics
        """
        return {op: self.get_stats(op) for op in self.timings}
    
    def print_summary(self) -> None:
        """Print a summary of all timing information."""
        print(f"\n=== {self.name} Performance Summary ===")
        
        for operation, stats in self.get_all_stats().items():
            if stats:
                print(f"\n{operation}:")
                print(f"  Count: {stats['count']}")
                print(f"  Total: {stats['total_time']:.4f}s")
                print(f"  Mean:  {stats['mean_time']:.4f}s")
                print(f"  Min:   {stats['min_time']:.4f}s")
                print(f"  Max:   {stats['max_time']:.4f}s")
                print(f"  Std:   {stats['std_time']:.4f}s")
        
        print("\n" + "=" * 40)


def _calculate_std(values: list) -> float:
    """Calculate standard deviation of a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Standard deviation
    """
    if len(values) < 2:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def time_function(func):
    """Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function with timing
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with Timer(func.__name__):
            return func(*args, **kwargs)
    return wrapper


@contextmanager
def profile_operation(name: str, log: bool = True):
    """Context manager for profiling operations.
    
    Args:
        name: Name of the operation
        log: Whether to log timing information
        
    Yields:
        Timer instance
    """
    timer = Timer(name, log)
    with timer:
        yield timer


def benchmark_function(func, *args, iterations: int = 100, **kwargs) -> Dict[str, float]:
    """Benchmark a function by running it multiple times.
    
    Args:
        func: Function to benchmark
        *args: Positional arguments for the function
        iterations: Number of iterations to run
        **kwargs: Keyword arguments for the function
        
    Returns:
        Dictionary with benchmark results
    """
    times = []
    
    # Warm up
    for _ in range(10):
        func(*args, **kwargs)
    
    # Benchmark
    for _ in range(iterations):
        start_time = time.time()
        func(*args, **kwargs)
        times.append(time.time() - start_time)
    
    # Calculate statistics
    total_time = sum(times)
    mean_time = total_time / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = _calculate_std(times)
    
    return {
        'iterations': iterations,
        'total_time': total_time,
        'mean_time': mean_time,
        'min_time': min_time,
        'max_time': max_time,
        'std_time': std_time,
        'throughput': iterations / total_time  # operations per second
    }


class MemoryProfiler:
    """Simple memory usage profiler."""
    
    def __init__(self, name: str = "memory_profiler"):
        """Initialize memory profiler.
        
        Args:
            name: Name of the profiler
        """
        self.name = name
        self.memory_usage = {}
    
    def __enter__(self):
        """Start memory profiling."""
        try:
            import psutil
            process = psutil.Process()
            self.memory_usage['start'] = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            self.memory_usage['start'] = 0
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End memory profiling."""
        try:
            import psutil
            process = psutil.Process()
            self.memory_usage['end'] = process.memory_info().rss / 1024 / 1024  # MB
            self.memory_usage['delta'] = self.memory_usage['end'] - self.memory_usage['start']
            
            logger.info(f"{self.name} memory usage: {self.memory_usage['delta']:+.2f} MB")
        except ImportError:
            pass
        
        return False
