#!/usr/bin/env python3
"""
CUDA Replace Benchmark Suite
Compares GPU-accelerated replacement against Python's bytes.replace()
"""

import argparse
import time
import statistics
from typing import List, Tuple
import sys

try:
    from cuda_replace_wrapper import CudaReplaceLib, gpu_replace_streaming
except ImportError:
    print("ERROR: cuda_replace_wrapper.py not found in current directory")
    sys.exit(1)


def format_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def format_throughput(bytes_processed: int, seconds: float) -> str:
    """Format throughput as GB/s."""
    gb_per_sec = bytes_processed / seconds / (1024**3)
    return f"{gb_per_sec:.2f} GB/s"


class BenchmarkCase:
    def __init__(self, name: str, data: bytes, pattern: bytes, replacement: bytes):
        self.name = name
        self.data = data
        self.pattern = pattern
        self.replacement = replacement
        self.data_size = len(data)


def create_benchmarks() -> List[BenchmarkCase]:
    """Create diverse benchmark scenarios."""
    
    benchmarks = []
    
    # 1. Dense replacements (many matches)
    benchmarks.append(BenchmarkCase(
        "Dense/Small (1MB, shrink)",
        data=b"hello " * (1024 * 1024 // 6),
        pattern=b"hello",
        replacement=b"hi"
    ))
    
    # 2. Sparse replacements (few matches)
    base = b"x" * (10 * 1024 * 1024)
    sparse_data = base[:len(base)//2] + b"NEEDLE" + base[len(base)//2:]
    benchmarks.append(BenchmarkCase(
        "Sparse/Large (10MB, 1 match)",
        data=sparse_data,
        pattern=b"NEEDLE",
        replacement=b"FOUND"
    ))
    
    # 3. Expansion case
    benchmarks.append(BenchmarkCase(
        "Expansion (5MB, 2x growth)",
        data=b"abc" * (5 * 1024 * 1024 // 3),
        pattern=b"abc",
        replacement=b"ABCDEF"
    ))
    
    # 4. Medium density
    benchmarks.append(BenchmarkCase(
        "Medium/Mid (20MB, ~10% matches)",
        data=(b"normal text " * 100 + b"REPLACE ") * (20 * 1024 // 12),
        pattern=b"REPLACE",
        replacement=b"DONE"
    ))
    
    # 5. Large single pattern (stress test)
    benchmarks.append(BenchmarkCase(
        "Large/Dense (50MB, shrink)",
        data=b"aaabaaabaaab" * (50 * 1024 * 1024 // 12),
        pattern=b"aaa",
        replacement=b"X"
    ))
    
    # 6. Tiny pattern, huge data
    benchmarks.append(BenchmarkCase(
        "Huge/Sparse (100MB, shrink)",
        data=b"x" * (100 * 1024 * 1024),
        pattern=b"xx",
        replacement=b"y"
    ))
    
    return benchmarks


def benchmark_cpu(case: BenchmarkCase, iterations: int = 3) -> Tuple[float, float, bytes]:
    """Benchmark Python's bytes.replace()."""
    times = []
    result = None
    
    for _ in range(iterations):
        start = time.perf_counter()
        result = case.data.replace(case.pattern, case.replacement)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0, result


def benchmark_gpu_unified(lib: CudaReplaceLib, case: BenchmarkCase, iterations: int = 3) -> Tuple[float, float, bytes]:
    """Benchmark GPU one-shot mode."""
    times = []
    result = None
    
    # Warmup
    _ = lib.unified(case.data, case.pattern, case.replacement)
    
    for _ in range(iterations):
        start = time.perf_counter()
        result = lib.unified(case.data, case.pattern, case.replacement)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0, result


def benchmark_gpu_streaming(lib: CudaReplaceLib, case: BenchmarkCase, iterations: int = 3) -> Tuple[float, float, bytes]:
    """Benchmark GPU streaming mode."""
    times = []
    result = None
    
    # Warmup
    _ = gpu_replace_streaming(lib, case.data, [(case.pattern, case.replacement)])
    
    for _ in range(iterations):
        start = time.perf_counter()
        result = gpu_replace_streaming(lib, case.data, [(case.pattern, case.replacement)])
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0, result


def verify_correctness(cpu_result: bytes, gpu_result: bytes, case_name: str) -> bool:
    """Verify GPU matches CPU output."""
    if cpu_result == gpu_result:
        return True
    else:
        print(f"\n⚠️  WARNING: Results mismatch for {case_name}")
        print(f"   CPU result length: {len(cpu_result)}")
        print(f"   GPU result length: {len(gpu_result)}")
        if len(cpu_result) < 200 and len(gpu_result) < 200:
            print(f"   CPU: {cpu_result[:100]}")
            print(f"   GPU: {gpu_result[:100]}")
        return False


def print_header():
    """Print benchmark header."""
    print("=" * 100)
    print(f"{'CUDA REPLACE BENCHMARK':^100}")
    print("=" * 100)
    print(f"{'Benchmark':<30} | {'Size':<10} | {'CPU (ms)':<12} | {'GPU Unified (ms)':<18} | {'GPU Stream (ms)':<18} | {'Speedup':<10}")
    print("-" * 100)


def print_result(case: BenchmarkCase, cpu_time: float, cpu_std: float, 
                 gpu_time: float, gpu_std: float,
                 stream_time: float, stream_std: float,
                 correct: bool):
    """Print single benchmark result."""
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    status = "✓" if correct else "✗"
    
    print(f"{case.name:<30} | {format_bytes(case.data_size):<10} | "
          f"{cpu_time*1000:>6.2f} ±{cpu_std*1000:>4.1f} | "
          f"{gpu_time*1000:>8.2f} ±{gpu_std*1000:>4.1f} | "
          f"{stream_time*1000:>8.2f} ±{stream_std*1000:>4.1f} | "
          f"{speedup:>5.2f}x {status}")


def print_summary(results: List[Tuple[str, float, float, float]]):
    """Print summary statistics."""
    print("\n" + "=" * 100)
    print(f"{'SUMMARY':^100}")
    print("=" * 100)
    
    total_cpu = sum(r[1] for r in results)
    total_gpu = sum(r[2] for r in results)
    total_stream = sum(r[3] for r in results)
    total_bytes = sum(r[4] for r in results)
    
    print(f"Total data processed: {format_bytes(total_bytes)}")
    print(f"Total CPU time:       {total_cpu:.3f}s ({format_throughput(total_bytes, total_cpu)})")
    print(f"Total GPU time:       {total_gpu:.3f}s ({format_throughput(total_bytes, total_gpu)})")
    print(f"Total Stream time:    {total_stream:.3f}s ({format_throughput(total_bytes, total_stream)})")
    print(f"Average speedup:      {total_cpu/total_gpu:.2f}x (unified), {total_cpu/total_stream:.2f}x (streaming)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA Replace vs Python bytes.replace()")
    parser.add_argument("--dll", default="./cuda_replace.dll", help="Path to CUDA replace library")
    parser.add_argument("--iterations", type=int, default=3, help="Iterations per benchmark")
    parser.add_argument("--quick", action="store_true", help="Run subset of benchmarks")
    args = parser.parse_args()
    
    # Load library
    try:
        lib = CudaReplaceLib(args.dll)
    except Exception as e:
        print(f"ERROR: Failed to load library: {e}")
        sys.exit(1)
    
    # Create benchmarks
    benchmarks = create_benchmarks()
    if args.quick:
        benchmarks = benchmarks[:3]
    
    print_header()
    
    results = []
    all_correct = True
    
    for case in benchmarks:
        # CPU
        cpu_time, cpu_std, cpu_result = benchmark_cpu(case, args.iterations)
        
        # GPU Unified
        gpu_time, gpu_std, gpu_result = benchmark_gpu_unified(lib, case, args.iterations)
        
        # GPU Streaming
        stream_time, stream_std, stream_result = benchmark_gpu_streaming(lib, case, args.iterations)
        
        # Verify
        correct = verify_correctness(cpu_result, gpu_result, case.name)
        correct = correct and verify_correctness(cpu_result, stream_result, f"{case.name} (streaming)")
        all_correct = all_correct and correct
        
        # Print
        print_result(case, cpu_time, cpu_std, gpu_time, gpu_std, stream_time, stream_std, correct)
        
        # Store for summary
        results.append((case.name, cpu_time, gpu_time, stream_time, case.data_size))
    
    print_summary(results)
    
    if not all_correct:
        print("\n⚠️  Some correctness checks FAILED!")
        return 1
    else:
        print("\n✅ All correctness checks PASSED!")
        return 0


if __name__ == "__main__":
    sys.exit(main())