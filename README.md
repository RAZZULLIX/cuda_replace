# CUDA Replace

High-throughput byte-pattern replacement on GPU. Python's `bytes.replace()` semantics without leaving device memory.

## What It Does

Performs leftmost, non-overlapping pattern replacement entirely on GPU. Designed for large-scale text processing, tokenization preprocessing, and streaming transformations.

**Use case:** Processing multi-GB files with repeated pattern replacement without CPU/GPU transfer overhead.

## Performance

- **Streaming mode**: Process unlimited file sizes with fixed GPU memory
- **Session reuse**: Amortize allocation costs across multiple operations
- **Fused pipeline**: Single-pass mark + suppress + scatter
- **Thread-safe**: Per-handle mutex allows safe concurrent access

## Installation

**Pre-built binaries:** See `bin/` directory

**Build from source:**
```bash
# Windows
nvcc -O3 -std=c++17 -m64 -Xcompiler "/MD" -shared -o cuda_replace.dll cuda_replace.cu

# Linux
nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC -o libcuda_replace.so cuda_replace.cu
```

## Usage

### One-Shot (Simple)

```python
from cuda_replace_wrapper import CudaReplaceLib

lib = CudaReplaceLib('./cuda_replace.dll')  # or .so on Linux

data = b"hello world hello universe"
result = lib.unified(data, b"hello", b"goodbye")
print(result)  # b"goodbye world goodbye universe"
```

### Session (Reusable GPU Buffer)

```python
from cuda_replace_wrapper import Session

with Session(lib, data) as sess:
    sess.apply(b"hello", b"hi")
    sess.apply(b"world", b"GPU")
    result = sess.result()
```

### Streaming (Large Files)

```python
from cuda_replace_wrapper import gpu_replace_streaming

# Process multi-GB files with limited GPU memory
with open('huge_file.txt', 'rb') as f:
    data = f.read()

pairs = [(b"pattern1", b"replacement1"), (b"pattern2", b"replacement2")]
result = gpu_replace_streaming(lib, data, pairs, chunk_bytes=256*1024*1024)
```

### Batch Operations

```python
with Session(lib, data) as sess:
    pairs = [
        (b"old1", b"new1"),
        (b"old2", b"new2"),
        (b"old3", b"new3")
    ]
    new_len = sess.apply_batch(pairs)
    result = sess.result()
```

## API Reference

### `CudaReplaceLib(dll_path: str)`

Loads the CUDA replace library.

**Methods:**
- `unified(src: bytes, pat: bytes, rep: bytes) -> bytes` - One-shot replacement

### `Session(lib, src: bytes)`

Persistent GPU session for efficient multi-operation workflows.

**Methods:**
- `apply(pat: bytes, rep: bytes) -> int` - Apply replacement, returns new length
- `apply_batch(pairs: List[Tuple[bytes, bytes]]) -> int` - Apply multiple patterns sequentially
- `apply_seeded(pat, rep, prev_last: int) -> (int, int)` - Apply with carry-in state for streaming
- `reset(src: bytes)` - Replace buffer contents (O(1) if fits capacity)
- `query_prefix(limit: int) -> (int, int)` - Count matches before position
- `result() -> bytes` - Retrieve result from GPU
- `length() -> int` - Get current buffer length
- `close()` - Free GPU resources

### `gpu_replace_streaming(lib, src, pairs, chunk_bytes=256MB) -> bytes`

Memory-bounded streaming processor. Handles files larger than GPU memory.

## Algorithm

1. **Mark Phase**: SIMD pattern matching across input (AVX2-style on GPU)
2. **Suppress Phase**: Leftmost greedy selection (non-overlapping semantics)
3. **Scatter Phase**: 
   - **Shrink**: Compact output when `len(replacement) <= len(pattern)`
   - **Expand**: Prefix-sum scatter when `len(replacement) > len(pattern)`

**Optimizations:**
- Fused mark+suppress in shared memory for small patterns
- Tiled prefix computation (Blelloch scan)
- Range-sliced expansion to avoid GPU TDR timeouts
- Presence index for sparse patterns (optional)

## Edge Cases

- **Handles embedded NULs** (binary-safe)
- **Denormal floats** (FTZ/DAZ enabled)
- **Thread-safe sessions** (internal mutex per handle)
- **Overflow protection** (streaming mode prevents 32-bit expansion overflow)

## Files

- `cuda_replace.cu` - CUDA kernel implementation
- `cuda_replace_wrapper.py` - Python ctypes interface
- `bin/` - Pre-compiled libraries
- `test/` - Usage examples

## Example: Tokenizer Preprocessing

```python
# Replace special tokens in bulk before tokenization
text = load_huge_corpus()  # 10GB text file

replacements = [
    (b"<br>", b"\n"),
    (b"&nbsp;", b" "),
    (b"&amp;", b"&"),
    # ... 100+ HTML entities
]

cleaned = gpu_replace_streaming(lib, text, replacements)
```

## Benchmarking

```python
import time

data = b"a" * (1024 * 1024 * 1024)  # 1GB of 'a's
pat = b"aa"
rep = b"X"

# CPU baseline
start = time.perf_counter()
cpu_result = data.replace(pat, rep)
cpu_time = time.perf_counter() - start

# GPU
start = time.perf_counter()
gpu_result = lib.unified(data, pat, rep)
gpu_time = time.perf_counter() - start

print(f"CPU: {cpu_time:.3f}s")
print(f"GPU: {gpu_time:.3f}s")
print(f"Speedup: {cpu_time/gpu_time:.2f}x")
```

## Requirements

- CUDA-capable GPU (compute capability 3.5+)
- CUDA Toolkit 11.0+ (for building)
- Python 3.7+ (for wrapper)

## License

MIT
