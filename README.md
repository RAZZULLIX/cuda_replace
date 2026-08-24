# CUDA Replace

High-throughput **byte-pattern replacement on the GPU**, with semantics that are
byte-for-byte identical to Python's `bytes.replace()` — leftmost, non-overlapping
matches, binary-safe (embedded NULs handled), and no CPU/GPU round-trip of the
whole buffer when you reuse a session.

Designed for large-scale text / tokenization preprocessing, bulk HTML-entity
cleaning, and any workload where you replace many patterns in multi-GB buffers.

```
┌─────────────┐   H2D once    ┌──────────────────────────────┐   D2H once   ┌─────────────┐
│  host bytes │ ────────────▶ │  GPU session (d_buf)         │ ───────────▶ │ result bytes │
└─────────────┘               │  apply() / apply_batch() × N  │              └─────────────┘
                              └──────────────────────────────┘
```

---

## Requirements

- A CUDA-capable GPU. The build covers compute capability **7.5 → 12.1** (plus PTX
  for future GPUs), so it runs on Turing (RTX 20xx), Ampere (RTX 30xx), Ada
  (RTX 40xx), Hopper, and Blackwell (RTX 50xx) cards.
- CUDA Toolkit **12.0+** (`nvcc`). Tested with CUDA 13.3.
- Python **3.7+** (wrapper is pure stdlib `ctypes`; no third-party deps).

---

## Installation

### Linux (recommended)

```bash
bash build.sh
```

(`chmod +x build.sh` lets you run it as `./build.sh` too.)

This produces `libcuda_replace.so` as a **fat binary** covering
`sm_75, sm_80, sm_86, sm_89, sm_90, sm_120, sm_121` plus PTX for forward
compatibility — one `.so` that runs on any supported GPU. `build.sh` is the
canonical build; edit the `GENS` array if you want to add/remove architectures.

### Windows

```bash
nvcc -O3 -std=c++17 -m64 -Xcompiler "/MD" -shared -o cuda_replace.dll cuda_replace.cu
```

### macOS

```bash
nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC -o libcuda_replace.dylib cuda_replace.cu
```

---

## Quick start

```python
from cuda_replace_wrapper import CudaReplaceLib, Session

lib = CudaReplaceLib('./libcuda_replace.so')   # or .dll on Windows

# One-shot
print(lib.unified(b"hello world hello universe", b"hello", b"goodbye"))
# b"goodbye world goodbye universe"

# Session (reusable GPU buffer — amortizes the H2D/D2D transfer)
with Session(lib, data) as sess:
    sess.apply(b"hello", b"hi")
    sess.apply(b"world", b"GPU")
    result = sess.result()
```

---

## API reference

### `CudaReplaceLib(dll_path: str)`

Loads the shared library and binds all symbols.

| Method | Returns | Description |
|---|---|---|
| `unified(src, pat, rep)` | `bytes` | One-shot replacement (open → apply → result → close). |
| `device_count()` | `int` | Number of CUDA devices visible to the process. |
| `set_device(device_id)` | `None` | Bind the calling thread to a device (for multi-GPU use). |

### `Session(lib, src: bytes)`

A persistent GPU-resident buffer. **Reuse one Session across many operations** to
amortize the host→device upload and device→host download; that is where the big
speedup lives.

| Method | Returns | Description |
|---|---|---|
| `apply(pat, rep)` | `int` | Replace `pat` with `rep` in-place on the GPU; returns new length. |
| `apply_batch(pairs)` | `int` | Apply a list of `(pat, rep)` sequentially (single transfer). |
| `apply_seeded(pat, rep, prev_last_rel)` | `(int, int)` | Low-level streaming primitive: apply with a carry-in "last kept start". |
| `query_prefix(limit)` | `(int, int)` | Low-level: count and last kept-start index strictly before `limit`. |
| `reset(src)` | `None` | Replace buffer contents (O(1) if it fits capacity). |
| `build_index(chunk_size=None)` | `None` | Optional presence index (no-op if unsupported). |
| `result()` | `bytes` | Download the current buffer to host as `bytes`. |
| `length()` | `int` | Current buffer length. |
| `close()` | `None` | Free GPU resources (also called by `with`). |

`Session` is a context manager and safe to share across threads for *distinct*
sessions; each handle has an internal mutex.

### `gpu_replace_streaming(lib, src, pairs, chunk_bytes=256*1024*1024) -> bytes`

Memory-bounded replacement for buffers **larger than GPU memory**. It splits the
input at match-free boundaries, processes each chunk independently, and
concatenates — so the result is still bit-for-bit identical to `bytes.replace`.
`chunk_bytes` bounds the per-chunk GPU footprint.

### `gpu_replace_multicard(lib, src, pairs, devices=None, chunk_bytes=256MB, oversplit=8) -> bytes`

Parallel multi-GPU replacement. Splits the buffer into independent segments and
processes them concurrently across the given CUDA devices (one worker thread per
device, each holding a persistent Session). Bit-for-bit identical to
`bytes.replace`.

```python
from cuda_replace_wrapper import gpu_replace_multicard

result = gpu_replace_multicard(lib, data, [(b"old", b"new")], devices=[0, 1, 2])
```

### `replace_unified(lib, src, pat, rep) -> bytes`

Thin alias for `lib.unified(...)`.

---

## Batch vs. single-shot (read this before benchmarking)

The GPU **compute** (mark + suppress + scatter) is much faster than CPython's
`bytes.replace`. The wall-clock win depends on whether the CPU↔GPU transfer is
amortized:

| Workload | Typical result |
|---|---|
| **Batch** — many `(pat, rep)` pairs on one buffer via `Session`/`apply_batch` | **~13× faster** than the equivalent `bytes.replace` chain (one H2D + one D2D) |
| **Session reuse** — many `apply()` calls, one `result()` | large win (same amortization) |
| **Single one-shot** `unified()` on CPU-resident data | roughly parity — dominated by the D2D + creating the output `bytes`, not by the kernel |

The floor for *returning* a fresh `bytes` of size `N` is the host memory
allocation (page faults), which `bytes.replace` pays once too. That's why the
single-shot case isn't "way faster", but batching is.

Numbers measured on an RTX 3090, enwik8 (100 MB), 100 two-byte patterns:

```
CPU  bytes.replace chain   : ~15.3 s
GPU  Session.apply_batch   : ~1.1 s   (13.4×)
```

---

## Correctness

- **Byte-for-byte identical to `bytes.replace`** for every input we tested:
  randomized fuzz, enwik8 (100 MB, 8 patterns), enwik9 (1 GB via
  streaming/multicard), dense overlapping patterns, and binary/NUL payloads.
- **Binary-safe**: embedded NULs and arbitrary bytes are fine.
- **Empty pattern** (`pat == b""`) matches Python (`rep` inserted between every
  byte).
- **Leftmost, non-overlapping** semantics — e.g. `b"aaaaa".replace(b"aaa", b"X")`
  yields `b"XaX"`, not `b"XXa"`.

### Algorithm

1. **Mark** — SIMD pattern matching across the buffer.
2. **Suppress** — leftmost greedy selection (non-overlapping).
3. **Scatter** — compact output (shrink) or prefix-sum expansion (expand).

The default path is the **legacy global-bitmap** pipeline (proven correct and
fast). A fused tiled path exists behind `CUDA_REPLACE_FUSED=1` but is
**experimental** — it has a known cross-tile carry bug on dense patterns and is
disabled by default.

---

## Testing

A quick smoke test after building:

```python
import cuda_replace_wrapper as W
lib = W.CudaReplaceLib('./libcuda_replace.so')
assert lib.unified(b"hello world hello", b"hello", b"goodbye") == b"goodbye world goodbye"
assert lib.unified(b"aaaaa", b"aaa", b"X") == b"XaX"
print("ok")
```

`unified()` is byte-identical to Python's `bytes.replace()` for every input we
tested — including dense/overlapping patterns, multi-GB buffers, and binary
payloads.

---

## Files

| File | Purpose |
|---|---|
| `cuda_replace.cu` | CUDA kernels + exported C API (the whole library). |
| `cuda_replace_wrapper.py` | Pure-stdlib `ctypes` Python wrapper. |
| `build.sh` | Linux multi-arch fat-binary build. |
| `README.md` | This file. |
| `LICENSE` | MIT license. |

---

## C API (for non-Python callers)

The `.so`/`.dll` exports a flat C ABI (declared in `cuda_replace.cu`):

- `int cuda_replace_unified(const uint8_t* src, size_t src_len, const uint8_t* pat, int pat_len, const uint8_t* rep, int rep_len, uint8_t** out_host, size_t* out_len)`
- `int cuda_replace_open(void** h, const uint8_t* src, size_t len)`
- `int cuda_replace_reset(void* h, const uint8_t* src, size_t len)`
- `int cuda_replace_apply(void* h, const uint8_t* pat, int pat_len, const uint8_t* rep, int rep_len, size_t* new_len)`
- `int cuda_replace_apply_batch(void* h, const uint8_t** pats, const int* pat_lens, const uint8_t** reps, const int* rep_lens, int count, size_t* new_len)`
- `int cuda_replace_build_index(void* h, size_t chunk_size)`
- `int cuda_replace_apply_seeded(void* h, const uint8_t* pat, int pat_len, const uint8_t* rep, int rep_len, long long prev_last_rel, long long* out_last_rel, size_t* new_len)`
- `int cuda_replace_query_prefix(void* h, size_t limit, int* kept_before, int* last_before)`
- `int cuda_replace_result(void* h, uint8_t** out_host, size_t* out_len)`
- `void cuda_free_host(void* p)`
- `void cuda_replace_close(void* h)`
- `const char* cuda_replace_version(void)`
- `int cuda_replace_device_count(void)`, `int cuda_replace_set_device(int)`, `int cuda_replace_get_device(void)`

Return codes: `0` = success, negative = argument error, otherwise a CUDA error
code (`cudaGetErrorString`-compatible).

---

## License

MIT — see [`LICENSE`](LICENSE).
