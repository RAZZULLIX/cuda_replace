# cuda_replace_wrapper.py
# Minimal, robust ctypes wrapper for cuda_replace.dll
# - Pure stdlib
# - Binary-safe (handles embedded NULs)
# - Exposes: unified(), Session(open/reset/build_index/apply/apply_seeded/apply_batch/query_prefix/result/length/close)
# - Includes a memory-bounded, correctness-preserving GPU streaming helper.

from __future__ import annotations
import ctypes as C
import os
from typing import List, Sequence, Tuple, Optional, Iterable

# ---- low-level helpers ----

_is_windows = os.name == "nt"

def _load_dll(path: str) -> C.CDLL:
    path = os.fspath(path)
    if _is_windows:
        dll_dir = os.path.abspath(os.path.dirname(path))
        try:
            os.add_dll_directory(dll_dir)  # type: ignore[attr-defined]
        except Exception:
            pass
    return C.CDLL(path)

# C types
c_size_t = C.c_size_t
c_uint8_p = C.POINTER(C.c_uint8)
c_void_p = C.c_void_p
c_int_p = C.POINTER(C.c_int)
c_longlong_p = C.POINTER(C.c_longlong)
c_size_t_p = C.POINTER(c_size_t)
c_uint8_pp = C.POINTER(c_uint8_p)

class CudaReplaceLib:
    """
    Loads cuda_replace.dll (or .so/.dylib) and binds function signatures.
    """
    def __init__(self, dll_path: str):
        self.lib = _load_dll(dll_path)
        lib = self.lib

        # int cuda_replace_unified(const uint8_t* src, size_t src_len,
        #                          const uint8_t* pat, int pat_len,
        #                          const uint8_t* rep, int rep_len,
        #                          uint8_t** out_host, size_t* outlen_host)
        lib.cuda_replace_unified.argtypes = [
            c_uint8_p, c_size_t,
            c_uint8_p, C.c_int,
            c_uint8_p, C.c_int,
            C.POINTER(c_uint8_p), c_size_t_p
        ]
        lib.cuda_replace_unified.restype = C.c_int

        # void cuda_free_host(void* p)
        lib.cuda_free_host.argtypes = [c_void_p]
        lib.cuda_free_host.restype = None

        # int cuda_replace_open(void** ph, const uint8_t* src, size_t len)
        lib.cuda_replace_open.argtypes = [C.POINTER(c_void_p), c_uint8_p, c_size_t]
        lib.cuda_replace_open.restype = C.c_int

        # int cuda_replace_build_index(void* h, size_t chunk_size)  // optional
        if hasattr(lib, "cuda_replace_build_index"):
            lib.cuda_replace_build_index.argtypes = [c_void_p, c_size_t]
            lib.cuda_replace_build_index.restype = C.c_int
        else:
            lib.cuda_replace_build_index = None  # type: ignore[assignment]

        # int cuda_replace_apply(void* h, const uint8_t* pat, int pat_len,
        #                        const uint8_t* rep, int rep_len, size_t* new_len)
        lib.cuda_replace_apply.argtypes = [
            c_void_p, c_uint8_p, C.c_int, c_uint8_p, C.c_int, c_size_t_p
        ]
        lib.cuda_replace_apply.restype = C.c_int

        # int cuda_replace_apply_batch(void* h, const uint8_t** pats, const int* pat_lens,
        #                              const uint8_t** reps, const int* rep_lens, int count, size_t* new_len)
        lib.cuda_replace_apply_batch.argtypes = [
            c_void_p,
            c_uint8_pp, c_int_p,
            c_uint8_pp, c_int_p,
            C.c_int, c_size_t_p
        ]
        lib.cuda_replace_apply_batch.restype = C.c_int

        # int cuda_replace_result(void* h, uint8_t** out_host, size_t* outlen_host)
        lib.cuda_replace_result.argtypes = [c_void_p, C.POINTER(c_uint8_p), c_size_t_p]
        lib.cuda_replace_result.restype = C.c_int

        # void cuda_replace_close(void* h)
        lib.cuda_replace_close.argtypes = [c_void_p]
        lib.cuda_replace_close.restype = None

        # --- NEW: reset / seeded apply / prefix query (thread-safe in DLL) ---
        if hasattr(lib, "cuda_replace_reset"):
            lib.cuda_replace_reset.argtypes = [c_void_p, c_uint8_p, c_size_t]
            lib.cuda_replace_reset.restype = C.c_int
        else:
            lib.cuda_replace_reset = None  # type: ignore[assignment]

        if hasattr(lib, "cuda_replace_apply_seeded"):
            lib.cuda_replace_apply_seeded.argtypes = [
                c_void_p,
                c_uint8_p, C.c_int,
                c_uint8_p, C.c_int,
                C.c_longlong, c_longlong_p,
                c_size_t_p
            ]
            lib.cuda_replace_apply_seeded.restype = C.c_int
        else:
            lib.cuda_replace_apply_seeded = None  # type: ignore[assignment]

        if hasattr(lib, "cuda_replace_query_prefix"):
            lib.cuda_replace_query_prefix.argtypes = [
                c_void_p, c_size_t, C.POINTER(C.c_int), C.POINTER(C.c_int)
            ]
            lib.cuda_replace_query_prefix.restype = C.c_int
        else:
            lib.cuda_replace_query_prefix = None  # type: ignore[assignment]

    # Convenience one-shot call
    def unified(self, src: bytes, pat: bytes, rep: bytes) -> bytes:
        src_buf = C.create_string_buffer(src)
        pat_buf = C.create_string_buffer(pat)
        rep_buf = C.create_string_buffer(rep)
        out_ptr = c_uint8_p()
        out_len = c_size_t()
        rc = self.lib.cuda_replace_unified(
            C.cast(src_buf, c_uint8_p), c_size_t(len(src)),
            C.cast(pat_buf, c_uint8_p), C.c_int(len(pat)),
            C.cast(rep_buf, c_uint8_p), C.c_int(len(rep)),
            C.byref(out_ptr), C.byref(out_len)
        )
        if rc != 0:
            raise RuntimeError(f"cuda_replace_unified failed rc={rc}")
        try:
            return C.string_at(out_ptr, out_len.value)
        finally:
            self.lib.cuda_free_host(out_ptr)

# ----- session -----

class Session:
    """
    RAII wrapper over a persistent cuda_replace session.
    Safe to share across threads for distinct Sessions; each Session's internal mutex lives in the DLL.
    """
    def __init__(self, lib: CudaReplaceLib, src: bytes):
        self.lib = lib
        self.h = c_void_p()
        src_buf = C.create_string_buffer(src)
        rc = lib.lib.cuda_replace_open(C.byref(self.h),
                                       C.cast(src_buf, c_uint8_p),
                                       c_size_t(len(src)))
        if rc != 0:
            raise RuntimeError(f"cuda_replace_open failed rc={rc}")
        self._src_keepalive = src_buf  # keep source alive for the call duration

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # NEW: reset existing session with new src (O(1) if fits capacity)
    def reset(self, src: bytes):
        fn = getattr(self.lib.lib, "cuda_replace_reset", None)
        if not fn:
            raise RuntimeError("cuda_replace_reset not available in this build")
        buf = C.create_string_buffer(src)
        rc = fn(self.h, C.cast(buf, c_uint8_p), c_size_t(len(src)))
        if rc != 0:
            raise RuntimeError(f"cuda_replace_reset failed rc={rc}")
        self._src_keepalive = buf  # keep alive for the duration of the call

    def build_index(self, chunk_size: Optional[int] = None):
        fn = self.lib.lib.cuda_replace_build_index
        if not fn:
            return
        cs = c_size_t(0 if chunk_size is None else int(chunk_size))
        rc = fn(self.h, cs)
        if rc != 0:
            raise RuntimeError(f"cuda_replace_build_index failed rc={rc}")

    def apply(self, pat: bytes, rep: bytes) -> int:
        pat_buf = C.create_string_buffer(pat)
        rep_buf = C.create_string_buffer(rep)
        new_len = c_size_t()
        rc = self.lib.lib.cuda_replace_apply(
            self.h,
            C.cast(pat_buf, c_uint8_p), C.c_int(len(pat)),
            C.cast(rep_buf, c_uint8_p), C.c_int(len(rep)),
            C.byref(new_len)
        )
        if rc != 0:
            raise RuntimeError(f"cuda_replace_apply failed rc={rc}")
        return int(new_len.value)

    # NEW: apply with leftmost-carry seed. Returns (new_len, last_kept_start_relative)
    def apply_seeded(self, pat: bytes, rep: bytes, prev_last_rel: int) -> tuple[int, int]:
        fn = getattr(self.lib.lib, "cuda_replace_apply_seeded", None)
        if not fn:
            raise RuntimeError("cuda_replace_apply_seeded not available in this build")
        pat_buf = C.create_string_buffer(pat)
        rep_buf = C.create_string_buffer(rep)
        new_len  = c_size_t()
        last_rel = C.c_longlong(-1)
        rc = fn(
            self.h,
            C.cast(pat_buf, c_uint8_p), C.c_int(len(pat)),
            C.cast(rep_buf, c_uint8_p), C.c_int(len(rep)),
            C.c_longlong(int(prev_last_rel)),
            C.byref(last_rel),
            C.byref(new_len)
        )
        if rc != 0:
            raise RuntimeError(f"cuda_replace_apply_seeded failed rc={rc}")
        return int(new_len.value), int(last_rel.value)

    def apply_batch(self, pairs: Sequence[Tuple[bytes, bytes]]) -> int:
        count = len(pairs)
        if count == 0:
            return self.length()

        pat_bufs: List[C.Array] = []
        rep_bufs: List[C.Array] = []
        pat_ptrs: List[c_uint8_p] = []
        rep_ptrs: List[c_uint8_p] = []
        pat_lens = (C.c_int * count)()
        rep_lens = (C.c_int * count)()

        for i, (pat, rep) in enumerate(pairs):
            pb = C.create_string_buffer(pat)
            rb = C.create_string_buffer(rep)
            pat_bufs.append(pb); rep_bufs.append(rb)
            pat_ptrs.append(C.cast(pb, c_uint8_p))
            rep_ptrs.append(C.cast(rb, c_uint8_p))
            pat_lens[i] = len(pat)
            rep_lens[i] = len(rep)

        PatPtrArray = c_uint8_p * count
        RepPtrArray = c_uint8_p * count
        pat_ptr_arr = PatPtrArray(*pat_ptrs)
        rep_ptr_arr = RepPtrArray(*rep_ptrs)

        new_len = c_size_t()
        rc = self.lib.lib.cuda_replace_apply_batch(
            self.h,
            C.cast(pat_ptr_arr, c_uint8_pp), C.cast(pat_lens, c_int_p),
            C.cast(rep_ptr_arr, c_uint8_pp), C.cast(rep_lens, c_int_p),
            C.c_int(count),
            C.byref(new_len)
        )
        if rc != 0:
            raise RuntimeError(f"cuda_replace_apply_batch failed rc={rc}")
        return int(new_len.value)

    # NEW: query how many kept-starts are strictly before 'limit'
    # and the last kept-start index (< limit). Returns (count, last_idx or -1).
    def query_prefix(self, limit: int) -> tuple[int, int]:
        fn = getattr(self.lib.lib, "cuda_replace_query_prefix", None)
        if not fn:
            raise RuntimeError("cuda_replace_query_prefix not available in this build")
        kept = C.c_int(0)
        last = C.c_int(-1)
        rc = fn(self.h, c_size_t(int(limit)), C.byref(kept), C.byref(last))
        if rc != 0:
            raise RuntimeError(f"cuda_replace_query_prefix failed rc={rc}")
        return int(kept.value), int(last.value)

    def result(self) -> bytes:
        out_ptr = c_uint8_p()
        out_len = c_size_t()
        rc = self.lib.lib.cuda_replace_result(self.h, C.byref(out_ptr), C.byref(out_len))
        if rc != 0:
            raise RuntimeError(f"cuda_replace_result failed rc={rc}")
        try:
            return C.string_at(out_ptr, out_len.value)
        finally:
            self.lib.lib.cuda_free_host(out_ptr)

    def length(self) -> int:
        out_ptr = c_uint8_p()
        out_len = c_size_t()
        rc = self.lib.lib.cuda_replace_result(self.h, C.byref(out_ptr), C.byref(out_len))
        if rc != 0:
            raise RuntimeError(f"cuda_replace_result(len) failed rc={rc}")
        self.lib.lib.cuda_free_host(out_ptr)
        return int(out_len.value)

    def close(self):
        if self.h:
            self.lib.lib.cuda_replace_close(self.h)
            self.h = c_void_p(None)
        self._src_keepalive = None  # type: ignore[assignment]

# ---- High-level helpers ----

def replace_unified(lib: CudaReplaceLib, src: bytes, pat: bytes, rep: bytes) -> bytes:
    """One-shot convenience around cuda_replace_unified()."""
    return lib.unified(src, pat, rep)

def gpu_replace_streaming(lib: CudaReplaceLib,
                          src: bytes,
                          pairs: Iterable[Tuple[bytes, bytes]],
                          chunk_bytes: int = 256 * 1024 * 1024) -> bytes:
    if chunk_bytes <= 0:
        chunk_bytes = 1
    out = src
    SEED_NEG_INF = -(1 << 29)
    MAX_SAFE_EXPANSION = (1 << 30)  # Stay well below 2^31 to prevent GPU kernel overflow
    
    with Session(lib, b"") as sess:
        for (pat, rep) in pairs:
            L = len(pat)
            if L == 0:
                continue
            diff = len(rep) - L
            pos = 0
            prev_last_global = SEED_NEG_INF
            result_chunks: List[bytes] = []
            n = len(out)
            
            while pos < n:
                # Dynamically adjust chunk size to prevent integer overflow in GPU kernels
                effective_chunk_size = chunk_bytes
                
                if diff > 0:  # Only worry about expansion
                    # Worst-case: every L bytes could be a match
                    max_possible_matches = chunk_bytes // L if L > 0 else chunk_bytes
                    estimated_expansion = max_possible_matches * diff
                    
                    # If expansion would exceed safe limits, reduce chunk size
                    if estimated_expansion > MAX_SAFE_EXPANSION:
                        safe_matches = MAX_SAFE_EXPANSION // diff
                        effective_chunk_size = safe_matches * L
                        effective_chunk_size = max(effective_chunk_size, L)  # At least one pattern length
                        effective_chunk_size = min(effective_chunk_size, chunk_bytes)  # Don't exceed original
                
                base_end = min(pos + effective_chunk_size, n)
                overlap = max(0, L - 1)
                take_end = min(base_end + overlap, n)
                chunk = out[pos:take_end]
                sess.reset(chunk)
                
                seed_rel = prev_last_global - pos
                if seed_rel < -(1 << 30):
                    seed_rel = -(1 << 30)
                if seed_rel > (1 << 30):
                    seed_rel = (1 << 30)
                
                _new_len, _last_rel = sess.apply_seeded(pat, rep, int(seed_rel))
                kept_before, last_before = sess.query_prefix(base_end - pos)
                out_prefix_end = (base_end - pos) + kept_before * diff
                gpu_out = sess.result()
                out_prefix_end = min(out_prefix_end, len(gpu_out))
                if out_prefix_end < 0:
                    out_prefix_end = 0
                result_chunks.append(gpu_out[:out_prefix_end])
                if last_before >= 0:
                    prev_last_global = pos + last_before
                pos = base_end
            out = b"".join(result_chunks)
    return out


# Optional: quick self-test (remove or adapt as needed)
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python cuda_replace_wrapper.py <path-to-cuda_replace.dll|.so|.dylib>")
        raise SystemExit(2)

    lib = CudaReplaceLib(sys.argv[1])

    # Simple correctness spot-checks
    s = b"aaabaaabaaa"
    pat = b"aaa"
    rep = b"X"

    # One-shot
    print("Unified:", lib.unified(s, pat, rep))

    # Session
    with Session(lib, s) as sess:
        sess.apply(pat, rep)
        print("Session:", sess.result())

    # Streaming (works even when chunking is tiny)
    print("Streaming:", gpu_replace_streaming(lib, s, [(pat, rep)], chunk_bytes=4))
