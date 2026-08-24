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

        # int cuda_replace_device_count() / set_device / get_device
        lib.cuda_replace_device_count.argtypes = []
        lib.cuda_replace_device_count.restype = C.c_int
        lib.cuda_replace_set_device.argtypes = [C.c_int]
        lib.cuda_replace_set_device.restype = C.c_int
        lib.cuda_replace_get_device.argtypes = []
        lib.cuda_replace_get_device.restype = C.c_int

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

    def device_count(self) -> int:
        return int(self.lib.cuda_replace_device_count())

    def set_device(self, device_id: int) -> None:
        rc = self.lib.cuda_replace_set_device(C.c_int(int(device_id)))
        if rc != 0:
            raise RuntimeError(f"cuda_replace_set_device({device_id}) failed rc={rc}")

    # Convenience one-shot call
    def unified(self, src: bytes, pat: bytes, rep: bytes) -> bytes:
        if len(pat) == 0:
            # Python's bytes.replace(b'', rep) inserts rep between every byte.
            return src.replace(b"", rep)
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
        if len(pat) == 0:
            cur = self.result()
            new = cur.replace(b"", rep)
            self.reset(new)
            return len(new)
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

        # Empty patterns are handled by the sequential path (which delegates to
        # Python's native bytes.replace for that edge case).
        if any(len(pat) == 0 for pat, _rep in pairs):
            for pat, rep in pairs:
                self.apply(pat, rep)
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
        # The returned pointer aliases the session's persistent pinned staging
        # buffer; it stays valid until close() and must NOT be freed here.
        return C.string_at(out_ptr, out_len.value)

    def length(self) -> int:
        out_ptr = c_uint8_p()
        out_len = c_size_t()
        rc = self.lib.lib.cuda_replace_result(self.h, C.byref(out_ptr), C.byref(out_len))
        if rc != 0:
            raise RuntimeError(f"cuda_replace_result(len) failed rc={rc}")
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

def _next_safe_boundary(data: bytes, pat: bytes, lo: int, hi: int) -> int:
    """Return the largest split index ``b`` in ``(lo, hi]`` such that no
    occurrence of ``pat`` starts in ``[b - len(pat) + 1, b - 1]`` (i.e. no match
    crosses the boundary ``b``).  Such a boundary lets each side be replaced
    independently with ``bytes.replace`` semantics.

    Returns ``lo`` when no safe boundary exists in ``(lo, hi]``.
    """
    L = len(pat)
    b = hi
    while b > lo:
        start = b - L + 1
        if start < 0:
            start = 0
        # Earliest occurrence of pat starting in [start, b) (and thus crossing b).
        p = data.find(pat, start, b + L - 1)
        if p == -1 or p >= b:
            return b
        b = p
    return lo


def gpu_replace_streaming(lib: CudaReplaceLib,
                          src: bytes,
                          pairs: Iterable[Tuple[bytes, bytes]],
                          chunk_bytes: int = 256 * 1024 * 1024) -> bytes:
    """Memory-bounded replacement that is bit-for-bit identical to sequential
    ``bytes.replace``.  Each pattern is applied to the whole buffer in
    independent chunks; chunk boundaries are chosen so that no match crosses
    them, which makes the concatenation of per-chunk results exactly correct.
    """
    if chunk_bytes <= 0:
        chunk_bytes = 1
    out = src
    with Session(lib, b"") as sess:
        for pat, rep in pairs:
            L = len(pat)
            if L == 0:
                out = out.replace(b"", rep)
                continue
            n = len(out)
            if n == 0:
                continue
            result: List[bytes] = []
            pos = 0
            while pos < n:
                end = pos + chunk_bytes
                if end < n:
                    end = _next_safe_boundary(out, pat, pos, end)
                    if end <= pos:
                        # Pathological dense-overlap input: no safe split found,
                        # so fall back to processing the entire remainder.
                        end = n
                else:
                    end = n
                chunk = out[pos:end]
                sess.reset(chunk)
                sess.apply(pat, rep)
                result.append(sess.result())
                pos = end
            out = b"".join(result)
    return out


# ---- Multi-GPU helpers ----

def _split_safe_segments(data: bytes, pat: bytes, num_parts: int,
                         max_segment: int = 1 << 30) -> List[Tuple[int, int]]:
    """Split ``data`` into roughly ``num_parts`` independent ``(start, end)``
    ranges at boundaries where no match of ``pat`` crosses.  Each range can be
    replaced independently and concatenated, exactly like ``bytes.replace``.

    ``max_segment`` caps the size of any single range (to bound GPU memory).
    """
    L = len(pat)
    n = len(data)
    if n == 0:
        return []
    if L <= 0 or num_parts <= 1:
        return [(0, n)]
    target = (n + num_parts - 1) // num_parts
    if target > max_segment:
        target = max_segment
    if target < 1:
        target = 1
    segments: List[Tuple[int, int]] = []
    pos = 0
    while pos < n:
        end = pos + target
        if end < n:
            end = _next_safe_boundary(data, pat, pos, end)
            if end <= pos:
                end = n  # pathological: no safe split; take the rest
        else:
            end = n
        segments.append((pos, end))
        pos = end
    return segments


def _multicard_worker(lib: "CudaReplaceLib", device_id: int,
                      in_q, out_q) -> None:
    """Persistent worker thread bound to one GPU. Reuses a single Session."""
    lib.set_device(device_id)
    sess = Session(lib, b"")
    while True:
        item = in_q.get()
        if item is None:
            break
        job_id, data, pat, rep = item
        try:
            sess.reset(data)
            sess.apply(pat, rep)
            out_q.put((job_id, sess.result(), None))
        except Exception as exc:  # pragma: no cover - error propagation
            out_q.put((job_id, None, exc))


def gpu_replace_multicard(lib: "CudaReplaceLib",
                          src: bytes,
                          pairs: Iterable[Tuple[bytes, bytes]],
                          devices: Optional[Sequence[int]] = None,
                          chunk_bytes: int = 256 * 1024 * 1024,
                          oversplit: int = 8) -> bytes:
    """Parallel multi-GPU replacement.

    Splits the buffer into independent segments (at match-free boundaries) and
    processes them concurrently across the given CUDA devices, one worker
    thread per device, each holding a persistent Session.  The result is
    bit-for-bit identical to ``bytes.replace`` (and to :func:`gpu_replace_streaming`).
    """
    if chunk_bytes <= 0:
        chunk_bytes = 1
    dev_list = list(devices) if devices is not None else list(range(lib.device_count()))
    dev_list = [int(d) for d in dev_list if int(d) < lib.device_count()]
    if not dev_list:
        raise RuntimeError("no CUDA devices available")
    if len(dev_list) == 1:
        return gpu_replace_streaming(lib, src, pairs, chunk_bytes)

    import queue
    import threading

    out = src
    nworkers = len(dev_list)
    in_q = queue.Queue()
    out_q = queue.Queue()
    threads = []
    for d in dev_list:
        t = threading.Thread(target=_multicard_worker, args=(lib, d, in_q, out_q), daemon=True)
        t.start()
        threads.append(t)

    try:
        for pat, rep in pairs:
            L = len(pat)
            if L == 0:
                out = out.replace(b"", rep)
                continue
            n = len(out)
            if n == 0:
                continue
            segments = _split_safe_segments(out, pat, nworkers * oversplit, chunk_bytes)
            for i, (a, b) in enumerate(segments):
                in_q.put((i, out[a:b], pat, rep))
            results: List[Optional[bytes]] = [None] * len(segments)
            for _ in segments:
                jid, res, err = out_q.get()
                if err is not None:
                    raise RuntimeError(f"multi-GPU worker failed: {err}")
                results[jid] = res
            out = b"".join(r for r in results if r is not None)
    finally:
        for _ in threads:
            in_q.put(None)
        for t in threads:
            t.join()
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
