// cuda_replace.cu
// High-throughput byte-pattern replace (GPU). Leftmost, non-overlapping semantics
// identical to Python's bytes.replace. Pure CUDA C++.
//
// Build (Windows):
//   nvcc -O3 -std=c++17 -m64 -Xcompiler "/MD" -shared -o cuda_replace.dll cuda_replace.cu --linker-options "/DEF:cuda_replace.def,/MACHINE:X64,/NODEFAULTLIB:LIBCMT"
//
// Exports:
//   int  cuda_replace_unified(...);
//   void cuda_free_host(void*);
//
//   // Persistent session API (GPU-resident buffer):
//   int  cuda_replace_open(void** ph, const uint8_t* src, size_t len);
//   int  cuda_replace_build_index(void* h, size_t chunk_size); // optional presence index
//   int  cuda_replace_apply(void* h, const uint8_t* pat, int pat_len,
//                           const uint8_t* rep, int rep_len, size_t* new_len);
//   int  cuda_replace_apply_batch(void* h,
//           const uint8_t** pats, const int* pat_lens,
//           const uint8_t** reps, const int* rep_lens,
//           int count, size_t* new_len);
//   int  cuda_replace_result(void* h, uint8_t** out_host, size_t* outlen_host);
//   void cuda_replace_close(void* h);
//   const char* cuda_replace_version();
//
//   // NEW (2025-09):
//   int  cuda_replace_reset(void* h, const uint8_t* src, size_t len);
//   int  cuda_replace_apply_seeded(void* h,
//         const uint8_t* pat, int pat_len,
//         const uint8_t* rep, int rep_len,
//         long long prev_last_rel, long long* out_last_rel,
//         size_t* new_len);
//   int  cuda_replace_query_prefix(void* h, size_t limit,
//         int* kept_before_limit, int* last_kept_before_limit);
//
// Notes:
// - Legacy path: proven-correct global bitmap + suppression.
// - Fused path (enable with CUDA_REPLACE_FUSED=1): tiled mark+greedy in shared memory.
//   Auto-fallback if SMEM is insufficient.
// - This version implements:
//     * Safe batch path (sync H2D copies, no host staging reuse).
//     * Range-sliced expansion scatter to avoid long kernels / TDR on Windows.
//     * **Split capacity** for main buffers vs aux arrays so we never clobber
//       flags/prefix/local between build and scatter (fixes intermittent mismatches).
// - 2025-09 Thread-Safety Fix:
//     * Per-handle (session) mutex serializes all public API calls that take `void* h`.
//     * Contract: A handle may be shared across threads, but calls on the SAME handle
//       are serialized internally. Do NOT call any API on a handle after `cuda_replace_close`.
//       Cross-handle parallelism is still fully supported.
//
// CRITICAL THREAD-SAFETY WARNING (now enforced):
//   • The session handle (`void* h`) is a mutable resource. This build adds a mutex to
//     `CRContext`, and every exported function that operates on a session acquires it,
//     ensuring calls on the same handle are serialized and race-free.
//   • You MUST NOT use a handle after `cuda_replace_close(h)` returns. The library cannot
//     protect against use-after-free if user code retains the raw pointer.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <algorithm>
#include <mutex> // per-handle mutex
#include <atomic>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef DEFAULT_CHUNK_SIZE
#define DEFAULT_CHUNK_SIZE (1u << 20) // 1 MiB presence-index chunking
#endif

#ifndef SCAN_TILE
#define SCAN_TILE 1024 // device prefix tile (power of 2)
#endif

#ifndef FUSED_TILE_BYTES
#define FUSED_TILE_BYTES 8192 // bytes of input per fused tile (before halo)
#endif

#define PADDING_MARKER_16 999

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

// ---------------- Error helpers ----------------
static inline const char *cu_errname(cudaError_t e) { return cudaGetErrorString(e); }

#define CUDA_OK(expr)            \
    do                           \
    {                            \
        cudaError_t _e = (expr); \
        if (_e != cudaSuccess)   \
        {                        \
            err = _e;            \
            goto cuda_fail;      \
        }                        \
    } while (0)

#define KERNEL_OK()                          \
    do                                       \
    {                                        \
        cudaError_t _e = cudaGetLastError(); \
        if (_e != cudaSuccess)               \
        {                                    \
            err = _e;                        \
            goto cuda_fail;                  \
        }                                    \
    } while (0)

// ---------------- Device utils ----------------
__device__ __forceinline__ size_t grid_stride_start() { return blockIdx.x * blockDim.x + threadIdx.x; }
__device__ __forceinline__ size_t grid_stride_step() { return (size_t)gridDim.x * (size_t)blockDim.x; }

__device__ __forceinline__ bool simple_pattern_match(
    const uint8_t *data, size_t pos, size_t data_len,
    const uint8_t *pat, int pat_len)
{
    if ((size_t)pat_len > data_len || pos + (size_t)pat_len > data_len)
        return false;
    for (int i = 0; i < pat_len; ++i)
        if (data[pos + i] != pat[i])
            return false;
    return true;
}

static inline uint64_t div_ceil_u64(uint64_t a, uint64_t b) { return (b ? (a + b - 1) / b : 0); }

// ===========================
// Core context & helpers
// ===========================
// ===========================
// Core context & helpers
// ===========================
struct CRContext
{
    // Thread-safety
    std::mutex mtx;    // serialize all API calls per handle
    bool alive = true; // set false at close() while under lock

    // Main buffer (input/output)
    uint8_t *d_buf = nullptr;
    uint8_t *d_tmp = nullptr;
    size_t len = 0;

    // **Split capacities**
    size_t cap = 0;     // capacity of d_buf / d_tmp
    size_t aux_cap = 0; // capacity for aux arrays below

    // temps (byte-level, length-dependent)
    bool *d_match = nullptr;      // [aux_cap] final kept-starts
    bool *d_match2 = nullptr;     // [aux_cap] scratch
    uint16_t *d_padded = nullptr; // [aux_cap]
    uint8_t *d_flags = nullptr;   // [aux_cap]
    int *d_local = nullptr;       // [aux_cap]
    int *d_bcounts = nullptr;     // [blocks]
    int *d_bprefix = nullptr;     // [blocks]
    size_t *d_outlen = nullptr;   // [1]
    int *d_totmatch = nullptr;    // [1]

    // kept info (device scalars)
    int *d_kept_count = nullptr;
    int *d_first_kept = nullptr;
    int *d_last_kept = nullptr;

    // device-wide scan temporaries
    int *d_tile_sums = nullptr; // [num_tiles]
    // --- FIX START ---
    // Promote tile_prefix to 64-bit to handle massive expansion offsets.
    size_t *d_tile_prefix = nullptr; // [num_tiles]
    // --- FIX END ---
    int tiles_cap = 0;

    // block launch cache for byte-grids
    int blocks_cap = 0;
    dim3 grid, block;

    // chunk presence index (optional)
    size_t chunk_size = DEFAULT_CHUNK_SIZE;
    int num_chunks = 0;
    size_t *d_chunk_addrs = nullptr;   // [num_chunks]
    uint64_t *d_chunk_masks = nullptr; // [num_chunks][4]
    uint8_t *d_chunk_flags = nullptr;  // [num_chunks]
    int *d_chunk_local = nullptr;      // [num_chunks]
    int *d_chunk_bcounts = nullptr;    // [(num_chunks+BLOCK_SIZE-1)/BLOCK_SIZE]
    int *d_chunk_bprefix = nullptr;    // [same]
    int *d_chunk_indices = nullptr;    // [num_chunks]
    int active_chunks = 0;

    // fused pipeline temps
    int fused_tile_bytes = FUSED_TILE_BYTES;
    int fused_num_tiles = 0;
    int *d_fused_last = nullptr; // [fused_num_tiles]
    int *d_fused_prev = nullptr; // [fused_num_tiles]

    // device staging for batch (single-buffer, sync H2D)
    uint8_t *d_pat_buf[1] = {nullptr};
    size_t d_pat_cap = 0;
    uint8_t *d_rep_buf[1] = {nullptr};
    size_t d_rep_cap = 0;

    // streams
    cudaStream_t stream = nullptr;
    cudaStream_t copy_stream = nullptr; // kept for ABI compat, unused

    // (unused now, but kept for ABI stability)
    cudaEvent_t ev_copy_done[2]{};
    cudaEvent_t ev_compute_done[2]{};

    // execution
    int use_fused = 0; // runtime toggle via env
};

static int dev_max_smem()
{
    int dev = 0, max_smem = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (max_smem == 0)
        cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock, dev);
    return (max_smem > 0 ? max_smem : 49152);
}

static void compute_launch(CRContext *c)
{
    c->block = dim3(BLOCK_SIZE, 1, 1);
    c->grid = dim3((unsigned)((c->len + BLOCK_SIZE - 1) / BLOCK_SIZE), 1, 1);
}

// ===========================
// Tiled device prefix (Blelloch)
// ===========================
__global__ void scan_counts_stage1_kernel(
    const int *__restrict__ in, int n,
    int *__restrict__ out_excl,  // exclusive per-element scan within tiles
    int *__restrict__ tile_sums) // per-tile totals
{
    __shared__ int s[SCAN_TILE];

    const int tile = blockIdx.x;
    const int base = tile * SCAN_TILE;
    int limit = n - base;
    if (limit <= 0)
    {
        if (threadIdx.x == 0)
            tile_sums[tile] = 0;
        return;
    }
    if (limit > SCAN_TILE)
        limit = SCAN_TILE;

    int t = threadIdx.x;
    s[t] = (t < limit) ? in[base + t] : 0;
    __syncthreads();

    // upsweep
    for (int d = 1; d < SCAN_TILE; d <<= 1)
    {
        int idx = (t + 1) * (d << 1) - 1;
        if (idx < SCAN_TILE)
            s[idx] += s[idx - d];
        __syncthreads();
    }

    int total = s[SCAN_TILE - 1];
    if (t == 0)
        s[SCAN_TILE - 1] = 0;
    __syncthreads();

    // downsweep
    for (int d = SCAN_TILE >> 1; d >= 1; d >>= 1)
    {
        int idx = (t + 1) * (d << 1) - 1;
        if (idx < SCAN_TILE)
        {
            int tmp = s[idx - d];
            s[idx - d] = s[idx];
            s[idx] += tmp;
        }
        __syncthreads();
    }

    if (t < limit)
        out_excl[base + t] = s[t];
    if (t == 0)
        tile_sums[tile] = total;
}

// Corrected scan_counts_stage2_kernel
__global__ void scan_counts_stage2_kernel(
    const int *__restrict__ tile_sums, int num_tiles,
    // --- FIX START ---
    // Match the 64-bit data type for the output buffer.
    size_t *__restrict__ tile_prefix) // exclusive
// --- FIX END ---
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    size_t acc = 0;
    for (int i = 0; i < num_tiles; ++i)
    {
        tile_prefix[i] = acc;
        acc += tile_sums[i];
    }
}

// Corrected scan_counts_stage3_add_kernel
__global__ void scan_counts_stage3_add_kernel(
    int *__restrict__ out_excl, int n,
    // --- FIX START ---
    // Match the 64-bit data type for the input buffer.
    const size_t *__restrict__ tile_prefix)
// --- FIX END ---
{
    const int tile = blockIdx.x;
    const int base = tile * SCAN_TILE;
    int limit = n - base;
    if (limit <= 0)
        return;
    if (limit > SCAN_TILE)
        limit = SCAN_TILE;

    // --- FIX START ---
    // Use a 64-bit variable to hold the prefix value before adding it.
    size_t add = tile_prefix[tile];
    // --- FIX END ---
    int t = threadIdx.x;
    if (t < limit)
        out_excl[base + t] += add;
}

__global__ void scan_counts_stage3_add_kernel(
    int *__restrict__ out_excl, int n,
    const int *__restrict__ tile_prefix)
{
    const int tile = blockIdx.x;
    const int base = tile * SCAN_TILE;
    int limit = n - base;
    if (limit <= 0)
        return;
    if (limit > SCAN_TILE)
        limit = SCAN_TILE;

    int add = tile_prefix[tile];
    int t = threadIdx.x;
    if (t < limit)
        out_excl[base + t] += add;
}

static cudaError_t ensure_tiles(CRContext *c, int n_elems)
{
    cudaError_t err = cudaSuccess;
    int num_tiles = (n_elems + SCAN_TILE - 1) / SCAN_TILE;
    if (num_tiles <= c->tiles_cap)
        return cudaSuccess;
    if (num_tiles == 0)
        return cudaSuccess;

    int *ts = nullptr;
    // --- FIX START ---
    // Match the type of tp and old_tp to the 64-bit d_tile_prefix in CRContext.
    size_t *tp = nullptr;
    CUDA_OK(cudaMalloc(&ts, (size_t)num_tiles * sizeof(int)));
    CUDA_OK(cudaMalloc(&tp, (size_t)num_tiles * sizeof(size_t)));

    int *old_ts = c->d_tile_sums;
    size_t *old_tp = c->d_tile_prefix;
    // --- FIX END ---

    c->d_tile_sums = ts;
    c->d_tile_prefix = tp;
    c->tiles_cap = num_tiles;
    if (old_ts)
        cudaFree(old_ts);
    if (old_tp)
        cudaFree(old_tp);
    return cudaSuccess;
cuda_fail:
    if (ts)
        cudaFree(ts);
    if (tp)
        cudaFree(tp);
    return err;
}
// ===========================
// Presence index over chunks (optional, kept minimal)
// ===========================
struct Bit256
{
    uint64_t w[4];
};
__device__ __forceinline__ void bit256_set(Bit256 &m, uint8_t byte) { m.w[byte >> 6] |= (1ull << (byte & 63)); }

__global__ void build_chunk_index_kernel(
    const uint8_t *__restrict__ data, size_t data_len,
    size_t chunk_size, int num_chunks,
    size_t *__restrict__ chunk_addrs,
    uint64_t *__restrict__ chunk_masks) // [num_chunks][4]
{
    int cid = blockIdx.x;
    if (cid >= num_chunks)
        return;

    size_t start = (size_t)cid * chunk_size;
    size_t end = start + chunk_size;
    if (start >= data_len)
    {
        if (threadIdx.x == 0)
        {
            chunk_addrs[cid] = start;
            chunk_masks[cid * 4 + 0] = chunk_masks[cid * 4 + 1] = chunk_masks[cid * 4 + 2] = chunk_masks[cid * 4 + 3] = 0;
        }
        return;
    }
    if (end > data_len)
        end = data_len;

    __shared__ Bit256 s_mask;
    if (threadIdx.x == 0)
        s_mask.w[0] = s_mask.w[1] = s_mask.w[2] = s_mask.w[3] = 0ull;
    __syncthreads();

    for (size_t i = start + threadIdx.x; i < end; i += blockDim.x)
    {
        uint8_t b = data[i];
        atomicOr((unsigned long long *)&s_mask.w[b >> 6], 1ull << (b & 63));
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        chunk_addrs[cid] = start;
        chunk_masks[cid * 4 + 0] = s_mask.w[0];
        chunk_masks[cid * 4 + 1] = s_mask.w[1];
        chunk_masks[cid * 4 + 2] = s_mask.w[2];
        chunk_masks[cid * 4 + 3] = s_mask.w[3];
    }
}

// ===========================
// Legacy mark / suppress path
// ===========================
__global__ void mark_match_positions_kernel_smem(
    bool *__restrict__ match_positions,
    const uint8_t *__restrict__ data, size_t data_len,
    const uint8_t *__restrict__ pattern, int pattern_len)
{
    extern __shared__ uint8_t smem[];
    uint8_t *shared_pattern = smem;
    for (int i = threadIdx.x; i < pattern_len; i += blockDim.x)
        shared_pattern[i] = pattern[i];
    __syncthreads();

    for (size_t pos = grid_stride_start(); pos < data_len; pos += grid_stride_step())
    {
        bool m = (pos + (size_t)pattern_len <= data_len) &&
                 simple_pattern_match(data, pos, data_len, shared_pattern, pattern_len);
        match_positions[pos] = m;
    }
}

__global__ void mark_match_positions_kernel_gmem(
    bool *__restrict__ match_positions,
    const uint8_t *__restrict__ data, size_t data_len,
    const uint8_t *__restrict__ pattern, int pattern_len)
{
    for (size_t pos = grid_stride_start(); pos < data_len; pos += grid_stride_step())
    {
        bool m = (pos + (size_t)pattern_len <= data_len) &&
                 simple_pattern_match(data, pos, data_len, pattern, pattern_len);
        match_positions[pos] = m;
    }
}

// pass1: local greedy (inside each block's byte-range), store last kept per block
__global__ void suppress_local_pass(
    const bool *__restrict__ match_in, size_t n, int L,
    bool *__restrict__ keep_tmp,      // zeroed before call
    int *__restrict__ block_last_keep // [gridDim.x]
)
{
    const size_t base = (size_t)blockIdx.x * blockDim.x;
    const size_t end = min(base + (size_t)blockDim.x, n);
    long long last = LLONG_MIN / 4;

    if (threadIdx.x == 0)
    {
        for (size_t i = base; i < end; ++i)
        {
            if (match_in[i])
            {
                long long dist = (long long)i - last;
                if (dist >= (long long)L)
                {
                    keep_tmp[i] = true;
                    last = (long long)i;
                }
            }
        }
        block_last_keep[blockIdx.x] = (last < 0 ? -1 : (int)last);
    }
}

// compute prev_keep[b] = last kept among all blocks [0..b-1], or -1
__global__ void scan_last_keep_prefix(
    const int *__restrict__ block_last_keep, int num_blocks,
    int *__restrict__ prev_keep // [num_blocks]
)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        int last = -1;
        for (int b = 0; b < num_blocks; ++b)
        {
            prev_keep[b] = last;
            if (block_last_keep[b] >= 0)
                last = block_last_keep[b];
        }
    }
}

// pass2: greedy seeded with prev_keep
__global__ void suppress_local_apply(
    const bool *__restrict__ match_in, size_t n, int L,
    const int *__restrict__ prev_keep, // [gridDim.x]
    bool *__restrict__ match_out       // zeroed before call
)
{
    const size_t base = (size_t)blockIdx.x * blockDim.x;
    const size_t end = min(base + (size_t)blockDim.x, n);
    long long last = (long long)prev_keep[blockIdx.x];

    if (threadIdx.x == 0)
    {
        for (size_t i = base; i < end; ++i)
        {
            if (match_in[i])
            {
                long long dist = (long long)i - last;
                if (dist >= (long long)L)
                {
                    match_out[i] = true;
                    last = (long long)i;
                }
            }
        }
    }
}

// ===========================
// Compaction / expansion path
// ===========================
__global__ void fill_u16_kernel(uint16_t *__restrict__ buf, size_t n, uint16_t value)
{
    for (size_t i = grid_stride_start(); i < n; i += grid_stride_step())
        buf[i] = value;
}

__global__ void comp_flags_localpos_kernel(
    const uint16_t *__restrict__ padded, size_t n,
    uint8_t *__restrict__ flags, int *__restrict__ local_pos, int *__restrict__ block_counts)
{
    __shared__ int s_scan[BLOCK_SIZE];
    const size_t base = (size_t)blockIdx.x * blockDim.x;
    const int lane = threadIdx.x;

    size_t idx = base + lane;
    int f = 0;
    if (idx < n)
    {
        f = (padded[idx] != (uint16_t)PADDING_MARKER_16) ? 1 : 0;
        flags[idx] = (uint8_t)f;
    }
    s_scan[lane] = f;
    __syncthreads();

    for (int off = 1; off < blockDim.x; off <<= 1)
    {
        int add = (lane >= off) ? s_scan[lane - off] : 0;
        __syncthreads();
        s_scan[lane] += add;
        __syncthreads();
    }

    if (idx < n)
        local_pos[idx] = s_scan[lane] - f;

    int valid = (int)((base < n) ? min((size_t)BLOCK_SIZE, n - base) : (size_t)0);
    if (valid > 0 && lane == (valid - 1))
        block_counts[blockIdx.x] = s_scan[lane];
}

__global__ void comp_scatter_kernel(
    const uint16_t *__restrict__ padded, size_t n,
    const uint8_t *__restrict__ flags, const int *__restrict__ local_pos,
    const int *__restrict__ block_prefix, uint8_t *__restrict__ out)
{
    const size_t base = (size_t)blockIdx.x * blockDim.x;
    const int lane = threadIdx.x;
    size_t idx = base + lane;
    if (idx >= n)
        return;
    if (flags[idx])
    {
        int pos = block_prefix[blockIdx.x] + local_pos[idx];
        out[pos] = (uint8_t)(padded[idx] & 0xFF);
    }
}

__global__ void compute_total_len_kernel(
    const int *__restrict__ block_prefix, const int *__restrict__ block_counts, int num_blocks,
    size_t *__restrict__ out_len)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        int last_prefix = (num_blocks > 0) ? block_prefix[num_blocks - 1] : 0;
        int last_count = (num_blocks > 0) ? block_counts[num_blocks - 1] : 0;
        *out_len = (size_t)(last_prefix + last_count);
    }
}

// Expand helpers (rely on kept-starts bitmap)
__global__ void build_match_flags_localpos_kernel(
    const bool *__restrict__ match_positions, size_t n,
    uint8_t *__restrict__ flags, int *__restrict__ local_pos, int *__restrict__ block_counts)
{
    __shared__ int s_scan[BLOCK_SIZE];
    const size_t base = (size_t)blockIdx.x * blockDim.x;
    const int lane = threadIdx.x;

    size_t idx = base + lane;
    int f = 0;
    if (idx < n)
    {
        f = match_positions[idx] ? 1 : 0;
        flags[idx] = (uint8_t)f;
    }
    s_scan[lane] = f;
    __syncthreads();

    for (int off = 1; off < blockDim.x; off <<= 1)
    {
        int add = (lane >= off) ? s_scan[lane - off] : 0;
        __syncthreads();
        s_scan[lane] += add;
        __syncthreads();
    }
    if (idx < n)
        local_pos[idx] = s_scan[lane] - f;

    int valid = (int)((base < n) ? min((size_t)BLOCK_SIZE, n - base) : (size_t)0);
    if (valid > 0 && lane == (valid - 1))
        block_counts[blockIdx.x] = s_scan[lane];
}

__global__ void compute_total_matches_kernel(
    const int *__restrict__ block_prefix, const int *__restrict__ block_counts, int num_blocks,
    int *__restrict__ total_matches)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        int last_prefix = (num_blocks > 0) ? block_prefix[num_blocks - 1] : 0;
        int last_count = (num_blocks > 0) ? block_counts[num_blocks - 1] : 0;
        *total_matches = last_prefix + last_count;
    }
}

__global__ void expansion_scatter_kernel(
    const uint8_t *__restrict__ src, size_t src_len,
    const bool *__restrict__ match_positions,
    const uint8_t *__restrict__ replacement, int pattern_len, int replacement_len,
    const int *__restrict__ match_block_prefix, const int *__restrict__ match_local_pos,
    uint8_t *__restrict__ dst, size_t /*dst_len*/)
{
    const int diff = replacement_len - pattern_len;

    for (size_t pos = grid_stride_start(); pos < src_len; pos += grid_stride_step())
    {
        int block = (int)(pos / (size_t)BLOCK_SIZE);
        int excl_before = match_block_prefix[block] + match_local_pos[pos];
        size_t dst_pos = (size_t)pos + (size_t)(excl_before * diff);

        if (match_positions[pos])
        {
            for (int i = 0; i < replacement_len; ++i)
                dst[dst_pos + i] = replacement[i];
        }
        else
        {
            bool covered = false;
            int start = (int)pos - (pattern_len - 1);
            if (start < 0)
                start = 0;
            for (int s = start; s <= (int)pos; ++s)
            {
                if (s < (int)src_len && match_positions[s] && (size_t)s + (size_t)pattern_len > pos)
                {
                    covered = true;
                    break;
                }
            }
            if (!covered)
                dst[dst_pos] = src[pos];
        }
    }
}

// ===========================
// NEW fused mark + suppression
// ===========================

// unaligned-safe 8B compares
__device__ __forceinline__ bool vec_match64_shared(const uint8_t *__restrict__ s,
                                                   const uint8_t *__restrict__ p,
                                                   int L)
{
    int i = 0;
    for (; i + 8 <= L; i += 8)
    {
        unsigned long long a, b;
        memcpy(&a, s + i, 8);
        memcpy(&b, p + i, 8);
        if (a != b)
            return false;
    }
    for (; i < L; ++i)
        if (s[i] != p[i])
            return false;
    return true;
}

// Pass 1: per-tile local greedy (seed = -inf). Output last kept global position per tile.
__global__ void fused_tile_lastkeep_kernel(
    const uint8_t *__restrict__ data, size_t n,
    const uint8_t *__restrict__ pat, int L,
    int tile_bytes,
    int *__restrict__ tile_last_keep // size = num_tiles
)
{
    const int tile = blockIdx.x;
    const size_t start = (size_t)tile * (size_t)tile_bytes;
    if (start >= n)
    {
        if (threadIdx.x == 0)
            tile_last_keep[tile] = -1;
        return;
    }
    const size_t end_excl = min(start + (size_t)tile_bytes, n);
    const int tlen = (int)(end_excl - start);

    extern __shared__ uint8_t smem[];
    uint8_t *sp = smem;
    uint8_t *sd = sp + L;
    uint8_t *sm = sd + tlen + max(0, L - 1);

    for (int i = threadIdx.x; i < L; i += blockDim.x)
        sp[i] = pat[i];
    const int halo = max(0, L - 1);
    for (int i = threadIdx.x; i < tlen + halo; i += blockDim.x)
    {
        size_t g = start + (size_t)i;
        sd[i] = (g < n ? data[g] : 0);
    }
    __syncthreads();

    for (int pos = threadIdx.x; pos < tlen; pos += blockDim.x)
    {
        bool m = (pos + L <= tlen + halo) && vec_match64_shared(sd + pos, sp, L);
        sm[pos] = (uint8_t)(m ? 1 : 0);
    }
    __syncthreads();

    int last = INT_MIN / 4;
    int last_kept = -1;
    if (threadIdx.x == 0)
    {
        for (int pos = 0; pos < tlen; ++pos)
        {
            if (sm[pos])
            {
                if (pos - last >= L)
                {
                    last = pos;
                    last_kept = pos;
                }
            }
        }
        tile_last_keep[tile] = (last_kept < 0) ? -1 : (int)(start + (size_t)last_kept);
    }
}

// tiny prefix: prev_last[t] = last kept across tiles [0..t-1]
__global__ void fused_tile_prevlast_kernel(
    const int *__restrict__ tile_last_keep, int num_tiles,
    int *__restrict__ tile_prev_last)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    int last = -1;
    for (int t = 0; t < num_tiles; ++t)
    {
        tile_prev_last[t] = last;
        int lk = tile_last_keep[t];
        if (lk >= 0)
            last = lk;
    }
}

// Pass 2: recompute matches + greedy with carry-in and write FINAL kept-starts bitmap.
__global__ void fused_tile_write_bitmap_kernel(
    bool *__restrict__ kept_bitmap, size_t n,
    const uint8_t *__restrict__ data,
    const uint8_t *__restrict__ pat, int L,
    int tile_bytes,
    const int *__restrict__ tile_prev_last)
{
    const int tile = blockIdx.x;
    const size_t start = (size_t)tile * (size_t)tile_bytes;
    if (start >= n)
        return;
    const size_t end_excl = min(start + (size_t)tile_bytes, n);
    const int tlen = (int)(end_excl - start);

    extern __shared__ uint8_t smem[];
    uint8_t *sp = smem;
    uint8_t *sd = sp + L;
    uint8_t *sm = sd + tlen + max(0, L - 1);

    for (int i = threadIdx.x; i < L; i += blockDim.x)
        sp[i] = pat[i];
    const int halo = max(0, L - 1);
    for (int i = threadIdx.x; i < tlen + halo; i += blockDim.x)
    {
        size_t g = start + (size_t)i;
        sd[i] = (g < n ? data[g] : 0);
    }
    __syncthreads();

    for (int pos = threadIdx.x; pos < tlen; pos += blockDim.x)
    {
        bool m = (pos + L <= tlen + halo) && vec_match64_shared(sd + pos, sp, L);
        sm[pos] = (uint8_t)(m ? 1 : 0);
    }
    __syncthreads();

    int last = tile_prev_last[tile]; // global coordinate
    if (threadIdx.x == 0)
    {
        for (int pos = 0; pos < tlen; ++pos)
        {
            if (!sm[pos])
                continue;
            size_t gpos = start + (size_t)pos;
            long long dist = (long long)gpos - (long long)last;
            if (last < 0 || dist >= (long long)L)
            {
                kept_bitmap[gpos] = true;
                last = (int)gpos;
            }
        }
    }
}

// ===== NEW: Seeded fused prefix composer (with initial carry-in) =====
__global__ void fused_tile_prevlast_kernel_seeded(
    const int *__restrict__ tile_last_keep, int num_tiles,
    int seed_last,
    int *__restrict__ tile_prev_last)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    int last = seed_last;
    for (int t = 0; t < num_tiles; ++t)
    {
        tile_prev_last[t] = last;
        int lk = tile_last_keep[t];
        if (lk >= 0)
            last = lk;
    }
}

// ===========================
// Extra helper kernels at file scope
// ===========================
__global__ void match_indices_scatter_kernel(
    const uint8_t *__restrict__ flags, const int *__restrict__ local_pos,
    const int *__restrict__ block_prefix, size_t n, int *__restrict__ out_indices)
{
    const size_t base = (size_t)blockIdx.x * blockDim.x;
    const int lane = threadIdx.x;
    size_t idx = base + lane;
    if (idx >= n)
        return;
    if (flags[idx])
    {
        int pos = block_prefix[blockIdx.x] + local_pos[idx];
        out_indices[pos] = (int)idx;
    }
}

__global__ void greedy_keep_on_indices_kernel(
    const int *__restrict__ indices, int m, int L,
    uint8_t *__restrict__ keep_flags,
    int *__restrict__ kept_count,
    int *__restrict__ first_kept_pos,
    int *__restrict__ last_kept_pos)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    int last = INT_MIN / 4;
    int kept = 0;
    int first = -1;
    for (int i = 0; i < m; ++i)
    {
        int p = indices[i];
        if (p - last >= L)
        {
            keep_flags[i] = 1;
            last = p;
            if (first < 0)
                first = p;
            ++kept;
        }
    }
    if (kept_count)
        *kept_count = kept;
    if (first_kept_pos)
        *first_kept_pos = first;
    if (last_kept_pos)
        *last_kept_pos = (kept > 0 ? last : -1);
}

// ===== NEW: Seeded greedy over indices =====
__global__ void greedy_keep_on_indices_kernel_seeded(
    const int *__restrict__ indices, int m, int L, int initial_last,
    uint8_t *__restrict__ keep_flags,
    int *__restrict__ kept_count,
    int *__restrict__ first_kept_pos,
    int *__restrict__ last_kept_pos)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    int last = initial_last;
    int kept = 0;
    int first = -1;
    for (int i = 0; i < m; ++i)
    {
        int p = indices[i];
        if (p - last >= L)
        {
            keep_flags[i] = 1;
            last = p;
            if (first < 0)
                first = p;
            ++kept;
        }
    }
    if (kept_count)
        *kept_count = kept;
    if (first_kept_pos)
        *first_kept_pos = first;
    if (last_kept_pos)
        *last_kept_pos = (kept > 0 ? last : -1);
}

__global__ void set_bitmap_from_kept_indices_kernel(
    bool *__restrict__ out_bitmap, size_t n,
    const int *__restrict__ indices, const uint8_t *__restrict__ keep_flags, int m)
{
    for (size_t i = grid_stride_start(); i < (size_t)m; i += grid_stride_step())
    {
        if (keep_flags[i])
        {
            int pos = indices[i];
            if ((size_t)pos < n)
                out_bitmap[pos] = true;
        }
    }
}

// ===== NEW: last kept-start strictly before a limit (atomicMax) =====
__global__ void last_true_before_limit_kernel(
    const bool *__restrict__ bm, size_t n, size_t limit, int *out_idx)
{
    size_t end = (limit < n ? limit : n);
    for (size_t pos = grid_stride_start(); pos < end; pos += grid_stride_step())
    {
        if (bm[pos])
            atomicMax(out_idx, (int)pos);
    }
}

// ===========================
// Atomic allocation helpers
// ===========================

// blocks helpers (independent from cap/aux_cap)
static cudaError_t ensure_blocks(CRContext *c, size_t for_len)
{
    cudaError_t err = cudaSuccess;
    int blocks = (int)((for_len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    if (blocks <= c->blocks_cap)
        return cudaSuccess;

    int *nbc = nullptr, *nbp = nullptr;
    CUDA_OK(cudaMalloc(&nbc, (size_t)blocks * sizeof(int)));
    CUDA_OK(cudaMalloc(&nbp, (size_t)blocks * sizeof(int)));

    int *old_bc = c->d_bcounts, *old_bp = c->d_bprefix;
    c->d_bcounts = nbc;
    c->d_bprefix = nbp;
    c->blocks_cap = blocks;
    if (old_bc)
        cudaFree(old_bc);
    if (old_bp)
        cudaFree(old_bp);
    return cudaSuccess;
cuda_fail:
    if (nbc)
        cudaFree(nbc);
    if (nbp)
        cudaFree(nbp);
    return err;
}

// **NEW**: grow only main IO buffers (preserve aux arrays)
static cudaError_t ensure_main_capacity(CRContext *c, size_t need)
{
    cudaError_t err = cudaSuccess;
    if (need <= c->cap)
        return cudaSuccess;

    size_t new_cap = c->cap ? c->cap : 1;
    while (new_cap < need)
        new_cap <<= 1;

    uint8_t *nd = nullptr, *nt = nullptr;
    CUDA_OK(cudaMalloc(&nd, new_cap));
    CUDA_OK(cudaMalloc(&nt, new_cap));

    if (c->d_buf && c->len)
    {
        CUDA_OK(cudaMemcpyAsync(nd, c->d_buf, c->len, cudaMemcpyDeviceToDevice, c->stream));
        CUDA_OK(cudaStreamSynchronize(c->stream));
    }

    uint8_t *old_buf = c->d_buf, *old_tmp = c->d_tmp;
    c->d_buf = nd;
    c->d_tmp = nt;
    c->cap = new_cap;
    if (old_buf)
        cudaFree(old_buf);
    if (old_tmp)
        cudaFree(old_tmp);
    return cudaSuccess;
cuda_fail:
    if (nd)
        cudaFree(nd);
    if (nt)
        cudaFree(nt);
    return err;
}

// **NEW**: ensure aux arrays sized for a given length
static cudaError_t ensure_aux_for_len(CRContext *c, size_t need)
{
    cudaError_t err = cudaSuccess;
    if (need == 0)
        return cudaSuccess;
    if (need <= c->aux_cap)
        return cudaSuccess;

    size_t new_cap = c->aux_cap ? c->aux_cap : 1;
    while (new_cap < need)
        new_cap <<= 1;

    bool *nm = nullptr, *nm2 = nullptr;
    uint16_t *np = nullptr;
    uint8_t *nf = nullptr;
    int *nl = nullptr;

    CUDA_OK(cudaMalloc(&nm, new_cap * sizeof(bool)));
    CUDA_OK(cudaMalloc(&nm2, new_cap * sizeof(bool)));
    CUDA_OK(cudaMalloc(&np, new_cap * sizeof(uint16_t)));
    CUDA_OK(cudaMalloc(&nf, new_cap * sizeof(uint8_t)));
    CUDA_OK(cudaMalloc(&nl, new_cap * sizeof(int)));

    bool *om = c->d_match, *om2 = c->d_match2;
    uint16_t *op = c->d_padded;
    uint8_t *of = c->d_flags;
    int *ol = c->d_local;

    c->d_match = nm;
    c->d_match2 = nm2;
    c->d_padded = np;
    c->d_flags = nf;
    c->d_local = nl;
    c->aux_cap = new_cap;

    if (om)
        cudaFree(om);
    if (om2)
        cudaFree(om2);
    if (op)
        cudaFree(op);
    if (of)
        cudaFree(of);
    if (ol)
        cudaFree(ol);

    // singletons (unchanged if already allocated)
    if (!c->d_outlen)
        CUDA_OK(cudaMalloc(&c->d_outlen, sizeof(size_t)));
    if (!c->d_totmatch)
        CUDA_OK(cudaMalloc(&c->d_totmatch, sizeof(int)));
    if (!c->d_kept_count)
        CUDA_OK(cudaMalloc(&c->d_kept_count, sizeof(int)));
    if (!c->d_first_kept)
        CUDA_OK(cudaMalloc(&c->d_first_kept, sizeof(int)));
    if (!c->d_last_kept)
        CUDA_OK(cudaMalloc(&c->d_last_kept, sizeof(int)));

    return cudaSuccess;
cuda_fail:
    if (nm)
        cudaFree(nm);
    if (nm2)
        cudaFree(nm2);
    if (np)
        cudaFree(np);
    if (nf)
        cudaFree(nf);
    if (nl)
        cudaFree(nl);
    return err;
}

// chunk buffers (atomic swap)
static cudaError_t ensure_chunk_buffers(CRContext *c, int num_chunks)
{
    cudaError_t err = cudaSuccess;
    if (num_chunks <= 0)
    {
        c->num_chunks = 0;
        return cudaSuccess;
    }
    if (num_chunks == c->num_chunks && c->d_chunk_addrs && c->d_chunk_masks)
        return cudaSuccess;

    size_t *n_addrs = nullptr;
    uint64_t *n_masks = nullptr;
    uint8_t *n_flags = nullptr;
    int *n_local = nullptr, *n_indices = nullptr;
    int *n_bcounts = nullptr, *n_bprefix = nullptr;

    CUDA_OK(cudaMalloc(&n_addrs, (size_t)num_chunks * sizeof(size_t)));
    CUDA_OK(cudaMalloc(&n_masks, (size_t)num_chunks * 4 * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc(&n_flags, (size_t)num_chunks * sizeof(uint8_t)));
    CUDA_OK(cudaMalloc(&n_local, (size_t)num_chunks * sizeof(int)));
    CUDA_OK(cudaMalloc(&n_indices, (size_t)num_chunks * sizeof(int)));

    int chunk_blocks = (num_chunks + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_OK(cudaMalloc(&n_bcounts, (size_t)chunk_blocks * sizeof(int)));
    CUDA_OK(cudaMalloc(&n_bprefix, (size_t)chunk_blocks * sizeof(int)));

    // success → swap, then free old
    size_t *o_addrs = c->d_chunk_addrs;
    c->d_chunk_addrs = n_addrs;
    uint64_t *o_masks = c->d_chunk_masks;
    c->d_chunk_masks = n_masks;
    uint8_t *o_flags = c->d_chunk_flags;
    c->d_chunk_flags = n_flags;
    int *o_local = c->d_chunk_local;
    c->d_chunk_local = n_local;
    int *o_bcounts = c->d_chunk_bcounts;
    c->d_chunk_bcounts = n_bcounts;
    int *o_bprefix = c->d_chunk_bprefix;
    c->d_chunk_bprefix = n_bprefix;
    int *o_indices = c->d_chunk_indices;
    c->d_chunk_indices = n_indices;

    c->num_chunks = num_chunks;

    if (o_addrs)
        cudaFree(o_addrs);
    if (o_masks)
        cudaFree(o_masks);
    if (o_flags)
        cudaFree(o_flags);
    if (o_local)
        cudaFree(o_local);
    if (o_bcounts)
        cudaFree(o_bcounts);
    if (o_bprefix)
        cudaFree(o_bprefix);
    if (o_indices)
        cudaFree(o_indices);

    return cudaSuccess;
cuda_fail:
    if (n_addrs)
        cudaFree(n_addrs);
    if (n_masks)
        cudaFree(n_masks);
    if (n_flags)
        cudaFree(n_flags);
    if (n_local)
        cudaFree(n_local);
    if (n_indices)
        cudaFree(n_indices);
    if (n_bcounts)
        cudaFree(n_bcounts);
    if (n_bprefix)
        cudaFree(n_bprefix);
    return err;
}

// device-scan over counts -> prefix (exclusive)
static int device_scan_counts(CRContext *c, int n_counts, const int *d_counts, int *d_prefix)
{
    cudaError_t err = cudaSuccess;
    if (n_counts <= 0)
        return 0;

    CUDA_OK(ensure_tiles(c, n_counts));
    {
        int num_tiles = (n_counts + SCAN_TILE - 1) / SCAN_TILE;
        if (num_tiles <= 0)
            return 0;
        scan_counts_stage1_kernel<<<num_tiles, SCAN_TILE, 0, c->stream>>>(d_counts, n_counts, d_prefix, c->d_tile_sums);
        KERNEL_OK();
        scan_counts_stage2_kernel<<<1, 1, 0, c->stream>>>(c->d_tile_sums, num_tiles, c->d_tile_prefix);
        KERNEL_OK();
        scan_counts_stage3_add_kernel<<<num_tiles, SCAN_TILE, 0, c->stream>>>(d_prefix, n_counts, c->d_tile_prefix);
        KERNEL_OK();
    }
    return 0;
cuda_fail:
    return (int)err;
}

// ===========================
// Marking — legacy dispatcher
// ===========================
static int mark_matches_dispatch(CRContext *c, const uint8_t *d_pat, int pat_len, bool /*use_active*/)
{
    cudaError_t err = cudaSuccess;
    if (c->len == 0)
        return 0;

    CUDA_OK(cudaMemsetAsync(c->d_match, 0, c->len * sizeof(bool), c->stream));

    const int max_smem = dev_max_smem();
    const bool use_smem = (pat_len > 0 && pat_len <= max_smem);

    if (c->grid.x == 0)
        return 0; // nothing to launch

    if (use_smem)
    {
        size_t smem = (size_t)pat_len;
        mark_match_positions_kernel_smem<<<c->grid, c->block, smem, c->stream>>>(c->d_match, c->d_buf, c->len, d_pat, pat_len);
    }
    else
    {
        mark_match_positions_kernel_gmem<<<c->grid, c->block, 0, c->stream>>>(c->d_match, c->d_buf, c->len, d_pat, pat_len);
    }
    KERNEL_OK();
    return 0;

cuda_fail:
    return (int)err;
}

// ===========================
// NEW fused mark+suppr driver (seedless and seeded)
// ===========================
static int fused_mark_suppress_bitmap(CRContext *c, const uint8_t *d_pat, int L)
{
    cudaError_t err = cudaSuccess;
    if (c->len == 0)
        return 0;
    if (c->fused_tile_bytes <= 0)
        return -2;

    int num_tiles = (int)((c->len + (size_t)c->fused_tile_bytes - 1) / (size_t)c->fused_tile_bytes);
    if (num_tiles <= 0)
    {
        CUDA_OK(cudaMemsetAsync(c->d_match, 0, c->len * sizeof(bool), c->stream));
        return 0;
    }

    // ensure fused buffers
    if (c->fused_num_tiles != num_tiles || !c->d_fused_last || !c->d_fused_prev)
    {
        int *n_last = nullptr, *n_prev = nullptr;
        CUDA_OK(cudaMalloc(&n_last, (size_t)num_tiles * sizeof(int)));
        CUDA_OK(cudaMalloc(&n_prev, (size_t)num_tiles * sizeof(int)));
        if (c->d_fused_last)
            cudaFree(c->d_fused_last);
        if (c->d_fused_prev)
            cudaFree(c->d_fused_prev);
        c->d_fused_last = n_last;
        c->d_fused_prev = n_prev;
        c->fused_num_tiles = num_tiles;
    }

    int halo = (L > 0 ? L - 1 : 0);
    size_t smem_need = (size_t)L + (size_t)c->fused_tile_bytes + (size_t)halo + (size_t)c->fused_tile_bytes;

    if (smem_need > (size_t)dev_max_smem())
        return -2;

    fused_tile_lastkeep_kernel<<<num_tiles, BLOCK_SIZE, smem_need, c->stream>>>(
        c->d_buf, c->len, d_pat, L, c->fused_tile_bytes, c->d_fused_last);
    KERNEL_OK();

    fused_tile_prevlast_kernel<<<1, 1, 0, c->stream>>>(c->d_fused_last, num_tiles, c->d_fused_prev);
    KERNEL_OK();

    CUDA_OK(cudaMemsetAsync(c->d_match, 0, c->len * sizeof(bool), c->stream));
    fused_tile_write_bitmap_kernel<<<num_tiles, BLOCK_SIZE, smem_need, c->stream>>>(
        c->d_match, c->len, c->d_buf, d_pat, L, c->fused_tile_bytes, c->d_fused_prev);
    KERNEL_OK();

    return 0;
cuda_fail:
    return (int)err;
}

static int fused_mark_suppress_bitmap_seeded(CRContext *c, const uint8_t *d_pat, int L, int initial_last)
{
    cudaError_t err = cudaSuccess;
    if (c->len == 0)
        return 0;
    if (c->fused_tile_bytes <= 0)
        return -2;

    int num_tiles = (int)((c->len + (size_t)c->fused_tile_bytes - 1) / (size_t)c->fused_tile_bytes);
    if (num_tiles <= 0)
    {
        CUDA_OK(cudaMemsetAsync(c->d_match, 0, c->len * sizeof(bool), c->stream));
        return 0;
    }

    if (c->fused_num_tiles != num_tiles || !c->d_fused_last || !c->d_fused_prev)
    {
        int *n_last = nullptr, *n_prev = nullptr;
        CUDA_OK(cudaMalloc(&n_last, (size_t)num_tiles * sizeof(int)));
        CUDA_OK(cudaMalloc(&n_prev, (size_t)num_tiles * sizeof(int)));
        if (c->d_fused_last)
            cudaFree(c->d_fused_last);
        if (c->d_fused_prev)
            cudaFree(c->d_fused_prev);
        c->d_fused_last = n_last;
        c->d_fused_prev = n_prev;
        c->fused_num_tiles = num_tiles;
    }

    int halo = (L > 0 ? L - 1 : 0);
    size_t smem_need = (size_t)L + (size_t)c->fused_tile_bytes + (size_t)halo + (size_t)c->fused_tile_bytes;

    if (smem_need > (size_t)dev_max_smem())
        return -2;

    fused_tile_lastkeep_kernel<<<num_tiles, BLOCK_SIZE, smem_need, c->stream>>>(
        c->d_buf, c->len, d_pat, L, c->fused_tile_bytes, c->d_fused_last);
    KERNEL_OK();

    fused_tile_prevlast_kernel_seeded<<<1, 1, 0, c->stream>>>(
        c->d_fused_last, num_tiles, initial_last, c->d_fused_prev);
    KERNEL_OK();

    CUDA_OK(cudaMemsetAsync(c->d_match, 0, c->len * sizeof(bool), c->stream));
    fused_tile_write_bitmap_kernel<<<num_tiles, BLOCK_SIZE, smem_need, c->stream>>>(
        c->d_match, c->len, c->d_buf, d_pat, L, c->fused_tile_bytes, c->d_fused_prev);
    KERNEL_OK();

    return 0;
cuda_fail:
    return (int)err;
}

// ===========================
// Shrink & Expand (use bitmap)
// ===========================
__global__ void shrink_pad_kernel_smem(
    const uint8_t *__restrict__ src, size_t src_len,
    const bool *__restrict__ match_positions,
    const uint8_t *__restrict__ pattern, int pattern_len,
    const uint8_t *__restrict__ replacement, int replacement_len,
    uint16_t *__restrict__ padded)
{
    extern __shared__ uint8_t smem[];
    uint8_t *shared_pattern = smem;
    uint8_t *shared_replacement = smem + pattern_len;

    for (int i = threadIdx.x; i < pattern_len; i += blockDim.x)
        shared_pattern[i] = pattern[i];
    for (int i = threadIdx.x; i < replacement_len; i += blockDim.x)
        shared_replacement[i] = replacement[i];
    __syncthreads();

    for (size_t pos = grid_stride_start(); pos < src_len; pos += grid_stride_step())
    {
        if (match_positions[pos] && pos + (size_t)pattern_len <= src_len)
        {
            for (int i = 0; i < pattern_len; ++i)
                padded[pos + i] = (i < replacement_len) ? (uint16_t)shared_replacement[i] : (uint16_t)PADDING_MARKER_16;
        }
        else
        {
            bool covered = false;
            int start = (int)pos - (pattern_len - 1);
            if (start < 0)
                start = 0;
            for (int s = start; s <= (int)pos; ++s)
            {
                if (s < (int)src_len && match_positions[s] && (size_t)s + (size_t)pattern_len > pos)
                {
                    covered = true;
                    break;
                }
            }
            if (!covered)
                padded[pos] = (uint16_t)src[pos];
        }
    }
}

__global__ void shrink_pad_kernel_gmem(
    const uint8_t *__restrict__ src, size_t src_len,
    const bool *__restrict__ match_positions,
    const uint8_t *__restrict__ pattern, int pattern_len,
    const uint8_t *__restrict__ replacement, int replacement_len,
    uint16_t *__restrict__ padded)
{
    for (size_t pos = grid_stride_start(); pos < src_len; pos += grid_stride_step())
    {
        if (match_positions[pos] && pos + (size_t)pattern_len <= src_len)
        {
            for (int i = 0; i < pattern_len; ++i)
                padded[pos + i] = (i < replacement_len) ? (uint16_t)replacement[i] : (uint16_t)PADDING_MARKER_16;
        }
        else
        {
            bool covered = false;
            int start = (int)pos - (pattern_len - 1);
            if (start < 0)
                start = 0;
            for (int s = start; s <= (int)pos; ++s)
            {
                if (s < (int)src_len && match_positions[s] && (size_t)s + (size_t)pattern_len > pos)
                {
                    covered = true;
                    break;
                }
            }
            if (!covered)
                padded[pos] = (uint16_t)src[pos];
        }
    }
}

// Fused-aware shrink: must still avoid copying bytes that are covered by a kept match.
// We only skip the extra *shared-memory* staging here; semantics stay identical.
__global__ void shrink_pad_kernel_fused(
    const uint8_t *__restrict__ src, size_t src_len,
    const bool *__restrict__ match_positions,
    const uint8_t *__restrict__ pattern, int pattern_len,
    const uint8_t *__restrict__ replacement, int replacement_len,
    uint16_t *__restrict__ padded)
{
    for (size_t pos = grid_stride_start(); pos < src_len; pos += grid_stride_step())
    {
        if (match_positions[pos] && pos + (size_t)pattern_len <= src_len)
        {
            // Write replacement bytes at the kept start; mark the tail as padding.
            for (int i = 0; i < pattern_len; ++i)
            {
                padded[pos + i] = (i < replacement_len)
                                      ? (uint16_t)replacement[i]
                                      : (uint16_t)PADDING_MARKER_16;
            }
        }
        else
        {
            // IMPORTANT: even in fused mode we must not copy source bytes that fall
            // inside the span of any kept match starting at s where s <= pos < s+L.
            bool covered = false;
            int start = (int)pos - (pattern_len - 1);
            if (start < 0)
                start = 0;

            for (int s = start; s <= (int)pos; ++s)
            {
                if ((size_t)s < src_len &&
                    match_positions[s] &&
                    (size_t)s + (size_t)pattern_len > pos)
                {
                    covered = true;
                    break;
                }
            }

            if (!covered)
                padded[pos] = (uint16_t)src[pos];
            // else: leave as-is; the position will be compacted away
        }
    }
}

// ===========================
// NEW: Range-limited expansion kernels (to avoid long-running launches)
// ===========================
__global__ void expansion_scatter_kernel_range(
    const uint8_t *__restrict__ src, size_t src_len,
    const bool *__restrict__ match_positions,
    const uint8_t *__restrict__ replacement, int pattern_len, int replacement_len,
    const int *__restrict__ match_block_prefix, const int *__restrict__ match_local_pos,
    uint8_t *__restrict__ dst, size_t /*dst_len*/,
    size_t start, size_t end)
{
    const int diff = replacement_len - pattern_len;
    for (size_t pos = grid_stride_start() + start; pos < end; pos += grid_stride_step())
    {
        int block = (int)(pos / (size_t)BLOCK_SIZE);
        // --- FIX: Promote excl_before to size_t before multiplication to avoid overflow ---
        size_t excl_before = (size_t)match_block_prefix[block] + (size_t)match_local_pos[pos];
        size_t dst_pos = pos + excl_before * (size_t)diff;

        if (match_positions[pos])
        {
            for (int i = 0; i < replacement_len; ++i)
                dst[dst_pos + i] = replacement[i];
        }
        else
        {
            bool covered = false;
            int start_s = (int)pos - (pattern_len - 1);
            if (start_s < 0)
                start_s = 0;
            for (int s = start_s; s <= (int)pos; ++s)
            {
                if ((size_t)s < src_len && match_positions[s] && (size_t)s + (size_t)pattern_len > pos)
                {
                    covered = true;
                    break;
                }
            }
            if (!covered)
                dst[dst_pos] = src[pos];
        }
    }
}

__global__ void expansion_scatter_kernel_fused_range(
    const uint8_t *__restrict__ src, size_t src_len,
    const bool *__restrict__ match_positions,
    const uint8_t *__restrict__ replacement, int pattern_len, int replacement_len,
    const int *__restrict__ match_block_prefix, const int *__restrict__ match_local_pos,
    uint8_t *__restrict__ dst, size_t /*dst_len*/,
    size_t start, size_t end)
{
    const size_t sdiff = (size_t)(replacement_len - pattern_len);
    for (size_t pos = grid_stride_start() + start; pos < end; pos += grid_stride_step())
    {
        const int block = (int)(pos / (size_t)BLOCK_SIZE);
        const int excl_before = match_block_prefix[block] + match_local_pos[pos];
        // --- FIX: Promote to size_t to avoid 32-bit overflow ---
        const size_t dst_pos = (size_t)pos + (size_t)excl_before * sdiff;
        if (match_positions[pos])
        {
            for (int i = 0; i < replacement_len; ++i)
                dst[dst_pos + i] = replacement[i];
        }
        else
        {
            if (pattern_len > 1)
            {
                bool covered = false;
                int s0 = (int)pos - (pattern_len - 1);
                if (s0 < 0)
                    s0 = 0;
                for (int s = s0; s <= (int)pos; ++s)
                {
                    if ((size_t)s < src_len &&
                        match_positions[s] &&
                        (size_t)s + (size_t)pattern_len > pos)
                    {
                        covered = true;
                        break;
                    }
                }
                if (!covered)
                    dst[dst_pos] = src[pos];
            }
            else
            {
                dst[dst_pos] = src[pos];
            }
        }
    }
}

// ===========================
// Suppression (seedless and seeded)
// ===========================
static int suppress_global_leftmost(CRContext *c, int pat_len, int *h_kept_count, int *h_first_kept, int *h_last_kept)
{
    cudaError_t err = cudaSuccess;
    if (c->len == 0)
    {
        if (h_kept_count)
            *h_kept_count = 0;
        if (h_first_kept)
            *h_first_kept = -1;
        if (h_last_kept)
            *h_last_kept = -1;
        return 0;
    }
    compute_launch(c);
    CUDA_OK(ensure_blocks(c, c->len));
    if (c->grid.x == 0)
    {
        if (h_kept_count)
            *h_kept_count = 0;
        if (h_first_kept)
            *h_first_kept = -1;
        if (h_last_kept)
            *h_last_kept = -1;
        return 0;
    }
    // Build match flags and local positions
    build_match_flags_localpos_kernel<<<c->grid, c->block, 0, c->stream>>>(c->d_match, c->len, c->d_flags, c->d_local, c->d_bcounts);
    KERNEL_OK();
    // Scan block counts
    int rc = device_scan_counts(c, c->grid.x, c->d_bcounts, c->d_bprefix);
    if (rc != 0)
    {
        err = cudaErrorUnknown;
        goto cuda_fail;
    }
    // Compute total matches
    compute_total_matches_kernel<<<1, 1, 0, c->stream>>>(c->d_bprefix, c->d_bcounts, c->grid.x, c->d_totmatch);
    KERNEL_OK();
    int h_matches = 0;
    CUDA_OK(cudaMemcpyAsync(&h_matches, c->d_totmatch, sizeof(int), cudaMemcpyDeviceToHost, c->stream));
    CUDA_OK(cudaStreamSynchronize(c->stream));
    if (h_matches == 0)
    {
        CUDA_OK(cudaMemsetAsync(c->d_match2, 0, c->len * sizeof(bool), c->stream));
        {
            bool *t = c->d_match;
            c->d_match = c->d_match2;
            c->d_match2 = t;
        }
        if (h_kept_count)
            *h_kept_count = 0;
        if (h_first_kept)
            *h_first_kept = -1;
        if (h_last_kept)
            *h_last_kept = -1;
        return 0;
    }
    // Allocate and compute kept indices
    int *d_indices = nullptr;
    uint8_t *d_keep_flags = nullptr;
    CUDA_OK(cudaMalloc(&d_indices, (size_t)h_matches * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_keep_flags, (size_t)h_matches * sizeof(uint8_t)));
    CUDA_OK(cudaMemsetAsync(d_keep_flags, 0, (size_t)h_matches * sizeof(uint8_t), c->stream));
    match_indices_scatter_kernel<<<c->grid, c->block, 0, c->stream>>>(c->d_flags, c->d_local, c->d_bprefix, c->len, d_indices);
    KERNEL_OK();
    // Apply greedy suppression
    greedy_keep_on_indices_kernel<<<1, 1, 0, c->stream>>>(d_indices, h_matches, pat_len, d_keep_flags, c->d_kept_count, c->d_first_kept, c->d_last_kept);
    KERNEL_OK();
    // Copy results back
    int kept = 0, first_kept = -1, last_kept = -1;
    CUDA_OK(cudaMemcpyAsync(&kept, c->d_kept_count, sizeof(int), cudaMemcpyDeviceToHost, c->stream));
    CUDA_OK(cudaMemcpyAsync(&first_kept, c->d_first_kept, sizeof(int), cudaMemcpyDeviceToHost, c->stream));
    CUDA_OK(cudaMemcpyAsync(&last_kept, c->d_last_kept, sizeof(int), cudaMemcpyDeviceToHost, c->stream));
    CUDA_OK(cudaStreamSynchronize(c->stream));
    if (h_kept_count)
        *h_kept_count = kept;
    if (h_first_kept)
        *h_first_kept = first_kept;
    if (h_last_kept)
        *h_last_kept = last_kept;
    // Update the kept-starts bitmap
    CUDA_OK(cudaMemsetAsync(c->d_match2, 0, c->len * sizeof(bool), c->stream));
    set_bitmap_from_kept_indices_kernel<<<c->grid, c->block, 0, c->stream>>>(c->d_match2, c->len, d_indices, d_keep_flags, h_matches);
    KERNEL_OK();
    {
        bool *t = c->d_match;
        c->d_match = c->d_match2;
        c->d_match2 = t;
    }
    // Cleanup
    cudaFree(d_indices);
    cudaFree(d_keep_flags);
    return 0;
cuda_fail:
    if (h_kept_count)
        *h_kept_count = 0;
    if (h_first_kept)
        *h_first_kept = -1;
    if (h_last_kept)
        *h_last_kept = -1;
    return (int)err;
}

// ===== NEW: seeded suppression over kept-start candidates =====
static int suppress_global_leftmost_seeded(
    CRContext *c, int pat_len, int initial_last,
    int *h_kept_count, int *h_first_kept, int *h_last_kept)
{
    cudaError_t err = cudaSuccess;
    if (c->len == 0)
    {
        if (h_kept_count)
            *h_kept_count = 0;
        if (h_first_kept)
            *h_first_kept = -1;
        if (h_last_kept)
            *h_last_kept = -1;
        return 0;
    }
    compute_launch(c);
    CUDA_OK(ensure_blocks(c, c->len));

    build_match_flags_localpos_kernel<<<c->grid, c->block, 0, c->stream>>>(
        c->d_match, c->len, c->d_flags, c->d_local, c->d_bcounts);
    KERNEL_OK();

    {
        int rc = device_scan_counts(c, c->grid.x, c->d_bcounts, c->d_bprefix);
        if (rc != 0)
        {
            err = cudaErrorUnknown;
            goto cuda_fail;
        }
    }

    compute_total_matches_kernel<<<1, 1, 0, c->stream>>>(c->d_bprefix, c->d_bcounts, c->grid.x, c->d_totmatch);
    KERNEL_OK();

    int m = 0;
    CUDA_OK(cudaMemcpyAsync(&m, c->d_totmatch, sizeof(int), cudaMemcpyDeviceToHost, c->stream));
    CUDA_OK(cudaStreamSynchronize(c->stream));

    CUDA_OK(cudaMemsetAsync(c->d_match2, 0, c->len * sizeof(bool), c->stream));
    if (m == 0)
    {
        if (h_kept_count)
            *h_kept_count = 0;
        if (h_first_kept)
            *h_first_kept = -1;
        if (h_last_kept)
            *h_last_kept = -1;
        {
            bool *t = c->d_match;
            c->d_match = c->d_match2;
            c->d_match2 = t;
        }
        return 0;
    }

    int *d_indices = nullptr;
    uint8_t *d_keep_flags = nullptr;
    CUDA_OK(cudaMalloc(&d_indices, (size_t)m * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_keep_flags, (size_t)m * sizeof(uint8_t)));
    CUDA_OK(cudaMemsetAsync(d_keep_flags, 0, (size_t)m * sizeof(uint8_t), c->stream));

    match_indices_scatter_kernel<<<c->grid, c->block, 0, c->stream>>>(
        c->d_flags, c->d_local, c->d_bprefix, c->len, d_indices);
    KERNEL_OK();

    greedy_keep_on_indices_kernel_seeded<<<1, 1, 0, c->stream>>>(
        d_indices, m, pat_len, initial_last,
        d_keep_flags, c->d_kept_count, c->d_first_kept, c->d_last_kept);
    KERNEL_OK();

    int kept = 0, first_kept = -1, last_kept = -1;
    CUDA_OK(cudaMemcpyAsync(&kept, c->d_kept_count, sizeof(int), cudaMemcpyDeviceToHost, c->stream));
    CUDA_OK(cudaMemcpyAsync(&first_kept, c->d_first_kept, sizeof(int), cudaMemcpyDeviceToHost, c->stream));
    CUDA_OK(cudaMemcpyAsync(&last_kept, c->d_last_kept, sizeof(int), cudaMemcpyDeviceToHost, c->stream));
    CUDA_OK(cudaStreamSynchronize(c->stream));
    if (h_kept_count)
        *h_kept_count = kept;
    if (h_first_kept)
        *h_first_kept = first_kept;
    if (h_last_kept)
        *h_last_kept = last_kept;

    set_bitmap_from_kept_indices_kernel<<<c->grid, c->block, 0, c->stream>>>(
        c->d_match2, c->len, d_indices, d_keep_flags, m);
    KERNEL_OK();
    {
        bool *t = c->d_match;
        c->d_match = c->d_match2;
        c->d_match2 = t;
    }

    cudaFree(d_indices);
    cudaFree(d_keep_flags);
    return 0;

cuda_fail:
    return (int)err;
}

// ===========================
// Prefix queries over kept-starts bitmap
// ===========================
// ===========================
// Prefix queries over kept-starts bitmap
// ===========================
// ===========================
// Prefix queries over kept-starts bitmap  (FIXED)
// ===========================
static int query_kept_prefix(CRContext *c, size_t limit, int *h_count, int *h_last_lt)
{
    cudaError_t err = cudaSuccess;

    // Defaults
    if (h_count)
        *h_count = 0;
    if (h_last_lt)
        *h_last_lt = -1;

    // Nothing to do
    if (limit == 0)
        return 0;

    // IMPORTANT: operate on the bitmap range [0, limit), NOT on c->len (which may be post-scatter).
    const size_t n = limit;

    // Ensure capacity for exactly 'n' bytes of bitmap work
    CUDA_OK(ensure_aux_for_len(c, n));
    CUDA_OK(ensure_blocks(c, n));

    // Local launch config for n bytes (do NOT use c->grid, which was computed for c->len)
    const int blocks = (int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    if (blocks <= 0)
        return 0;
    const dim3 grid_n((unsigned)blocks, 1, 1);
    const dim3 block_n(BLOCK_SIZE, 1, 1);

    // Build flags/local positions for KEPT bitmap over [0, n)
    build_match_flags_localpos_kernel<<<grid_n, block_n, 0, c->stream>>>(
        c->d_match, n, c->d_flags, c->d_local, c->d_bcounts);
    KERNEL_OK();

    // Scan block counts -> block prefix for 'blocks' only
    {
        int rc = device_scan_counts(c, blocks, c->d_bcounts, c->d_bprefix);
        if (rc != 0)
        {
            err = cudaErrorUnknown;
            goto cuda_fail;
        }
    }

    // Count kept-starts strictly before 'limit' == exactly all kept within [0, n)
    if (h_count)
    {
        compute_total_matches_kernel<<<1, 1, 0, c->stream>>>(c->d_bprefix, c->d_bcounts, blocks, c->d_totmatch);
        KERNEL_OK();

        int cnt = 0;
        CUDA_OK(cudaMemcpyAsync(&cnt, c->d_totmatch, sizeof(int), cudaMemcpyDeviceToHost, c->stream));
        CUDA_OK(cudaStreamSynchronize(c->stream));
        *h_count = cnt;
    }

    // Last kept-start index (< limit). Evaluate only within [0, n).
    if (h_last_lt)
    {
        int *d_out = nullptr;
        CUDA_OK(cudaMalloc(&d_out, sizeof(int)));
        CUDA_OK(cudaMemsetAsync(d_out, 0xFF, sizeof(int), c->stream)); // -1
        last_true_before_limit_kernel<<<grid_n, block_n, 0, c->stream>>>(c->d_match, n, n, d_out);
        KERNEL_OK();
        CUDA_OK(cudaMemcpyAsync(h_last_lt, d_out, sizeof(int), cudaMemcpyDeviceToHost, c->stream));
        CUDA_OK(cudaStreamSynchronize(c->stream));
        cudaFree(d_out);
    }

    return 0;

cuda_fail:
    return (int)err;
}

// ===========================
// Apply (shrink / expand) — internals
// ===========================
static int apply_shrink(CRContext *c, const uint8_t *d_pat, int pat_len,
                        const uint8_t *d_rep, int rep_len, size_t *new_len_out,
                        bool /*use_active_chunks*/,
                        int *o_kept, int *o_first_kept, int *o_last_kept,
                        int used_fused, int /*initial_last*/)
{
    cudaError_t err = cudaSuccess;
    if (c->len == 0)
    {
        if (new_len_out)
            *new_len_out = 0;
        return 0;
    }

    compute_launch(c);
    CUDA_OK(ensure_blocks(c, c->len));
    CUDA_OK(ensure_aux_for_len(c, c->len));

    // Mark + suppress already executed by caller; bitmap in c->d_match.
    // Pad with replacement or marker
    if (used_fused)
    {
        shrink_pad_kernel_fused<<<c->grid, c->block, 0, c->stream>>>(
            c->d_buf, c->len, c->d_match, d_pat, pat_len, d_rep, rep_len, c->d_padded);
    }
    else
    {
        const int max_smem = dev_max_smem();
        bool use_smem = (pat_len + rep_len) <= max_smem && pat_len > 0 && rep_len >= 0;
        if (use_smem)
        {
            size_t smem = (size_t)pat_len + (size_t)rep_len;
            shrink_pad_kernel_smem<<<c->grid, c->block, smem, c->stream>>>(
                c->d_buf, c->len, c->d_match, d_pat, pat_len, d_rep, rep_len, c->d_padded);
        }
        else
        {
            shrink_pad_kernel_gmem<<<c->grid, c->block, 0, c->stream>>>(
                c->d_buf, c->len, c->d_match, d_pat, pat_len, d_rep, rep_len, c->d_padded);
        }
    }
    KERNEL_OK();

    // Compact non-marker u16 → bytes
    comp_flags_localpos_kernel<<<c->grid, c->block, 0, c->stream>>>(
        c->d_padded, c->len, c->d_flags, c->d_local, c->d_bcounts);
    KERNEL_OK();

    {
        int rc = device_scan_counts(c, c->grid.x, c->d_bcounts, c->d_bprefix);
        if (rc != 0)
        {
            err = cudaErrorUnknown;
            goto cuda_fail;
        }
    }

    compute_total_len_kernel<<<1, 1, 0, c->stream>>>(c->d_bprefix, c->d_bcounts, c->grid.x, c->d_outlen);
    KERNEL_OK();

    size_t new_len = 0;
    CUDA_OK(cudaMemcpyAsync(&new_len, c->d_outlen, sizeof(size_t), cudaMemcpyDeviceToHost, c->stream));
    CUDA_OK(cudaStreamSynchronize(c->stream));

    CUDA_OK(ensure_main_capacity(c, new_len ? new_len : 1));
    comp_scatter_kernel<<<c->grid, c->block, 0, c->stream>>>(
        c->d_padded, c->len, c->d_flags, c->d_local, c->d_bprefix, c->d_tmp);
    KERNEL_OK();
    CUDA_OK(cudaStreamSynchronize(c->stream));

    // swap
    {
        uint8_t *t = c->d_buf;
        c->d_buf = c->d_tmp;
        c->d_tmp = t;
    }
    c->len = new_len;
    if (new_len_out)
        *new_len_out = new_len;

    // report kept info if requested
    if (o_kept || o_first_kept || o_last_kept)
    {
        int kept = 0, first = -1, last = -1;
        CUDA_OK(cudaMemcpyAsync(&kept, c->d_kept_count, sizeof(int), cudaMemcpyDeviceToHost, c->stream));
        CUDA_OK(cudaMemcpyAsync(&first, c->d_first_kept, sizeof(int), cudaMemcpyDeviceToHost, c->stream));
        CUDA_OK(cudaMemcpyAsync(&last, c->d_last_kept, sizeof(int), cudaMemcpyDeviceToHost, c->stream));
        CUDA_OK(cudaStreamSynchronize(c->stream));
        if (o_kept)
            *o_kept = kept;
        if (o_first_kept)
            *o_first_kept = first;
        if (o_last_kept)
            *o_last_kept = last;
    }
    return 0;

cuda_fail:
    return (int)err;
}

static int apply_expand(CRContext *c, const uint8_t *d_pat, int pat_len,
                        const uint8_t *d_rep, int rep_len, size_t *new_len_out,
                        bool /*use_active_chunks*/,
                        int *o_kept, int *o_first_kept, int *o_last_kept,
                        int used_fused, int /*initial_last*/)
{
    cudaError_t err = cudaSuccess;
    if (c->len == 0)
    {
        if (new_len_out)
            *new_len_out = 0;
        return 0;
    }

    compute_launch(c);
    CUDA_OK(ensure_blocks(c, c->len));
    CUDA_OK(ensure_aux_for_len(c, c->len));

    // We already have kept-starts bitmap in c->d_match.
    // Build flags/local for kept-starts (count & positions)
    build_match_flags_localpos_kernel<<<c->grid, c->block, 0, c->stream>>>(
        c->d_match, c->len, c->d_flags, c->d_local, c->d_bcounts);
    KERNEL_OK();

    {
        int rc = device_scan_counts(c, c->grid.x, c->d_bcounts, c->d_bprefix);
        if (rc != 0)
        {
            err = cudaErrorUnknown;
            goto cuda_fail;
        }
    }

    // total kept
    compute_total_matches_kernel<<<1, 1, 0, c->stream>>>(c->d_bprefix, c->d_bcounts, c->grid.x, c->d_totmatch);
    KERNEL_OK();

    int kept = 0, first_kept = -1, last_kept = -1;
    CUDA_OK(cudaMemcpyAsync(&kept, c->d_totmatch, sizeof(int), cudaMemcpyDeviceToHost, c->stream));
    // also fetch first/last if produced earlier
    CUDA_OK(cudaMemcpyAsync(&first_kept, c->d_first_kept, sizeof(int), cudaMemcpyDeviceToHost, c->stream));
    CUDA_OK(cudaMemcpyAsync(&last_kept, c->d_last_kept, sizeof(int), cudaMemcpyDeviceToHost, c->stream));
    CUDA_OK(cudaStreamSynchronize(c->stream));

    const int diff = rep_len - pat_len;
    size_t new_len = c->len + (size_t)kept * (size_t)diff;

    CUDA_OK(ensure_main_capacity(c, new_len ? new_len : 1));

    // Range-sliced expansion to avoid long-running kernels on huge inputs
    const size_t RANGE = (size_t)64 * 1024 * 1024; // 64 MiB slices of source positions
    for (size_t s = 0; s < c->len; s += RANGE)
    {
        size_t e = (s + RANGE < c->len ? s + RANGE : c->len);
        if (used_fused)
        {
            expansion_scatter_kernel_fused_range<<<c->grid, c->block, 0, c->stream>>>(
                c->d_buf, c->len, c->d_match, d_rep, pat_len, rep_len,
                c->d_bprefix, c->d_local, c->d_tmp, new_len, s, e);
        }
        else
        {
            expansion_scatter_kernel_range<<<c->grid, c->block, 0, c->stream>>>(
                c->d_buf, c->len, c->d_match, d_rep, pat_len, rep_len,
                c->d_bprefix, c->d_local, c->d_tmp, new_len, s, e);
        }
        KERNEL_OK();
    }
    CUDA_OK(cudaStreamSynchronize(c->stream));

    // swap
    {
        uint8_t *t = c->d_buf;
        c->d_buf = c->d_tmp;
        c->d_tmp = t;
    }
    c->len = new_len;
    if (new_len_out)
        *new_len_out = new_len;

    if (o_kept)
        *o_kept = kept;
    if (o_first_kept)
        *o_first_kept = first_kept;
    if (o_last_kept)
        *o_last_kept = last_kept;

    return 0;

cuda_fail:
    return (int)err;
}

// mark + suppress + scatter (one pass selector)
static int apply_once(CRContext *c, const uint8_t *d_pat, int pat_len,
                      const uint8_t *d_rep, int rep_len, size_t *new_len_out,
                      bool use_active_chunks,
                      int *o_kept, int *o_first_kept, int *o_last_kept,
                      int initial_last)
{
    cudaError_t err = cudaSuccess;
    if (pat_len <= 0)
    {
        if (new_len_out)
            *new_len_out = c->len;
        return 0;
    }

    compute_launch(c);
    CUDA_OK(ensure_blocks(c, c->len));
    CUDA_OK(ensure_aux_for_len(c, c->len));

    // mark & suppress
    int used_fused = 0;
    if (c->use_fused)
    {
        int rc = (initial_last <= INT_MIN / 8)
                     ? fused_mark_suppress_bitmap(c, d_pat, pat_len)
                     : fused_mark_suppress_bitmap_seeded(c, d_pat, pat_len, initial_last);
        if (rc == 0)
            used_fused = 1;
        else if (rc != -2)
        {
            err = cudaErrorUnknown;
            goto cuda_fail;
        }
    }
    if (!used_fused)
    {
        int rc = mark_matches_dispatch(c, d_pat, pat_len, use_active_chunks);
        if (rc != 0)
        {
            err = cudaErrorUnknown;
            goto cuda_fail;
        }
        if (initial_last <= INT_MIN / 8)
            rc = suppress_global_leftmost(c, pat_len, o_kept, o_first_kept, o_last_kept);
        else
            rc = suppress_global_leftmost_seeded(c, pat_len, initial_last, o_kept, o_first_kept, o_last_kept);
        if (rc != 0)
        {
            err = cudaErrorUnknown;
            goto cuda_fail;
        }
    }

    // scatter
    if (rep_len <= pat_len)
        return apply_shrink(c, d_pat, pat_len, d_rep, rep_len, new_len_out, use_active_chunks,
                            o_kept, o_first_kept, o_last_kept, used_fused, initial_last);
    else
        return apply_expand(c, d_pat, pat_len, d_rep, rep_len, new_len_out, use_active_chunks,
                            o_kept, o_first_kept, o_last_kept, used_fused, initial_last);

cuda_fail:
    return (int)err;
}

// ===========================
// Exports — session & one-shot
// ===========================
extern "C"
{

    static void cr_destroy(CRContext *c)
    {
        if (!c)
            return;
        // Streams first
        if (c->stream)
            cudaStreamDestroy(c->stream);
        if (c->copy_stream)
            cudaStreamDestroy(c->copy_stream);

        // Main buffers
        if (c->d_buf)
            cudaFree(c->d_buf);
        if (c->d_tmp)
            cudaFree(c->d_tmp);

        // Aux
        if (c->d_match)
            cudaFree(c->d_match);
        if (c->d_match2)
            cudaFree(c->d_match2);
        if (c->d_padded)
            cudaFree(c->d_padded);
        if (c->d_flags)
            cudaFree(c->d_flags);
        if (c->d_local)
            cudaFree(c->d_local);
        if (c->d_bcounts)
            cudaFree(c->d_bcounts);
        if (c->d_bprefix)
            cudaFree(c->d_bprefix);
        if (c->d_outlen)
            cudaFree(c->d_outlen);
        if (c->d_totmatch)
            cudaFree(c->d_totmatch);
        if (c->d_kept_count)
            cudaFree(c->d_kept_count);
        if (c->d_first_kept)
            cudaFree(c->d_first_kept);
        if (c->d_last_kept)
            cudaFree(c->d_last_kept);

        if (c->d_tile_sums)
            cudaFree(c->d_tile_sums);
        if (c->d_tile_prefix)
            cudaFree(c->d_tile_prefix);

        // Presence index
        if (c->d_chunk_addrs)
            cudaFree(c->d_chunk_addrs);
        if (c->d_chunk_masks)
            cudaFree(c->d_chunk_masks);
        if (c->d_chunk_flags)
            cudaFree(c->d_chunk_flags);
        if (c->d_chunk_local)
            cudaFree(c->d_chunk_local);
        if (c->d_chunk_bcounts)
            cudaFree(c->d_chunk_bcounts);
        if (c->d_chunk_bprefix)
            cudaFree(c->d_chunk_bprefix);
        if (c->d_chunk_indices)
            cudaFree(c->d_chunk_indices);

        // Fused
        if (c->d_fused_last)
            cudaFree(c->d_fused_last);
        if (c->d_fused_prev)
            cudaFree(c->d_fused_prev);

        delete c;
    }

    static int env_flag(const char *name, int defv)
    {
        const char *v = getenv(name);
        if (!v)
            return defv;
        if (v[0] == '0')
            return 0;
        if (v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N')
            return 0;
        return 1;
    }

    EXPORT const char *cuda_replace_version()
    {
        return "cuda_replace 2025-09-08 fused+seeded r3";
    }

    EXPORT int cuda_replace_open(void **ph, const uint8_t *src, size_t len)
    {
        if (!ph)
            return -1;
        *ph = nullptr;

        CRContext *c = new (std::nothrow) CRContext();
        if (!c)
            return -2;

        cudaError_t err = cudaSuccess;

        // Streams
        CUDA_OK(cudaStreamCreate(&c->stream));
        CUDA_OK(cudaStreamCreate(&c->copy_stream));

        c->use_fused = env_flag("CUDA_REPLACE_FUSED", 1);

        // Capacity & aux
        CUDA_OK(ensure_main_capacity(c, len ? len : 1));
        CUDA_OK(ensure_aux_for_len(c, len ? len : 1));
        c->len = len;

        // Upload
        if (len)
            CUDA_OK(cudaMemcpyAsync(c->d_buf, src, len, cudaMemcpyHostToDevice, c->stream));
        CUDA_OK(cudaStreamSynchronize(c->stream));

        compute_launch(c);
        CUDA_OK(ensure_blocks(c, c->len));

        *ph = (void *)c;
        return 0;

    cuda_fail:
        cr_destroy(c);
        return (int)err;
    }

    EXPORT int cuda_replace_reset(void *h, const uint8_t *src, size_t len)
    {
        if (!h)
            return -1;
        CRContext *c = (CRContext *)h;
        std::lock_guard<std::mutex> lock(c->mtx);
        if (!c->alive)
            return -3;

        cudaError_t err = cudaSuccess;

        CUDA_OK(ensure_main_capacity(c, len ? len : 1));
        CUDA_OK(ensure_aux_for_len(c, len ? len : 1));
        if (len)
            CUDA_OK(cudaMemcpyAsync(c->d_buf, src, len, cudaMemcpyHostToDevice, c->stream));
        c->len = len;
        CUDA_OK(cudaStreamSynchronize(c->stream));

        compute_launch(c);
        CUDA_OK(ensure_blocks(c, c->len));
        return 0;
    cuda_fail:
        return (int)err;
    }

    EXPORT int cuda_replace_build_index(void *h, size_t chunk_size)
    {
        if (!h)
            return -1;
        CRContext *c = (CRContext *)h;
        std::lock_guard<std::mutex> lock(c->mtx);
        if (!c->alive)
            return -3;

        if (chunk_size == 0)
            chunk_size = c->chunk_size;
        c->chunk_size = chunk_size;

        if (c->len == 0)
        {
            c->num_chunks = 0;
            return 0;
        }

        int num_chunks = (int)div_ceil_u64((uint64_t)c->len, (uint64_t)chunk_size);
        if (num_chunks <= 0)
            return 0;

        cudaError_t err = cudaSuccess;
        CUDA_OK(ensure_chunk_buffers(c, num_chunks));

        build_chunk_index_kernel<<<num_chunks, 256, 0, c->stream>>>(
            c->d_buf, c->len, chunk_size, num_chunks, c->d_chunk_addrs, c->d_chunk_masks);
        KERNEL_OK();
        CUDA_OK(cudaStreamSynchronize(c->stream));

        return 0;
    cuda_fail:
        return (int)err;
    }

    EXPORT int cuda_replace_apply(void *h,
                                  const uint8_t *pat, int pat_len,
                                  const uint8_t *rep, int rep_len,
                                  size_t *new_len)
    {
        if (!h)
            return -1;
        CRContext *c = (CRContext *)h;
        std::lock_guard<std::mutex> lock(c->mtx);
        if (!c->alive)
            return -3;

        cudaError_t err = cudaSuccess;
        uint8_t *d_pat = nullptr, *d_rep = nullptr;
        if (pat_len > 0)
            CUDA_OK(cudaMalloc(&d_pat, (size_t)pat_len));
        if (rep_len > 0)
            CUDA_OK(cudaMalloc(&d_rep, (size_t)rep_len));
        if (pat_len > 0)
            CUDA_OK(cudaMemcpyAsync(d_pat, pat, (size_t)pat_len, cudaMemcpyHostToDevice, c->stream));
        if (rep_len > 0)
            CUDA_OK(cudaMemcpyAsync(d_rep, rep, (size_t)rep_len, cudaMemcpyHostToDevice, c->stream));
        CUDA_OK(cudaStreamSynchronize(c->stream));

        int kept = 0, first = -1, last = -1;
        int rc = apply_once(c, d_pat, pat_len, d_rep, rep_len, new_len, /*use_active*/ false,
                            &kept, &first, &last, INT_MIN / 4);
        if (rc != 0)
        {
            err = cudaErrorUnknown;
            goto cuda_fail;
        }

        if (d_pat)
            cudaFree(d_pat);
        if (d_rep)
            cudaFree(d_rep);
        return 0;

    cuda_fail:
        if (d_pat)
            cudaFree(d_pat);
        if (d_rep)
            cudaFree(d_rep);
        return (int)err;
    }

    EXPORT int cuda_replace_apply_seeded(void *h,
                                         const uint8_t *pat, int pat_len,
                                         const uint8_t *rep, int rep_len,
                                         long long prev_last_rel, long long *out_last_rel,
                                         size_t *new_len)
    {
        if (!h)
            return -1;
        CRContext *c = (CRContext *)h;
        std::lock_guard<std::mutex> lock(c->mtx);
        if (!c->alive)
            return -3;

        cudaError_t err = cudaSuccess;
        uint8_t *d_pat = nullptr, *d_rep = nullptr;
        if (pat_len > 0)
            CUDA_OK(cudaMalloc(&d_pat, (size_t)pat_len));
        if (rep_len > 0)
            CUDA_OK(cudaMalloc(&d_rep, (size_t)rep_len));
        if (pat_len > 0)
            CUDA_OK(cudaMemcpyAsync(d_pat, pat, (size_t)pat_len, cudaMemcpyHostToDevice, c->stream));
        if (rep_len > 0)
            CUDA_OK(cudaMemcpyAsync(d_rep, rep, (size_t)rep_len, cudaMemcpyHostToDevice, c->stream));
        CUDA_OK(cudaStreamSynchronize(c->stream));

        int kept = 0, first = -1, last = -1;
        int rc = apply_once(c, d_pat, pat_len, d_rep, rep_len, new_len, /*use_active*/ false,
                            &kept, &first, &last, (int)prev_last_rel);
        if (rc != 0)
        {
            err = cudaErrorUnknown;
            goto cuda_fail;
        }
        if (out_last_rel)
            *out_last_rel = (long long)last;

        if (d_pat)
            cudaFree(d_pat);
        if (d_rep)
            cudaFree(d_rep);
        return 0;
    cuda_fail:
        if (d_pat)
            cudaFree(d_pat);
        if (d_rep)
            cudaFree(d_rep);
        return (int)err;
    }

    EXPORT int cuda_replace_query_prefix(void *h, size_t limit,
                                         int *kept_before_limit,
                                         int *last_kept_before_limit)
    {
        if (!h)
            return -1;
        CRContext *c = (CRContext *)h;
        std::lock_guard<std::mutex> lock(c->mtx);
        if (!c->alive)
            return -3;
        return query_kept_prefix(c, limit, kept_before_limit, last_kept_before_limit);
    }

    EXPORT int cuda_replace_apply_batch(void *h,
                                        const uint8_t **pats, const int *pat_lens,
                                        const uint8_t **reps, const int *rep_lens,
                                        int count, size_t *new_len_out)
    {
        if (!h)
            return -1;
        CRContext *c = (CRContext *)h;
        std::lock_guard<std::mutex> lock(c->mtx);
        if (!c->alive)
            return -3;

        cudaError_t err = cudaSuccess;
        size_t cur_len = c->len;

        for (int i = 0; i < count; ++i)
        {
            const uint8_t *p = pats ? pats[i] : nullptr;
            const uint8_t *r = reps ? reps[i] : nullptr;
            int pl = pat_lens ? pat_lens[i] : 0;
            int rl = rep_lens ? rep_lens[i] : 0;

            // ✅ Allocate device memory for this pattern/replacement
            uint8_t *d_pat = nullptr, *d_rep = nullptr;
            if (pl > 0)
            {
                CUDA_OK(cudaMalloc(&d_pat, (size_t)pl));
                CUDA_OK(cudaMemcpyAsync(d_pat, p, (size_t)pl,
                                        cudaMemcpyHostToDevice, c->stream));
            }
            if (rl > 0)
            {
                CUDA_OK(cudaMalloc(&d_rep, (size_t)rl));
                CUDA_OK(cudaMemcpyAsync(d_rep, r, (size_t)rl,
                                        cudaMemcpyHostToDevice, c->stream));
            }
            CUDA_OK(cudaStreamSynchronize(c->stream));

            // Apply with device pointers
            int kept = 0, first = -1, last = -1;
            int rc = apply_once(c, d_pat, pl, d_rep, rl, &cur_len,
                                false, &kept, &first, &last, INT_MIN / 4);

            // Cleanup
            if (d_pat)
                cudaFree(d_pat);
            if (d_rep)
                cudaFree(d_rep);

            if (rc != 0)
            {
                err = cudaErrorUnknown;
                goto cuda_fail;
            }
        }

        if (new_len_out)
            *new_len_out = cur_len;
        return 0;

    cuda_fail:
        return (int)err;
    }
    EXPORT int cuda_replace_result(void *h, uint8_t **out_host, size_t *outlen_host)
    {
        if (!h || !out_host || !outlen_host)
            return -1;
        CRContext *c = (CRContext *)h;
        std::lock_guard<std::mutex> lock(c->mtx);
        if (!c->alive)
            return -3;

        *out_host = nullptr;
        *outlen_host = c->len;

        if (c->len == 0)
            return 0;

        void *host_ptr = nullptr;
        cudaError_t err = cudaSuccess;
        CUDA_OK(cudaMallocHost(&host_ptr, c->len));
        CUDA_OK(cudaMemcpyAsync(host_ptr, c->d_buf, c->len, cudaMemcpyDeviceToHost, c->stream));
        CUDA_OK(cudaStreamSynchronize(c->stream));

        *out_host = (uint8_t *)host_ptr;
        return 0;
    cuda_fail:
        if (host_ptr)
            cudaFreeHost(host_ptr);
        *out_host = nullptr;
        *outlen_host = 0;
        return (int)err;
    }

    EXPORT void cuda_free_host(void *p)
    {
        if (p)
            cudaFreeHost(p);
    }

    EXPORT void cuda_replace_close(void *h)
    {
        if (!h)
            return;
        CRContext *c = (CRContext *)h;
        {
            std::lock_guard<std::mutex> lock(c->mtx);
            if (!c->alive)
                return;
            c->alive = false;
        }
        cr_destroy(c);
    }

    EXPORT int cuda_replace_unified(
        const uint8_t *src, size_t src_len,
        const uint8_t *pat, int pat_len,
        const uint8_t *rep, int rep_len,
        uint8_t **out_host, size_t *out_len)
    {
        if (!out_host || !out_len)
            return -1;
        *out_host = nullptr;
        *out_len = 0;

        void *h = nullptr;
        int rc = cuda_replace_open(&h, src, src_len);
        if (rc != 0)
            return rc;

        rc = cuda_replace_apply(h, pat, pat_len, rep, rep_len, out_len);
        if (rc == 0)
            rc = cuda_replace_result(h, out_host, out_len);

        cuda_replace_close(h);
        return rc;
    }

} // extern "C"
