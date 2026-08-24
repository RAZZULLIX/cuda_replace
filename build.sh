#!/usr/bin/env bash
# Build the CUDA replace shared library for Linux.
#
# Produces a fat binary covering the GPU architectures below (plus PTX for
# forward compatibility).  Architectures unsupported by the installed nvcc are
# skipped automatically, so the build succeeds on older toolkits (you just get
# fewer embedded targets).
set -euo pipefail

cd "$(dirname "$0")"

# Architectures to embed as SASS (native code).  Edit to add/remove.
# Format: "compute_XY,sm_XY"  (the "compute_XY,compute_XY" entry is PTX for JIT).
GENS=(
  "compute_75,sm_75"          # Turing (RTX 20xx / GTX 16xx)
  "compute_80,sm_80"          # A100
  "compute_86,sm_86"          # Ampere (RTX 30xx)
  "compute_89,sm_89"          # Ada Lovelace (RTX 40xx)
  "compute_90,sm_90"          # Hopper (H100)
  "compute_120,sm_120"        # Blackwell (RTX 50xx)
  "compute_121,sm_121"        # Blackwell B200
  "compute_121,compute_121"   # PTX for future GPUs (JIT)
)

# Query the installed nvcc for the architectures it actually supports.
SUPPORTED="$(nvcc --list-gpu-arch 2>/dev/null || true)"

GENCODE_ARGS=()
EMBEDDED=()
for g in "${GENS[@]}"; do
  arch="${g%,*}"       # e.g. compute_86
  if [[ -z "$SUPPORTED" ]] || grep -qw "$arch" <<<"$SUPPORTED"; then
    GENCODE_ARGS+=( -gencode "arch=$arch,code=${g#*,}" )
    EMBEDDED+=("$g")
  fi
done

if [[ ${#GENCODE_ARGS[@]} -eq 0 ]]; then
  echo "error: no compatible GPU architecture found for this nvcc" >&2
  exit 1
fi

echo "Embedding: ${EMBEDDED[*]}"
nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC \
  "${GENCODE_ARGS[@]}" \
  -o libcuda_replace.so cuda_replace.cu

echo "Built libcuda_replace.so"
