// Back-to-back correctness check for the NCCL shim on two GPUs, one process.
//
// nccl-tests -c re-initializes buffers, runs ONE collective, synchronizes and
// checks it; its timed iterations are never checked. This harness checks
// every iteration of a back-to-back loop instead. Per iteration, on each GPU's
// stream and with no host synchronization until the end:
//   init kernel (inputs depend on rank, index, iteration) -> collective ->
//   check kernel (expected values computed in-kernel, mismatches counted).
// If the shim let a GPU return while its peer still read its send buffer or
// wrote its receive buffer, the next iteration's init kernel or this
// iteration's check would observe it.
//
// Data pattern. Each element is a 64-bit hash of (rank, index, iteration,
// seed) mapped to a signed integer whose magnitude keeps every sum exact:
// |v| <= 2^23-1 (fp32), 1023 (fp16), 127 (bf16, int8), 2^30-1 (int32). The
// previous (7*idx + 13*iter + 5*rank) mod 16 pattern repeated every 16
// elements, so data displaced by any multiple of 16 elements -- including
// every MiB-sized shim chunk boundary -- compared equal. Outputs are compared
// bit for bit against the converted expected value.
//
// Guards. Every send and receive buffer sits between two STRESS_GUARD_BYTES
// regions (default 4096) filled with 0xA5 and checked after the case, so a
// write one element past an odd tail or before a slot fails the case. Set
// STRESS_GUARD_BYTES=0 under compute-sanitizer memcheck so allocations end
// exactly at the buffer end and out-of-bounds reads are visible to it.
//
// Differential mode (STRESS_DIFF=1). Inputs are full-range random bit
// patterns: finite values of every exponent, subnormals, +-0, +-inf, the
// largest binade (sums overflow) and, on rank 1, near-negations of rank 0
// (catastrophic cancellation). Copy collectives also carry NaNs with random
// payloads. Each iteration first runs the same collective through libnccl's
// own entry points (looked up in libnccl.so.2 itself, bypassing any
// LD_PRELOAD interposer) into a shadow buffer, then through the interposed
// API, and compares the two outputs bitwise. A two-rank sum is one correctly
// rounded a+b per element, so stock and shim must agree exactly; positions
// where both outputs are NaN are skipped for reductions (NaN payload
// propagation is implementation-defined), never for copies.
//
// Boundary cases (bnd_*) are never scaled by STRESS_SCALE. Their sizes are
// derived from the frozen slim-shim policy, or explicit TEST_* tuning, defaulting to
// the competition "fixed" settings (peer-read 32 MiB, inputhybrid 192 MiB,
// 128 MiB NVLink chunks, 24 MiB host and tail chunks, divisors 7/7, AG/RS
// 4 MiB and Broadcast 8 MiB minimums, AG/RS/BC host divisors 7/7/6): each
// threshold and chunk edge at -1, 0 and +1 element, the host-share rounding
// edges, and 1 GiB. Pass the same TEST_* tuning to a stock run (no
// LD_PRELOAD) to get identical case sizes.
//
// Cases cover the four collectives, fp32/fp16/bf16/int32/int8, in-place and
// out-of-place, Broadcast roots 0 and 1, odd element counts, non-sum
// (NCCL fallback) ops, and a mixed AllGather+Broadcast group.
//
// Build: nvcc -O2 -std=c++17 -I$NCCL/include nccl_shim_stress_check.cu
//          -L$NCCL/lib -lnccl -ldl -Xcompiler -pthread -o nccl_shim_stress_check
// Env:   STRESS_ITERS (default 20; cases >= 1 GiB use ITERS/2, at least 2)
//        STRESS_SCALE (divide the standard cases' element counts by this,
//                      keeping count mod 16 so odd tails survive; default 1)
//        STRESS_CASE_FILTER (run only the case whose label exactly matches)
//        STRESS_BOUNDARY (0 drops the bnd_* cases; default 1)
//        STRESS_LIST_CASES (1: print the selected labels and exit)
//        STRESS_THREADS (1 or 2; independent communicator pairs and seeds)
//        STRESS_EXPECT_SUBMIT_ERROR (1: expect one CUDA error and check recovery)
//        STRESS_SHIM_TEST_TUNING (1: oracle reads TEST_* tuning)
//        STRESS_DIFF (1: differential mode against stock libnccl)
//        STRESS_SEED (data seed; default 1)
//        STRESS_GUARD_BYTES (guard size, multiple of 256; default 4096)
//        STRESS_TEST_CORRUPT_SHIFT (checker self-test: displace device 0's
//                      output by this many elements; affected cases must FAIL)
// Output: one TSV row per case:
//   case iters bytes_per_rank mismatches first_bad_iter status guard_bad
//   expect_route kind mode nan_skipped
// expect_route is the route the shim's documented auto policy should take
// (named custom route or nccl). Exit status 1 on
// any mismatch or error. Case labels also go to stderr ("CASE <label>") so
// NCCL_SHIM_DEBUG routing lines can be attributed to a case.
#include <nccl.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)
#define NCCL_OK(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) { \
  fprintf(stderr, "NCCL %s at %s:%d\n", ncclGetErrorString(r_), __FILE__, __LINE__); exit(2); } } while (0)

enum Coll { AR, AG, RS, BC, MIXED };
enum Mode { M_REDUCE, M_GATHER, M_RSCAT, M_BCAST };

// ---------------------------------------------------------------------------
// Data generation.

__host__ __device__ __forceinline__ uint64_t mix64(uint64_t z) {  // splitmix64
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Distinct (rank, idx, iter) give distinct keys (idx < 2^44), and mix64 is a
// bijection, so no two elements of a run share a hash.
__host__ __device__ __forceinline__ uint64_t elem_hash(int rank, size_t idx, int iter, uint64_t seed) {
  const uint64_t key = static_cast<uint64_t>(idx) ^
                       (static_cast<uint64_t>(iter * 2 + rank) << 44);
  return mix64(key ^ seed);
}

// Largest |v| whose two-rank sum is exactly representable in T.
template <typename T> struct Lim;
template <> struct Lim<float> { static constexpr int A = (1 << 23) - 1; };
template <> struct Lim<__half> { static constexpr int A = 1023; };
template <> struct Lim<__nv_bfloat16> { static constexpr int A = 127; };
template <> struct Lim<int32_t> { static constexpr int A = (1 << 30) - 1; };
template <> struct Lim<int8_t> { static constexpr int A = 127; };

template <typename T>
__host__ __device__ __forceinline__ int gen(int rank, size_t idx, int iter, uint64_t seed) {
  constexpr uint64_t span = 2ull * Lim<T>::A + 1;
  return static_cast<int>(elem_hash(rank, idx, iter, seed) % span) - Lim<T>::A;
}

template <typename T> __device__ __forceinline__ T from_int(int v);
template <> __device__ __forceinline__ float from_int<float>(int v) { return static_cast<float>(v); }
template <> __device__ __forceinline__ __half from_int<__half>(int v) { return __int2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_int<__nv_bfloat16>(int v) { return __int2bfloat16_rn(v); }
template <> __device__ __forceinline__ int32_t from_int<int32_t>(int v) { return v; }
template <> __device__ __forceinline__ int8_t from_int<int8_t>(int v) { return static_cast<int8_t>(v); }

template <typename T> __host__ __device__ __forceinline__ uint32_t bits_of(T v) {
  if constexpr (sizeof(T) == 4) { uint32_t u; memcpy(&u, &v, 4); return u; }
  else if constexpr (sizeof(T) == 2) { uint16_t u; memcpy(&u, &v, 2); return u; }
  else { uint8_t u; memcpy(&u, &v, 1); return u; }
}
template <typename T> __host__ __device__ __forceinline__ T from_bits(uint32_t u) {
  T v;
  if constexpr (sizeof(T) == 4) { memcpy(&v, &u, 4); }
  else if constexpr (sizeof(T) == 2) { const uint16_t s = static_cast<uint16_t>(u); memcpy(&v, &s, 2); }
  else { const uint8_t s = static_cast<uint8_t>(u); memcpy(&v, &s, 1); }
  return v;
}

// IEEE layout (total bits, exponent bits, mantissa bits); MB == 0 marks an
// integer type.
template <typename T> struct Fp { static constexpr int NB = 8 * sizeof(T), EB = 0, MB = 0; };
template <> struct Fp<float> { static constexpr int NB = 32, EB = 8, MB = 23; };
template <> struct Fp<__half> { static constexpr int NB = 16, EB = 5, MB = 10; };
template <> struct Fp<__nv_bfloat16> { static constexpr int NB = 16, EB = 8, MB = 7; };

template <typename T> __host__ __device__ __forceinline__ bool is_nan_bits(uint32_t u) {
  if constexpr (Fp<T>::MB == 0) { return false; }
  else {
    constexpr uint32_t emax = (1u << Fp<T>::EB) - 1, mmask = (1u << Fp<T>::MB) - 1;
    return ((u >> Fp<T>::MB) & emax) == emax && (u & mmask) != 0;
  }
}

// Full-range random bits for the differential mode. Classes, by six hash
// bits: +-0, +-inf, subnormals, largest binade, smallest normal binade, NaN
// with random payload (copies only), otherwise any finite exponent.
template <typename T> __host__ __device__ __forceinline__ uint32_t rand_bits_from(uint64_t h, bool allow_nan) {
  if constexpr (Fp<T>::MB == 0) {
    return static_cast<uint32_t>(h);
  } else {
    constexpr int EB = Fp<T>::EB, MB = Fp<T>::MB, NB = Fp<T>::NB;
    constexpr uint32_t emax = (1u << EB) - 1;
    const uint32_t sign = static_cast<uint32_t>(h >> 63);
    uint32_t mant = static_cast<uint32_t>(h >> 8) & ((1u << MB) - 1);
    const uint32_t cls = static_cast<uint32_t>(h >> 56) & 63;
    uint32_t e;
    if (cls < 2) { e = 0; mant = 0; }                        // +-0
    else if (cls < 4) { e = emax; mant = 0; }                // +-inf
    else if (cls < 10) { e = 0; if (!mant) mant = 1; }       // subnormal
    else if (cls < 13) { e = emax - 1; }                     // overflow on sum
    else if (cls < 15) { e = 1; }                            // underflow on difference
    else if (cls < 17 && allow_nan) { e = emax; if (!mant) mant = 1; }  // NaN
    else { e = static_cast<uint32_t>((h >> 32) & 0xFFFFFF) % emax; }    // finite
    return (sign << (NB - 1)) | (e << MB) | mant;
  }
}

template <typename T>
__host__ __device__ __forceinline__ uint32_t rand_bits(int rank, size_t idx, int iter, uint64_t seed,
                                              bool allow_nan) {
  const uint64_t h = elem_hash(rank, idx, iter, seed);
  if constexpr (Fp<T>::MB != 0) {
    // About 1/16 of rank 1's elements are rank 0's value negated with the
    // two low mantissa bits perturbed, so sums cancel to tiny or zero results.
    if (rank == 1 && ((h >> 4) & 15) == 0) {
      const uint32_t other = rand_bits_from<T>(elem_hash(0, idx, iter, seed), allow_nan);
      const uint32_t negated = other ^ (1u << (Fp<T>::NB - 1));
      const uint32_t near = negated ^ static_cast<uint32_t>(h & 3);
      return !allow_nan && is_nan_bits<T>(near) ? negated : near;  // inf must stay inf
    }
  }
  return rand_bits_from<T>(h, allow_nan);
}

// buf[i] = value of (rank, idx0 + i, iter)
template <typename T>
__global__ void init_kernel(T* buf, size_t n, size_t idx0, int rank, int iter, uint64_t seed,
                            int diff, int allow_nan) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    buf[i] = diff ? from_bits<T>(rand_bits<T>(rank, idx0 + i, iter, seed, allow_nan != 0))
                  : from_int<T>(gen<T>(rank, idx0 + i, iter, seed));
  }
}

__global__ void delay_cycles_kernel(unsigned long long cycles) {
  const unsigned long long start = clock64();
  while (clock64() - start < cycles) { }
}

__device__ __forceinline__ int combine(int a, int b, int op) {
  return op == 1 ? (a > b ? a : b) : a + b;  // 0 sum, 1 max
}

// n = elements checked in `out`; cnt = per-rank element count of the case.
template <typename T>
__global__ void check_kernel(const T* out, size_t n, size_t cnt, int mode, int rank,
                             int root, int op, int iter, uint64_t seed, unsigned long long* bad) {
  unsigned long long local = 0;
  for (size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x; j < n;
       j += (size_t)gridDim.x * blockDim.x) {
    int expect;
    if (mode == M_REDUCE) {
      expect = combine(gen<T>(0, j, iter, seed), gen<T>(1, j, iter, seed), op);
    } else if (mode == M_GATHER) {
      expect = gen<T>(static_cast<int>(j / cnt), j % cnt, iter, seed);
    } else if (mode == M_RSCAT) {
      const size_t idx = static_cast<size_t>(rank) * cnt + j;
      expect = combine(gen<T>(0, idx, iter, seed), gen<T>(1, idx, iter, seed), op);
    } else {
      expect = gen<T>(root, j, iter, seed);
    }
    if (bits_of<T>(out[j]) != bits_of<T>(from_int<T>(expect))) ++local;
  }
  if (local) atomicAdd(bad, local);
}

// Bitwise comparison of an output with the stock-NCCL shadow output.
template <typename T>
__global__ void diff_kernel(const T* out, const T* ref, size_t n, int nan_skip,
                            unsigned long long* bad, unsigned long long* nans) {
  unsigned long long local = 0, skipped = 0;
  for (size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x; j < n;
       j += (size_t)gridDim.x * blockDim.x) {
    const uint32_t a = bits_of<T>(out[j]), b = bits_of<T>(ref[j]);
    if (a == b) continue;
    if (nan_skip && is_nan_bits<T>(a) && is_nan_bits<T>(b)) ++skipped;
    else ++local;
  }
  if (local) atomicAdd(bad, local);
  if (skipped) atomicAdd(nans, skipped);
}

static constexpr unsigned char kGuardByte = 0xA5;

__global__ void guard_kernel(const unsigned char* p, size_t n, unsigned long long* bad) {
  unsigned long long local = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    if (p[i] != kGuardByte) ++local;
  if (local) atomicAdd(bad, local);
}

// ---------------------------------------------------------------------------
// Cases.

struct Case {
  std::string label;
  Coll coll;
  ncclDataType_t dt;
  ncclRedOp_t op;
  size_t count;   // AR/BC: count; AG: send count; RS: recv count
  bool inplace;
  int root;
  bool boundary;  // exact size, never scaled
};

static size_t dt_size(ncclDataType_t dt) {
  switch (dt) {
    case ncclInt8: return 1;
    case ncclFloat16: case ncclBfloat16: return 2;
    default: return 4;
  }
}

static const char* dt_name(ncclDataType_t dt) {
  switch (dt) {
    case ncclInt8: return "i8";
    case ncclFloat16: return "f16";
    case ncclBfloat16: return "bf16";
    case ncclInt32: return "i32";
    default: return "f32";
  }
}

static bool is_float(ncclDataType_t dt) {
  return dt == ncclFloat32 || dt == ncclFloat16 || dt == ncclBfloat16;
}

// Independent oracle for the slim shim's frozen policy. Test tuning is
// explicit: production runs ignore TEST_* even when those variables are set.
static bool contract_in_place(const Case& c,int dev) {
  const char* mask=getenv("STRESS_INPLACE_MASK");
  return mask ? ((atoi(mask) & (1<<dev))!=0) : c.inplace;
}
struct ShimCfg {
  size_t peerread_min = 0, inputhybrid_min = 0, chunk = 0, host_chunk = 0, tail_chunk = 0;
  size_t in_div = 7, out_div = 7;
  size_t ag_min = 0, rs_min = 0, bc_min = 0;
  double ag_div = 7, rs_div = 7, bc_div = 6;
};

static size_t env_size(const char* name, size_t def) {
  const char* e = getenv(name);
  return e && *e ? static_cast<size_t>(strtoull(e, nullptr, 10)) : def;
}
static ShimCfg read_shim_cfg() {
  constexpr size_t M = 1ull << 20;
  ShimCfg s;
  s.peerread_min = s.inputhybrid_min = 256 * M;
  s.chunk = 128 * M; s.host_chunk = s.tail_chunk = 24 * M;
  s.ag_min = size_t(4) << 30; s.rs_min = 256 * M; s.bc_min = 32 * M;
  if (env_size("STRESS_SHIM_TEST_TUNING", 0)) {
    s.peerread_min = env_size("NCCL_SHIM_TEST_PEERREAD_MIN_BYTES", s.peerread_min);
    s.inputhybrid_min = env_size("NCCL_SHIM_TEST_INPUTHYBRID_MIN_BYTES", s.inputhybrid_min);
    s.chunk = env_size("NCCL_SHIM_TEST_CHUNK_MIB", 128) * M;
    s.host_chunk = env_size("NCCL_SHIM_TEST_HOST_CHUNK_MIB", 24) * M;
    s.tail_chunk = env_size("NCCL_SHIM_TEST_TAIL_CHUNK_MIB", 24) * M;
    s.ag_min = s.rs_min = env_size("NCCL_SHIM_TEST_COLL_MIN_BYTES", s.ag_min);
    s.bc_min = env_size("NCCL_SHIM_TEST_COLL_MIN_BYTES", s.bc_min);
  }
  return s;
}

// Named routes are checked against the exit counters in a one-case process.
static const char* expect_route(const Case& c, const ShimCfg&) {
  const size_t n=c.count*dt_size(c.dt), logical=(c.coll==AG||c.coll==RS)?2*n:n;
  if(logical>(size_t(8)<<30))return "nccl";
  if(c.coll==AR && is_float(c.dt) && c.op==ncclSum && n>=(size_t(64)<<20))return "inputhybrid";
  if(c.coll==AG && logical>=(size_t(32)<<20))return "allgather";
  if(c.coll==RS && is_float(c.dt) && c.op==ncclSum && logical>=(size_t(64)<<20) && n%16==0)return "reducescatter";
  if(c.coll==BC && n>=(size_t(32)<<20))return "broadcast";
  return "nccl";
}

// Bytes the inputhybrid input phase stages through pinned host for one
// half of `part` bytes (shim_input_host_bytes with the divisor policy).
static size_t ih_host_bytes(const ShimCfg& s, size_t part) {
  if (s.in_div == 0 || s.host_chunk == 0) return 0;
  return ((part / s.in_div) / s.host_chunk) * s.host_chunk;
}

// Boundary cases: every edge at -1, 0 and +1 element of the given type.
static void add_boundary_cases(std::vector<Case>& v, const ShimCfg& s) {
  constexpr size_t G = 1ull << 30;
  // Adds sizes edge-step, edge (unless with_eq is false) and edge+step, in
  // elements; labels end in _m<step>el, _eq, _p<step>el.
  auto add3 = [&](const std::string& what, Coll coll, ncclDataType_t dt, ncclRedOp_t op,
                  size_t bytes, size_t step_elems, bool inplace, int root,
                  bool with_eq = true) {
    const size_t es = dt_size(dt);
    const size_t base = bytes / es;
    const std::string step = std::to_string(step_elems) + "el";
    for (int k = 0; k < 3; ++k) {
      if (k == 0 && base < step_elems + 1) continue;
      if (k == 1 && !with_eq) continue;
      const size_t cnt = k == 0 ? base - step_elems : k == 1 ? base : base + step_elems;
      std::string label = std::string("bnd_") +
                          (coll == AR ? "ar" : coll == AG ? "ag" : coll == RS ? "rs" : "bc") +
                          "_" + dt_name(dt) + "_" + what + "_" +
                          (k == 0 ? "m" + step : k == 1 ? std::string("eq") : "p" + step);
      if (inplace) label += "_inplace";
      if (coll == BC) label += "_root" + std::to_string(root);
      v.push_back({label, coll, dt, op, cnt, inplace, root, true});
    }
  };
  const ncclDataType_t f32 = ncclFloat32, f16 = ncclFloat16, bf16 = ncclBfloat16, i8 = ncclInt8;

  // AllReduce route thresholds (NCCL / peer-read / inputhybrid).
  add3("peerread_min", AR, f32, ncclSum, s.peerread_min, 1, false, 0);
  add3("peerread_min", AR, bf16, ncclSum, s.peerread_min, 1, false, 0);
  add3("inputhybrid_min", AR, f32, ncclSum, s.inputhybrid_min, 1, false, 0);
  add3("inputhybrid_min", AR, f16, ncclSum, s.inputhybrid_min, 1, true, 0);
  // Inputhybrid NVLink chunk edges: half sizes whose NVLink part is exactly
  // one and two chunks (the tail-chunk rule switches at remaining == chunk).
  if (s.chunk) {
    for (size_t k = 1; k <= 2; ++k) {
      for (size_t j = 0; j < 64; ++j) {
        const size_t part = k * s.chunk + j * s.host_chunk;
        if (ih_host_bytes(s, part) == j * s.host_chunk) {
          add3("nvchunk" + std::to_string(k), AR, f32, ncclSum, 2 * part, 1, false, 0);
          break;
        }
        if (!s.host_chunk) break;
      }
    }
  }
  // Host-share rounding: the half where part/divisor reaches one host chunk.
  if (s.in_div && s.host_chunk)
    add3("inhost_edge", AR, f32, ncclSum, 2 * s.in_div * s.host_chunk, 1, false, 0);
  if (s.out_div && s.host_chunk && s.out_div != s.in_div)
    add3("outhost_edge", AR, f32, ncclSum, 2 * s.out_div * s.host_chunk, 1, false, 0);
  add3("1GiB", AR, f32, ncclSum, G, 1, false, 0);
  add3("1GiB", AR, bf16, ncclSum, G, 1, true, 0);

  // ReduceScatter slots must stay 16-byte multiples to stay on the shim, so
  // its edges step by 16 bytes; +-1 element exercises the unaligned fallback.
  add3("rs_min", RS, f32, ncclSum, s.rs_min / 2, 4, false, 0);
  if (s.tail_chunk) {
    add3("tailchunk", RS, f32, ncclSum, s.tail_chunk, 4, false, 0);
    add3("tailchunk", RS, f32, ncclSum, s.tail_chunk, 1, false, 0, false);  // unaligned: NCCL
  }
  if (s.chunk) add3("nvchunk1", RS, f32, ncclSum, s.chunk, 4, true, 0);
  if (s.host_chunk && s.rs_div > 0) {
    const size_t edge = static_cast<size_t>(std::ceil(s.rs_div * s.host_chunk));
    add3("host_edge", RS, f32, ncclSum, edge, 4, false, 0);
    add3("host_edge", RS, bf16, ncclSum, edge, 8, false, 0);
  }
  add3("1GiB", RS, f32, ncclSum, G / 2, 4, false, 0);

  // AllGather: per-rank send bytes; the minimum applies to the output size.
  add3("ag_min", AG, i8, ncclSum, s.ag_min / 2, 1, false, 0);
  if (s.host_chunk && s.ag_div > 0) {
    const size_t edge = static_cast<size_t>(std::ceil(s.ag_div * s.host_chunk));
    add3("host_edge", AG, f16, ncclSum, edge, 1, false, 0);
    add3("host_edge", AG, f32, ncclSum, edge, 1, true, 0);
  }
  add3("1GiB", AG, i8, ncclSum, G / 2, 1, false, 0);

  // Broadcast, both roots.
  add3("bc_min", BC, i8, ncclSum, s.bc_min, 1, false, 1);
  if (s.host_chunk && s.bc_div > 0) {
    const size_t edge = static_cast<size_t>(std::ceil(s.bc_div * s.host_chunk));
    add3("host_edge", BC, f32, ncclSum, edge, 1, false, 1);
    add3("host_edge", BC, f32, ncclSum, edge, 1, true, 0);
  }
  add3("1GiB", BC, bf16, ncclSum, G, 1, false, 1);
}

// ---------------------------------------------------------------------------
// Stock NCCL reference for the differential mode.

struct RealNccl {
  decltype(&ncclGroupStart) group_start = nullptr;
  decltype(&ncclGroupEnd) group_end = nullptr;
  decltype(&ncclAllReduce) all_reduce = nullptr;
  decltype(&ncclAllGather) all_gather = nullptr;
  decltype(&ncclReduceScatter) reduce_scatter = nullptr;
  decltype(&ncclBroadcast) broadcast = nullptr;
};

template <typename F> static void load_sym(void* h, const char* name, F& out) {
  out = reinterpret_cast<F>(dlsym(h, name));
  if (!out) { fprintf(stderr, "dlsym(libnccl, %s) failed: %s\n", name, dlerror()); exit(2); }
}

// dlsym on libnccl's own handle starts the search at libnccl, so it returns
// libnccl's definitions even when an LD_PRELOAD library interposes them.
static RealNccl load_real_nccl() {
  void* h = dlopen("libnccl.so.2", RTLD_NOW | RTLD_NOLOAD);
  if (!h) h = dlopen("libnccl.so", RTLD_NOW | RTLD_NOLOAD);
  if (!h) { fprintf(stderr, "cannot locate the loaded libnccl: %s\n", dlerror()); exit(2); }
  RealNccl r;
  load_sym(h, "ncclGroupStart", r.group_start);
  load_sym(h, "ncclGroupEnd", r.group_end);
  load_sym(h, "ncclAllReduce", r.all_reduce);
  load_sym(h, "ncclAllGather", r.all_gather);
  load_sym(h, "ncclReduceScatter", r.reduce_scatter);
  load_sym(h, "ncclBroadcast", r.broadcast);
  // RTLD_NEXT from the executable resolves to the first definition after it:
  // the interposer when one is preloaded, libnccl otherwise.
  void* next = dlsym(RTLD_NEXT, "ncclAllReduce");
  Dl_info ref_info{}, next_info{};
  dladdr(reinterpret_cast<void*>(r.all_reduce), &ref_info);
  dladdr(next, &next_info);
  fprintf(stderr, "[stress-test] diff reference=%s interposer=%s reference_distinct=%d\n",
          ref_info.dli_fname ? ref_info.dli_fname : "?",
          next_info.dli_fname ? next_info.dli_fname : "?",
          next && next != reinterpret_cast<void*>(r.all_reduce) ? 1 : 0);
  return r;
}

// ---------------------------------------------------------------------------
// Launch helpers.

struct Run {
  uint64_t seed;
  bool diff;
};

template <typename T>
static void launch_init(T* p, size_t n, size_t idx0, int rank, int iter, const Run& run,
                        bool allow_nan, cudaStream_t s) {
  if (n) init_kernel<T><<<528, 256, 0, s>>>(p, n, idx0, rank, iter, run.seed, run.diff,
                                             allow_nan);
}
template <typename T>
static void launch_check(const T* p, size_t n, size_t cnt, int mode, int rank, int root,
                         int op, int iter, const Run& run, unsigned long long* bad,
                         cudaStream_t s) {
  if (n) check_kernel<T><<<528, 256, 0, s>>>(p, n, cnt, mode, rank, root, op, iter, run.seed,
                                              bad);
}
template <typename T>
static void launch_diff(const void* out, const void* ref, size_t n, bool nan_skip,
                        unsigned long long* bad, unsigned long long* nans, cudaStream_t s) {
  if (n) diff_kernel<T><<<528, 256, 0, s>>>(static_cast<const T*>(out),
                                             static_cast<const T*>(ref), n, nan_skip, bad,
                                             nans);
}

#define DISPATCH(dt, FN, ...) do { switch (dt) { \
  case ncclFloat32: FN<float>(__VA_ARGS__); break; \
  case ncclFloat16: FN<__half>(__VA_ARGS__); break; \
  case ncclBfloat16: FN<__nv_bfloat16>(__VA_ARGS__); break; \
  case ncclInt32: FN<int32_t>(__VA_ARGS__); break; \
  case ncclInt8: FN<int8_t>(__VA_ARGS__); break; \
  default: fprintf(stderr, "unsupported dtype\n"); exit(2); } } while (0)

static bool reduces(const Case& c) { return c.coll == AR || c.coll == RS; }

template <typename T>
static void init_case(const Case& c, int rank, void* send, void* recv, int iter, const Run& run,
                      cudaStream_t s) {
  T* sp = static_cast<T*>(send);
  const bool nan_ok = !reduces(c);  // NaN payloads only where the output is a copy
  switch (c.coll) {
    case AR: launch_init<T>(sp, c.count, 0, rank, iter, run, nan_ok, s); break;
    case AG: launch_init<T>(sp, c.count, 0, rank, iter, run, nan_ok, s); break;  // sp is the slot when in-place
    case RS: launch_init<T>(sp, 2 * c.count, 0, rank, iter, run, nan_ok, s); break;
    case BC:
      if (rank == c.root) launch_init<T>(sp, c.count, 0, rank, iter, run, nan_ok, s);
      break;
    case MIXED:
      launch_init<T>(sp, c.count, 0, rank, iter, run, nan_ok, s);  // AllGather input
      if (rank == c.root)  // Broadcast input after the gather slots
        launch_init<T>(static_cast<T*>(recv) + 2 * c.count, c.count, 0, rank, iter, run, nan_ok, s);
      break;
  }
}

template <typename T>
static void check_case(const Case& c, int rank, void* recv, int iter, const Run& run,
                       unsigned long long* bad, cudaStream_t s) {
  const T* rp = static_cast<const T*>(recv);
  const int op = c.op == ncclMax ? 1 : 0;
  switch (c.coll) {
    case AR: launch_check<T>(rp, c.count, c.count, M_REDUCE, rank, 0, op, iter, run, bad, s); break;
    case AG: launch_check<T>(rp, 2 * c.count, c.count, M_GATHER, rank, 0, 0, iter, run, bad, s); break;
    case RS: launch_check<T>(rp, c.count, c.count, M_RSCAT, rank, 0, op, iter, run, bad, s); break;
    case BC: launch_check<T>(rp, c.count, c.count, M_BCAST, rank, c.root, 0, iter, run, bad, s); break;
    case MIXED:
      launch_check<T>(rp, 2 * c.count, c.count, M_GATHER, rank, 0, 0, iter, run, bad, s);
      launch_check<T>(rp + 3 * c.count, c.count, c.count, M_BCAST, rank, c.root, 0, iter, run, bad, s);
      break;
  }
}

// Compares `out` with the shadow output over the regions the collective
// writes (MIXED skips its Broadcast source slot).
template <typename T>
static void diff_case(const Case& c, const void* out, const void* shadow,
                      unsigned long long* bad, unsigned long long* nans, cudaStream_t s) {
  const bool nan_skip = reduces(c);
  const T* o = static_cast<const T*>(out);
  const T* r = static_cast<const T*>(shadow);
  switch (c.coll) {
    case AR: case RS: case BC: launch_diff<T>(o, r, c.count, nan_skip, bad, nans, s); break;
    case AG: launch_diff<T>(o, r, 2 * c.count, nan_skip, bad, nans, s); break;
    case MIXED:
      launch_diff<T>(o, r, 2 * c.count, false, bad, nans, s);
      launch_diff<T>(o + 3 * c.count, r + 3 * c.count, c.count, false, bad, nans, s);
      break;
  }
}

// A buffer of `bytes` between two guard regions; `p` is the usable pointer.
struct Guarded {
  char* base = nullptr;
  char* p = nullptr;
  size_t bytes = 0;
};

static void v2_alloc(char** p,size_t n) {
  if(getenv("STRESS_NATIVE_ALLOC")) NCCL_OK(ncclMemAlloc(reinterpret_cast<void**>(p),n));
  else CUDA_OK(cudaMalloc(p,n));
}
static void v2_free(char* p) {
  if(getenv("STRESS_NATIVE_ALLOC")) NCCL_OK(ncclMemFree(p));
  else CUDA_OK(cudaFree(p));
}
static Guarded guarded_alloc(size_t bytes, size_t guard, int fill) {
  Guarded g;
  g.bytes = bytes;
  v2_alloc(&g.base, bytes + 2 * guard + (bytes + 2 * guard == 0));
  if (guard) CUDA_OK(cudaMemset(g.base, kGuardByte, bytes + 2 * guard));
  g.p = g.base + guard;
  if (fill >= 0 && bytes) CUDA_OK(cudaMemset(g.p, fill, bytes));
  return g;
}

static void guard_check(const Guarded& g, size_t guard, unsigned long long* bad,
                        cudaStream_t s) {
  if (!guard) return;
  guard_kernel<<<64, 256, 0, s>>>(reinterpret_cast<unsigned char*>(g.base), guard, bad);
  guard_kernel<<<64, 256, 0, s>>>(reinterpret_cast<unsigned char*>(g.p) + g.bytes, guard, bad);
}

// Reusable per-case rendezvous: both host threads own independent
// communicator pairs and streams, and enter the enqueue loop together.
static int g_workers = 1;
static std::mutex g_barrier_mutex;
static std::condition_variable g_barrier_cv;
static int g_barrier_arrived = 0, g_barrier_generation = 0;
static void case_barrier() {
  if (g_workers == 1) return;
  std::unique_lock<std::mutex> lock(g_barrier_mutex);
  const int generation = g_barrier_generation;
  if (++g_barrier_arrived == g_workers) {
    g_barrier_arrived = 0;
    ++g_barrier_generation;
    g_barrier_cv.notify_all();
  } else {
    g_barrier_cv.wait(lock, [generation] { return generation != g_barrier_generation; });
  }
}

static int run_stress(int worker) {
  const int iters_env = getenv("STRESS_ITERS") ? atoi(getenv("STRESS_ITERS")) : 20;
  const size_t scale = getenv("STRESS_SCALE") ? strtoull(getenv("STRESS_SCALE"), nullptr, 10) : 1;
  const bool boundary = !getenv("STRESS_BOUNDARY") || atoi(getenv("STRESS_BOUNDARY")) != 0;
  const bool list_only = getenv("STRESS_LIST_CASES") && atoi(getenv("STRESS_LIST_CASES")) != 0;
  Run run;
  run.diff = getenv("STRESS_DIFF") && atoi(getenv("STRESS_DIFF")) != 0;
  run.seed = mix64((getenv("STRESS_SEED") ? strtoull(getenv("STRESS_SEED"), nullptr, 10) : 1) + worker);
  const size_t guard = getenv("STRESS_GUARD_BYTES")
      ? (strtoull(getenv("STRESS_GUARD_BYTES"), nullptr, 10) + 255) & ~size_t(255) : 4096;
  const size_t M = 1ull << 20;
  // Scale the bulk of the count but keep count mod 16, so odd tails, 16-byte
  // slot alignment and the unaligned-fallback cases survive scaling.
  auto S = [&](size_t n) {
    if (scale <= 1) return n;
    return std::max<size_t>(16, (n / scale) & ~size_t(15)) + n % 16;
  };
  const ShimCfg cfg = read_shim_cfg();

  std::vector<Case> cases = {
    // AllReduce: NCCL (< 32 MiB), peer-read, inputhybrid; odd counts; dtypes; max fallback.
    {"ar_f32_8MiB", AR, ncclFloat32, ncclSum, S(2 * M), false, 0, false},
    {"ar_f32_64MiB_odd", AR, ncclFloat32, ncclSum, S(16 * M + 5), false, 0, false},
    {"ar_f32_300MiB_odd_inplace", AR, ncclFloat32, ncclSum, S(75 * M + 3), true, 0, false},
    {"ar_f32_1GiB", AR, ncclFloat32, ncclSum, S(256 * M), false, 0, false},
    // Odd size with a non-zero host share in every setting: misaligned host
    // suffix before the job-1445 fix.
    {"ar_f32_1GiB_odd", AR, ncclFloat32, ncclSum, S(256 * M + 3), false, 0, false},
    {"ar_f16_300MiB", AR, ncclFloat16, ncclSum, S(150 * M + 1), false, 0, false},
    {"ar_bf16_1GiB_inplace", AR, ncclBfloat16, ncclSum, S(512 * M), true, 0, false},
    {"ar_f32_300MiB_max_fallback", AR, ncclFloat32, ncclMax, S(75 * M), false, 0, false},
    // AllGather: copy path for any type; odd byte counts; in-place.
    {"ag_i8_6MiB_odd", AG, ncclInt8, ncclSum, S(3 * M + 7), false, 0, false},
    {"ag_f32_256MiB", AG, ncclFloat32, ncclSum, S(32 * M), false, 0, false},
    {"ag_f32_256MiB_inplace", AG, ncclFloat32, ncclSum, S(32 * M), true, 0, false},
    {"ag_f16_1GiB_odd", AG, ncclFloat16, ncclSum, S(256 * M + 3), false, 0, false},
    {"ag_i32_1GiB_inplace", AG, ncclInt32, ncclSum, S(128 * M), true, 0, false},
    // ReduceScatter: shim slots must be 16-byte multiples; others fall back.
    {"rs_f32_8MiB", RS, ncclFloat32, ncclSum, S(1 * M), false, 0, false},
    {"rs_f32_256MiB_inplace", RS, ncclFloat32, ncclSum, S(32 * M + 4), true, 0, false},
    {"rs_f32_256MiB_unaligned_fallback", RS, ncclFloat32, ncclSum, S(32 * M + 1), false, 0, false},
    {"rs_bf16_1GiB", RS, ncclBfloat16, ncclSum, S(256 * M), false, 0, false},
    {"rs_f16_512MiB_inplace", RS, ncclFloat16, ncclSum, S(128 * M), true, 0, false},
    {"rs_f32_256MiB_max_fallback", RS, ncclFloat32, ncclMax, S(32 * M), false, 0, false},
    // Broadcast: both roots, odd counts, in-place.
    {"bc_f32_256MiB_root0", BC, ncclFloat32, ncclSum, S(64 * M), false, 0, false},
    {"bc_f32_256MiB_root1", BC, ncclFloat32, ncclSum, S(64 * M), false, 1, false},
    {"bc_i8_10MiB_odd_root1", BC, ncclInt8, ncclSum, S(10 * M + 3), false, 1, false},
    {"bc_bf16_1GiB_inplace_root1", BC, ncclBfloat16, ncclSum, S(512 * M), true, 1, false},
    // Mixed group: AllGather + Broadcast in one group must replay in order on NCCL.
    {"mixed_ag_bc_f32_128MiB", MIXED, ncclFloat32, ncclSum, S(16 * M), false, 1, false},
  };
  if (getenv("STRESS_CONTRACT_CAPS") && atoi(getenv("STRESS_CONTRACT_CAPS"))) {
    const std::vector<Case> caps={
    {"cap_ar_below8GiB", AR, ncclFloat32, ncclSum, size_t(2147483647ULL), false, 0, true},
    {"cap_ar_at8GiB", AR, ncclFloat32, ncclSum, size_t(2147483648ULL), false, 0, true},
    {"cap_ar_above8GiB", AR, ncclFloat32, ncclSum, size_t(2147483649ULL), false, 0, true},
    {"cap_ag_below8GiB", AG, ncclFloat32, ncclSum, size_t(1073741823ULL), false, 0, true},
    {"cap_ag_at8GiB", AG, ncclFloat32, ncclSum, size_t(1073741824ULL), false, 0, true},
    {"cap_ag_above8GiB", AG, ncclFloat32, ncclSum, size_t(1073741825ULL), false, 0, true},
    {"cap_rs_below8GiB", RS, ncclFloat32, ncclSum, size_t(1073741823ULL), false, 0, true},
    {"cap_rs_at8GiB", RS, ncclFloat32, ncclSum, size_t(1073741824ULL), false, 0, true},
    {"cap_rs_above8GiB", RS, ncclFloat32, ncclSum, size_t(1073741825ULL), false, 0, true},
    {"cap_bc_below8GiB", BC, ncclFloat32, ncclSum, size_t(2147483647ULL), false, 0, true},
    {"cap_bc_at8GiB", BC, ncclFloat32, ncclSum, size_t(2147483648ULL), false, 0, true},
    {"cap_bc_above8GiB", BC, ncclFloat32, ncclSum, size_t(2147483649ULL), false, 0, true},
    };
    cases.insert(cases.end(),caps.begin(),caps.end());
  }
  { const std::vector<Case> extra={
    {"v2_ar_f32_2GiB_m1el", AR, ncclFloat32, ncclSum, size_t(536870911ULL), false, 0, true},
    {"v2_ar_f32_2GiB", AR, ncclFloat32, ncclSum, size_t(536870912ULL), false, 0, true},
    {"v2_ar_f32_2GiB_p1el", AR, ncclFloat32, ncclSum, size_t(536870913ULL), false, 0, true},
    {"v2_ag_f32_4GiB_m1el", AG, ncclFloat32, ncclSum, size_t(536870911ULL), false, 0, true},
    {"v2_ag_f32_4GiB", AG, ncclFloat32, ncclSum, size_t(536870912ULL), false, 0, true},
    {"v2_ag_f32_4GiB_p1el", AG, ncclFloat32, ncclSum, size_t(536870913ULL), false, 0, true},
    {"v2_ar_f16_2GiB_m1el", AR, ncclFloat16, ncclSum, size_t(1073741823ULL), false, 0, true},
    {"v2_ar_f16_2GiB", AR, ncclFloat16, ncclSum, size_t(1073741824ULL), false, 0, true},
    {"v2_ar_f16_2GiB_p1el", AR, ncclFloat16, ncclSum, size_t(1073741825ULL), false, 0, true},
    {"v2_ag_f16_4GiB_m1el", AG, ncclFloat16, ncclSum, size_t(1073741823ULL), false, 0, true},
    {"v2_ag_f16_4GiB", AG, ncclFloat16, ncclSum, size_t(1073741824ULL), false, 0, true},
    {"v2_ag_f16_4GiB_p1el", AG, ncclFloat16, ncclSum, size_t(1073741825ULL), false, 0, true},
    {"v2_ar_bf16_2GiB_m1el", AR, ncclBfloat16, ncclSum, size_t(1073741823ULL), false, 0, true},
    {"v2_ar_bf16_2GiB", AR, ncclBfloat16, ncclSum, size_t(1073741824ULL), false, 0, true},
    {"v2_ar_bf16_2GiB_p1el", AR, ncclBfloat16, ncclSum, size_t(1073741825ULL), false, 0, true},
    {"v2_ag_bf16_4GiB_m1el", AG, ncclBfloat16, ncclSum, size_t(1073741823ULL), false, 0, true},
    {"v2_ag_bf16_4GiB", AG, ncclBfloat16, ncclSum, size_t(1073741824ULL), false, 0, true},
    {"v2_ag_bf16_4GiB_p1el", AG, ncclBfloat16, ncclSum, size_t(1073741825ULL), false, 0, true},
  }; cases.insert(cases.end(),extra.begin(),extra.end()); }
  { const std::vector<Case> v4extra={
    {"v4_k0_dt3_n134217727_root0_ip0", AR, ncclFloat16, ncclSum, size_t(134217727ULL), false, 0, true},
    {"v4_k0_dt3_n134217728_root0_ip0", AR, ncclFloat16, ncclSum, size_t(134217728ULL), false, 0, true},
    {"v4_k0_dt3_n33554431_root0_ip0", AR, ncclFloat16, ncclSum, size_t(33554431ULL), false, 0, true},
    {"v4_k0_dt3_n33554431_root0_ip1", AR, ncclFloat16, ncclSum, size_t(33554431ULL), true, 0, true},
    {"v4_k0_dt3_n33554432_root0_ip0", AR, ncclFloat16, ncclSum, size_t(33554432ULL), false, 0, true},
    {"v4_k0_dt3_n33554432_root0_ip1", AR, ncclFloat16, ncclSum, size_t(33554432ULL), true, 0, true},
    {"v4_k0_dt4_n134217727_root0_ip1", AR, ncclBfloat16, ncclSum, size_t(134217727ULL), true, 0, true},
    {"v4_k0_dt4_n134217728_root0_ip1", AR, ncclBfloat16, ncclSum, size_t(134217728ULL), true, 0, true},
    {"v4_k0_dt4_n33554431_root0_ip0", AR, ncclBfloat16, ncclSum, size_t(33554431ULL), false, 0, true},
    {"v4_k0_dt4_n33554431_root0_ip1", AR, ncclBfloat16, ncclSum, size_t(33554431ULL), true, 0, true},
    {"v4_k0_dt4_n33554432_root0_ip0", AR, ncclBfloat16, ncclSum, size_t(33554432ULL), false, 0, true},
    {"v4_k0_dt4_n33554432_root0_ip1", AR, ncclBfloat16, ncclSum, size_t(33554432ULL), true, 0, true},
    {"v4_k0_dt7_n1073741824_root0_ip0", AR, ncclFloat32, ncclSum, size_t(1073741824ULL), false, 0, true},
    {"v4_k0_dt7_n1073741824_root0_ip1", AR, ncclFloat32, ncclSum, size_t(1073741824ULL), true, 0, true},
    {"v4_k0_dt7_n134217728_root0_ip0", AR, ncclFloat32, ncclSum, size_t(134217728ULL), false, 0, true},
    {"v4_k0_dt7_n134217728_root0_ip1", AR, ncclFloat32, ncclSum, size_t(134217728ULL), true, 0, true},
    {"v4_k0_dt7_n16777215_root0_ip0", AR, ncclFloat32, ncclSum, size_t(16777215ULL), false, 0, true},
    {"v4_k0_dt7_n16777215_root0_ip1", AR, ncclFloat32, ncclSum, size_t(16777215ULL), true, 0, true},
    {"v4_k0_dt7_n16777216_root0_ip0", AR, ncclFloat32, ncclSum, size_t(16777216ULL), false, 0, true},
    {"v4_k0_dt7_n16777216_root0_ip1", AR, ncclFloat32, ncclSum, size_t(16777216ULL), true, 0, true},
    {"v4_k0_dt7_n2147483648_root0_ip0", AR, ncclFloat32, ncclSum, size_t(2147483648ULL), false, 0, true},
    {"v4_k0_dt7_n2147483648_root0_ip1", AR, ncclFloat32, ncclSum, size_t(2147483648ULL), true, 0, true},
    {"v4_k0_dt7_n268435456_root0_ip1", AR, ncclFloat32, ncclSum, size_t(268435456ULL), true, 0, true},
    {"v4_k0_dt7_n33554432_root0_ip0", AR, ncclFloat32, ncclSum, size_t(33554432ULL), false, 0, true},
    {"v4_k0_dt7_n33554432_root0_ip1", AR, ncclFloat32, ncclSum, size_t(33554432ULL), true, 0, true},
    {"v4_k0_dt7_n536870912_root0_ip1", AR, ncclFloat32, ncclSum, size_t(536870912ULL), true, 0, true},
    {"v4_k0_dt7_n67108863_root0_ip1", AR, ncclFloat32, ncclSum, size_t(67108863ULL), true, 0, true},
    {"v4_k0_dt7_n67108864_root0_ip0", AR, ncclFloat32, ncclSum, size_t(67108864ULL), false, 0, true},
    {"v4_k0_dt7_n67108864_root0_ip1", AR, ncclFloat32, ncclSum, size_t(67108864ULL), true, 0, true},
    {"v4_k1_dt3_n1073741823_root0_ip1", AG, ncclFloat16, ncclSum, size_t(1073741823ULL), true, 0, true},
    {"v4_k1_dt3_n1073741824_root0_ip1", AG, ncclFloat16, ncclSum, size_t(1073741824ULL), true, 0, true},
    {"v4_k1_dt3_n134217727_root0_ip0", AG, ncclFloat16, ncclSum, size_t(134217727ULL), false, 0, true},
    {"v4_k1_dt3_n134217727_root0_ip1", AG, ncclFloat16, ncclSum, size_t(134217727ULL), true, 0, true},
    {"v4_k1_dt3_n134217728_root0_ip0", AG, ncclFloat16, ncclSum, size_t(134217728ULL), false, 0, true},
    {"v4_k1_dt3_n134217728_root0_ip1", AG, ncclFloat16, ncclSum, size_t(134217728ULL), true, 0, true},
    {"v4_k1_dt3_n8388607_root0_ip0", AG, ncclFloat16, ncclSum, size_t(8388607ULL), false, 0, true},
    {"v4_k1_dt3_n8388607_root0_ip1", AG, ncclFloat16, ncclSum, size_t(8388607ULL), true, 0, true},
    {"v4_k1_dt3_n8388608_root0_ip0", AG, ncclFloat16, ncclSum, size_t(8388608ULL), false, 0, true},
    {"v4_k1_dt3_n8388608_root0_ip1", AG, ncclFloat16, ncclSum, size_t(8388608ULL), true, 0, true},
    {"v4_k1_dt4_n1073741823_root0_ip1", AG, ncclBfloat16, ncclSum, size_t(1073741823ULL), true, 0, true},
    {"v4_k1_dt4_n1073741824_root0_ip1", AG, ncclBfloat16, ncclSum, size_t(1073741824ULL), true, 0, true},
    {"v4_k1_dt4_n134217727_root0_ip0", AG, ncclBfloat16, ncclSum, size_t(134217727ULL), false, 0, true},
    {"v4_k1_dt4_n134217727_root0_ip1", AG, ncclBfloat16, ncclSum, size_t(134217727ULL), true, 0, true},
    {"v4_k1_dt4_n134217728_root0_ip0", AG, ncclBfloat16, ncclSum, size_t(134217728ULL), false, 0, true},
    {"v4_k1_dt4_n134217728_root0_ip1", AG, ncclBfloat16, ncclSum, size_t(134217728ULL), true, 0, true},
    {"v4_k1_dt4_n8388607_root0_ip0", AG, ncclBfloat16, ncclSum, size_t(8388607ULL), false, 0, true},
    {"v4_k1_dt4_n8388607_root0_ip1", AG, ncclBfloat16, ncclSum, size_t(8388607ULL), true, 0, true},
    {"v4_k1_dt4_n8388608_root0_ip0", AG, ncclBfloat16, ncclSum, size_t(8388608ULL), false, 0, true},
    {"v4_k1_dt4_n8388608_root0_ip1", AG, ncclBfloat16, ncclSum, size_t(8388608ULL), true, 0, true},
    {"v4_k1_dt7_n134217728_root0_ip0", AG, ncclFloat32, ncclSum, size_t(134217728ULL), false, 0, true},
    {"v4_k1_dt7_n134217728_root0_ip1", AG, ncclFloat32, ncclSum, size_t(134217728ULL), true, 0, true},
    {"v4_k1_dt7_n16777216_root0_ip0", AG, ncclFloat32, ncclSum, size_t(16777216ULL), false, 0, true},
    {"v4_k1_dt7_n16777216_root0_ip1", AG, ncclFloat32, ncclSum, size_t(16777216ULL), true, 0, true},
    {"v4_k1_dt7_n268435456_root0_ip0", AG, ncclFloat32, ncclSum, size_t(268435456ULL), false, 0, true},
    {"v4_k1_dt7_n268435456_root0_ip1", AG, ncclFloat32, ncclSum, size_t(268435456ULL), true, 0, true},
    {"v4_k1_dt7_n4194303_root0_ip0", AG, ncclFloat32, ncclSum, size_t(4194303ULL), false, 0, true},
    {"v4_k1_dt7_n4194303_root0_ip1", AG, ncclFloat32, ncclSum, size_t(4194303ULL), true, 0, true},
    {"v4_k1_dt7_n4194304_root0_ip0", AG, ncclFloat32, ncclSum, size_t(4194304ULL), false, 0, true},
    {"v4_k1_dt7_n4194304_root0_ip1", AG, ncclFloat32, ncclSum, size_t(4194304ULL), true, 0, true},
    {"v4_k1_dt7_n536870911_root0_ip1", AG, ncclFloat32, ncclSum, size_t(536870911ULL), true, 0, true},
    {"v4_k1_dt7_n536870912_root0_ip1", AG, ncclFloat32, ncclSum, size_t(536870912ULL), true, 0, true},
    {"v4_k1_dt7_n67108863_root0_ip0", AG, ncclFloat32, ncclSum, size_t(67108863ULL), false, 0, true},
    {"v4_k1_dt7_n67108863_root0_ip1", AG, ncclFloat32, ncclSum, size_t(67108863ULL), true, 0, true},
    {"v4_k1_dt7_n67108864_root0_ip0", AG, ncclFloat32, ncclSum, size_t(67108864ULL), false, 0, true},
    {"v4_k1_dt7_n67108864_root0_ip1", AG, ncclFloat32, ncclSum, size_t(67108864ULL), true, 0, true},
    {"v4_k1_dt7_n8388608_root0_ip0", AG, ncclFloat32, ncclSum, size_t(8388608ULL), false, 0, true},
    {"v4_k1_dt7_n8388608_root0_ip1", AG, ncclFloat32, ncclSum, size_t(8388608ULL), true, 0, true},
    {"v4_k2_dt3_n16777215_root0_ip0", RS, ncclFloat16, ncclSum, size_t(16777215ULL), false, 0, true},
    {"v4_k2_dt3_n16777215_root0_ip1", RS, ncclFloat16, ncclSum, size_t(16777215ULL), true, 0, true},
    {"v4_k2_dt3_n16777216_root0_ip0", RS, ncclFloat16, ncclSum, size_t(16777216ULL), false, 0, true},
    {"v4_k2_dt3_n16777216_root0_ip1", RS, ncclFloat16, ncclSum, size_t(16777216ULL), true, 0, true},
    {"v4_k2_dt3_n536870911_root0_ip0", RS, ncclFloat16, ncclSum, size_t(536870911ULL), false, 0, true},
    {"v4_k2_dt3_n536870911_root0_ip1", RS, ncclFloat16, ncclSum, size_t(536870911ULL), true, 0, true},
    {"v4_k2_dt3_n536870912_root0_ip0", RS, ncclFloat16, ncclSum, size_t(536870912ULL), false, 0, true},
    {"v4_k2_dt3_n536870912_root0_ip1", RS, ncclFloat16, ncclSum, size_t(536870912ULL), true, 0, true},
    {"v4_k2_dt3_n67108863_root0_ip0", RS, ncclFloat16, ncclSum, size_t(67108863ULL), false, 0, true},
    {"v4_k2_dt3_n67108863_root0_ip1", RS, ncclFloat16, ncclSum, size_t(67108863ULL), true, 0, true},
    {"v4_k2_dt3_n67108864_root0_ip0", RS, ncclFloat16, ncclSum, size_t(67108864ULL), false, 0, true},
    {"v4_k2_dt3_n67108864_root0_ip1", RS, ncclFloat16, ncclSum, size_t(67108864ULL), true, 0, true},
    {"v4_k2_dt4_n16777215_root0_ip0", RS, ncclBfloat16, ncclSum, size_t(16777215ULL), false, 0, true},
    {"v4_k2_dt4_n16777215_root0_ip1", RS, ncclBfloat16, ncclSum, size_t(16777215ULL), true, 0, true},
    {"v4_k2_dt4_n16777216_root0_ip0", RS, ncclBfloat16, ncclSum, size_t(16777216ULL), false, 0, true},
    {"v4_k2_dt4_n16777216_root0_ip1", RS, ncclBfloat16, ncclSum, size_t(16777216ULL), true, 0, true},
    {"v4_k2_dt4_n536870911_root0_ip0", RS, ncclBfloat16, ncclSum, size_t(536870911ULL), false, 0, true},
    {"v4_k2_dt4_n536870911_root0_ip1", RS, ncclBfloat16, ncclSum, size_t(536870911ULL), true, 0, true},
    {"v4_k2_dt4_n536870912_root0_ip0", RS, ncclBfloat16, ncclSum, size_t(536870912ULL), false, 0, true},
    {"v4_k2_dt4_n536870912_root0_ip1", RS, ncclBfloat16, ncclSum, size_t(536870912ULL), true, 0, true},
    {"v4_k2_dt4_n67108863_root0_ip0", RS, ncclBfloat16, ncclSum, size_t(67108863ULL), false, 0, true},
    {"v4_k2_dt4_n67108863_root0_ip1", RS, ncclBfloat16, ncclSum, size_t(67108863ULL), true, 0, true},
    {"v4_k2_dt4_n67108864_root0_ip0", RS, ncclBfloat16, ncclSum, size_t(67108864ULL), false, 0, true},
    {"v4_k2_dt4_n67108864_root0_ip1", RS, ncclBfloat16, ncclSum, size_t(67108864ULL), true, 0, true},
    {"v4_k2_dt7_n134217728_root0_ip0", RS, ncclFloat32, ncclSum, size_t(134217728ULL), false, 0, true},
    {"v4_k2_dt7_n134217728_root0_ip1", RS, ncclFloat32, ncclSum, size_t(134217728ULL), true, 0, true},
    {"v4_k2_dt7_n16777216_root0_ip0", RS, ncclFloat32, ncclSum, size_t(16777216ULL), false, 0, true},
    {"v4_k2_dt7_n16777216_root0_ip1", RS, ncclFloat32, ncclSum, size_t(16777216ULL), true, 0, true},
    {"v4_k2_dt7_n268435455_root0_ip0", RS, ncclFloat32, ncclSum, size_t(268435455ULL), false, 0, true},
    {"v4_k2_dt7_n268435455_root0_ip1", RS, ncclFloat32, ncclSum, size_t(268435455ULL), true, 0, true},
    {"v4_k2_dt7_n268435456_root0_ip0", RS, ncclFloat32, ncclSum, size_t(268435456ULL), false, 0, true},
    {"v4_k2_dt7_n268435456_root0_ip1", RS, ncclFloat32, ncclSum, size_t(268435456ULL), true, 0, true},
    {"v4_k2_dt7_n33554431_root0_ip0", RS, ncclFloat32, ncclSum, size_t(33554431ULL), false, 0, true},
    {"v4_k2_dt7_n33554431_root0_ip1", RS, ncclFloat32, ncclSum, size_t(33554431ULL), true, 0, true},
    {"v4_k2_dt7_n33554432_root0_ip1", RS, ncclFloat32, ncclSum, size_t(33554432ULL), true, 0, true},
    {"v4_k2_dt7_n67108864_root0_ip0", RS, ncclFloat32, ncclSum, size_t(67108864ULL), false, 0, true},
    {"v4_k2_dt7_n67108864_root0_ip1", RS, ncclFloat32, ncclSum, size_t(67108864ULL), true, 0, true},
    {"v4_k2_dt7_n8388607_root0_ip0", RS, ncclFloat32, ncclSum, size_t(8388607ULL), false, 0, true},
    {"v4_k2_dt7_n8388607_root0_ip1", RS, ncclFloat32, ncclSum, size_t(8388607ULL), true, 0, true},
    {"v4_k2_dt7_n8388608_root0_ip0", RS, ncclFloat32, ncclSum, size_t(8388608ULL), false, 0, true},
    {"v4_k2_dt7_n8388608_root0_ip1", RS, ncclFloat32, ncclSum, size_t(8388608ULL), true, 0, true},
    {"v4_k3_dt3_n134217727_root0_ip0", BC, ncclFloat16, ncclSum, size_t(134217727ULL), false, 0, true},
    {"v4_k3_dt3_n134217727_root0_ip1", BC, ncclFloat16, ncclSum, size_t(134217727ULL), true, 0, true},
    {"v4_k3_dt3_n134217728_root0_ip0", BC, ncclFloat16, ncclSum, size_t(134217728ULL), false, 0, true},
    {"v4_k3_dt3_n134217728_root0_ip1", BC, ncclFloat16, ncclSum, size_t(134217728ULL), true, 0, true},
    {"v4_k3_dt3_n75497471_root0_ip0", BC, ncclFloat16, ncclSum, size_t(75497471ULL), false, 0, true},
    {"v4_k3_dt3_n75497471_root0_ip1", BC, ncclFloat16, ncclSum, size_t(75497471ULL), true, 0, true},
    {"v4_k3_dt3_n75497472_root0_ip0", BC, ncclFloat16, ncclSum, size_t(75497472ULL), false, 0, true},
    {"v4_k3_dt3_n75497472_root0_ip1", BC, ncclFloat16, ncclSum, size_t(75497472ULL), true, 0, true},
    {"v4_k3_dt4_n134217727_root0_ip0", BC, ncclBfloat16, ncclSum, size_t(134217727ULL), false, 0, true},
    {"v4_k3_dt4_n134217727_root0_ip1", BC, ncclBfloat16, ncclSum, size_t(134217727ULL), true, 0, true},
    {"v4_k3_dt4_n134217728_root0_ip0", BC, ncclBfloat16, ncclSum, size_t(134217728ULL), false, 0, true},
    {"v4_k3_dt4_n134217728_root0_ip1", BC, ncclBfloat16, ncclSum, size_t(134217728ULL), true, 0, true},
    {"v4_k3_dt4_n75497471_root0_ip0", BC, ncclBfloat16, ncclSum, size_t(75497471ULL), false, 0, true},
    {"v4_k3_dt4_n75497471_root0_ip1", BC, ncclBfloat16, ncclSum, size_t(75497471ULL), true, 0, true},
    {"v4_k3_dt4_n75497472_root0_ip0", BC, ncclBfloat16, ncclSum, size_t(75497472ULL), false, 0, true},
    {"v4_k3_dt4_n75497472_root0_ip1", BC, ncclBfloat16, ncclSum, size_t(75497472ULL), true, 0, true},
    {"v4_k3_dt7_n37748735_root0_ip0", BC, ncclFloat32, ncclSum, size_t(37748735ULL), false, 0, true},
    {"v4_k3_dt7_n37748736_root0_ip0", BC, ncclFloat32, ncclSum, size_t(37748736ULL), false, 0, true},
    {"v4_k3_dt7_n37748736_root0_ip1", BC, ncclFloat32, ncclSum, size_t(37748736ULL), true, 0, true},
    {"v4_k3_dt7_n67108863_root0_ip0", BC, ncclFloat32, ncclSum, size_t(67108863ULL), false, 0, true},
    {"v4_k3_dt7_n67108863_root0_ip1", BC, ncclFloat32, ncclSum, size_t(67108863ULL), true, 0, true},
    {"v4_k3_dt7_n67108864_root0_ip1", BC, ncclFloat32, ncclSum, size_t(67108864ULL), true, 0, true},
  }; cases.insert(cases.end(),v4extra.begin(),v4extra.end()); }
  if (boundary) add_boundary_cases(cases, cfg);

  const char* case_filter = getenv("STRESS_CASE_FILTER");
  if (list_only) {
    int n = 0;
    for (const Case& c : cases) {
      if (case_filter && *case_filter && c.label != case_filter) continue;
      printf("%s\t%zu\t%s\n", c.label.c_str(), c.count * dt_size(c.dt), expect_route(c, cfg));
      ++n;
    }
    return n ? 0 : 1;
  }
  fprintf(stderr,
          "[stress-test] mode=%s seed=%llu guard_bytes=%zu scale=%zu boundary=%d "
          "cfg: peerread_min=%zu inputhybrid_min=%zu chunk=%zu host_chunk=%zu tail_chunk=%zu "
          "in_div=%zu out_div=%zu ag_min=%zu rs_min=%zu bc_min=%zu auto=%d\n",
          run.diff ? "diff" : "check", static_cast<unsigned long long>(run.seed), guard, scale,
          boundary ? 1 : 0, cfg.peerread_min, cfg.inputhybrid_min, cfg.chunk, cfg.host_chunk,
          cfg.tail_chunk, cfg.in_div, cfg.out_div, cfg.ag_min, cfg.rs_min, cfg.bc_min,
          1);

  RealNccl real;
  if (run.diff) real = load_real_nccl();

  int devs[2] = {0, 1};
  if (getenv("STRESS_REVERSE_RANKS") && atoi(getenv("STRESS_REVERSE_RANKS"))) std::swap(devs[0],devs[1]);
  ncclComm_t initialized[2],comms[2];
  NCCL_OK(ncclCommInitAll(initialized,2,devs));
  for(int index=0;index<2;++index)comms[devs[index]]=initialized[index];
  cudaStream_t streams[2];
  for (int d = 0; d < 2; ++d) {
    CUDA_OK(cudaSetDevice(d));
    CUDA_OK(cudaStreamCreateWithFlags(&streams[d], cudaStreamNonBlocking));
  }
  int rank_of[2];
  for (int d = 0; d < 2; ++d) NCCL_OK(ncclCommUserRank(comms[d], &rank_of[d]));

  int failed = 0;
  const bool expect_submit_error = env_size("STRESS_EXPECT_SUBMIT_ERROR", 0) != 0;
  const char* delay_dev_env = getenv("STRESS_DELAY_INPUT_DEVICE");
  const int delay_input_dev = delay_dev_env ? atoi(delay_dev_env) : -1;
  const unsigned long delay_input_ms = getenv("STRESS_DELAY_INPUT_MS")
      ? strtoul(getenv("STRESS_DELAY_INPUT_MS"), nullptr, 10) : 250;
  const int skip_check_dev = getenv("STRESS_SKIP_CHECK_DEVICE")
      ? atoi(getenv("STRESS_SKIP_CHECK_DEVICE")) : -1;
  const bool check_current_device = getenv("STRESS_CHECK_CURRENT_DEVICE") &&
      atoi(getenv("STRESS_CHECK_CURRENT_DEVICE")) != 0;
  const size_t corrupt_shift = getenv("STRESS_TEST_CORRUPT_SHIFT")
      ? strtoull(getenv("STRESS_TEST_CORRUPT_SHIFT"), nullptr, 10) : 0;
  printf("case\titers\tbytes_per_rank\tmismatches\tfirst_bad_iter\tstatus\tguard_bad\t"
         "expect_route\tkind\tmode\tnan_skipped\n");
  int selected_cases = 0;
  for (const Case& c : cases) {
    if (case_filter && *case_filter && c.label != case_filter) continue;
    ++selected_cases;
    const size_t es = dt_size(c.dt);
    const size_t bytes = c.count * es;
    const int iters = bytes >= (1ull << 30)
        ? std::max(std::min(2, iters_env), iters_env / 2) : iters_env;
    // Buffer sizes in elements.
    size_t send_n = 0, recv_n = 0;
    switch (c.coll) {
      case AR: send_n = recv_n = c.count; break;
      case AG: send_n = c.count; recv_n = 2 * c.count; break;
      case RS: send_n = 2 * c.count; recv_n = c.count; break;
      case BC: send_n = recv_n = c.count; break;
      case MIXED: send_n = c.count; recv_n = 4 * c.count; break;  // [gather 2n][bc src n][bc dst n]
    }
    Guarded send[2], recv[2], shadow[2];
    unsigned long long* bad[2];   // [iters] data mismatches, then guard, then NaN skips
    for (int d = 0; d < 2; ++d) {
      CUDA_OK(cudaSetDevice(d));
      recv[d] = guarded_alloc(recv_n * es, guard, 0xff);
      send[d] = guarded_alloc(send_n * es, guard, 0x00);
      if (run.diff) shadow[d] = guarded_alloc(recv_n * es, 0, 0xff);
      CUDA_OK(cudaMalloc(&bad[d], (iters + 2) * sizeof(unsigned long long)));
      CUDA_OK(cudaMemset(bad[d], 0, (iters + 2) * sizeof(unsigned long long)));
      CUDA_OK(cudaDeviceSynchronize());
    }
    case_barrier();
    int expected_errors = 0, recovery_iters = 0;
    const char* route = expect_route(c, cfg);
    fprintf(stderr, "CASE %s\n", c.label.c_str());
    fprintf(stderr, "ROUTE_EXPECT %s %s bytes=%zu\n", c.label.c_str(), route, bytes);
    int context_errors = 0;
    int first_context_bad = -1;
    for (int it = 0; it < iters; ++it) {
      if(env_size("V4_FRESH_STREAMS",0)) for(int d=0;d<2;++d){
        CUDA_OK(cudaSetDevice(d)); CUDA_OK(cudaStreamSynchronize(streams[d]));
        CUDA_OK(cudaStreamDestroy(streams[d]));
        CUDA_OK(cudaStreamCreateWithFlags(&streams[d],cudaStreamNonBlocking));
      }
      void* in_ptr[2];
      for (int d = 0; d < 2; ++d) {
        CUDA_OK(cudaSetDevice(d));
        if (c.coll == AR && it == 1 && d == delay_input_dev && delay_input_ms > 0) {
          int clock_khz = 0;
          CUDA_OK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, d));
          const unsigned long long cycles =
              static_cast<unsigned long long>(clock_khz) * delay_input_ms;
          delay_cycles_kernel<<<1, 1, 0, streams[d]>>>(cycles);
          CUDA_OK(cudaGetLastError());
          fprintf(stderr, "[stress-test] delayed AllReduce input producer on device %d for %lu ms\n",
                  d, delay_input_ms);
        }
        const int r = rank_of[d];
        char* sp = send[d].p;
        char* rp = recv[d].p;
        // Where this rank's input lives. In-place AllReduce/Broadcast use
        // recvbuff; in-place AllGather uses this rank's slot of recvbuff;
        // in-place ReduceScatter keeps the full input in sendbuff and writes
        // its result to sendbuff + rank*recvcount.
        void* in = sp;
        if (contract_in_place(c,d) && (c.coll == AR || c.coll == BC)) in = rp;
        if (contract_in_place(c,d) && c.coll == AG) in = rp + r * bytes;
        in_ptr[d] = in;
        DISPATCH(c.dt, init_case, c, r, in, rp, it, run, streams[d]);
      }
      if (run.diff) {
        // Stock reference, always out-of-place into the shadow buffer, on the
        // same streams so it reads exactly the inputs the shim will read.
        NCCL_OK(real.group_start());
        for (int d = 0; d < 2; ++d) {
          CUDA_OK(cudaSetDevice(d));
          char* rp = recv[d].p;
          char* sh = shadow[d].p;
          switch (c.coll) {
            case AR: NCCL_OK(real.all_reduce(in_ptr[d], sh, c.count, c.dt, c.op, comms[d], streams[d])); break;
            case AG: NCCL_OK(real.all_gather(in_ptr[d], sh, c.count, c.dt, comms[d], streams[d])); break;
            case RS: NCCL_OK(real.reduce_scatter(in_ptr[d], sh, c.count, c.dt, c.op, comms[d], streams[d])); break;
            case BC: NCCL_OK(real.broadcast(in_ptr[d], sh, c.count, c.dt, c.root, comms[d], streams[d])); break;
            case MIXED:
              NCCL_OK(real.all_gather(in_ptr[d], sh, c.count, c.dt, comms[d], streams[d]));
              NCCL_OK(real.broadcast(rp + 2 * bytes, sh + 3 * bytes, c.count, c.dt, c.root,
                                     comms[d], streams[d]));
              break;
          }
        }
        NCCL_OK(real.group_end());
      }
      NCCL_OK(ncclGroupStart());
      for (int step = 0; step < 2; ++step) {
        // Reverse call order every iteration while using each communicator
        // under its matching current CUDA device.
        const int d = (check_current_device && ((it & 1) ^ int(env_size("V4_REVERSE_CALLS",0)))) ? 1 - step : step;
        CUDA_OK(cudaSetDevice(d));
        const int r = rank_of[d];
        char* sp = send[d].p;
        char* rp = recv[d].p;
        switch (c.coll) {
          case AR:
            NCCL_OK(ncclAllReduce(contract_in_place(c,d) ? rp : sp, rp, c.count, c.dt, c.op, comms[d], streams[d]));
            break;
          case AG:
            NCCL_OK(ncclAllGather(contract_in_place(c,d) ? rp + r * bytes : sp, rp, c.count, c.dt, comms[d], streams[d]));
            break;
          case RS:
            NCCL_OK(ncclReduceScatter(sp, contract_in_place(c,d) ? sp + r * bytes : rp, c.count, c.dt, c.op,
                                      comms[d], streams[d]));
            break;
          case BC:
            NCCL_OK(ncclBroadcast(contract_in_place(c,d) ? rp : sp, rp, c.count, c.dt, c.root, comms[d], streams[d]));
            break;
          case MIXED:
            NCCL_OK(ncclAllGather(sp, rp, c.count, c.dt, comms[d], streams[d]));
            NCCL_OK(ncclBroadcast(rp + 2 * bytes, rp + 3 * bytes, c.count, c.dt, c.root,
                                  comms[d], streams[d]));
            break;
        }
      }
      int expected_current_device = -1;
      if (check_current_device) CUDA_OK(cudaGetDevice(&expected_current_device));
      const ncclResult_t end_result = ncclGroupEnd();
      if (end_result != ncclSuccess && expect_submit_error &&
          end_result == ncclUnhandledCudaError && expected_errors == 0) {
        ++expected_errors;
        fprintf(stderr, "[stress-test] expected ncclUnhandledCudaError case=%s iter=%d\n",
                c.label.c_str(), it);
        // Output for this iteration is undefined after partial submission.
        // Synchronize, skip its checker, then reinitialize and check recovery.
        for (int d = 0; d < 2; ++d) {
          CUDA_OK(cudaSetDevice(d));
          CUDA_OK(cudaDeviceSynchronize());
        }
        continue;
      }
      NCCL_OK(end_result);
      if (expected_errors) ++recovery_iters;
      if (check_current_device) {
        int actual_current_device = -1;
        CUDA_OK(cudaGetDevice(&actual_current_device));
        if (actual_current_device != expected_current_device) {
          fprintf(stderr,
                  "[stress-test] CUDA current device changed across group end: expected %d got %d (%s)\n",
                  expected_current_device, actual_current_device, c.label.c_str());
          ++context_errors;
          if (first_context_bad < 0) first_context_bad = it;
        }
      }
      // Deterministic lifetime-race control for AllGather: when explicitly
      // enabled, overwrite each out-of-place send buffer on its owning stream
      // as soon as the collective is enqueued. With the shim's completion
      // join removed, a delayed peer read must then observe the overwrite. The
      // normal checker verifies the peer slot still contains this iteration's input.
      const char* overwrite = getenv("STRESS_TEST_OVERWRITE_AFTER_GROUP");
      if (c.coll == AG && (!contract_in_place(c,0) || !contract_in_place(c,1)) && overwrite && strcmp(overwrite, "allgather") == 0) {
        for (int d = 0; d < 2; ++d) {
          CUDA_OK(cudaSetDevice(d));
          if (!contract_in_place(c,d)) CUDA_OK(cudaMemsetAsync(send[d].p, 0, bytes, streams[d]));
        }
      }
      const char* overwrite_reducer_output = getenv("STRESS_TEST_OVERWRITE_REDUCER_OUTPUT");
      if (c.coll == AR && !contract_in_place(c,0) && it == 1 && overwrite_reducer_output &&
          atoi(overwrite_reducer_output) != 0) {
        CUDA_OK(cudaSetDevice(0));
        CUDA_OK(cudaMemsetAsync(recv[0].p, 0, bytes, streams[0]));
      }
      for (int d = 0; d < 2; ++d) {
        if (c.coll == AR && d == skip_check_dev) continue;
        CUDA_OK(cudaSetDevice(d));
        const int r = rank_of[d];
        char* out = recv[d].p;
        if (contract_in_place(c,d) && c.coll == RS) out = send[d].p + r * bytes;
        // Checker self-test: displace device 0's output by corrupt_shift
        // elements (out[j] = out[j + shift] over a non-overlapping prefix),
        // the way a chunk-offset bug would. Every such case must FAIL.
        const size_t out_n = (c.coll == AG || c.coll == MIXED) ? 2 * c.count : c.count;
        if (d == 0 && corrupt_shift > 0 && out_n > corrupt_shift) {
          CUDA_OK(cudaMemcpyAsync(out, out + corrupt_shift * es,
                                  std::min(corrupt_shift, out_n - corrupt_shift) * es,
                                  cudaMemcpyDeviceToDevice, streams[d]));
          if (it == 0)
            fprintf(stderr, "[stress-test] corrupted %s shift=%zu\n", c.label.c_str(), corrupt_shift);
        }
        if (run.diff) {
          DISPATCH(c.dt, diff_case, c, out, shadow[d].p, bad[d] + it, bad[d] + iters + 1,
                   streams[d]);
        } else {
          DISPATCH(c.dt, check_case, c, r, out, it, run, bad[d] + it, streams[d]);
        }
      }
    }
    if (expect_submit_error) {
      fprintf(stderr, "[stress-test] fault_observed=%d recovery_iters=%d case=%s\n",
              expected_errors, recovery_iters, c.label.c_str());
      if (expected_errors != 1 || recovery_iters < 1) {
        fprintf(stderr, "[stress-test] expected exactly one submit error and checked recovery\n");
        ++context_errors;
      }
    }
    unsigned long long total = context_errors, guard_bad = 0, nan_skipped = 0;
    int first_bad = first_context_bad;
    std::vector<unsigned long long> h(iters + 2);
    for (int d = 0; d < 2; ++d) {
      CUDA_OK(cudaSetDevice(d));
      guard_check(send[d], guard, bad[d] + iters, streams[d]);
      guard_check(recv[d], guard, bad[d] + iters, streams[d]);
      CUDA_OK(cudaGetLastError());
      CUDA_OK(cudaStreamSynchronize(streams[d]));
      CUDA_OK(cudaMemcpy(h.data(), bad[d], (iters + 2) * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));
      for (int it = 0; it < iters; ++it) {
        total += h[it];
        if (h[it] && (first_bad < 0 || it < first_bad)) first_bad = it;
      }
      guard_bad += h[iters];
      nan_skipped += h[iters + 1];
      v2_free(send[d].base);
      v2_free(recv[d].base);
      if (shadow[d].base) v2_free(shadow[d].base);
      CUDA_OK(cudaFree(bad[d]));
    }
    if (guard_bad)
      fprintf(stderr, "[stress-test] %s: %llu guard bytes overwritten\n", c.label.c_str(), guard_bad);
    const bool ok = total == 0 && guard_bad == 0;
    failed |= !ok;
    printf("%s\t%d\t%zu\t%llu\t%d\t%s\t%llu\t%s\t%s\t%s\t%llu\n", c.label.c_str(), iters, bytes,
           total, first_bad, ok ? "PASS" : "FAIL", guard_bad, route,
           c.boundary ? "bnd" : "std", run.diff ? "diff" : "check", nan_skipped);
    fflush(stdout);
  }
  if (selected_cases == 0) {
    fprintf(stderr, "no stress case matched STRESS_CASE_FILTER=%s\n",
            case_filter ? case_filter : "");
    failed = 1;
  }
  for (int d = 0; d < 2; ++d) NCCL_OK(ncclCommDestroy(comms[d]));
  printf("overall\t%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}

int main() {
  g_workers = static_cast<int>(env_size("STRESS_THREADS", 1));
  if (g_workers != 1 && g_workers != 2) {
    fprintf(stderr, "STRESS_THREADS must be 1 or 2\n");
    return 2;
  }
  if (g_workers == 1) return run_stress(0);
  if (env_size("STRESS_EXPECT_SUBMIT_ERROR", 0) || env_size("STRESS_LIST_CASES", 0)) {
    fprintf(stderr, "thread mode cannot combine fault expectation or case listing\n");
    return 2;
  }
  int result[2] = {2, 2};
  std::thread first([&] { result[0] = run_stress(0); });
  std::thread second([&] { result[1] = run_stress(1); });
  first.join(); second.join();
  const bool ok = result[0] == 0 && result[1] == 0;
  printf("threads_overall\t%s\tworkers=2\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
