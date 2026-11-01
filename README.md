# NCCL shim for two NVLink-connected GPUs

An `LD_PRELOAD` shim for NCCL, written for the SC26 Student Cluster Competition benchmark report.
It handles large AllReduce, AllGather, ReduceScatter and Broadcast calls when one process drives
two GPUs. It sends most of the data over NVLink with CUDA copy engines and, at the same time, a
disjoint share through pinned host memory over PCIe. Every other call goes to `libnccl.so`.

Tested on two H100 NVL GPUs (NV12) with CUDA 13.2 and NCCL 2.28.3. It does not engage with one
process per GPU.

## Files

| Path | Contents |
|---|---|
| `shim/nccl_collectives_shim.cu` | The exact source we measured. SHA-256 `e121036ad68e3748e307b18e7619ff5aa65051b8b43b45622946cdb041360fb7` |
| `shim/nccl_collectives_shim_clean.cu` | The same code with shorter comments, for reading |
| `tests/stress_check.cu` | The GPU test program used for the stress, coverage and sanitizer runs |
| `tla/` | TLA+ models of group fallback and host-buffer reuse, with their configurations and deliberately broken versions |

The exact source keeps its original comments, so that its hash stays the one in the report. Some
of those comments are out of date, including "V3 screening candidate". The clean copy differs only in
comments and whitespace, so it has a different file hash.

## Build

```sh
src=shim/nccl_collectives_shim.cu
sha=$(sha256sum "$src" | cut -d' ' -f1)
native=/path/to/nccl/build
nvcc -O3 -std=c++17 -arch=sm_90 --cudart shared -include cstring \
     -I "$native/include" -shared -Xcompiler -fPIC \
     "-DNCCL_SHIM_SOURCE_SHA256=\"$sha\"" "$src" \
     -L "$native/lib" -lnccl -lcuda -ldl -Xcompiler -pthread -o libnccl_ce.so
```

`nvcc` output differs between builds of the same file, so rebuilt libraries will not match the
measured library's hash.

## Run

```sh
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_CTAS=32 NCCL_BUFFSIZE=1048576
LD_PRELOAD=$PWD/libnccl_ce.so NCCL_SHIM_SUMMARY=1 \
  numactl --cpunodebind=0 --membind=0 \
  ./all_reduce_perf -t 1 -g 2 -b 8 -e 8G -f 2 -d float -a 1 -R 0 -c 1 -n 20 -w 5
```

Bind to the GPUs' NUMA node. These settings are for AllReduce; the benchmark report lists the
settings for each collective. `NCCL_SHIM_SUMMARY=1` prints routes, fallbacks and host-buffer placement
at exit.

## Results

Out-of-place `busbw` at 8 GiB, mean of 10 paired rounds, FP32.

| Collective | Tuned NCCL (GB/s) | With shim (GB/s) |
|---|---:|---:|
| AllReduce | 247.37 | 301.60 |
| AllGather | 246.37 | 303.75 |
| ReduceScatter | 246.52 | 295.44 |
| Broadcast | 251.29 | 316.68 |

## Limits

The checks cover two GPUs driven by one process with NCCL 2.28.3. They are not a formal proof of the
code, and the TLA+ models abstract CUDA's ordering rules. Stable buffer mappings and the absence of
physical aliases remain up to the caller.

## Licence

MIT. See `LICENSE`.
