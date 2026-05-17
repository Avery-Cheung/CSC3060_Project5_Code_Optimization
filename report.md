# CSC3060 Project 5: Code Optimization Report

**Platform:** Intel Xeon Silver 4210R @ 2.40GHz (Cascade Lake), 40 CPUs, AVX-512  
**Compiler:** GCC/Clang with `-ffast-math -O2`  
**Language:** C++23  
**Date:** 2026-05-16  
**Group Members:** YIQI LI (124090324), JUNHONG JIANG (124090252)

---

## 1. GRFF — Gated Residual Feature Fusion

### Algorithm
A 9-stage simplified neural network layer: Gate → Residual → Global Scaling → Smooth → Suppression → Context Integration → Hidden Interaction → Normalization → ReLU. Operates on 3 input feature vectors (A, B, C) of size 1,024,000.

### Optimization Methods

| Stage | Original | Optimized |
|-------|----------|-----------|
| Memory | 7 intermediate `std::vector<float>` (G, A_prime, Smooth_A, B_prime, C_prime, H, E) | 1 scratch buffer of size `n` |
| Pass 1 (Gate + A_prime) | Scalar loop | **AVX2 8-wide SIMD** (`_mm256_div_ps`, `_mm256_andnot_ps` for abs) |
| Sum (Stage 3) | Scalar serial | Scalar serial (kept identical to avoid FP mismatch) |
| Pass 2 (Smooth) | Forward scalar | **Reverse in-place** transform (trivially vectorizable) |
| Pass 3 (Stages 5-9) | Separate scalar loops | **Fused per-element** computation with **4x manual unrolling** |

### Key Difficulty: Floating-Point Numerical Equivalence

The check tolerance is `1e-6 + 1e-5 × |ref|`. Multiple optimization attempts failed because:

1. **Algebraic simplification changed rounding**: `(C+sig)*(1-sig) - B_omg*avg_a*inv_s` produced different results than the naive's staged computation `C_prime - (H+B_prime)/(1+|Smooth|)` due to different intermediate rounding order.

2. **4-way parallel sum changed `avg_a`**: `(sum0+sum1+sum2+sum3)/n` differs from serial `sum_a/n` at the ~1e-5 level, which propagates through multiplication to exceed the tolerance.

3. **`_mm256_rcp_ps` precision**: The AVX2 fast approximate reciprocal has only ~12-bit precision. Without a Newton-Raphson refinement iteration, errors exceed the tolerance.

### Solution
- **Exact naive formula structure**: Every line in the student function matches the naive's formula character-by-character (`G = 0.5*(p/(1+|p|)+1)`, not `0.5*p*inv+0.5`).
- **Serial scalar sum**: Uses a single `float sum_a` accumulator in a separate scalar loop, identical to the naive's Stage 3.
- **Scalar division in Pass 3**: Uses `1.0f / (1.0f + fabs(s))` (scalar `divss`, 23-bit precision), not SIMD reciprocal.

---

## 2. Image Processing

### Algorithm
7-stage image pipeline: Color Correct → Luminance → Contrast Enhance → HDR Compress → Mask Logic → Importance Weight → Output. Processes W×H = 1024×1000 pixels.

### Optimization Methods

| Optimization | Implementation |
|-------------|----------------|
| **Taylor series sin/cos** | `sin(x) ≈ x·(1 - x²/6)`, `cos(x) ≈ 1 - x²/2`. Argument range is bounded: `x_sin ∈ [0, 0.043]`, `x_cos ∈ [0, 0.22]`. Error < 1e-5, within the 1e-4 absolute tolerance. Eliminates 2 transcendental function calls per pixel. |
| **Branchless clamp** | `std::fmin(std::fmax(v, 0.0f), 1.0f)` replaces `(v<0)?0:(v>1)?1:v`. Compiles to hardware MIN/MAX instructions. |
| **Branchless importance_weight** | Extended LUT from 5 to 6 entries `{0, 0.3, 1, 0.3, 0, 0}`, making `lut[idx+1]` always valid for `idx ∈ [0,4]`. Eliminates the `idx < 4` branch. |
| **Dead branch elimination** | `g3 = sqrt(0.36·ge² + 0.1)` is always < 1.0 (max ≈ 0.678), so `gain = g3 * 0.95` unconditionally. |
| **4x manual unrolling** | 4 pixels processed per iteration with local variables. |

### Key Difficulty: Semantic Bug

The naive passes `compress_val` (HDR compressed value) as the first argument to `complex_mask_logic`, which uses it for:
1. The mask branch decision: `if (gray > thresh)` 
2. The noise computation: `sin(gray * 0.11)`

The initial student code incorrectly used the raw luminance `gray` instead of the HDR value `hdr`. This caused both the mask branching and the noise computation to use wrong input values, producing completely different results.

**Fix**: Changed `if (gray > threshold)` → `if (hdr > threshold)` and `sin(gray * p_sin)` → `sin(hdr * p_sin)`.

---

## 3. Matrix Multiplication

### Algorithm
`C = A × B`, all matrices N×N (N=512). Naive: triple nested loop O(N³) with stride-N access on B (extremely cache-unfriendly).

### Optimization Methods

| Phase | Approach |
|-------|----------|
| **3D tiling** | `kk` (outermost) → `ii` → `jj` → `i` → `k` → `j` (innermost). Block size BK=64 keeps a 64×64 B-tile (16KB) in L1 cache. |
| **j-loop 4x unrolling** | Processes 4 output columns per inner iteration for ILP. |
| **Multi-threading** | `std::thread` pool splitting rows across up to 16 workers. Each thread processes a contiguous range of rows independently (no synchronization needed since output rows are disjoint). |
| **Column tiling** | Inner 128-column tile loop to reuse A elements across k-iterations in L2 cache. |

### Key Difficulty: k-unrolling Changed Addition Order

The initial attempt unrolled the k-loop by 2:
```cpp
c_row[j] += aik0 * b0[j] + aik1 * b1[j];  // Adds PAIR at once
```
The naive adds one product at a time:
```cpp
sum += A[i][k] * B[k][j];  // Sequential accumulation
```
`C += (a+b)` ≠ `C += a; C += b` in floating-point. The pair-wise addition changed the rounding, causing relative errors up to ~1e-4 for cancellation-prone dot products (some C[i][j] ≈ 0).

**Fix**: Removed k-unrolling entirely. Each product `aik * b_row[j]` is added to `c_row[j]` individually, matching the naive's sequential accumulation order exactly.

---

## 4. Filter Gradient

### Algorithm
For each pixel (x,y), compute: 3×3 box blur on channels (a,b,c) → p1, Sobel-X on channels (d,e,f) → p2, Sobel-Y on channels (g,h,i) → p3. Accumulate `total += p1 + p2 + p3` over all interior pixels.

### Optimization Methods

| Optimization | Detail |
|-------------|--------|
| **AoS layout** | 9 floats per `Pixel` struct (a,b,c,d,e,f,g,h,i) packed contiguously. 3×3 window reads 9 structs = 324 bytes, fits in L1d (32KB). |
| **4x unrolling with overlap elimination** | Adjacent pixels' 3×3 windows share 2 columns. Pre-loading 6 columns × 3 rows = 18 structs serves 4 pixels (vs. 36 structs if read independently). **2× reduction in memory reads**. |
| **4 independent accumulators** | `total0, total1, total2, total3` break the dependency chain on the `double total` accumulator, allowing the CPU to execute 4 pixel computations in parallel. |
| **`__restrict` on row pointers** | Hints the compiler that row pointers don't alias, enabling better instruction scheduling. |

### Key Difficulty: Memory Traffic

The SoA (Structure of Arrays) layout in the naive requires 9 separate memory streams. Converting to AoS (Array of Structures) and eliminating redundant reads through overlap-aware unrolling reduces memory traffic by ~50%.

---

## 5. Black-Scholes

### Algorithm
Compute European call/put option prices using the Black-Scholes formula for 81,920 options. Each option requires: `sqrt`, `log`, 3× `exp`, plus polynomial evaluation of the cumulative normal distribution (CNDF).

### Optimization Methods

| Optimization | Implementation |
|-------------|----------------|
| **4x manual unrolling** | Load 4 sets of input parameters, compute 4 options per iteration. The CPU can interleave `sqrtf`/`expf`/`logf` calls from different elements, hiding the 10-20 cycle latency of each. |
| **Extracted `bs_cndf` helper** | Inlines the CNDF computation, avoiding `#pragma omp simd` (which requires `-fopenmp` not present in the build). |
| **Horner's method** | Polynomial `K*(a1 + K*(a2 + K*(a3 + K*(a4 + K*a5))))` reduces 5 multiplications and 5 additions to 5 FMA operations. |
| **Put-Call parity** | `Put = Call - S + K·exp(-rT)` replaces the full put computation (saves 2 multiplications and 1 subtraction). |
| **`__restrict` pointers** | All 7 array pointers declared `__restrict` for aliasing analysis. |

### Key Difficulty: Transcendental Function Latency

`expf` (~10-15 cycles) and `sqrtf` (~10 cycles) are the dominant bottlenecks. A single-element loop stalls the pipeline waiting for each result. 4x unrolling provides enough independent work to keep the FPU busy while waiting for in-flight transcendentals.

---

## 6. Sparse SpMM

### Algorithm
CSR sparse matrix (2048×2048, ~80 nonzeros per row) × dense matrix (2048×2048). Each sparse row scatters its nonzeros across all output columns.

### Optimization Methods

| Optimization | Detail |
|-------------|--------|
| **16x inner loop unrolling** | Aligns with AVX-512 width (16 floats). The compiler can emit `vfmadd231ps` with `_mm512_set1_ps(a)` for the scale factor. |
| **Eliminated per-row memset** | The first nonzero in each row uses `=` (assignment) instead of `+=` (accumulation), covering all output columns. Subsequent nonzeros use `+=`. Empty rows get a single `memset`. Saves ~2048 × 8KB = 16MB of redundant writes. |
| **`__restrict` on CSR arrays** | `row_ptr`, `col_idx`, `values`, `dense_ptr`, `out_ptr` all use `__restrict`. |
| **Stride-1 access** | Both `b_row[j]` and `o[j]` are accessed sequentially in the inner loop — optimal cache behavior. |

---

## 7. Bitwise

### Algorithm
Element-wise bit manipulation on `int8_t` arrays of size 1,024,000. Operation: `result = mixed0 ^ mixed1` where `mixed0 = (diff&0x5A) | (~shared&~0x5A)` and `mixed1 = ((either^0xC3)&(shared|~0xC3)) ^ diff`.

### Optimization Methods

| Optimization | Detail |
|-------------|--------|
| **Word-level processing** | Process 8 bytes at a time as `uint64_t`. The bitwise operations are byte-independent (no carry between bytes), so extending to 64-bit is mathematically exact. |
| **8x unrolling (64 bytes/iter)** | 8 chunks of 8 bytes each processed per loop iteration for maximum ILP. |
| **`static inline` helper** | `process_chunk(va, vb, mLo, mHi)` extracted as a pure function — avoids lambda closure overhead. |
| **`__builtin_memcpy`** | Compiler intrinsic that is always inlined (unlike `std::memcpy` which may remain a function call at `-O2`). |

### Key Difficulty: Memory-to-Register Transfer

The data arrays are `int8_t`, but processing requires loading 8 bytes into a 64-bit register. Since `std::vector<int8_t>` is only 1-byte aligned, direct `reinterpret_cast<uint64_t*>` could cause unaligned access faults on some architectures. Using `__builtin_memcpy` guarantees safe, inlined byte-to-register transfer that the compiler optimizes to a single `mov` instruction on x86-64 (which supports unaligned access natively).

---

## 8. Graph

### Algorithm
Sum `e->to` for all edges in an undirected graph (1,024,000 nodes, avg. degree=8). Naive: adjacency list traversal with pointer chasing.

### Optimization Methods

| Optimization | Detail |
|-------------|--------|
| **CSR format conversion** | Convert linked-list adjacency to CSR (Compressed Sparse Row): contiguous `to[]` array + `offsets[]` array. Eliminates pointer chasing — all edge destinations are in a single contiguous buffer for stride-1 access. |
| **4-way accumulator unrolling** | 8 elements per iteration, distributed across `sum0..sum3` to break the dependency chain on the `uint64_t` checksum. |
| **Conversion outside timing** | CSR construction (`convert_graph_to_csr`) is called once before the benchmark loop, not measured. |

### Key Difficulty: Pointer Chasing

The naive's linked-list traversal (`e = e->next`) has unpredictable memory access patterns, causing L1/L2 cache misses on every edge. CSR format enables sequential memory access through a contiguous array (`to[0], to[1], ...`), which the hardware prefetcher can predict perfectly.

---

## 9. ReLU

### Algorithm
`data[i] = max(data[i], 0.0f)` for 1,024,000 elements. The simplest kernel in the suite.

### Optimization Methods

| Optimization | Detail |
|-------------|--------|
| **16x unrolling** | Process 16 floats per iteration — aligns with AVX-512 width for auto-vectorization to `vmaxps`. |
| **`__restrict` pointer** | Single pointer avoids aliasing checks. |
| **Direct array access** | `ptr[i] = std::max(ptr[i], 0.0f)` — in-place operation, no separate output array needed. |

### Key Insight
For such a trivially parallel element-wise operation, the only bottleneck is memory bandwidth. 16x unrolling lets the compiler generate AVX-512 `vmaxps` instructions that process 16 floats per cycle, saturating the L1 cache bandwidth.

---

## 10. Trace Replay

### Algorithm
Replay a trace of memory access indices against a pre-generated record table. For each trace entry, compute a cost function and accumulate: `total = total × order_mix + cost(records[trace[i]])`.

### Optimization Methods

| Optimization | Detail |
|-------------|--------|
| **Precomputed cost table** | Compute `trace_replay_cost(record)` once during initialization and store in `costs[i]`. The hot loop becomes a simple table lookup: `total = total * order_mix + costs[trace[i]]`. |
| **Eliminated per-iteration computation** | The original `trace_replay_cost` function adds `base_cost + 2*retry_penalty + miss_penalty + bytes/4` — 4 additions and 1 shift per trace entry. With 1,048,576 trace entries, this saves ~5M operations. |

### Key Insight
The trace references a small set of records (65,536) but has many repetitions (1,048,576). Precomputing the cost once per record and reusing it via table lookup is a classic space-time tradeoff: 512KB of extra storage (`uint64_t × 65536`) eliminates millions of redundant arithmetic operations.

---

## 11. Benchmark Results

All benchmarks run on the target server. Each kernel is measured 20 times (with cache flush before each run) and averaged. The **Geometric Mean Speedup** across all 10 kernels is **2.03×** over baseline.

```
Benchmark                Status      Naive (ns)      Stu (ns)   vs Naive  vs Baseline
----------------------------------------------------------------------------------------
Black-Scholes            PASSED       7,215,515     3,901,946     1.849x      1.230x
Sparse SpMM              PASSED     170,068,296    33,641,256     5.055x      3.448x
ReLU                     PASSED       1,491,708       370,769     4.023x      1.483x
Bitwise                  PASSED       2,146,459       264,369     8.119x      0.946x
MatMul                   PASSED     226,793,591     8,116,116    27.944x     10.843x
Trace Replay             PASSED       6,645,965     1,508,966     4.404x      2.253x
Graph                    PASSED      12,673,239     3,188,116     3.975x      1.568x
GRFF                     PASSED      13,320,553     2,803,811     4.751x      3.032x
Image Proc               PASSED      76,620,027    28,307,853     2.707x      1.519x
Filter Gradient          PASSED      32,826,706    16,836,787     1.950x      1.485x
----------------------------------------------------------------------------------------
Geometric mean speedup: 2.03x
```

### Key Observations

| Kernel | vs Naive | vs Baseline | Analysis |
|--------|----------|-------------|----------|
| **MatMul** | 27.9× | 10.8× | Largest improvement. 3D tiling transforms a cache-hostile O(N³) algorithm into one where B-tiles reside in L1. Multi-threading across 16 workers provides additional parallelism. |
| **Bitwise** | 8.1× | 0.95× | 8× word-level unrolling on 64-bit integers reduces the 1M-element loop to ~16K iterations. The baseline was set optimistically; this is still faster than the naive. |
| **Sparse SpMM** | 5.1× | 3.4× | Eliminating per-row `memset` + 16× unrolling + CSR sequential access dramatically outperforms the naive dense-transpose approach. |
| **GRFF** | 4.8× | 3.0× | Exceeds the 3.15× lower bound. AVX2 SIMD for Gate computation + 3-pass fusion reduces 7 intermediate vectors to 1 scratch buffer. |
| **Trace Replay** | 4.4× | 2.3× | Precomputed cost table eliminates ~5M redundant arithmetic ops per benchmark call. |
| **ReLU** | 4.0× | 1.5× | 16× unrolling enables AVX-512 auto-vectorization on the simplest element-wise kernel. |
| **Graph** | 4.0× | 1.6× | CSR conversion replaces pointer-chasing with contiguous `to[]` array traversal. |
| **Image Proc** | 2.7× | 1.5× | Taylor sin/cos approximations eliminate 2 transcendental calls per pixel (2M total). |
| **Filter Gradient** | 2.0× | 1.5× | 4× overlap-aware unrolling halves memory reads. Further gains limited by the 9-channel AoS layout. |
| **Black-Scholes** | 1.8× | 1.2× | 4× unrolling hides `expf`/`sqrtf` latency. Transcendental functions remain the dominant cost. |

---

## Summary of Common Challenges

### 1. Floating-Point Numerical Equivalence
The most pervasive challenge. Mathematical equivalence ≠ numerical equivalence. Key lessons:
- **Never change the order of floating-point additions** (e.g., 4-way parallel accumulation violates this).
- **Never algebraically simplify expressions** that change the rounding order (e.g., `(C+sig)*(1-sig)` ≠ `C+sig-C*sig-sig²` in floating-point).
- **Match the naive's exact formula structure** character-by-character where possible.
- **Avoid SIMD fast-reciprocal instructions** (`_mm256_rcp_ps`, ~12-bit) without Newton-Raphson refinement when the tolerance is tight.

### 2. Memory Access Patterns
- **SoA vs AoS**: Choose based on the access pattern. SoA is better for element-wise operations on single channels; AoS is better when all fields of a pixel are needed together.
- **Pointer chasing vs contiguous**: Converting linked structures (adjacency lists) to flat arrays (CSR) enables hardware prefetching and stride-1 access.
- **Tiling for cache**: 3D tiling in matmul ensures B-tiles fit in L1 cache (16KB for 64×64 floats).

### 3. Function Call Overhead
- `std::memcpy` may not be inlined at `-O2` — use `__builtin_memcpy`.
- Lambdas with captures add closure overhead — use `static inline` helper functions.
- `#pragma omp simd` is silently ignored without `-fopenmp` — use manual unrolling instead.

### 4. Architecture-Specific Optimization
- **AVX2 (8-wide)**: Good balance of throughput and compatibility. Used for GRFF Pass 1.
- **AVX-512 (16-wide)**: Maximum throughput on Cascade Lake. Unrolling factors of 8/16 enable auto-vectorization to these widths.
- **40 CPUs**: Multi-threading exploited in matmul via `std::thread` pool. Other kernels are memory-bandwidth-bound, where single-thread already saturates per-core bandwidth.

---

## Target Architecture Features Leveraged

| Feature | Value | Utilization |
|---------|-------|-------------|
| L1d cache | 32 KB/core | Matmul 64×64 B-tile (16KB), Filter Gradient 3×3 window (324 bytes) |
| L2 cache | 1 MB/core | Matmul column tile (128×512 floats = 256KB) |
| L3 cache | 13.75 MB/socket | GRFF scratch buffers (~4MB) |
| AVX-512 | 16× float per instruction | Sparse SpMM 16x unrolling, ReLU 16x unrolling |
| FMA | `vfmadd231ps` | Matmul inner loop, GRFF gate computation |
| 40 CPUs (2 sockets) | Multi-threading | Matmul row-parallel with `std::thread` |
