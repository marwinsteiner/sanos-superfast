# Low-Latency Data Structures in sanos-superfast

## 1. Flat Arrays & Contiguous Memory (SOA)

**Where**: `ExpiryMarket::block_` — single `AVec<double>` holding all 9 per-strike fields.

**Layout**: `[strikes₀..ₙ | bids₀..ₙ | asks₀..ₙ | mids₀..ₙ | spreads₀..ₙ | weights₀..ₙ | iv_bids₀..ₙ | iv_asks₀..ₙ | w_prev₀..ₙ]`

**Why it matters**: The alternative is 9 separate `std::vector` heap allocations scattered across memory. When `compute_gradient` iterates over all weights and mids, the SOA layout means each field lives in a contiguous block — one cache line (64 bytes) services 8 consecutive doubles. With separate vectors, each vector's data might be on a different memory page, causing TLB misses and defeating the hardware prefetcher.

**Measured impact**: -40μs warm recalibration (370→340μs median).

## 2. Lock-Free SPSC Ring Buffer

**Where**: `TickQueue<Capacity>` in `tick_queue.hpp`.

**Structure**: Fixed-size circular buffer indexed by `head` (producer writes) and `tail` (consumer reads), both `std::atomic<int>`. Capacity is a power of two so `index & (Capacity - 1)` replaces modulo.

**Why it matters**: In a live trading system, the market data thread must never block — a blocked feed handler means missed quotes and stale prices. The SPSC queue lets the producer thread push `TickUpdate` messages (24 bytes each: expiry index, strike index, new bid, new ask) without any mutex, lock, or CAS loop. The consumer (fitting thread) drains all pending updates in one batch via `drain()`, then re-solves only the affected expiries once.

**Key invariant**: Only the producer writes `head`, only the consumer writes `tail`. Each is on its own cache line (see below), so neither core ever waits for the other's cache line.

## 3. Object Pool / Flyweight Pattern

**Where**: `ExpiryFit::w2`, `ExpiryFit::w2mid` — persistent scratch buffers. `DenseMat::resize()` — reuses capacity.

**The problem they solve**: The original code allocated temporary `AVec<double> w2(m)` and `AVec<double> w2mid(m)` inside `compute_hessian` and `compute_gradient` on every call — including every tick update. Each `AVec` constructor calls `_aligned_malloc` (Windows) or `posix_memalign` (Linux), which enters the kernel allocator. At ~200ns per allocation, two allocations per tick = 400ns of pure overhead on the critical path.

**The fix**: Store `w2` and `w2mid` as members of `ExpiryFit`. They're allocated once during `setup_expiry` and reused across all subsequent calls. `DenseMat::resize()` was also changed to only grow the underlying buffer, never shrink — if the new logical size fits the existing capacity, it just zeroes the memory with `memset` (no system call).

**Measured impact**: -100μs per tick update, -500μs cold calibration.

## 4. Cache Line Alignment & False Sharing Prevention

**Where**: `ThreadPool::PaddedAtomic` — `alignas(64)` wrapper around `std::atomic<int>`.

**The problem**: The thread pool uses two atomic counters: `next_task_` (incremented by every worker to claim work) and `done_count_` (incremented when a task completes). Without padding, both 4-byte atomics sit on the same 64-byte cache line. When core 0 does `next_task_.fetch_add(1)` and core 1 does `done_count_.fetch_add(1)`, the cache coherence protocol (MESI) bounces the cache line between cores. Each bounce costs ~50-100ns depending on the interconnect.

**The fix**: `alignas(64)` ensures each atomic occupies its own cache line. Now `next_task_` and `done_count_` are on different cache lines, so concurrent updates don't interfere.

Same pattern is used in `TickQueue::PaddedIdx` for the `head_` and `tail_` indices.

## 5. Cache Warming (Software Prefetch)

**Where**: `solve_kkt()` H-matrix gather, `compute_hessian()` column pairs.

**The problem**: The QP solver's `solve_kkt` function gathers a reduced Hessian from the full Hessian using indirect indices: `H_r[i,j] = H[free_idx[i]*n + free_idx[j]]`. This scattered access pattern defeats the hardware prefetcher, which predicts stride-1 or stride-N patterns. Each cache miss stalls the pipeline ~100 cycles waiting for L2/L3.

**The fix**: Before the gather loop, issue `_mm_prefetch(H + free_idx[i] * n, _MM_HINT_T0)` for each row that will be read. This gives the memory controller a ~100-cycle head start to bring the data into L1. By the time the gather loop reaches row `i`, it's already in cache.

In `compute_hessian`, the inner loop reads two columns `cj` and `ck` of the kernel matrix C. While computing the dot product for column pair (j, k), we prefetch column k+1 into L2 with `_MM_HINT_T1`. This hides the latency of accessing the next column, which is `n_eval` doubles away — too far apart for the hardware prefetcher to detect.

## 6. Priority Queue (Urgency-Ordered Dispatch)

**Where**: `Surface::calibrate()` — sort expiry indices by ascending DTE before `parallel_for`.

**Why it matters**: In options trading, near-term expiries (0DTE, 1DTE) have the highest gamma and the most urgent hedging needs. When calibrating all 20 expiries in parallel, the thread pool picks up tasks in the order they're submitted. By sorting shortest-DTE first, the 0DTE surface is available for quoting/hedging while the 2-year LEAPS is still being fitted in the background.

This is a priority queue by scheduling order, not a heap data structure — the sort is O(M log M) where M is ~20 expiries, taking <1μs, negligible compared to the calibration work.
