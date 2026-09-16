# HERM optimization lab

Status: experimental and opt-in. The current machine has `torch 2.14.0+cpu`
and no CUDA device. The modules below define bounded seams and executable
contracts; `BulkPrefixCache` is connected to generation when explicitly
selected, but the repository still has no measured GPU speedup, memory-quality
improvement or end-to-end batching integration.

## Verdict

GPU-first is the right direction for HERM, but “GPU-first” does not mean that
every operation should be moved to a GPU or that a Python thread creates
parallelism by itself. The useful order is to measure the end-to-end critical
path, remove allocations and padding, then fuse only the operation that the
profile identifies as dominant.

## Delivered seams

| Area | Module | Contract | Current status |
| --- | --- | --- | --- |
| CUDA scan | `koemi.model.cuda_scan` | CUDA-only affine scan with chunk carry, validation and synchronized diagnostics | PyTorch tensor-op backend; no native `.cu` kernel; not wired into `KoemiModel` |
| State memory | `koemi.model.gpu_memory` | Reusable fixed-layout buffers with explicit reset/resize and optional stream scope | CPU-tested; CUDA path is conditional and unmeasured |
| Precision | `koemi.model.gpu_precision` | Device-safe FP32/BF16/FP16 policy, scoped TF32 flags and FP32 comparison | CPU-tested; CUDA AMP and numerical tolerance are unmeasured |
| Context summary | `koemi.model.context_summary` | Multi-rate bounded EMA slots, evidence and confidence-gated reads | Opt-in; validation and serialization cross to host memory |
| Context index | `koemi.model.context_index` | Exact namespace-aware longest-prefix index with TTL/LRU and explicit data serialization | CPU-side exact index; not semantic retrieval |
| Context admission | `koemi.model.context_policy` | Causal bounded selection by surprise, recency and novelty | Opt-in; bounded but `O(batch × candidates × capacity)` |
| Training batch | `koemi.training.batching_mode` | Length buckets, padded-token budget, stable permutation and accumulation boundaries | Plan-only; trainer and collation are unchanged |
| Inference batch | `koemi.runtime.inference_batching` | Contract-compatible FIFO queues, padding, deadlines and request handles | Scheduler-only; caller executes and completes the model batch |
| Exact blocks | `koemi.runtime.bulk_blocks` + `koemi.runtime.bulk_prefix_cache` | RAM/SSD fixed-token blocks with digest, TTL, capacity and validated prefix-state restore | `BulkPrefixCache` is connected to generation; SSD payloads are integrity-checked, not encrypted |
| Async bulk | `koemi.runtime.bulk_executor` | Bounded CPU preparation plus optional CUDA stream/event enqueue | Pipeline seam; it does not execute a model or promise overlap |

Nine seams remain separate from the default forward path. `BulkPrefixCache` is
connected to generation only when explicitly selected. The exact context and
bulk stores reject fuzzy reuse: a cache hit must identify the namespace and
prove the complete token sequence before a bounded state is released.

## What was actually verified

Before these seams, the local baseline was Python 3.13.14 with PyTorch
`2.14.0+cpu`: `compileall` passed and the existing suite ran 159 tests with one
conditional CUDA skip. After the six model seams, the focused suite ran 63
tests with 11 CUDA skips. After all ten seams, the complete suite ran 271 tests
with 13 conditional CUDA skips, and `compileall` passed.

After connecting `BulkPrefixCache` to the opt-in generation path, the complete
local suite ran 275 tests with 13 conditional CUDA skips, and `compileall`
passed. This validates exact reuse and state restoration; it does not measure
the cost of key hashing, SSD I/O or CUDA execution.

The result is contract verification, not performance evidence. No A100/T4
execution, CUDA kernel timing, GPU memory profile, end-to-end batch throughput,
context recall ablation, cache hit-rate study, or quality comparison was run on
this host. The default HERM model and trainer were intentionally not changed.

## GPU-first design

### Affine scan

`cuda_affine_scan` is a drop-in CUDA backend seam for the affine recurrence. It
uses PyTorch tensor operations and a Hillis–Steele-style scan inside bounded
chunks, carrying the final state between chunks. That makes the algebra and
gradient contract testable now, but it is not a custom CUDA kernel. A native
fused kernel is justified only if a real CUDA profile shows that scan launches,
temporary tensors, or dispatch dominate the forward and backward pass.

### State and precision

`StateBufferPool` reuses exact shapes, dtypes and devices, avoiding repeated
allocation of fixed HERM state. It deliberately performs no implicit stream
synchronization; the caller owns stream ordering. `gpu_precision` makes AMP
selection explicit and provides an FP32 comparison probe before a reduced
precision run is accepted. Diagnostics are allowed to synchronize or read
scalars because they are observability boundaries, not token-level kernels.

### Shape policy

Length-aware training batches reduce padded work. Inference batches group only
requests with the same model, device, dtype and namespace, then report real and
padded tokens. Static length buckets are a prerequisite for reliable CUDA Graph
experiments; dynamic shapes should be enabled only when the measured workload
needs them and recompilation is controlled.

## Context and “Jenga” blocks

The cheap memory proposal has three different jobs rather than one unbounded
memory object:

1. `ContextSummary` compresses recent observations into a fixed number of
   exponential time scales. Evidence and confidence bound the read; it is a
   lossy summary, not a claim of exact recall.
2. `ContextIndex`, `BulkBlockStore` and `BulkPrefixCache` identify exact prefixes
   or fixed token blocks. `BulkPrefixCache` restores the bounded state at the
   end of the longest complete block and runs only its suffix, but it cannot
   infer that two similar questions are equivalent.
3. `ContextPolicy` admits a bounded set of causal candidates using scores that
   are already available. It never consults future positions and does not
   replace HERM's recurrent state transition.

The safe block flow is:

```text
token sequence
    -> namespace + exact digest chain
    -> candidate metadata without payload release
    -> full sequence and TTL validation
    -> restore bounded state
    -> compute only the uncached suffix
```

RAM/VRAM is the active working set. SSD is a persistence tier and can be slower
than recomputation; it must not be placed in the critical path for every token.
Bulk block payloads can contain prompt-derived state, so the current explicit
warning about unencrypted local storage remains in force.

## What “extreme async” can mean physically

CUDA launches are asynchronous with respect to the host, but useful overlap
requires dependencies and resources to be explicit. Host-to-device copies need
appropriate pinned host memory for true asynchronous transfer; separate streams
and events express when a consumer may read a result. The bulk executor exposes
that boundary with bounded worker backpressure and optional stream events. It
does not claim that CPU preparation, SSD I/O and GPU compute always overlap.

The implementation must preserve the following sequence:

```text
CPU prepare -> pinned/non-blocking transfer -> GPU stream compute
       \-> next CPU prepare ----------------^ event dependency
```

The transfer policy must be tuned rather than maximized: pinned memory is a
finite resource, and copying more data can erase the gain. CUDA Graphs are a
later option for static shapes, stable control flow and stable memory addresses;
they are not a substitute for fixing graph breaks or dynamic allocation.

These constraints follow the [CUDA asynchronous execution guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html),
the [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/),
and [PyTorch CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html).
For host-loader overlap, use the [PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide).
For compilation, check [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
and [CUDA Graph constraints](https://docs.pytorch.org/docs/main/notes/cuda.html#constraints).

## Acceptance gate before default promotion

Opt-in seams may expose an explicit caller boundary, as `BulkPrefixCache` does
for generation. Promotion into the default HERM path still requires all of
these measurements:

- CUDA execution on the target GPU passes forward, backward and state
  equivalence against the sequential scan oracle for FP32 and the selected AMP
  dtype.
- Three warmed repetitions report tokens/s, p50/p95 step latency, peak VRAM,
  allocation count and host-to-device bytes for the baseline and optimized
  paths. A kernel is kept only when end-to-end time improves after its launch,
  staging and synchronization overhead.
- Length bucketing reports padding fraction and does not change sample order,
  supervision masks, gradient-accumulation boundaries or optimizer-step count.
- Inference batching reports queue wait, batch occupancy, real/padded tokens,
  cancellation and deadline behavior; no request can receive another request's
  output or state.
- Context experiments use exact key/value or detail-recall tasks across at
  least three seeds. They report hit rate, false-reuse rate, state bytes and
  quality against the unchanged HERM baseline. False reuse must remain zero for
  exact blocks.
- SSD tests cover crash recovery, TTL, namespace isolation, capacity and the
  decision for encryption/key ownership before prompt-derived data is persisted.

## Recommended order

1. Profile the unchanged HERM forward and backward on the target CUDA device.
2. Integrate buffer reuse, pinned staging and explicit precision checks; measure
   memory and transfer overhead before changing the recurrence.
3. Integrate length-aware training batches and continuous inference batching;
   compare padding and queue metrics with the baseline.
4. Evaluate exact prefix/block reuse and the EMA summary on quality tasks, with
   the default path still available as a control.
5. Only then prototype a native fused scan or CUDA Graph capture for the exact
   static bucket that the profile selects.

Until that gate is closed, “GPU-first”, `BulkPrefixCache` and “Jenga memory” are
opt-in product paths or design directions, not measured performance capabilities.
