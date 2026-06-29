# Reading vLLM Traces in Perfetto

This is the analysis playbook. Pair it with `setup.md` (capture) and
`optimization.md` (turning findings into config changes).

---

## 1. Open the trace

- <https://ui.perfetto.dev> → "Open trace file" → pick `*.pt.trace.json`.
- For files > ~2 GB, use Perfetto's `trace_processor` (see `troubleshooting.md`).
- For `.json.gz`, Perfetto loads it directly; don't decompress first.

---

## 2. Lane layout to expect

Top-down, a healthy vLLM trace has roughly:

1. **API server / asyncio threads** (online server only) — short
   request-handling spans, JSON encode/decode, queue puts.
2. **Engine / scheduler thread(s)** — `Scheduler.schedule`,
   `EngineCore.step`, `add_request`, etc.
3. **Worker process(es)** — one per TP rank. Contains:
   - `Worker.execute_model` / `model_runner.execute_model` blocks (one per step).
   - Sub-blocks for forward, attention, sampling, logits processing.
4. **CUDA/XPU streams** under each worker — actual kernel timeline.
5. **NCCL streams** (TP > 1) — collective comms.

If a lane is missing, see `troubleshooting.md`.

---

## 3. Identifying prefill vs decode

Anchors in the trace:

- **Prefill steps** are *one-shot, large*. Expect:
  - A single `execute_model` block far wider than its neighbors.
  - Attention kernel time roughly proportional to `batch_seq_len^2`
    (or `batch_seq_len * head_dim` for FlashAttention).
  - Few of them (one per request, one per chunk if chunked prefill).
- **Decode steps** are *many, small, near-uniform*. Expect:
  - Cadence: a regular `execute_model` block every ~2-15 ms depending
    on model/hardware.
  - Attention kernel uses paged attention path; size grows linearly with
    accumulated KV.
  - Sampling/logits blocks are visible and non-trivial.

Look for vLLM's annotations (NVTX/ITT-style spans named `prefill`,
`decode`, attention backend names). They're the easiest hook to grab.

---

## 4. Common pathology patterns

### 4.1 GPU idle between decode steps

What it looks like: GPU stream has a kernel ⇒ gap ⇒ kernel, repeating.
CPU lane shows a long `Scheduler.schedule` or `EngineCore.step` block in
the gap.

Diagnoses (most common first):

- **Batch too small to hide CPU overhead.** Increase `--max-num-seqs` or
  the workload's concurrency.
- **Python overhead in the step path.** v0 engine is more Python-heavy
  than v1; check `vllm.__version__` and engine version.
- **`enforce_eager=True`** — for production runs disable it; for
  profiling keep it on but expect lower throughput.
- **Synchronous host-device copies** — look for `cudaMemcpy` /
  `xpuMemcpy` blocks on the gap. Often a logits-postprocessing or
  guided-decoding hook.

### 4.2 Long single block on CPU before any kernels

Tokenization, sampling-params validation, or guided-decoding compile
(outlines/lm-format-enforcer FSM build). Diagnoses:

- Tokenizer is slow / not in fast mode (`use_fast=True`).
- Guided decoding building grammar on every request — check for an
  `outlines` / `xgrammar` block.
- Image/multimodal preprocessing on CPU.

### 4.3 Attention kernel dominates decode time

Diagnoses:

- Wrong backend selected (e.g. legacy paged-attention v1 on long context).
  Set `VLLM_ATTENTION_BACKEND=FLASH_ATTN` (or hardware equivalent).
- Block size mismatched to typical seq length — try `--block-size 16`
  (default) vs `32`.
- Long context with many short decodes ⇒ each step re-reads a long KV;
  this is expected. Only a problem if it's larger than the model
  context warrants.

### 4.4 KV-cache fragmentation / allocator churn

What it looks like: many small `cudaMalloc` / `cudaFree` (or XPU
equivalents) interleaved with kernels; allocator events visible on the
"profiler memory" track if `profile_memory=True`.

Diagnoses:

- `--gpu-memory-utilization` too high — vLLM has no headroom and is
  spilling. Drop to 0.85-0.90.
- Mixed long+short sequences forcing block reallocation. Increase
  `--block-size` to reduce metadata churn.
- An external CUDA library (e.g. NCCL workspace, custom kernels) is
  allocating during steady state — check who issued the alloc.

### 4.5 Tensor-parallel: NCCL stripes longer than compute

What it looks like: `all_reduce` blocks on the NCCL stream are wider
than the surrounding GEMMs.

Diagnoses:

- TP size too high for batch — small batches don't amortize NCCL.
  Try lower TP and higher concurrency.
- Cross-NUMA / cross-PCIe path — verify NVLink/Xe Link topology
  (`nvidia-smi topo -m`, `xpu-smi topology`).
- Network plugin issue (RDMA mistuned). Check NCCL env: `NCCL_DEBUG=INFO`.

### 4.6 Speculative decoding looks worse than no spec

What it looks like: a `draft_model.execute` block is roughly the same
width as `target_model.execute`, and acceptance counters (in stats logs)
are low.

Diagnoses:

- Draft model too large relative to target.
- `num_speculative_tokens` too high for the workload's hit rate.
- Verification step isn't batching rejections efficiently — check the
  scheduler block right after each spec step.

---

## 5. Quantitative starting points

When you open a new trace, eyeball these first:

- **Step cadence** (median time between `execute_model` starts in
  steady-state decode). Compare to model size + hw expectation
  (e.g. 8B fp16 on a single Hopper GPU: 5-12 ms is reasonable;
  on Intel BMG / Xe2 expect higher).
- **Forward fraction** (sum of forward-pass kernel time / step duration).
  Healthy: > 0.85 in pure decode. Below 0.7 means CPU/scheduler
  overhead is the bottleneck.
- **Attention fraction within forward** (attention kernels / forward).
  Long-context decode: 0.5-0.8 expected. Short-context decode:
  0.2-0.4. Out-of-range numbers point to backend/config issues.
- **NCCL fraction** (TP only) — collectives / step duration. If > 0.2
  steady-state, TP is overprovisioned for the batch.

These aren't hard thresholds, just smell checks. Always compare a
suspect run against a baseline configuration before recommending
changes.

---

## 6. Useful Perfetto features

- **Pivot Table** (top-right "Query" → "Slice ↦ Stats by name") — get
  total time per kernel name. The fastest way to find dominant kernels
  without scrolling.
- **SQL queries** on the trace processor — for traces too big to scroll.
  Example:

  ```sql
  SELECT name, COUNT(*) AS n, SUM(dur)/1e6 AS total_ms
  FROM slice
  WHERE category = 'kernel'
  GROUP BY name ORDER BY total_ms DESC LIMIT 20;
  ```

- **Flow events** — vLLM emits some; they connect a Python op on CPU
  to its dispatched kernel on GPU. Toggle "Flow events" in the view
  menu.
- **Pinning lanes** — pin the worker stream + NCCL stream side by
  side to align comm with compute.

---

## 7. Multi-rank correlation

- Open each rank's trace in a separate tab, line up by wall clock —
  Perfetto syncs clocks if the metadata events are present.
- For numerical analysis (e.g. straggler detection across 8 ranks),
  use `hta` (HolisticTraceAnalysis) on the directory of trace files
  rather than eyeballing.

---

## 8. Annotating your own ROIs

If you fork vLLM or write a driver, you can add NVTX/ITT-style
ranges that show up as Perfetto blocks:

```python
import torch
with torch.profiler.record_function("my_roi"):
    # whatever you want labeled
    ...
```

Combined with `start_profile/stop_profile`, this is the cleanest way
to mark "this is the prefill step I care about" without forking
the engine.
