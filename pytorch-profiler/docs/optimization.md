# From Trace Findings to vLLM Config Changes

This file maps concrete trace symptoms to vLLM knobs. Always validate
with a before/after capture — don't ship config changes off a single
trace.

---

## 1. Batching: `--max-num-seqs`, `--max-num-batched-tokens`

When to touch:

- Decode steps show GPU idle gaps with non-trivial CPU work in between.
- Throughput plateau at low GPU utilization (< 60%) under load.

Tune:

- `--max-num-seqs` (default 256): cap on concurrent sequences in
  decode. Increase if GPU is idle and KV memory has room; decrease if
  TTFT is suffering due to scheduler latency.
- `--max-num-batched-tokens` (default ≈ 8192 or model-dependent):
  total tokens per scheduler step. For prefill-heavy workloads, raise
  to capture larger chunks; for decode-heavy, the batch size is usually
  not the binding constraint.
- **Chunked prefill** (`--enable-chunked-prefill`, default on in v1):
  splits long prefills into smaller chunks so decode requests aren't
  starved. If your trace shows a single huge prefill block blocking
  decode, confirm chunked prefill is enabled.

Verify by re-profiling: GPU idle gaps shrink, step cadence becomes
denser, forward-fraction rises.

---

## 2. PagedAttention: `--block-size`, `--gpu-memory-utilization`

- `--block-size` (default 16): KV-cache page size. Larger blocks =
  less metadata + fewer allocator events, but more wasted slots when
  sequences end mid-block. Try 16 → 32 if you see allocator churn in
  the trace.
- `--gpu-memory-utilization` (default 0.90): fraction of device memory
  reserved for KV cache. Lower it to 0.85 if you see allocator
  pressure or other libs (NCCL, custom kernels) being squeezed.

---

## 3. Attention backend selection

Selection knob: `VLLM_ATTENTION_BACKEND` env var, or `--attention-backend`
on supported builds.

Common values (CUDA): `FLASH_ATTN`, `FLASHINFER`, `XFORMERS`, `TORCH_SDPA`.
On Intel XPU: built-in IPEX/XeTLA paths are selected automatically;
fewer knobs.

Trace-driven choice:

- **Long context, decode-heavy** → FlashInfer / FlashAttention paged
  paths. Check for paged-attention kernel names in the trace.
- **Short prompts, large batch** → most modern backends are similar.
  Pick the one with the cleanest kernel naming for analysis.
- **Numerical issues / debugging** → `TORCH_SDPA` (slowest but most
  predictable).

After switching, re-capture and compare the attention-fraction metric
from `perfetto-analysis.md` §5.

---

## 4. Quantization

Trace-relevant effects:

- **Compute-bound layers** (large GEMMs): quantization shrinks the
  kernel; expect MM kernel time to drop, attention time roughly
  unchanged.
- **Memory-bound decode**: quantizing weights helps weight-load
  bandwidth; attention is mostly bound by KV reads, so KV-cache
  quantization (`--kv-cache-dtype fp8` or `int8`) matters more.

Methods (CUDA): AWQ, GPTQ, FP8, SmoothQuant. On Intel XPU, FP8 / int8
flows are model- and version-dependent — verify against the running
build. We've observed BMG cases where int8 hit thermal/power limits
rather than compute limits, which masqueraded as a kernel slowdown.

A profile that's "slower than expected" at int8 is often *not* a
kernel issue. Cross-check power/thermal counters before blaming
the kernel.

---

## 5. Tensor parallel scaling

Read `perfetto-analysis.md` §4.5 first; use this section after
identifying TP as the bottleneck.

- Lower `--tensor-parallel-size` if NCCL fraction > 0.2 in steady
  state and request concurrency is low.
- Raise concurrency (clients, `--max-num-seqs`) if you must keep TP
  high — comm cost is mostly fixed, so amortize it across more work.
- Pipeline parallel (`--pipeline-parallel-size`) shifts the comm
  pattern from many small all-reduces to fewer larger sends; useful
  when interconnect is weak but devices are many.

---

## 6. Speculative decoding

Knobs:

- `--speculative-config` / `--speculative-model` / `--num-speculative-tokens`
  (exact flag set is version-dependent — check `vllm serve --help`).
- For Eagle/Medusa-style decoders: model-specific config.

Trace pattern for healthy spec decode:

- A short `draft.execute_model` block followed by a
  `target.execute_model` block ~5-10× wider, then sampling.
- Acceptance counters (in vLLM logs) > 50% — below that, spec is
  rarely a win.

If draft and target blocks are similar widths, the draft model is too
large relative to the target. Pick a smaller draft or a different
spec method.

---

## 7. Engine version (v0 vs v1)

vLLM v1 (default in current releases) has a leaner Python step path
and different scheduler. If you're on a build that supports both:

- v0: more Python overhead per step; trace shows wider scheduler blocks.
- v1: tighter step cadence; the v1 worker path is what
  [[reference_vllm_v1_apis]] documents for the Intel XPU build we use.

Don't mix v0/v1 advice. Confirm with `VLLM_USE_V1` / engine-version
log line at startup.

---

## 8. Order of changes

When multiple findings compete, change one knob per capture, in this
order:

1. Engine version + `--enforce-eager` for analysis sanity.
2. Batching (`--max-num-seqs`, `--max-num-batched-tokens`,
   chunked prefill).
3. Memory (`--gpu-memory-utilization`, `--block-size`).
4. Attention backend.
5. Quantization (weights, then KV cache).
6. Parallelism (TP/PP).
7. Speculative decoding.

Capture between steps. It's easy to "improve" one fraction while
regressing the overall throughput — only the end-to-end numbers
matter.
