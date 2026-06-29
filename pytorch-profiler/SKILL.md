---
name: vllm-pytorch-profiler
description: >
  Expert guidance for profiling vLLM serving workloads using vLLM's official
  profiling framework, the PyTorch profiler, and Perfetto / chrome://tracing
  for trace visualization. Use this skill when the user asks about
  VLLM_TORCH_PROFILER_DIR, --profile / --profile-result-dir, vllm bench
  serve --profile, AsyncLLMEngine.start_profile/stop_profile, profile_run,
  torch.profiler.profile inside vLLM, JSON / .pt.trace.json traces, Perfetto
  analysis of vLLM, prefill vs decode phase isolation, continuous batching
  timeline analysis, PagedAttention/KV-cache patterns, attention backend
  comparison (FlashAttention/xFormers/Triton/torch SDPA), tensor-parallel
  NCCL overhead, speculative decoding traces, or any
  "how do I profile vLLM with PyTorch profiler" / "open this trace in
  Perfetto" / "explain this gap in the vLLM timeline" question. Also trigger
  for partial asks like "vllm profile decode only", "perfetto vllm trace
  too big", or "torch profiler with vllm tensor parallel".
---

# vLLM PyTorch Profiler + Perfetto Analysis

Companion to [[vtune-vllm-profiling]]. That skill covers Intel VTune
GPU-Hotspots on Intel Xe; this skill covers vLLM's **official PyTorch-profiler
based** flow and Perfetto/chrome-tracing analysis (vendor-agnostic, works on
NVIDIA CUDA and Intel XPU builds).

Authoritative reference: <https://docs.vllm.ai/en/stable/contributing/profiling/profiling_index.html>

---

## How to Use This Skill

- **This file** — env vars, CLI flags, API entry points, capture recipes, Perfetto workflow.
- **`references/setup.md`** — full setup + config matrix (server, offline, online bench, distributed/TP).
- **`references/perfetto-analysis.md`** — what to look for in a vLLM trace: prefill vs decode, batching, KV-cache, NCCL, spec decode.
- **`references/optimization.md`** — turning trace findings into vLLM config changes (`--max-num-seqs`, `--block-size`, attention backend, quant, TP).
- **`references/troubleshooting.md`** — trace too big, missing kernels, CUPTI/XPU issues, Perfetto load failures.

---

## 1. The vLLM Profiling Contract

vLLM exposes profiling through **three layers**, all backed by `torch.profiler`:

| Layer | Entry point | When to use |
|------|-------------|-------------|
| Env-var gated server | `VLLM_TORCH_PROFILER_DIR=/path` | Always-on profile endpoint on `vllm serve` |
| Programmatic | `LLM.start_profile()` / `LLM.stop_profile()` (offline) and `AsyncLLMEngine.start_profile()` / `stop_profile()` (online) | Offline scripts / custom drivers |
| Bench-driven | `vllm bench serve --profile ...` (after `VLLM_TORCH_PROFILER_DIR` is set) | Reproducible end-to-end captures |

> Setting `VLLM_TORCH_PROFILER_DIR` is **required** before the server/engine
> starts. The endpoints/methods are no-ops without it. This is the single most
> common "my profile is empty" cause.

The worker loop is decorated so each `execute_model` step lands in the
trace as its own block. With `--enforce-eager` you get readable kernel
names; with CUDA graphs / `torch.compile` enabled you'll see fused
"CUDAGraphLaunch" / compiled-region blocks instead.

---

## 2. Quick-Start Recipes

### A. Server + bench (recommended)

```bash
# Terminal 1 — start server with profiler dir set
export VLLM_TORCH_PROFILER_DIR=$HOME/vllm_profile
mkdir -p "$VLLM_TORCH_PROFILER_DIR"

vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enforce-eager \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192

# Terminal 2 — drive a controlled workload and trigger profiling around it
vllm bench serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --num-prompts 8 \
  --request-rate inf \
  --profile
```

`--profile` calls the server's `/start_profile` before the run and
`/stop_profile` after. Trace `.pt.trace.json` (sometimes `.json.gz`)
files appear in `$VLLM_TORCH_PROFILER_DIR`.

### B. HTTP endpoints directly

```bash
curl -X POST http://localhost:8000/start_profile
# ... drive your traffic ...
curl -X POST http://localhost:8000/stop_profile
```

### C. Offline `LLM` API

```python
import os
os.environ["VLLM_TORCH_PROFILER_DIR"] = "/home/me/vllm_profile"

from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", enforce_eager=True)

# Warmup OUTSIDE the profile window
llm.generate(["warmup"] * 4, SamplingParams(max_tokens=8))

llm.start_profile()
out = llm.generate(prompts, SamplingParams(max_tokens=64))
llm.stop_profile()
```

### D. Async engine

```python
await engine.start_profile()
# ... await engine.generate(...) over your ROI ...
await engine.stop_profile()
```

> Always **warm up before** `start_profile` (compile, autotune, KV cache
> alloc, CUDA-graph capture). Otherwise the first step dominates the trace
> and the steady-state numbers are wrong.

---

## 3. Keep the Trace Small and Useful

Traces are *huge* by default; Perfetto chokes around 1-2 GB and `chrome://tracing`
much sooner. Tactics, in order of preference:

1. **Tight ROI** — start/stop around 1-2 prefill steps + 5-20 decode steps.
   Don't profile a whole benchmark run.
2. **Small batch / low concurrency** — `--num-prompts 4-8`, short outputs.
3. **`--enforce-eager`** — disables CUDA graphs so kernel names are real
   names, and avoids capturing graph build noise.
4. **Disable shape/stack recording** unless you need them
   (`record_shapes=False`, `with_stack=False` in any custom `torch.profiler`
   wrapper). vLLM's defaults are reasonable; only override if you've
   forked the profiling code.
5. **Tensor parallel**: each rank writes its own `.pt.trace.json`. Open
   them **separately** in Perfetto first; only merge if you need
   cross-rank correlation (see `references/setup.md`).

---

## 4. Open in Perfetto

1. Go to <https://ui.perfetto.dev>.
2. "Open trace file" → pick `*.pt.trace.json` (or `.json.gz`).
3. If it's > ~2 GB, follow Perfetto's "trace_processor" large-file path
   (see `references/troubleshooting.md`).

What you should see in a healthy vLLM trace:

- A **CPU thread** for the API server / scheduler with `step()` /
  `schedule()` blocks.
- A **worker process** with `Worker.execute_model` / `model_runner.execute_model`
  blocks, one per scheduler step.
- A **CUDA / XPU stream** lane with kernels (attention, GEMMs, all-reduce
  on TP).
- ITT/NVTX-style annotations from vLLM (`prefill`, `decode`, attention
  backend names) — use these to anchor the timeline.

`references/perfetto-analysis.md` has the full reading guide
(prefill vs decode signatures, KV-cache pattern, batching gaps,
spec-decode shape, NCCL overhead).

---

## 5. What to Investigate, by Symptom

| Symptom in trace | Likely cause | Where to look |
|------------------|--------------|---------------|
| Long single block before any kernels | Tokenization / scheduler stall on CPU | `references/perfetto-analysis.md` §"CPU stalls" |
| Big GPU idle between decode steps | Scheduler/Python overhead, batch too small | `--max-num-seqs`, `--max-num-batched-tokens`; `references/optimization.md` §"Batching" |
| Attention kernel >> GEMMs in decode | Wrong/legacy backend, long context not paged efficiently | `references/optimization.md` §"Attention backend" |
| Many small `cudaMalloc` / allocator events | KV-cache fragmentation, gpu-mem-util too high | `--gpu-memory-utilization`, `--block-size` |
| Tall NCCL all-reduce stripes | TP comm-bound layer, undersized batch for TP=N | `references/optimization.md` §"Tensor parallel" |
| Draft model time ~ target model time | Spec decode mis-tuned | `references/optimization.md` §"Speculative decoding" |

---

## 6. Sanity Checks Before You Trust a Trace

- `VLLM_TORCH_PROFILER_DIR` was set **before** the server/engine started.
- A warmup ran **before** `start_profile`.
- ROI is bounded — you stopped profiling, you didn't ctrl-C the process.
- File size is plausible (tens to a few hundred MB, not multi-GB unless intentional).
- Trace contains both CPU step blocks **and** GPU/XPU kernels — if
  GPU lane is empty, CUPTI/XPU profiler isn't attached
  (see `references/troubleshooting.md`).

---

## 7. Cross-Skill Notes

- For Intel Xe (PVC/BMG) + ITT-driven ROI control with VTune
  GPU-Hotspots, use [[vtune-vllm-profiling]] instead. The trace formats
  are different: VTune `.vtune` result-dir vs Perfetto-readable
  `.pt.trace.json`.
- For Intel XPU vLLM builds, prefer `--enforce-eager` (same reason
  as in [[feedback_vllm_xpu_enforce_eager]]): SYCL JIT + CUDA-graph-like
  capture paths produce unreadable / sometimes crashing traces.

---

## 8. When the User Asks Something Specific

Before answering from this skill, **verify against the running version**:

- `python -c "import vllm; print(vllm.__version__)"` — features (e.g. v1
  scheduler, spec decode flags, bench CLI) move between releases.
- `vllm serve --help | grep -i profile` — confirm `--profile-result-dir`
  / endpoint flags exist on this build.
- Read the actual trace file header (first few KB) before giving advice
  about its contents.

If the user names a different profiler (Nsight Systems, VTune, perf),
defer to that tool's skill or say so — don't shoehorn it into the
PyTorch-profiler workflow.
