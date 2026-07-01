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

> **Version note (July 2026).** vLLM 0.17+ removed `--disable-log-requests`;
> request logging is off by default and the toggle is now
> `--enable-log-requests` / `--no-enable-log-requests`. Both `run_pt_profile_vllm.sh`
> and the quick-start recipes below have been updated. If you're on an older
> build (<= 0.14.x), you can add `--no-enable-log-requests` back — the wrapper
> forwards unknown flags to the api_server unchanged via `runpy`.

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
| CLI-gated server (vLLM >= 0.17) | `--profiler-config '{"profiler":"torch","torch_profiler_dir":"/abs/path"}'` | Current path — required for `/start_profile` route to register |
| Env-var gated server (vLLM <= 0.14) | `VLLM_TORCH_PROFILER_DIR=/path` | Legacy path — silently no-ops on >= 0.17 |
| Programmatic | `LLM.start_profile()` / `LLM.stop_profile()` (offline) and `AsyncLLMEngine.start_profile()` / `stop_profile()` (online) | Offline scripts / custom drivers |
| Bench-driven | `vllm bench serve --profile ...` (after the server is started with one of the gates above) | Reproducible end-to-end captures |

> **vLLM 0.17 changed the profiler gate.** The env-var `VLLM_TORCH_PROFILER_DIR`
> is silently ignored on modern builds — you now pass a `ProfilerConfig` JSON
> via `--profiler-config`. Symptom on 0.17 with only the env var: server logs
> `Unknown vLLM environment variable detected: VLLM_TORCH_PROFILER_DIR`, and
> `curl -X POST /start_profile` returns 404 because the profile router only
> attaches when `profiler_config.profiler is not None`. `run_pt_profile_vllm.sh`
> sets both — env var and CLI arg — so it works either way.

The worker loop is decorated so each `execute_model` step lands in the
trace as its own block. With `--enforce-eager` you get readable kernel
names; with CUDA graphs / `torch.compile` enabled you'll see fused
"CUDAGraphLaunch" / compiled-region blocks instead.

---

## 2. Quick-Start Recipes

### A. Server + bench (recommended)

```bash
# Terminal 1 — start server with the profiler configured.
# vLLM >= 0.17: pass --profiler-config with a JSON ProfilerConfig.
TRACE_DIR="$HOME/vllm_profile"
mkdir -p "$TRACE_DIR"

vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enforce-eager \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$TRACE_DIR\",\"torch_profiler_use_gzip\":true}"

# vLLM <= 0.14 (legacy): use the env var instead
# export VLLM_TORCH_PROFILER_DIR="$TRACE_DIR"
# vllm serve meta-llama/Llama-3.1-8B-Instruct --enforce-eager ...

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
files appear in the configured `torch_profiler_dir`.

### B. HTTP endpoints directly

```bash
curl -X POST http://localhost:8000/start_profile
# ... drive your traffic ...
curl -X POST http://localhost:8000/stop_profile
```

### C. Offline `LLM` API

```python
import os
# vLLM <= 0.14: env var still honored offline.
# vLLM >= 0.17: pass profiler_config directly to LLM(...) instead.
os.environ["VLLM_TORCH_PROFILER_DIR"] = "/home/me/vllm_profile"

from vllm import LLM, SamplingParams
# On 0.17+, prefer:
#   from vllm.config import ProfilerConfig
#   llm = LLM(model=..., enforce_eager=True,
#             profiler_config=ProfilerConfig(profiler="torch",
#                                            torch_profiler_dir="/home/me/vllm_profile"))
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

- Profiler was configured **before** the server/engine started, via
  `--profiler-config` (vLLM >= 0.17) or `VLLM_TORCH_PROFILER_DIR` (vLLM <= 0.14).
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
