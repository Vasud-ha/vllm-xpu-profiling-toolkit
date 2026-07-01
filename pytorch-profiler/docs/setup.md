# vLLM PyTorch Profiler — Setup & Configuration

Authoritative source: <https://docs.vllm.ai/en/stable/contributing/profiling/profiling_index.html>.
Treat this file as a working reference; verify flags/endpoints against
the version actually running (`vllm --version`).

---

## 1. Enabling the profiler

The gate changed at vLLM 0.17. Pick the right one for the build you're running.

### vLLM >= 0.17 — `--profiler-config` CLI arg (current)

```bash
TRACE_DIR=/abs/path/to/trace_dir
mkdir -p "$TRACE_DIR"

vllm serve <model> \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$TRACE_DIR\",\"torch_profiler_use_gzip\":true}"
```

`ProfilerConfig` fields you can toggle in the same JSON blob:
`torch_profiler_with_stack` (default true), `torch_profiler_with_flops`,
`torch_profiler_use_gzip`, `torch_profiler_record_shapes`,
`torch_profiler_with_memory`, `ignore_frontend`, `delay_iterations`,
`max_iterations`. Full schema: `vllm/config/profiler.py`.

### vLLM <= 0.14 — `VLLM_TORCH_PROFILER_DIR` env var (legacy)

```bash
export VLLM_TORCH_PROFILER_DIR=/abs/path/to/trace_dir
mkdir -p "$VLLM_TORCH_PROFILER_DIR"
```

On vLLM >= 0.17 this env var is silently ignored (`Unknown vLLM environment
variable detected: VLLM_TORCH_PROFILER_DIR` in the server log). `run_pt_profile_vllm.sh`
sets both — env var and CLI arg — so it works either way.

### Rules (both mechanisms)

- Must be set/passed **before** `vllm serve` / `LLM(...)` / `AsyncLLMEngine.from_engine_args` runs.
- Must be writable by the server process (watch for Docker uid mismatch).
- One directory holds **all** ranks' traces in tensor-parallel runs;
  filenames disambiguate by hostname/PID/rank.

If neither is provided, all of the following silently no-op / 404:

- `POST /start_profile`, `POST /stop_profile` HTTP endpoints
- `LLM.start_profile()` / `LLM.stop_profile()`
- `AsyncLLMEngine.start_profile()` / `AsyncLLMEngine.stop_profile()`
- `vllm bench serve --profile`

---

## 2. Server-side configurations

### 2.1 OpenAI-compatible server

```bash
# vLLM >= 0.17
TRACE_DIR=$HOME/vllm_profile
mkdir -p "$TRACE_DIR"
vllm serve <model> \
  --enforce-eager \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$TRACE_DIR\",\"torch_profiler_use_gzip\":true}"

# vLLM <= 0.14 (legacy)
# export VLLM_TORCH_PROFILER_DIR=$HOME/vllm_profile
# vllm serve <model> --enforce-eager ...
```

Profile control:

```bash
curl -X POST http://localhost:8000/start_profile
# drive workload
curl -X POST http://localhost:8000/stop_profile
```

### 2.2 `vllm bench serve --profile`

`vllm bench serve` (formerly `benchmark_serving.py`) calls
`/start_profile` and `/stop_profile` for you when `--profile` is passed.
The server **must** have been started with the profiler enabled —
`--profiler-config` on vLLM >= 0.17 or `VLLM_TORCH_PROFILER_DIR` on <= 0.14 —
otherwise `--profile` is a no-op.

```bash
vllm bench serve \
  --backend vllm \
  --model <model> \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 128 \
  --num-prompts 8 \
  --request-rate inf \
  --profile
```

Tips:

- Keep `--num-prompts` small (4-16). Profiling all 1000 prompts of a
  bench run is rarely useful and produces a multi-GB trace.
- `--request-rate inf` for steady decode-phase analysis; finite rate
  to study queueing/scheduling instead.

---

## 3. Offline `LLM` API

```python
# vLLM >= 0.17 — preferred: pass ProfilerConfig
from vllm import LLM, SamplingParams
from vllm.config import ProfilerConfig

llm = LLM(
    model=...,
    enforce_eager=True,
    profiler_config=ProfilerConfig(
        profiler="torch",
        torch_profiler_dir="/abs/path",
    ),
)

# vLLM <= 0.14 legacy — env var still honored
# import os
# os.environ["VLLM_TORCH_PROFILER_DIR"] = "/abs/path"
# from vllm import LLM
# llm = LLM(model=..., enforce_eager=True)

# 1. Warmup OUTSIDE the profile
llm.generate(["warmup"] * 4, SamplingParams(max_tokens=8))

# 2. Profile a tight ROI
llm.start_profile()
llm.generate(prompts, SamplingParams(max_tokens=64))
llm.stop_profile()
```

Why warmup matters:

- First call triggers CUDA graph capture (if not eager), Triton/torch.compile
  autotuning, KV-cache allocation, weight CUDA-paging. All of these dominate
  a cold profile and lie about steady-state.

---

## 4. Async engine (custom servers)

```python
from vllm import AsyncEngineArgs, AsyncLLMEngine

engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model=...))

await _drive_warmup(engine)
await engine.start_profile()
await _drive_roi(engine)
await engine.stop_profile()
```

Both `start_profile` and `stop_profile` are coroutines — `await` them.

---

## 5. Tensor parallel / multi-rank

- Each worker rank writes its **own** `.pt.trace.json` into
  `VLLM_TORCH_PROFILER_DIR`. File names include rank/PID.
- For first-pass analysis, open rank 0 in Perfetto.
- Cross-rank correlation (e.g. NCCL all-reduce timing) needs the
  trace files merged. Two options:
  - **Easy**: open each rank in a separate Perfetto tab, eyeball the
    annotated `all_reduce` blocks.
  - **Rigorous**: use `torch.profiler`'s built-in distributed merge
    or [HolisticTrace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis)
    (`hta`) to load the directory of per-rank traces and analyze
    collectively.
- File timestamps across ranks share a base — Perfetto's clock-sync
  marks line them up, but only after merging.

---

## 6. Shapes / stacks / memory — when to enable

vLLM's defaults capture activities, GPU kernels, and Python ops. If you
need more, you'll need to fork/edit the profiler call sites in
`vllm/v1/worker/...` (or v0 equivalent) and pass:

- `record_shapes=True` — adds tensor shapes to ops. Useful for
  detecting padding/recompiles. **Doubles+ trace size.**
- `with_stack=True` — Python stacks per op. Heavy; only when you can't
  identify which call site issued a kernel.
- `profile_memory=True` — allocator events. Use when chasing OOMs or
  KV-cache fragmentation. Significantly larger trace.

Default off > on for these. Turn on, capture *briefly*, turn off.

---

## 7. CUDA / XPU activity coverage

PyTorch profiler relies on:

- **CUDA**: CUPTI. Ensure CUPTI is installed and not blocked by the
  driver (e.g., NVIDIA `nvidia-smi -pm 1` and CUPTI permission can
  matter on locked-down hosts).
- **Intel XPU**: Level Zero / oneAPI profiling tools. Versions matter —
  see [[reference_vllm_v1_apis]] for the v1 worker path on
  intel/vllm:0.14.1-xpu, and prefer `--enforce-eager`
  ([[feedback_vllm_xpu_enforce_eager]]).
- **CPU-only**: still valid, but you only see Python/op blocks, no
  GPU lane.

If the GPU lane is empty in Perfetto, almost always one of:

- CUPTI not loadable (CUDA) — `LD_LIBRARY_PATH` missing.
- XPU profiler ext not loaded — wrong torch / IPEX version.
- Container missing capabilities (`--cap-add SYS_ADMIN`, `/proc` access).

---

## 8. Verifying the trace before deep analysis

Cheap pre-checks before opening a multi-hundred-MB file:

```bash
TRACE_DIR=<the dir you passed to --profiler-config or exported as VLLM_TORCH_PROFILER_DIR>
ls -lh "$TRACE_DIR"
# Plausible: tens to low-hundreds of MB per rank for a tight ROI.

# Peek at the JSON header (first event names):
python -c "
import json, sys
with open(sys.argv[1]) as f:
    head = f.read(200_000)
# Trace JSON is a single object with traceEvents; the header has metadata.
print(head[:2000])
" "$TRACE_DIR"/<file>.pt.trace.json
```

If the file is `.json.gz`, `zcat | head -c 200000` instead.

You're looking for:

- `"traceEvents"` array.
- A few `process_name` / `thread_name` metadata events identifying
  ranks/streams.
- Kernel names you recognize (`vllm`, `flash_attn`, `paged_attn`,
  `gemm`, `all_reduce`, `xetla` on Intel, etc).
