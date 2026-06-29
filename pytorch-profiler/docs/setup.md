# vLLM PyTorch Profiler — Setup & Configuration

Authoritative source: <https://docs.vllm.ai/en/stable/contributing/profiling/profiling_index.html>.
Treat this file as a working reference; verify flags/endpoints against
the version actually running (`vllm --version`).

---

## 1. The single required env var

```bash
export VLLM_TORCH_PROFILER_DIR=/abs/path/to/trace_dir
mkdir -p "$VLLM_TORCH_PROFILER_DIR"
```

Rules:

- Must be set **before** `vllm serve` / `LLM(...)` / `AsyncLLMEngine.from_engine_args` runs.
- Must be writable by the server process (watch for Docker uid mismatch).
- One directory holds **all** ranks' traces in tensor-parallel runs;
  filenames disambiguate by hostname/PID/rank.

If unset, all of the following silently no-op:

- `POST /start_profile`, `POST /stop_profile` HTTP endpoints
- `LLM.start_profile()` / `LLM.stop_profile()`
- `AsyncLLMEngine.start_profile()` / `AsyncLLMEngine.stop_profile()`
- `vllm bench serve --profile`

---

## 2. Server-side configurations

### 2.1 OpenAI-compatible server

```bash
export VLLM_TORCH_PROFILER_DIR=$HOME/vllm_profile
vllm serve <model> \
  --enforce-eager \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85
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
The server **must** have been started with `VLLM_TORCH_PROFILER_DIR` set.

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
import os
os.environ["VLLM_TORCH_PROFILER_DIR"] = "/abs/path"

from vllm import LLM, SamplingParams

llm = LLM(model=..., enforce_eager=True)

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
ls -lh "$VLLM_TORCH_PROFILER_DIR"
# Plausible: tens to low-hundreds of MB per rank for a tight ROI.

# Peek at the JSON header (first event names):
python -c "
import json, sys
with open(sys.argv[1]) as f:
    head = f.read(200_000)
# Trace JSON is a single object with traceEvents; the header has metadata.
print(head[:2000])
" "$VLLM_TORCH_PROFILER_DIR"/<file>.pt.trace.json
```

If the file is `.json.gz`, `zcat | head -c 200000` instead.

You're looking for:

- `"traceEvents"` array.
- A few `process_name` / `thread_name` metadata events identifying
  ranks/streams.
- Kernel names you recognize (`vllm`, `flash_attn`, `paged_attn`,
  `gemm`, `all_reduce`, `xetla` on Intel, etc).
