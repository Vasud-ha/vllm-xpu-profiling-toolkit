---
name: vtune-vllm-profiling
description: >
  Expert-level guidance for Intel VTune Profiler GPU-Hotspots conditional profiling
  integrated with vLLM serving workloads on Intel Xe GPU architecture. Use this skill
  whenever the user asks about VTune, GPU-Hotspots, ITT API, Level Zero profiling,
  vLLM performance analysis, ROI-based profiling, prefill/decode phase profiling,
  or any Intel GPU profiling for LLM inference. Also trigger for questions about
  vtune -start-paused, vtune -command resume/pause, __itt_resume/__itt_pause,
  collection control APIs, reducing profiling overhead in serving workloads, or
  isolating GPU kernels from tokenizer/network overhead in PyTorch/vLLM pipelines.
  Trigger this skill even for partial questions like "how do I profile only decode
  in vLLM" or "vtune gpu-hotspots paused mode" or "ITT API Python".
---

# VTune Conditional Profiling for vLLM on Intel Xe GPU

## How to Use This Skill

- **This file** — architecture, cheat sheet reference, quick-start commands, workflow
- **`validation-and-flow.txt`** — annotated capture timeline, pre-flight + post-run verification checks (read this if you want to know *why* the ROI looks the way it does)
- **`references/itt-api.md`** — ITT API deep dive, C bindings, Python wrappers
- **`references/vllm-integration.md`** — All vLLM patch patterns (prefill, decode, per-request, middleware)
- **`references/troubleshooting.md`** — GPU timeline failures, ZE conflicts, sync issues, **healthy-trace cheat sheet** (§11)

---

## 1. VTune Cheat Sheet: Control Collection Commands

From the official Intel VTune Profiler Cheat Sheet
(https://www.intel.com/content/dam/develop/external/us/en/documents/vtune-profiler-cheat-sheet.pdf):

```
vtune <-action> [-action-option] [-global-option] [[--] <target> [target-options]]

ACTION 1 — Run Analysis:
  vtune -collect gpu-hotspots [options] -- <target>
  Key flags:
    -r, -result-dir=<str>       Result directory path
    -start-paused               Start collection paused, resume later  <- KEY FLAG
    -resume-after=<seconds>     Delay collection N seconds after start
    -k, -knob=<str>             Analysis-specific modifier (repeatable)
    -target-pid=<uint>          Attach to running process by PID
    -target-process=<str>       Attach to running process by name

ACTION 3 — Control Running Analysis:
  vtune -command pause   -r <result-dir>   <- Pause collection
  vtune -command resume  -r <result-dir>   <- Resume collection
  vtune -command stop    -r <result-dir>   <- Stop and finalize
  vtune -command status  -r <result-dir>   <- Print current status
  vtune -command mark    -r <result-dir>   <- Place reference timestamp

Example from cheat sheet:
  vtune -command resume -r r000hs
```

---

## 2. Architecture: Global vs Conditional Profiling

### Global Profiling (avoid for vLLM)

VTune collects from process start — captures everything:
- Python interpreter startup (~2-5s noise)
- vLLM engine initialization, model load, KV cache alloc
- Tokenizer overhead (CPU-bound, pollutes GPU timeline)
- HTTP server / asyncio event loop overhead
- Warmup requests (skews kernel timing distributions)
- Network I/O between requests

**Result**: 10-50GB trace files, GPU timeline dominated by non-inference events.

### Conditional / ROI-Based Profiling (target approach)

```
[Process Start]
     |
     v
[VTune: PAUSED] <- Zero collection overhead outside ROI
     |
     v
[vLLM init, model load, KV alloc, warmup] <- EXCLUDED
     |
     v
[ROI Start] -> itt_resume() or vtune -command resume
     |
     v
[Inference: prefill + decode GPU kernels] <- COLLECTED
     |
     v
[GPU Sync: torch.xpu.synchronize()]       <- MANDATORY
     |
     v
[ROI End] -> itt_pause() or vtune -command pause
     |
     v
[VTune: PAUSED again]
```

**Quantitative benefits**:
| Metric              | Global     | Conditional    |
|---------------------|------------|----------------|
| Result size         | 10-50 GB   | 100 MB - 2 GB  |
| GPU timeline noise  | Very high  | Minimal        |
| Warmup contamination| Always     | None           |
| Analysis time in GUI| Minutes    | Seconds        |
| Kernel attribution  | Low        | High           |

---

## 3. Complete VTune Command Examples

### 3.1 Basic GPU-Hotspots, Start Paused

```bash
vtune \
  -collect gpu-hotspots \
  -start-paused \
  -result-dir ./vtune_results/vllm_roi \
  -- python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Llama-3-8B \
       --port 8000 \
       --dtype bfloat16
```

### 3.2 With GPU-Hotspots Knobs for Xe Architecture

> Knob names vary across VTune releases. Verify against your install with
> `vtune -help collect gpu-hotspots` before pasting. The CPU-collector knobs
> `sampling-interval` and `enable-stack-collection` are NOT valid here and
> will fail with `Cannot find knob ...`.

```bash
vtune \
  -collect gpu-hotspots \
  -start-paused \
  -knob characterization-mode=overview \
  -knob collect-programming-api=true \
  -knob gpu-sampling-interval=1 \
  -result-dir ./vtune_results/vllm_roi \
  -- python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Llama-3-8B \
       --port 8000
```

Common valid knobs (gpu-hotspots, VTune 2024.x–2025.x):

| Knob | Values | Purpose |
|------|--------|---------|
| `profiling-mode` | `characterization` *(default)*, `source-analysis` | Top-level mode. `source-analysis` enables EU-stall / instruction-level attribution (one kernel at a time). |
| `characterization-mode` | `overview` *(default)*, `global-local-accesses`, `instruction-count`, `dynamic-instruction-count`, `full-compute` | Sub-mode under `characterization`. |
| `computing-task-of-interest` | `<kernel-name>` | Restrict source-analysis to one kernel. |
| `gpu-sampling-interval` | float ms (default 1.0) | EU sampling rate. |
| `collect-programming-api` | `true` / `false` | Capture Level Zero / SYCL / OpenCL API timeline (recommended for vLLM). |
| `collect-host-gpu-data-transfers` | `true` / `false` | H2D / D2H copies. |
| `enable-gpu-runtimes` | `true` / `false` | Per-runtime breakdown. |

### 3.3 Source-Analysis Mode (EU Stalls, Xe-Specific)

```bash
vtune \
  -collect gpu-hotspots \
  -start-paused \
  -knob profiling-mode=source-analysis \
  -knob computing-task-of-interest=<kernel-name> \
  -result-dir ./vtune_results/vllm_xe_source \
  -- python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Llama-3-8B \
       --port 8000
```

`gpu-vendor=intel` was removed in newer releases — VTune auto-detects.
Source-analysis is most useful when you've already identified a hotspot
in characterization mode and want EU stalls for that one kernel.

### 3.4 External Collection Control (from Benchmark Script)

```bash
# Terminal 1: Launch vLLM under VTune (paused)
export VTUNE_RESULT=$(realpath ./vtune_results/run_$(date +%Y%m%d_%H%M%S))
vtune -collect gpu-hotspots -start-paused -result-dir $VTUNE_RESULT \
  -- python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Llama-3-8B --port 8000 &

# Terminal 2: Benchmark script controls collection timing
sleep 30                                      # Wait for server ready + warmup
vtune -command resume -r $VTUNE_RESULT        # START profiling
python run_benchmark.py --requests 50         # Run profiled workload
vtune -command pause  -r $VTUNE_RESULT        # STOP profiling
vtune -command stop   -r $VTUNE_RESULT        # Finalize result
```

### 3.5 Attach to Running vLLM Process

```bash
python -m vllm.entrypoints.openai.api_server --model ... --port 8000 &
VLLM_PID=$!

vtune -collect gpu-hotspots \
      -start-paused \
      -target-pid $VLLM_PID \
      -result-dir ./vtune_results/attach_run

vtune -command resume -r ./vtune_results/attach_run
```

---

## 4. ITT API: Inline Collection Control

The ITT (Instrumentation and Tracing Technology) API allows controlling
collection from within the profiled process — tightest possible ROI.

### 4.1 Python ittapi Package

```bash
pip install ittapi   # Intel official Python binding
```

```python
import ittapi

ittapi.pause()    # Ensure paused at startup (belt-and-suspenders)

# Enter ROI
ittapi.resume()
# ... GPU inference work ...
import torch
torch.xpu.synchronize()   # CRITICAL: flush GPU queue before pause
ittapi.pause()
```

### 4.2 ctypes Wrapper (Zero Dependencies)

For production use without pip. Full implementation in `references/itt-api.md`.

```python
from vtune_itt import itt_resume, itt_pause, vtune_roi

# As context manager:
with vtune_roi():            # auto-resume, auto-sync, auto-pause
    model.forward(input_ids)

# As decorator:
@roi()
def execute_model(self, ...):
    ...
```

---

## 5. ROI Boundaries: Best Choices for LLM

vLLM v1 (default ≥0.10, including `intel/vllm:0.14.1-xpu`) split the engine
across processes. The Worker class moved; entry-point symbols changed:

| Concern | v0 path | v1 path (current) |
|---------|---------|-------------------|
| Worker class | `vllm.worker.worker.Worker` | `vllm.v1.worker.gpu_worker.Worker` |
| `execute_model` arg | `ExecuteModelRequest` | `SchedulerOutput` |
| Phase detection | `seq_group_metadata_list[*].is_prompt` | `bool(scheduler_output.scheduled_new_reqs)` |
| API server entry | `from ... import main; main()` | `runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__", alter_sys=True)` |

| Target | Integration Point (v1) | Granularity | Noise Level |
|--------|------------------------|-------------|-------------|
| Full inference (recommended) | `vllm.v1.worker.gpu_worker.Worker.execute_model` | Per-step | Low |
| Pure GPU compute (tightest) | `model.forward` (post-load patch) | Per-forward | Minimal |
| Prefill phase only | Conditional in patched `execute_model` (check `scheduled_new_reqs`) | Per-prefill | Low |
| Decode phase only | Conditional in patched `execute_model` (check `scheduled_cached_reqs`) | Per-decode | Low |
| Per-request | `AsyncLLM.generate` (parent process) | Per-request | Medium |
| Full pipeline | `EngineCore.step` (in subprocess) | Per-step | Medium |

**Never ROI above `EngineCore.step` / `LLMEngine.step`** — includes HTTP,
tokenizer, asyncio noise.

### 5.1 Per-step ROI vs Batched (window) ROI

The default patch in `references/vllm-integration.md` opens an ROI on every
`Worker.execute_model` call: 2N transitions for N steps and many fragmented
sub-ROIs in the trace. For most analyses, a **single window-level ROI** that
spans the whole profiled batch is cleaner and cheaper.

| Scheme | Resume/pause count | Trace fragmentation | Phase tagging | When to use |
|--------|-------------------|---------------------|---------------|-------------|
| Per-step | 2 × steps | High (many short ROIs) | Hardest signal | Phase-isolated runs (decode-only, prefill-only) and per-step kernel attribution |
| Batched window (recommended) | 2 (open once, close once) | Low (one continuous ROI) | Use ITT *tasks* (not resume/pause) | Steady-state hotspot ranking, BW/EU summaries, regression baselines |
| Hybrid | 2 + ITT tasks per step | Low + per-step task lanes | Best | Most production runs — open ROI once, tag each step with an ITT task |

Hybrid pattern (recommended default):

```python
# Worker patch — single resume/pause, but every step is an ITT task
import ittapi, torch
from vtune_itt import itt_resume, itt_pause

_dom    = ittapi.domain_create("vllm.window")
_open   = False

def _vtune_execute_model(self, scheduler_output, *args, **kwargs):
    global _open
    if not _open:
        itt_resume()       # opened once for the whole profiled batch
        _open = True

    label = classify_step(scheduler_output)   # see §3 of vllm-integration.md
    h     = ittapi.string_handle_create(label)
    ittapi.task_begin(_dom, h)
    try:
        return _orig_execute(self, scheduler_output, *args, **kwargs)
    finally:
        torch.xpu.synchronize()  # per-step sync keeps task end accurate
        ittapi.task_end(_dom)

# Close the window once at the end of the benchmark, e.g. via SIGTERM handler:
import atexit
def _close_roi():
    if _open:
        torch.xpu.synchronize()
        itt_pause()
atexit.register(_close_roi)
```

The benchmark script controls when the window opens with `vtune -command resume`
(workers re-resume idempotently); the per-step ITT *tasks* give phase lanes in
the Tasks tab without paying the per-step resume/pause cost.

---

## 6. GPU Synchronization Requirement

Always call `torch.xpu.synchronize()` before pausing. This is the #1 source
of empty GPU timelines when omitted.

```
Without sync:                    With sync:
[resume]                         [resume]
  GPU work submitted             GPU work submitted
[pause] <- too early!           torch.xpu.synchronize() <- wait for GPU
  GPU kernels still running!    [pause] <- safe, all kernels captured
  VTune misses them entirely!
```

Overhead of `synchronize()`: 1-50 us depending on queue depth.
For LLM forward passes (10-500 ms), this is negligible.

---

## 7. Complete Production Workflow

```bash
#!/bin/bash
source /opt/intel/oneapi/setvars.sh
unset ZE_ENABLE_TRACING_LAYER   # Let VTune manage this

MODEL="meta-llama/Llama-3-8B"
PORT=8000
RESULT_DIR=$(realpath ./vtune_results/$(date +%Y%m%d_%H%M%S))

# Step 1: Launch vLLM under VTune (paused)
vtune -collect gpu-hotspots \
      -start-paused \
      -knob collect-programming-api=true \
      -result-dir $RESULT_DIR \
      -- python -m vllm.entrypoints.openai.api_server \
           --model $MODEL --port $PORT --disable-log-requests &

# Step 2: Wait for server ready
until curl -sf http://localhost:$PORT/health > /dev/null; do sleep 2; done

# Step 3: Warmup (VTune stays PAUSED)
python benchmark_requests.py --num-requests 10 --quiet
sleep 3   # Drain async GPU work

# Step 4: Resume -> Benchmark -> Pause
vtune -command resume -r $RESULT_DIR
python benchmark_requests.py --num-requests 50 --max-tokens 256
vtune -command pause -r $RESULT_DIR

# Step 5: Finalize
vtune -command stop -r $RESULT_DIR
echo "Open: vtune-gui $RESULT_DIR"
```

---

## 8. Multi-Process Considerations

vLLM v1 always runs the engine core in a separate process from the API server,
even with TP=1. Tensor parallelism adds one GPU worker process per rank.

```
api_server (parent)  --ZMQ-->  EngineCore (subprocess)  --spawn-->  Worker rank 0
                                                                    Worker rank 1
                                                                    ...
```

| Control Method | Scope |
|----------------|-------|
| `vtune -command resume` | All processes under VTune umbrella (parent + all descendants) |
| `ittapi.resume()` / `__itt_resume()` | Only the calling process |

**Implications for v1:**
- ITT calls in the API server parent are useless for GPU profiling — the GPU
  work happens in the worker subprocess. Patch `Worker.execute_model` instead;
  the patch survives because the subprocess re-imports the (already-patched)
  class from the parent's monkey-patched `sys.modules`. If your launcher uses
  `multiprocessing` `spawn` start-method, monkey-patches don't transfer — apply
  them inside the worker, e.g. via `Worker.__init__` patching or a vLLM plugin.
- For multi-rank ITT control, each rank must call ITT independently.
- External `vtune -command resume/pause` is the simplest control surface for v1
  because it covers all descendants regardless of spawn semantics.

See `references/vllm-integration.md` for per-rank patterns.

---

## 9. Xe GPU Architecture Caveats

1. **Level Zero tracing**: VTune manages `ZE_ENABLE_TRACING_LAYER` automatically.
   Never set it manually — causes empty GPU timelines.

2. **Tile affinity** (multi-tile Xe, e.g., PVC):
   ```bash
   export ZE_AFFINITY_MASK=0   # Profile tile 0 only
   ```

3. **Decode is BW-bound**: Expect 10-40% EU utilization for decode on large
   models — correct, not a profiling failure.

4. **Prefill is compute-bound**: Expect 50-80% EU utilization for long sequences.

---

## 10. VTune GUI Validation

```
Open result: vtune-gui <result-dir>

Validation checklist:
[x] Summary tab -> GPU Time > 0, matches ROI duration
[x] Bottom-up tab -> Top kernels are GEMM/attention (sort by GPU Time desc)
[x] Timeline tab -> GPU activity shows bursts only during ROI window
[x] Platform tab -> Level Zero queue shows kernel submissions
[x] Top Hotspots -> Kernel names: xetla_gemm, esimd_attention, sycl_native_*

Red flags:
[!] GPU Time = 0        -> Missing torch.xpu.synchronize() before pause
[!] Empty GPU timeline  -> ZE tracing env conflict (see troubleshooting.md)
[!] Huge CPU fraction   -> ROI too wide; move to Worker.execute_model()
[!] Truncated kernels   -> pause() fired before GPU drain
```

For numeric health bounds (EU active %, GPU BW %, expected GPU Time per
prefill/decode step on BMG/PVC) and the explicit pre-flight + post-run check
sequence, see `validation-and-flow.txt` (sections C-F).

---

## 11. Headless Reporting & A/B Diff

The GUI is not required to extract hotspots. Use this when:

- running profiling in CI / on a headless node
- you need a numeric baseline vs. a candidate (regression gate)
- you want grep-able artifacts archived alongside the result dir

### 11.1 Single-result reports

```bash
# Top hotspots (CSV, machine-parseable)
vtune -report hotspots -r $RESULT_DIR \
      -group-by computing-task \
      -format csv -limit 25 > $RESULT_DIR/hotspots.csv

# Top-line summary (GPU Time, EU active %, BW %, elapsed)
vtune -report summary -r $RESULT_DIR \
      -format csv > $RESULT_DIR/summary.csv

# Per-task GPU Time + EU stalls (good for ranking attention vs. GEMM)
vtune -report hw-events -r $RESULT_DIR \
      -group-by computing-task \
      -format csv -limit 25 > $RESULT_DIR/hw-events.csv

# Phase breakdown when ITT tasks were emitted (see vllm-integration.md §3)
vtune -report tasks -r $RESULT_DIR \
      -group-by task -format csv > $RESULT_DIR/tasks.csv
```

### 11.2 Baseline vs. candidate diff

```bash
BASE=./vtune_results/baseline
CAND=./vtune_results/candidate

# Side-by-side hotspots (sorted by GPU Time, top 30 each)
for D in $BASE $CAND; do
  vtune -report hotspots -r $D \
        -group-by computing-task -format csv -limit 30 \
    | awk -F, 'NR==1 || $0!~/^,*$/' \
    > $D/hotspots.csv
done

# Quick regression diff: kernel name + GPU Time delta
python3 - <<'PY'
import csv, os, sys
def load(p):
    with open(p) as f:
        return {r['Computing Task']: float(r.get('GPU Time', 0) or 0)
                for r in csv.DictReader(f) if r.get('Computing Task')}
b, c = load(f"{os.environ['BASE']}/hotspots.csv"), load(f"{os.environ['CAND']}/hotspots.csv")
keys = sorted(set(b)|set(c), key=lambda k: -(c.get(k,0) - b.get(k,0)))
print(f"{'kernel':<60} {'base(s)':>10} {'cand(s)':>10} {'delta(s)':>10}")
for k in keys[:25]:
    print(f"{k[:60]:<60} {b.get(k,0):>10.3f} {c.get(k,0):>10.3f} "
          f"{c.get(k,0)-b.get(k,0):>+10.3f}")
PY
```

Use `vtune -command mark -r $RESULT_DIR -message "request_42_start"` from a
benchmark script to drop named timeline markers — these appear in the report
output and let you correlate hotspot deltas with specific requests without
ITT plumbing.

---

## Reference Files

| File | Contents |
|------|----------|
| `validation-and-flow.txt` | End-to-end captured-flow timeline (T0..T10), pre-flight + post-run verification, healthy numeric bounds |
| `references/itt-api.md` | ITT API bindings, ctypes wrapper, C extension, task/domain/frame API, overhead table |
| `references/vllm-integration.md` | Monkey-patch patterns, batched/per-step ROI, sharper phase classifier, prefill/decode/mixed/cache-hit, per-request ROI, async, benchmark script |
| `references/troubleshooting.md` | Empty timelines, ZE conflicts, sync issues, environment checklist, GUI validation, **healthy-trace cheat sheet** (§11) |
