---
name: unitrace-vllm-profiling
description: >
  Build, install, and run Intel pti-gpu unitrace against vLLM serving workloads on
  Intel Xe / Xe2 (BMG) GPUs, with ITT-driven /start_profile and /stop_profile
  endpoints so any HTTP client (curl, OpenAI SDK, vllm bench serve) can bracket
  the unitrace ROI. Use this skill for: "build unitrace", "install pti-gpu
  unitrace", "unitrace BMG", "unitrace Level Zero trace", "unitrace --start-paused",
  unitrace + vLLM integration, ITT API resume/pause from a FastAPI route, Chrome
  trace from unitrace, ze_ipc_event_counter_based_handle_t build error in
  pti-gpu, libunitrace_tool.so symbol resolution. Trigger on any partial
  question matching unitrace, pti-gpu, libunitrace_tool, or "profile vLLM with
  unitrace".
---

# unitrace + vLLM Conditional Profiling on Intel Xe GPU

## How to use this skill

- **This file** — what unitrace is, build steps (incl. the BMG/Xe2 patch needed today), launch pattern, ROI gating contract
- **`scripts/serve_with_unitrace.py`** — drop-in vLLM server wrapper that adds `/start_profile` and `/stop_profile` driven by ITT `__itt_resume()`/`__itt_pause()`
- **`scripts/run_unitrace_vllm.sh`** — launcher that wraps the wrapper in `unitrace --start-paused`. Has an `EXIT` trap that auto-runs `generate_report.py` against `$RESULT_DIR` when the launcher exits, so every run produces an HTML report next to the trace JSONs
- **`scripts/generate_report.py`** — standalone report builder: given a result dir, parses `python.*.json`, segments iterations, ranks kernels, and writes a self-contained `unitrace_vllm_report.html` (CSS + SVG plots inline, no external assets). Run it manually on any old result dir: `python3 scripts/generate_report.py /path/to/result_dir/`
- **`references/build-troubleshooting.md`** — the `(uintptr_t)(val)` cast failure on newer L0 headers, generator patch, regen flow
- **`references/itt-resolution.md`** — why ITT symbols come from `libunitrace_tool.so` (not `libittnotify.so`) when running under unitrace
- **`references/output-and-analysis.md`** — Chrome trace JSON layout, device-timing summary, healthy ROI heuristics, Perfetto loading

Developed and validated on an Intel Xe (BMG-class) GPU host running `intel/vllm:0.14.1-xpu` with vLLM `0.1.dev14456+gde3f7fe65`, oneAPI 2025.3, and kernel 6.17.

---

## 1. What unitrace is, and why ROI matters

unitrace is the GPU tracer in [intel/pti-gpu](https://github.com/intel/pti-gpu/tree/master/tools/unitrace). It hooks Level Zero (and optionally OpenCL/SYCL/oneCCL/oneDNN) at runtime and writes:

- a Chrome-trace JSON per process (`python.<pid>.json`) viewable in `chrome://tracing` or Perfetto
- an aggregated device-timing summary printed at exit when `-d` is set

For vLLM serving, capturing the entire server lifetime is wasteful — load_format, KV-cache allocation, warmup, chat-template processing, and idle scheduler ticks all land in the trace. We want to capture only the request window.

unitrace supports **`--start-paused`**: launch with collection paused, then ITT API calls (`__itt_resume()` / `__itt_pause()`) from the target process toggle collection. This is the same control mechanism VTune uses; the [[vtune-vllm-profiling]] skill describes the equivalent flow for VTune GPU-Hotspots.

---

## 2. Build (Linux)

Verified on Intel BMG (Xe2). Requires CMake ≥ 3.22, C++17, oneAPI Base Toolkit, Python ≥ 3.9.

```bash
git clone --depth 1 https://github.com/intel/pti-gpu.git
cd pti-gpu/tools/unitrace
mkdir build && cd build

source /opt/intel/oneapi/setvars.sh --force

cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_WITH_MPI=0 \
      -DBUILD_WITH_ITT=1 \
      -DBUILD_WITH_XPTI=1 \
      -DBUILD_WITH_OPENCL=1 \
      ..
make -j$(nproc)
```

Artifacts:

- `build/unitrace` — the launcher binary (`./unitrace --help` lists all flags)
- `build/libunitrace_tool.so` — injected into the target process; **must stay alongside the binary**, or move both together
- `build/ittapi/bin/libittnotify.a` — static ITT lib unitrace links against; you typically don't need to touch it

### 2.1 BMG/Xe2 build break — `(uintptr_t)(val)` cast on struct handle

Newer Level Zero headers (those bundled with oneAPI 2025.3 / `compute-runtime` recent enough to have `ze_event_counter_based_*`) make `ze_ipc_event_counter_based_handle_t` a struct. The pti-gpu generator emits a `(uintptr_t)(val)` cast that fails:

```
tracing.gen: error: invalid cast from type 'ze_ipc_event_counter_based_handle_t'
{aka '_ze_ipc_event_counter_based_handle_t'} to type 'uintptr_t'
```

Patch `scripts/gen_tracing_callbacks.py::gen_to_hex_string_functions` to emit a templated helper before the macro:

```python
f.write("#include <type_traits>\n")
f.write("template <typename T>\n")
f.write("static inline uintptr_t to_hex_value_(const T& v) {\n")
f.write("  if constexpr (std::is_pointer_v<T>) {\n")
f.write("    return reinterpret_cast<uintptr_t>(v);\n")
f.write("  } else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {\n")
f.write("    return static_cast<uintptr_t>(v);\n")
f.write("  } else {\n")
f.write("    uintptr_t out = 0;\n")
f.write("    std::memcpy(&out, &v, sizeof(v) < sizeof(out) ? sizeof(v) : sizeof(out));\n")
f.write("    return out;\n")
f.write("  }\n")
f.write("}\n")
f.write("#define TO_HEX_STRING(str, val) \\\n")
f.write("    {char buffer[32]; \\\n")
f.write("    std::sprintf(buffer, \"0x%lx\", (unsigned long)to_hex_value_(val)); \\\n")
f.write("    str += std::string(buffer); \\\n")
f.write("    }\n")
```

Then regenerate and rebuild:

```bash
cd build
rm -f tracing.gen common_header.gen l0_loader.gen
make -j$(nproc)
```

See `references/build-troubleshooting.md` for the full diff and rationale.

---

## 3. ROI control contract: how `/start_profile` and `/stop_profile` drive unitrace

vLLM ships a built-in `/start_profile` / `/stop_profile` pair that drives **torch.profiler**. Those endpoints will not affect unitrace. This skill replaces them with ITT-driven equivalents:

```
client (curl / OpenAI SDK / vllm bench serve)
      │
      │ POST /start_profile
      ▼
FastAPI handler   ──►  __itt_resume()      ──►  unitrace begins capture
      │                + create gate file
      │
      │ ... inference traffic ...
      │
      │ POST /stop_profile
      ▼
FastAPI handler   ──►  torch.xpu.synchronize()
                  ──►  __itt_pause()       ──►  unitrace stops capture
                  ──►  remove gate file
```

The gate file (filesystem sentinel) is the cross-process glue: vLLM v1 spawns the EngineCore in a subprocess, and the API-server-process ITT calls don't reach the worker process. We patch `vllm.v1.worker.gpu_worker.Worker.execute_model` so each forward step in the worker process re-checks the gate file and toggles its own `__itt_resume`/`__itt_pause` around the kernel launches. Belt-and-suspenders: even if the outer ITT pause was missed, the worker's per-step gate keeps coverage tight.

This is the **same pattern** used by `[[vtune-vllm-profiling]]` (`serve_with_vtune.py` + `vtune_itt.py`) — just retargeted at unitrace.

### 3.1 Why any client works

The ROI is bounded by curl-able endpoints, not by the request payload. So:

- `curl /start_profile && <inference> && curl /stop_profile` — manual loop
- `curl /start_profile && vllm bench serve --base-url ... && curl /stop_profile` — benchmarks
- An OpenAI SDK script that calls `/start_profile`, runs traffic, then `/stop_profile`
- LangChain/LlamaIndex chains, as long as something brackets the run with the two POSTs

The **wrapper does not care which client**, because both endpoints fire ITT in the same process where unitrace injected its tool .so.

---

## 4. Launch

Two-file launcher (both live in this skill's `scripts/`):

```bash
# scripts/serve_with_unitrace.py  — vLLM server wrapper
# scripts/run_unitrace_vllm.sh    — unitrace --start-paused entrypoint
```

Run from inside the vLLM container (mount the skill into the container or copy both files in):

```bash
cd /path/to/skills/unitrace-vllm-profiling/scripts
bash run_unitrace_vllm.sh
```

The shell script bakes in your env vars (`HF_HOME=/hf_cache`, `SYCL_UR_USE_LEVEL_ZERO_V2=0`, `TORCH_LLM_ALLREDUCE=1`, `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, `VLLM_USE_V1=1`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`) and your CLI flags (`--model`, `--block-size 64`, `--dtype float16`, `--enforce-eager`, etc.). Override via env: `MODEL=... PORT=... bash run_unitrace_vllm.sh`.

unitrace flags applied by default:

```
--start-paused
-d                          # device-timing summary at exit
--chrome-kernel-logging     # GPU kernels in Chrome trace
--chrome-sycl-logging       # SYCL runtime calls
--chrome-itt-logging        # ITT API regions (so /start_profile shows up)
```

### 4.0 Picking a preset (unitrace flag matrix)

The launcher resolves flags from `UNITRACE_PRESET` (shorthand) and `UNITRACE_EXTRA` (appended verbatim). Setting `UNITRACE_FLAGS` directly overrides everything. Always-on under any preset: `--start-paused -d --chrome-itt-logging`.

| Preset | Adds | Use it for | Cost |
|---|---|---|---|
| `lite` | `--chrome-kernel-logging` | Smallest trace; just GPU kernels and ROI markers. Good for first pass / large benchmarks. | low |
| `default` (default) | `--chrome-kernel-logging --chrome-sycl-logging` | Adds SYCL runtime / UR plugin calls — see what the host is doing between launches. | medium |
| `call` | `default` + `--chrome-call-logging` | Trace Level Zero / OpenCL host API calls (`zeCommandListAppend...`, `urKernelSetArg...`). Use when host overhead is the suspect. | medium-high |
| `ccl` | `default` + `--chrome-ccl-logging` | Multi-rank / TP > 1 / DP runs that go through oneCCL allreduce. | medium |
| `dnn` | `default` + `--chrome-dnn-logging` | Anything that calls oneDNN primitives (some SDPA / fused ops). | medium |
| `mpi` | `default` + `--chrome-mpi-logging` | Multi-node MPI runs. | medium |
| `device` | `--chrome-device-logging` | Device activities only — skip per-thread / per-engine breakdown to keep traces small. | low |
| `full` | kernel + device + sycl + ccl + dnn + mpi + call + `--verbose` | Maximum visibility for one-off deep-dives. **Big traces** — keep ROI tight. | high |

#### Per-flag reference

| Flag | What it traces | When to use |
|---|---|---|
| `--chrome-kernel-logging` | GPU + host kernel activities | Almost always. The default GPU timeline. |
| `--chrome-device-logging` | Device-side activities only | When you don't need per-thread or per-engine info — gives a smaller trace. |
| `--chrome-sycl-logging` | SYCL runtime + UR plugin | Gap analysis between kernel launches. |
| `--chrome-call-logging` | Level Zero / OpenCL host calls | Investigating L0 dispatch overhead, command-list builds, sync waits. |
| `--chrome-itt-logging` | `__itt_resume`/`__itt_pause` events | Always on (under any preset). Surfaces ROI markers on the timeline. |
| `--chrome-ccl-logging` | oneCCL collectives | Multi-rank / TP / DP runs to see allreduce/allgather costs. |
| `--chrome-dnn-logging` | oneDNN primitives | Workloads that lower into oneDNN (some attention/conv paths). |
| `--chrome-mpi-logging` | MPI calls | Multi-node MPI runs. |
| `--chrome-no-thread-on-device` | (modifier) drop per-thread breakdown for device events | Reduce trace size when you don't care which CPU thread launched what. |
| `--chrome-no-engine-on-device` | (modifier) drop per-L0-engine / per-OpenCL-queue breakdown | Reduce trace size when you have a single device queue. |
| `--chrome-event-buffer-size <N>` | Per-host-thread event buffer (-1 = unlimited) | Cap memory in long captures (e.g. `1000000` ≈ 1M events/thread). |
| `--verbose`, `-v` | Show kernel shapes in timeline labels | Always nice on Level Zero (shapes are already shown); affects OpenCL output. |
| `--demangle` | Demangle OpenCL kernel names | OpenCL only — Level Zero names are already demangled. |

### 4.0.1 Examples

```bash
# Tight GPU-only trace (fast, small):
UNITRACE_PRESET=lite bash run_unitrace_vllm.sh

# Multi-rank/TP run with allreduce visibility:
TP_SIZE=2 UNITRACE_PRESET=ccl bash run_unitrace_vllm.sh

# Hunting host overhead — show every L0 dispatch call:
UNITRACE_PRESET=call bash run_unitrace_vllm.sh

# Same as default plus verbose + 1M-event buffer cap:
UNITRACE_EXTRA="--verbose --chrome-event-buffer-size 1000000" bash run_unitrace_vllm.sh

# Smaller traces by dropping per-thread / per-engine subdivision:
UNITRACE_EXTRA="--chrome-no-thread-on-device --chrome-no-engine-on-device" \
  bash run_unitrace_vllm.sh

# One-off "show me everything" deep dive (big trace; keep ROI short):
UNITRACE_PRESET=full bash run_unitrace_vllm.sh

# Manual override — full control, ignores preset:
UNITRACE_FLAGS="--start-paused -d --chrome-kernel-logging --chrome-call-logging" \
  bash run_unitrace_vllm.sh
```

### 4.1 Driving the ROI

```bash
PORT=9090

# 1. Start the ROI
curl -X POST http://localhost:$PORT/start_profile

# 2. Issue inference traffic — any client works
curl -X POST http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role":"user","content":"San Francisco is a"}],
        "max_tokens": 24
      }'
# OR a benchmark sweep:
# vllm bench serve --base-url http://localhost:$PORT --model meta-llama/Llama-3.1-8B-Instruct ...

# 3. Stop the ROI
curl -X POST http://localhost:$PORT/stop_profile

# 4. Stop the server (Ctrl-C in the launching shell). unitrace flushes the
#    Chrome trace and prints the device-timing summary to stdout.
```

Result lives in `unitrace_results/<timestamp>_<model>/`:
- `python.<api_server_pid>.json`
- `python.<engine_core_pid>.json`  ← contains the GPU kernels
- `sh.<pid>.json`
- `unitrace_vllm_report.html`  ← auto-generated by the launcher's `EXIT` trap; self-contained HTML with iteration timeline, prefill/decode kernel tables, and findings
- `roi_gate` is auto-removed when `/stop_profile` is called

The launcher's `trap report_on_exit EXIT` calls `generate_report.py "$RESULT_DIR"` after unitrace tears down, so a single Ctrl-C produces both the trace JSONs and the HTML report. To regenerate the report against an old run (or one captured outside the launcher):

```bash
python3 scripts/generate_report.py /path/to/unitrace_results/<timestamp>_<model>/
```

To skip auto-report generation (e.g. during a quick smoke test): `UNITRACE_SKIP_REPORT=1 bash run_unitrace_vllm.sh`.

---

## 5. Sanity checks for a healthy capture

After the run:

```bash
python3 - <<'PY'
import json, collections, glob, os, sys
d = sorted(glob.glob("unitrace_results/*/"))[-1]
files = glob.glob(os.path.join(d, "python.*.json"))
for p in files:
    data = json.load(open(p))
    ev = data.get("traceEvents", data) if isinstance(data, dict) else data
    cats = collections.Counter(e.get("cat","-") for e in ev)
    gpu = [e for e in ev if e.get("cat")=="gpu_op" and e.get("ph")=="X"]
    if gpu:
        ts = sorted(e["ts"] for e in gpu if "ts" in e)
        span = (ts[-1]-ts[0])/1e6 if ts else 0
        print(f"{os.path.basename(p)}: {len(ev)} events, {cats.most_common(3)}, gpu span {span:.2f}s")
    else:
        print(f"{os.path.basename(p)}: {len(ev)} events, no gpu_op")
PY
```

What to look for:

- One `python.<pid>.json` should contain `gpu_op` events. If none has them, the EngineCore subprocess wasn't traced — usually `--follow-child-process 1` is the default but check unitrace flags.
- GPU span should match your inference window (sub-second for a single short chat; several seconds for a benchmark sweep). If span ≈ server lifetime, the ROI gate didn't engage — check that `/start_profile` returned `200` and that `ITT available: True` appeared in the launcher log.
- `--chrome-itt-logging` adds an `ITT` series in Perfetto; you should see resume/pause markers framing the kernel cluster.

`references/output-and-analysis.md` has Perfetto loading tips and how to convert the per-PID JSONs into one merged trace.

---

## 6. Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `ITT available: False` in launcher log | wrapper opened libittnotify.so, didn't find it; pti-gpu builds it as `.a` only | wrapper falls back to `libunitrace_tool.so` (which exports `__itt_resume`/`__itt_pause`); verify `UNITRACE_TOOL_LIB` env or default path |
| Server starts but `/start_profile` 404 | `runpy.run_module(..., __main__)` re-imported api_server fresh, dropping the build_app monkey-patch | use the `run_server(...)` direct-invocation path in this skill's wrapper, NOT `runpy` |
| `[ERROR] Failed to launch target application: --` | `unitrace -- python ...` rejects the `--` separator | drop the `--`: `unitrace <flags> python ...` |
| GPU span = entire server lifetime | ROI gate never closed; client crashed mid-run | the worker patch reads the gate file every step, so leftover `roi_gate` files cause unbounded coverage. Delete stale `roi_gate` files between runs (the launcher script puts them in `$RESULT_DIR` for easy isolation) |
| `_patch_build_app.<locals>.wrapped() takes 1 positional argument but 2 were given` | `build_app` signature widened | wrapper uses `*args, **kwargs` — keep it that way |
| `llvm-foreach SEGV` inside SYCL JIT | torch.compile interaction with intel/vllm-xpu | always launch with `--enforce-eager` (see [[feedback-vllm-xpu-enforce-eager]]) |

---

## 7. Files in this skill

- `SKILL.md` (this file) — overview, build, launch, ROI contract
- `scripts/serve_with_unitrace.py` — vLLM wrapper with ITT-driven endpoints
- `scripts/run_unitrace_vllm.sh` — unitrace launcher (auto-runs `generate_report.py` on exit)
- `scripts/generate_report.py` — builds `unitrace_vllm_report.html` for a given result dir
- `references/build-troubleshooting.md` — generator patch details
- `references/itt-resolution.md` — symbol resolution model
- `references/output-and-analysis.md` — trace artifacts and viewing

Related: the sibling `vtune/` skill applies the same ROI pattern with VTune as the backend.
