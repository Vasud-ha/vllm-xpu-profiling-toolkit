# vllm-xpu-profiling-toolkit

Profiling recipes, wrapper scripts, and analysis notes for **vLLM serving on Intel Xe (XPU / BMG-class) GPUs**, covering three complementary profilers:

| Folder | Profiler | When to use |
| --- | --- | --- |
| [`vtune/`](./vtune) | **Intel VTune Profiler — GPU Hotspots** | Kernel-level attribution on the GPU (GEMM vs attention vs norm vs memcpy). Use when you need cycle-accurate per-kernel cost and EU/XVE occupancy. |
| [`unitrace/`](./unitrace) | **pti-gpu unitrace** | Low-overhead Level Zero (and SYCL / oneCCL / oneDNN) timeline tracing. Use when you want a Chrome/Perfetto-loadable timeline of host + device activity without the heavy collector. |
| [`pytorch-profiler/`](./pytorch-profiler) | **`torch.profiler` (XPU backend)** | Python-side framework view — module/op-level timing, CPU↔XPU hand-off, autograd. Use when the bottleneck might be Python/scheduling rather than a single kernel. |

All three share the same **ROI-gated collection pattern**: instead of profiling the full vLLM serve run, the wrapper toggles the collector on only during a specific phase (prefill / decode / a particular request range) so the resulting trace stays small and the overhead doesn't perturb the steady state.

For a **side-by-side comparison** of what each profiler sees, when to reach for each, and how to combine them, see [`profiler_comparison.txt`](./profiler_comparison.txt) — GitHub renders it in-browser.

## Layout

Each profiler folder follows the same shape:

```
<profiler>/
├── SKILL.md                # end-to-end guide: setup, ROI design, gotchas
├── docs/                   # deeper references (troubleshooting, internals)
└── scripts/                # ready-to-run wrappers
    ├── run_*_vllm.sh       # launcher: env-driven, sets ROI + collector flags
    ├── serve_with_*.py     # vLLM serve wrapper that emits ROI start/stop signals
    └── *.py                # post-run analysis (report / summary generators)
```

## Quickstart

### 1. VTune GPU-Hotspots — prefill phase of Llama-3.1-8B

```bash
cd vtune/scripts
source /opt/intel/oneapi/setvars.sh
MODEL=meta-llama/Llama-3.1-8B-Instruct \
MAX_MODEL_LEN=4096 VTUNE_PHASE=prefill \
INPUT_LEN=2048 OUTPUT_LEN=1 \
NUM_PROMPTS=20 MAX_CONCURRENCY=1 \
./run_vtune_vllm.sh
```

Result lands in `vtune_results/<timestamp>_<phase>/` — open the `.vtune` project file in the VTune GUI for the GPU-Hotspots view.

### 2. unitrace — Level Zero timeline

```bash
cd unitrace/scripts
UNITRACE_BIN=/opt/pti-gpu/tools/unitrace/build/unitrace \
MODEL=meta-llama/Llama-3.1-8B-Instruct \
UNITRACE_PRESET=default \
./run_unitrace_vllm.sh
```

Drop the resulting Chrome trace JSON into [ui.perfetto.dev](https://ui.perfetto.dev) or `chrome://tracing`. Use `scripts/generate_report.py` for a device-timing summary.

### 3. PyTorch profiler — module-level view

```bash
cd pytorch-profiler/scripts
MODEL=meta-llama/Llama-3.1-8B-Instruct \
PT_PROFILE_PHASE=decode \
./run_pt_profile_vllm.sh
python summarize_trace.py pt_profile_results/<timestamp>/trace.json
```

## Prerequisites

Full list — apt packages, pip modules, library symlinks, kernel-driver
caveats — in [`PREREQUISITES.md`](./PREREQUISITES.md). At a glance:

- Intel oneAPI Base Toolkit ≥ 2025.x (preinstalled in `intel/vllm:*-xpu`)
- `intel/vllm:*-xpu` container ≥ 0.14.1-xpu (validated on 0.17.0-xpu)
- A discrete Intel GPU (BMG / Arc / Max) with a recent Level Zero driver
- **VTune skill only:** `apt install intel-oneapi-vtune intel-metrics-discovery`, plus the `libigdmd.so` symlink and `pip install ittapi`
- **unitrace skill only:** a local build of [intel/pti-gpu](https://github.com/intel/pti-gpu) → `tools/unitrace` (see `unitrace/SKILL.md` §2)

Verify a machine before invoking any wrapper:

```bash
./scripts/check_prereqs.sh
```

Prints a PASS/WARN/FAIL per skill and exits non-zero on any FAIL.

## Known limitations

- **VTune GPU-Hotspots on BMG/Xe2 (`xe` kernel driver).** VTune 2025.x was
  validated on `i915`; on `xe` the collect completes but the per-kernel
  "Hottest GPU Computing Tasks" table is often empty. `check_prereqs.sh` and
  `run_vtune_vllm.sh` both `[WARN]` when they detect `xe`. For BMG kernel
  attribution, prefer the **unitrace** skill until VTune 2026+ ships xe support.

## What was validated

Every wrapper below was executed end-to-end against a fresh checkout of
`main` — no local edits, no manual patching — with the environment listed
here. If your setup differs, expect to hit different edges; the scripts
are best-effort on other hardware/versions.

| Item | Value |
|---|---|
| Validation date | 2026-07-01 |
| Host | `gnrsp-bmg3.iind.intel.com` (Ubuntu 24.04, kernel 6.17) |
| Container image | `intel/vllm:0.17.0-xpu` |
| vLLM | `0.1.dev14456+gde3f7fe65.xpu` (v1 engine) |
| GPU | Intel BMG / Xe2 (device `0xe223`), `xe` kernel driver |
| oneAPI Base Toolkit | 2025.3 (preinstalled in the image) |
| VTune Profiler | 2025.10.0 (apt-installed from Intel oneAPI repo) |
| Metrics Discovery | `intel-metrics-discovery 1.14.180-1111~24.04` |
| unitrace | pti-gpu build at `/data/workspace/vasudha/pti-gpu/tools/unitrace/build/unitrace` |
| Model used | `meta-llama/Llama-3.1-8B-Instruct` (float16, `--enforce-eager`) |

Wrapper results:

| Wrapper | Result |
|---|---|
| `pytorch-profiler/scripts/run_pt_profile_vllm.sh` | ✅ PASS — server up, warmup + bench under `--profiler-config`, traces + summary |
| `unitrace/scripts/run_unitrace_vllm.sh` | ✅ PASS — unitrace `--start-paused`, ITT-driven ROI, EXIT trap → `unitrace_vllm_report.html` |
| `vtune/scripts/run_vtune_vllm.sh` | ⚠️ Preflight PASS, collect hits the documented BMG/xe VTune limitation (see [Known limitations](#known-limitations)); wrapper WARNs before starting and exits with an actionable error |
| `scripts/check_prereqs.sh` | ✅ PASS — all three skills reported ready |

## Status & scope

These are working recipes — not a packaged product. Paths and version pins reflect what we validated on; expect to edit env vars and adjust ROI bounds for your model and workload. See each folder's `SKILL.md` for the full set of knobs and known pitfalls.

## License

[Apache 2.0](./LICENSE)
