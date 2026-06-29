# vllm-xpu-profiling-toolkit

Profiling recipes, wrapper scripts, and analysis notes for **vLLM serving on Intel Xe (XPU / BMG-class) GPUs**, covering three complementary profilers:

| Folder | Profiler | When to use |
| --- | --- | --- |
| [`vtune/`](./vtune) | **Intel VTune Profiler — GPU Hotspots** | Kernel-level attribution on the GPU (GEMM vs attention vs norm vs memcpy). Use when you need cycle-accurate per-kernel cost and EU/XVE occupancy. |
| [`unitrace/`](./unitrace) | **pti-gpu unitrace** | Low-overhead Level Zero (and SYCL / oneCCL / oneDNN) timeline tracing. Use when you want a Chrome/Perfetto-loadable timeline of host + device activity without the heavy collector. |
| [`pytorch-profiler/`](./pytorch-profiler) | **`torch.profiler` (XPU backend)** | Python-side framework view — module/op-level timing, CPU↔XPU hand-off, autograd. Use when the bottleneck might be Python/scheduling rather than a single kernel. |

All three share the same **ROI-gated collection pattern**: instead of profiling the full vLLM serve run, the wrapper toggles the collector on only during a specific phase (prefill / decode / a particular request range) so the resulting trace stays small and the overhead doesn't perturb the steady state.

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

- Intel oneAPI Base Toolkit ≥ 2025.x (for VTune, ITT, oneAPI runtimes)
- `intel/vllm:*-xpu` container or an equivalent local install (vLLM + IPEX + torch.xpu)
- A discrete Intel GPU (BMG / Arc / Max) with a recent Level Zero driver
- For unitrace: a local build of [intel/pti-gpu](https://github.com/intel/pti-gpu) → `tools/unitrace`

## Status & scope

These are working recipes — not a packaged product. Paths and version pins reflect what we validated on; expect to edit env vars and adjust ROI bounds for your model and workload. See each folder's `SKILL.md` for the full set of knobs and known pitfalls.

## License

[Apache 2.0](./LICENSE)
