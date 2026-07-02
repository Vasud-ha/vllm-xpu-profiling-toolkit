# Prerequisites

Everything you need on the host / in the container before any of the three
`run_*_vllm.sh` wrappers can run cleanly.

Validated on **`intel/vllm:0.17.0-xpu`** container running on an Intel BMG
(Xe2) GPU host with the `xe` kernel driver (Ubuntu 24.04, kernel 6.17).

For a one-shot machine check, run:

```bash
bash <(curl -sfL https://raw.githubusercontent.com/Vasud-ha/vllm-xpu-profiling-toolkit/main/scripts/check_prereqs.sh)
```

or clone the repo and run `./scripts/check_prereqs.sh` locally.

---

## 1. Container / OS

| Requirement | Why | Check |
|---|---|---|
| Intel discrete GPU (Arc / BMG / PVC / Max) | XPU backend | `xpu-smi discovery` shows at least one card |
| `intel/vllm:*-xpu` image (>= 0.14.1-xpu, validated on 0.17.0-xpu) | vLLM v1 worker path with XPU support | `python -c "from vllm.v1.worker.gpu_worker import Worker"` |
| Container run with `--device /dev/dri` (or full DRM access) | GPU submission | `ls /dev/dri/renderD*` inside container |
| oneAPI Base Toolkit >= 2025.x (comes with the image) | compiler / runtimes | `source /opt/intel/oneapi/setvars.sh --force` |
| Root or user in `render`/`video` group | `/dev/dri/renderD*` access | `id -nG` shows the group |
| >= 5 GB free disk under `RESULT_ROOT` | trace + report space | `df -h` |
| HF cache warm with the model you're profiling | avoids first-run download | `ls /hf_cache/hub/models--<org>--<name>/snapshots/` |

---

## 2. VTune skill (vtune/)

Beyond the base container:

```bash
# Install VTune Profiler (Base Toolkit alone does NOT include it).
# ~547 MB download; adds vtune to PATH after re-sourcing setvars.sh.
apt install -y intel-oneapi-vtune

# Intel Metrics Discovery library (libigdmd.so + libmd.so).
# VTune's gpu-hotspots / gpu-offload dlopen this at collect time.
apt install -y intel-metrics-discovery

# The apt package ships /usr/lib/x86_64-linux-gnu/libigdmd.so.1 but not
# the unversioned symlink VTune looks for. Create it manually.
ln -sf libigdmd.so.1 /usr/lib/x86_64-linux-gnu/libigdmd.so
ldconfig

# Preferred ITT backend for the wrapper (ctypes fallback works too, but
# ittapi handles more edge cases on newer VTune installs).
pip install ittapi
```

**BMG requires VTune 2026.0.** VTune 2025.x fails on BMG with
`Cannot collect GPU hardware metrics because neither libigdmd.so nor libmd.so
was found` — even with `intel-metrics-discovery` installed and the
unversioned `libigdmd.so` symlink in place. Confirmed working on VTune
2026.0 against the same BMG hardware (prior runs). VTune 2026.0 is not in
the Intel apt repo as of writing — install it via the standalone Intel
installer from
<https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler-download.html>
or use **unitrace** for BMG kernel attribution.

---

## 3. unitrace skill (unitrace/)

Beyond the base container:

```bash
# Build pti-gpu unitrace from source. See unitrace/SKILL.md §2 for the
# build steps and the BMG/Xe2 (uintptr_t) generator patch.
git clone --depth 1 https://github.com/intel/pti-gpu.git
cd pti-gpu/tools/unitrace
mkdir build && cd build
source /opt/intel/oneapi/setvars.sh --force
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_MPI=0 -DBUILD_WITH_ITT=1 \
      -DBUILD_WITH_XPTI=1 -DBUILD_WITH_OPENCL=1 ..
make -j$(nproc)

# Point the wrapper at your build.
export UNITRACE_BIN=/path/to/pti-gpu/tools/unitrace/build/unitrace
```

`run_unitrace_vllm.sh` auto-detects `UNITRACE_BIN` under
`/opt/pti-gpu/...`, `/data/workspace/*/pti-gpu/...`, or
`$HOME/pti-gpu/...`. If none match, set `UNITRACE_BIN=<path>` explicitly.

---

## 4. PyTorch profiler skill (pytorch-profiler/)

No extra installs — vLLM's built-in torch.profiler is the collector.

**Version-sensitive gate:**
- vLLM >= 0.17 uses `--profiler-config` (JSON blob). The wrapper builds
  this automatically.
- vLLM <= 0.14 uses `VLLM_TORCH_PROFILER_DIR` env var. The wrapper
  exports both for backward compat.

If neither is set at server start, `/start_profile` returns 404. See
`pytorch-profiler/docs/troubleshooting.md` §1.

---

## 5. Environment variables the wrappers respect

Common:

| Var | Default | Use |
|---|---|---|
| `MODEL` | `Qwen/Qwen2.5-7B-Instruct` (vtune/pt) / `meta-llama/Llama-3.1-8B-Instruct` (unitrace) | HF model id |
| `PORT` | `8000` (vtune/pt), `9090` (unitrace) | serve port |
| `HOST` | `127.0.0.1` (vtune/pt), `0.0.0.0` (unitrace) | bind addr |
| `DTYPE` | `bfloat16` (vtune/pt), `float16` (unitrace) | model dtype |
| `MAX_MODEL_LEN` | 8192 / 4352 / 8192 | context length |
| `GPU_MEM_UTIL` | 0.90 | vLLM `--gpu-memory-utilization` |
| `ENFORCE_EAGER` | 1 | pass `--enforce-eager` (recommended on xpu) |
| `RESULT_ROOT` | `$PWD/{vtune,pt,unitrace}_results` | result dir parent |
| `HF_HOME` | `/hf_cache` (unitrace default) or user's default | HF cache location |

VTune-specific:

| Var | Default | Use |
|---|---|---|
| `VTUNE_PHASE` | `both` | `prefill \| decode \| mixed \| both` |
| `VTUNE_ROI_MODE` | `window` | `window \| per_step \| per_step_isolate` |
| `TARGET_GPU` | (empty; auto) | GPU BDF (`0000:97:00.0`) on multi-GPU hosts |
| `VTUNE_KEEP_RAW` | 0 | 1 to preserve raw collector data (result dir stays ~2-3x larger) |

PyTorch-profiler-specific:

| Var | Default | Use |
|---|---|---|
| `PT_PHASE` | `both` | annotation phase filter |
| `PT_LABEL_EVERY_STEP` | 1 | 0 to label only phase transitions |

unitrace-specific:

| Var | Default | Use |
|---|---|---|
| `UNITRACE_BIN` | auto-detected | path to unitrace binary |
| `UNITRACE_PRESET` | `default` | `lite \| default \| call \| ccl \| dnn \| mpi \| device \| full` |
| `UNITRACE_EXTRA` | (empty) | extra flags appended verbatim |
| `UNITRACE_FLAGS` | (built from preset) | full override, ignores preset |
| `UNITRACE_SKIP_REPORT` | 0 | 1 to skip `generate_report.py` on exit |

---

## 6. Running from scratch (fresh container walkthrough)

```bash
# 0. Start the container (if not already running)
docker run --rm -it --device /dev/dri --shm-size=8g \
  -v /path/to/hf_cache:/hf_cache \
  intel/vllm:0.17.0-xpu bash

# 1. Install profiler prereqs (inside the container)
apt update
apt install -y intel-oneapi-vtune intel-metrics-discovery
ln -sf libigdmd.so.1 /usr/lib/x86_64-linux-gnu/libigdmd.so
ldconfig
pip install ittapi
source /opt/intel/oneapi/setvars.sh --force

# 2. Clone the toolkit
git clone https://github.com/Vasud-ha/vllm-xpu-profiling-toolkit.git
cd vllm-xpu-profiling-toolkit

# 3. Sanity-check the machine
./scripts/check_prereqs.sh

# 4. Run whichever profiler you want, e.g.:
cd pytorch-profiler/scripts
MODEL=meta-llama/Llama-3.1-8B-Instruct \
  NUM_PROMPTS=4 MAX_CONCURRENCY=2 \
  ./run_pt_profile_vllm.sh
```
