#!/usr/bin/env bash
# check_prereqs.sh - Verify a machine has everything the three profiler
# wrappers need. Prints a colored PASS/WARN/FAIL summary per skill.
#
# Usage:  ./scripts/check_prereqs.sh
#         curl -sfL <raw>/scripts/check_prereqs.sh | bash
#
# Exit code: 0 if all skills are runnable (WARN OK), non-zero if any FAIL.

# Deliberately NOT using `set -u` -- sourcing /opt/intel/oneapi/setvars.sh
# below trips on unset internal vars in some oneAPI releases.
set +e   # keep going after each check

# ---- Terminal colors (optional) ----
if [[ -t 1 ]]; then
  R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'; C=$'\033[36m'; B=$'\033[1m'; N=$'\033[0m'
else
  R=""; G=""; Y=""; C=""; B=""; N=""
fi

pass() { echo "  ${G}[PASS]${N} $1"; }
warn() { echo "  ${Y}[WARN]${N} $1"; }
fail() { echo "  ${R}[FAIL]${N} $1"; FAILURES=$((FAILURES+1)); }
info() { echo "  ${C}[INFO]${N} $1"; }
section() { echo; echo "${B}== $1 ==${N}"; }

FAILURES=0

# ---- Section 1: common prerequisites ----
section "Common (all three skills)"

# Python
if command -v python >/dev/null 2>&1; then
  PYV=$(python --version 2>&1)
  pass "python: $PYV"
else
  fail "python not on PATH"
fi

# vLLM v1 Worker
if python -c 'from vllm.v1.worker.gpu_worker import Worker' 2>/dev/null; then
  VLLM_VER=$(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null)
  pass "vllm.v1.worker.gpu_worker importable (vllm=$VLLM_VER)"
else
  fail "vllm v1 Worker not importable — install intel/vllm >= 0.14.1-xpu or upstream >= 0.10"
fi

# torch XPU
if python -c 'import torch; import sys; sys.exit(0 if torch.xpu.is_available() else 1)' 2>/dev/null; then
  XPU_CNT=$(python -c 'import torch; print(torch.xpu.device_count())' 2>/dev/null)
  XPU_NAME=$(python -c 'import torch; print(torch.xpu.get_device_name(0))' 2>/dev/null)
  pass "torch.xpu: $XPU_CNT device(s), $XPU_NAME"
else
  fail "torch.xpu not available — check IPEX / intel-extension-for-pytorch"
fi

# /dev/dri
if [[ "$(id -u)" -eq 0 ]]; then
  pass "running as root (full /dev/dri access)"
elif id -nG | tr ' ' '\n' | grep -qE '^(render|video)$'; then
  pass "user in render/video group"
else
  warn "not in render/video group — /dev/dri/renderD* may be inaccessible"
fi

# oneAPI setvars
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
  pass "oneAPI setvars.sh present"
else
  warn "oneAPI setvars.sh not at /opt/intel/oneapi/setvars.sh"
fi

# HF cache
HF_HOME_DEFAULT="${HF_HOME:-$HOME/.cache/huggingface}"
if [[ -d "$HF_HOME_DEFAULT/hub" ]]; then
  pass "HF cache at $HF_HOME_DEFAULT"
else
  info "HF cache empty at $HF_HOME_DEFAULT (models download on first run)"
fi

# ---- Section 2: VTune skill ----
section "VTune skill (vtune/scripts/run_vtune_vllm.sh)"

if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
fi

VTUNE_OK=1

if command -v vtune >/dev/null 2>&1; then
  VTV=$(vtune --version 2>&1 | head -1 | sed 's/^ *//')
  pass "vtune on PATH: $VTV"
else
  fail "vtune NOT on PATH — apt install -y intel-oneapi-vtune (then re-source setvars.sh)"
  VTUNE_OK=0
fi

if ldconfig -p 2>/dev/null | grep -qE 'libigdmd\.so'; then
  pass "libigdmd.so resolvable"
else
  fail "libigdmd.so NOT resolvable — apt install -y intel-metrics-discovery && ln -sf libigdmd.so.1 /usr/lib/x86_64-linux-gnu/libigdmd.so"
  VTUNE_OK=0
fi

if python -c 'import ittapi' 2>/dev/null; then
  pass "ittapi python module installed"
else
  warn "ittapi not installed — pip install ittapi (ctypes fallback used otherwise)"
fi

# xe-driver warning (BMG/Xe2/Lunar Lake)
if readlink /sys/class/drm/renderD128/device/driver 2>/dev/null | grep -q '/xe$'; then
  warn "GPU uses 'xe' kernel driver — VTune 2025.x gpu-hotspots may produce empty per-kernel table on BMG/Xe2 (prefer unitrace for this HW)"
elif readlink /sys/class/drm/renderD128/device/driver 2>/dev/null | grep -q '/i915$'; then
  pass "GPU uses 'i915' kernel driver (VTune-supported)"
fi

# ---- Section 3: unitrace skill ----
section "unitrace skill (unitrace/scripts/run_unitrace_vllm.sh)"

UT=""
for CAND in "${UNITRACE_BIN:-}" \
            /opt/pti-gpu/tools/unitrace/build/unitrace \
            /data/workspace/*/pti-gpu/tools/unitrace/build/unitrace \
            "$HOME/pti-gpu/tools/unitrace/build/unitrace"; do
  if [[ -n "$CAND" && -x "$CAND" ]]; then
    UT="$CAND"
    break
  fi
done

if [[ -n "$UT" ]]; then
  UT_INFO=$("$UT" --help 2>&1 | head -1 | tr -d '\r')
  pass "unitrace: $UT"
  info "  build flags: $UT_INFO"
else
  fail "unitrace not found — build from https://github.com/intel/pti-gpu (see unitrace/SKILL.md §2) and set UNITRACE_BIN=<path>"
fi

# ---- Section 4: PyTorch-profiler skill ----
section "PyTorch-profiler skill (pytorch-profiler/scripts/run_pt_profile_vllm.sh)"

if python -c 'from vllm.config import ProfilerConfig' 2>/dev/null; then
  pass "ProfilerConfig importable (vLLM >= 0.17 CLI-arg gate available)"
elif python -c 'import vllm; assert int(vllm.__version__.split(".")[1]) <= 14 or "dev" in vllm.__version__' 2>/dev/null; then
  info "vLLM <= 0.14 detected — will use VLLM_TORCH_PROFILER_DIR env var gate"
else
  warn "ProfilerConfig not importable and vLLM version unclear"
fi

# ---- Summary ----
section "Summary"

if [[ $FAILURES -eq 0 ]]; then
  echo "  ${G}${B}All checks passed.${N} The three run_*_vllm.sh wrappers should work end-to-end."
  exit 0
else
  echo "  ${R}${B}$FAILURES failure(s).${N} Fix the [FAIL] items above before running the wrappers."
  echo "  See PREREQUISITES.md for install commands."
  exit 1
fi
