#!/usr/bin/env bash
# Launch vLLM OpenAI server under unitrace, with ROI gated by curl.
#
# Usage:
#   ./run_unitrace_vllm.sh                   # uses defaults below
#   PORT=9090 MODEL=... ./run_unitrace_vllm.sh
#
# After launch, drive ROI with ANY HTTP client:
#   curl -X POST http://localhost:$PORT/start_profile
#   <send inference requests: curl, OpenAI SDK, vllm bench serve, ...>
#   curl -X POST http://localhost:$PORT/stop_profile
#
# Result trace lives at $RESULT_DIR/<pid>.json (Chrome-trace) plus
# device-timing summary printed at server exit.
set -euo pipefail

# ---------------- Config ----------------
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${PORT:-9090}"
HOST="${HOST:-0.0.0.0}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4352}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4352}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
TP_SIZE="${TP_SIZE:-1}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/hf_cache}"

# unitrace binary. Override via env: UNITRACE_BIN=/path/to/unitrace
# Common install locations (searched in order if UNITRACE_BIN is unset):
#   /opt/pti-gpu/tools/unitrace/build/unitrace         (upstream instructions)
#   /data/workspace/*/pti-gpu/tools/unitrace/build/unitrace  (shared host build)
if [[ -z "${UNITRACE_BIN:-}" ]]; then
  for _cand in \
    /opt/pti-gpu/tools/unitrace/build/unitrace \
    /data/workspace/*/pti-gpu/tools/unitrace/build/unitrace \
    "$HOME/pti-gpu/tools/unitrace/build/unitrace"; do
    if [[ -x "$_cand" ]]; then UNITRACE_BIN="$_cand"; break; fi
  done
fi
UNITRACE_BIN="${UNITRACE_BIN:-/opt/pti-gpu/tools/unitrace/build/unitrace}"
if [[ ! -x "$UNITRACE_BIN" ]]; then
  echo "ERROR: unitrace binary not found at UNITRACE_BIN=$UNITRACE_BIN" >&2
  echo "       Build it from https://github.com/intel/pti-gpu (tools/unitrace)," >&2
  echo "       then re-run with UNITRACE_BIN=<path> $0" >&2
  exit 1
fi

RESULT_ROOT="${RESULT_ROOT:-$PWD/unitrace_results}"
RESULT_DIR="$RESULT_ROOT/$(date +%Y%m%d_%H%M%S)_$(echo "${MODEL}" | tr '/' '_')"
mkdir -p "$RESULT_DIR"

# ---------------- unitrace flag selection ----------------
# Two ways to control flags (in order of precedence):
#
#   1. UNITRACE_FLAGS="..."           # full override; ignores PRESET
#   2. UNITRACE_PRESET=<preset>       # shorthand (default: "default")
#
# Presets:
#   lite      = kernel + ITT (smallest trace; just GPU kernels and ROI markers)
#   default   = kernel + SYCL + oneDNN + L0/OpenCL host calls + --verbose
#   call      = lite + SYCL + Level-Zero/OpenCL host calls
#   ccl       = default + oneCCL          (multi-rank / TP runs)
#   dnn       = default + oneDNN          (oneDNN primitive timing)
#   mpi       = default + MPI             (multi-node MPI runs)
#   device    = lite + device-only (no per-thread / per-engine subdivision)
#   full      = everything: kernel, device, sycl, ccl, dnn, mpi, call, itt + verbose
#
# Always-on regardless of preset: --start-paused -d --chrome-itt-logging
# Add UNITRACE_EXTRA="..." to append extra flags to whichever preset you pick:
#   UNITRACE_EXTRA="--verbose --chrome-event-buffer-size 1000000"
UNITRACE_PRESET="${UNITRACE_PRESET:-default}"
UNITRACE_EXTRA="${UNITRACE_EXTRA:-}"

_base="--start-paused -d --chrome-itt-logging"
case "$UNITRACE_PRESET" in
  lite)    _preset="--chrome-kernel-logging" ;;
  default) _preset="--chrome-kernel-logging --chrome-sycl-logging --chrome-dnn-logging --chrome-call-logging --verbose" ;;
  call)    _preset="--chrome-kernel-logging --chrome-sycl-logging --chrome-call-logging" ;;
  ccl)     _preset="--chrome-kernel-logging --chrome-sycl-logging --chrome-ccl-logging" ;;
  dnn)     _preset="--chrome-kernel-logging --chrome-sycl-logging --chrome-dnn-logging" ;;
  mpi)     _preset="--chrome-kernel-logging --chrome-sycl-logging --chrome-mpi-logging" ;;
  device)  _preset="--chrome-device-logging" ;;
  full)    _preset="--chrome-kernel-logging --chrome-device-logging --chrome-sycl-logging --chrome-ccl-logging --chrome-dnn-logging --chrome-mpi-logging --chrome-call-logging --verbose" ;;
  *)       echo "[run_unitrace_vllm] unknown UNITRACE_PRESET=$UNITRACE_PRESET" >&2; exit 2 ;;
esac
UNITRACE_FLAGS="${UNITRACE_FLAGS:-$_base $_preset $UNITRACE_EXTRA}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="$SCRIPT_DIR/serve_with_unitrace.py"
if [[ ! -f "$WRAPPER" ]]; then
  echo "ERROR: serve_with_unitrace.py not found next to this script ($WRAPPER)" >&2
  exit 1
fi

# ---------------- vLLM env ----------------
export HF_HOME="${HF_HOME:-/hf_cache}"
export SYCL_UR_USE_LEVEL_ZERO_V2=0
export TORCH_LLM_ALLREDUCE=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Cold weight-load on NFS-mounted /hf_cache can exceed vLLM 0.21's 600s
# default engine-ready timeout (observed: ~10 min for Llama-3.1-8B).
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"

# Sentinel-file path shared by API server and EngineCore subprocess.
export UNITRACE_ROI_GATE="$RESULT_DIR/roi_gate"

# ---------------- oneAPI ----------------
# unitrace links against libxptifw / libittnotify from oneAPI.
# setvars.sh references unset vars (e.g. OCL_ICD_FILENAMES) under `set -u`,
# so temporarily drop pipefail/nounset around the source and re-enable after.
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
  set +euo pipefail
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
  set -euo pipefail
fi

echo "==== unitrace ROI vLLM serve ===="
echo "Model:           $MODEL"
echo "Port:            $PORT"
echo "Result dir:      $RESULT_DIR"
echo "ROI gate file:   $UNITRACE_ROI_GATE"
echo "Preset:          $UNITRACE_PRESET"
echo "unitrace flags:  $UNITRACE_FLAGS"
echo "================================="

# ---------------- Env snapshot ----------------
# Write env_snapshot.json into RESULT_DIR so generate_report.py can render
# the actual runtime environment (container image, vLLM/torch versions,
# oneAPI, kernel, GPU) instead of hard-coded strings. Best-effort — any
# probe that fails becomes null in the JSON.
_probe() { local out; out=$("$@" 2>/dev/null); printf '%s' "$out"; }
_json_str() {
  # JSON-escape a string (quotes, backslashes, newlines, tabs). Prints "null" for empty.
  python3 - "$1" <<'PY' 2>/dev/null || printf 'null'
import json, sys
s = sys.argv[1]
print("null" if s == "" else json.dumps(s))
PY
}

_snap="$RESULT_DIR/env_snapshot.json"
_container_image="${UNITRACE_CONTAINER_IMAGE:-}"
if [[ -z "$_container_image" && -r /etc/os-release ]]; then
  # No reliable in-container way to read our own image tag. Prefer explicit
  # env override; else leave blank so the report shows "unknown".
  :
fi
_vllm_ver="$(_probe python3 -c 'import vllm;print(vllm.__version__)')"
_torch_ver="$(_probe python3 -c 'import torch;print(torch.__version__)')"
_torch_xpu="$(_probe python3 -c 'import torch;print(getattr(torch,"xpu",None) is not None and torch.xpu.is_available())')"
_oneapi_ver="${ONEAPI_ROOT:-}"
if [[ -z "$_oneapi_ver" && -d /opt/intel/oneapi ]]; then
  _oneapi_ver="$(readlink -f /opt/intel/oneapi 2>/dev/null || echo /opt/intel/oneapi)"
fi
_icpx_ver="$(_probe icpx --version | head -1)"
_kernel="$(_probe uname -r)"
_kmd="$(_probe uname -sm)"
_hostname="$(_probe hostname)"
_gpu_info="$(_probe xpu-smi discovery)"
if [[ -z "$_gpu_info" ]]; then
  _gpu_info="$(_probe sycl-ls)"
fi
_unitrace_ver="$(_probe "$UNITRACE_BIN" --version)"
_python_ver="$(_probe python3 --version)"

cat > "$_snap" <<EOF
{
  "container_image": $(_json_str "$_container_image"),
  "hostname": $(_json_str "$_hostname"),
  "kernel": $(_json_str "$_kernel"),
  "arch": $(_json_str "$_kmd"),
  "python": $(_json_str "$_python_ver"),
  "vllm": $(_json_str "$_vllm_ver"),
  "torch": $(_json_str "$_torch_ver"),
  "torch_xpu_available": $(_json_str "$_torch_xpu"),
  "oneapi_root": $(_json_str "$_oneapi_ver"),
  "icpx_version": $(_json_str "$_icpx_ver"),
  "unitrace_bin": $(_json_str "$UNITRACE_BIN"),
  "unitrace_version": $(_json_str "$_unitrace_ver"),
  "unitrace_preset": $(_json_str "$UNITRACE_PRESET"),
  "unitrace_flags": $(_json_str "$UNITRACE_FLAGS"),
  "model": $(_json_str "$MODEL"),
  "port": $(_json_str "$PORT"),
  "dtype": $(_json_str "$DTYPE"),
  "tp_size": $(_json_str "$TP_SIZE"),
  "gpu_info": $(_json_str "$_gpu_info"),
  "vllm_env": {
    "VLLM_USE_V1": $(_json_str "${VLLM_USE_V1:-}"),
    "VLLM_WORKER_MULTIPROC_METHOD": $(_json_str "${VLLM_WORKER_MULTIPROC_METHOD:-}"),
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": $(_json_str "${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-}"),
    "SYCL_UR_USE_LEVEL_ZERO_V2": $(_json_str "${SYCL_UR_USE_LEVEL_ZERO_V2:-}"),
    "TORCH_LLM_ALLREDUCE": $(_json_str "${TORCH_LLM_ALLREDUCE:-}")
  }
}
EOF
echo "[run_unitrace_vllm] env snapshot: $_snap"

# When the launcher exits (Ctrl-C or unitrace returns), build the HTML report.
# Skipped if explicitly disabled via UNITRACE_SKIP_REPORT=1.
report_on_exit() {
  rc=$?
  if [[ "${UNITRACE_SKIP_REPORT:-0}" == "1" ]]; then
    return $rc
  fi
  echo
  echo "[run_unitrace_vllm] generating HTML report from $RESULT_DIR ..."
  if python3 "$SCRIPT_DIR/generate_report.py" "$RESULT_DIR"; then
    echo "[run_unitrace_vllm] report: $RESULT_DIR/unitrace_vllm_report.html"
  else
    echo "[run_unitrace_vllm] report generation failed (non-fatal)" >&2
  fi
  return $rc
}
trap report_on_exit EXIT

cd "$RESULT_DIR"

# Tee unitrace's stdout+stderr to unitrace.log so the `-d` Device Timing Summary
# (printed at inferior exit) and every other unitrace message are preserved on
# disk alongside the trace JSONs. Without this, the summary is lost when the
# terminal closes. PIPESTATUS preserves unitrace's exit code under pipefail.
UNITRACE_LOG="$RESULT_DIR/unitrace.log"
echo "unitrace log:    $UNITRACE_LOG"

set +e
"$UNITRACE_BIN" $UNITRACE_FLAGS \
  python "$WRAPPER" \
       --model "$MODEL" \
       --block-size "$BLOCK_SIZE" \
       --dtype "$DTYPE" \
       --enforce-eager \
       --host "$HOST" \
       --trust-remote-code \
       --no-enable-prefix-caching \
       --disable-sliding-window \
       --tensor-parallel-size "$TP_SIZE" \
       --download_dir "$DOWNLOAD_DIR" \
       --max-model-len "$MAX_MODEL_LEN" \
       --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
       --gpu-memory-utilization "$GPU_MEM_UTIL" \
       --port "$PORT" 2>&1 | tee "$UNITRACE_LOG"
unitrace_rc=${PIPESTATUS[0]}
set -e

# Extract the Device Timing Summary block into its own file for quick top-kernel lookup.
if grep -q "Device Timing Summary" "$UNITRACE_LOG" 2>/dev/null; then
  awk '/=== Device Timing Summary/{flag=1} flag{print} /^===.*===$/ && !/Device Timing Summary/ && flag>1{exit} flag{flag++}' \
    "$UNITRACE_LOG" > "$RESULT_DIR/device_timing_summary.txt" || true
  echo "[run_unitrace_vllm] device timing summary: $RESULT_DIR/device_timing_summary.txt"
else
  echo "[run_unitrace_vllm] WARNING: Device Timing Summary not found in unitrace log." >&2
  echo "[run_unitrace_vllm]   This usually means unitrace didn't observe a clean inferior exit." >&2
  echo "[run_unitrace_vllm]   Next time SIGINT the python process (vLLM), not unitrace itself." >&2
fi

exit $unitrace_rc
