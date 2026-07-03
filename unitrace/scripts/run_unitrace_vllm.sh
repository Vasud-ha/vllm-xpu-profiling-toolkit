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
if [[ -z "${SETVARS_COMPLETED:-}" && -f /opt/intel/oneapi/setvars.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/setvars.sh --force >/dev/null
fi

echo "==== unitrace ROI vLLM serve ===="
echo "Model:           $MODEL"
echo "Port:            $PORT"
echo "Result dir:      $RESULT_DIR"
echo "ROI gate file:   $UNITRACE_ROI_GATE"
echo "Preset:          $UNITRACE_PRESET"
echo "unitrace flags:  $UNITRACE_FLAGS"
echo "================================="

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
