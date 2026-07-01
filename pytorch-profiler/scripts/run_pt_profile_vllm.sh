#!/usr/bin/env bash
set -euo pipefail

# PyTorch-profiler ROI capture for vLLM (OpenAI API server).
#
# Companion to ../run_vtune_vllm.sh. That script captures Intel VTune
# GPU-Hotspots traces; this one captures torch.profiler traces, written as
# .pt.trace.json files for Perfetto / chrome://tracing analysis.
#
# Phase isolation strategy:
#   1. WORKLOAD SHAPING - INPUT_LEN/OUTPUT_LEN choose the dominant phase:
#        PT_PHASE=prefill INPUT_LEN=2048 OUTPUT_LEN=1   ./run_pt_profile_vllm.sh
#        PT_PHASE=decode  INPUT_LEN=128  OUTPUT_LEN=512 ./run_pt_profile_vllm.sh
#
#   2. WORKER PATCH - serve_with_pt_profile.py wraps Worker.execute_model in
#      a torch.profiler.record_function span tagged with the classified phase
#      (prefill / decode / mixed) and shape (bs / tok). These spans appear in
#      Perfetto as named blocks so you can pivot/filter cleanly.
#
#   3. ROI = vLLM's official /start_profile + /stop_profile endpoints. We
#      drive them around a tight benchmark window so model load, warmup and
#      drain are excluded. VLLM_TORCH_PROFILER_DIR MUST be set before the
#      server starts; otherwise the endpoints are no-ops.
#
# After collection, summarize.py walks the trace and prints step cadence,
# forward fraction, top kernels, etc. (validation cribbed from
# vllm-pytorch-profiler/references/perfetto-analysis.md).

# ---- Config ----
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# Eager mode is recommended for profiling: torch.compile fuses ops into
# Inductor-named compiled regions which obscures kernel attribution. It also
# avoids an llvm-foreach SEGV seen on intel/vllm:0.14.1-xpu + oneAPI 2025.3
# during the SYCL JIT step. Set ENFORCE_EAGER=0 to use compiled mode anyway.
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

PT_PHASE="${PT_PHASE:-both}"
PT_LABEL_EVERY_STEP="${PT_LABEL_EVERY_STEP:-1}"

INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
WARMUP_PROMPTS="${WARMUP_PROMPTS:-10}"
DRAIN_SECONDS="${DRAIN_SECONDS:-3}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-5}"

RESULT_ROOT="${RESULT_ROOT:-$(pwd)/pt_results}"
RESULT_DIR="$(realpath -m "$RESULT_ROOT")/$(date +%Y%m%d_%H%M%S)_${PT_PHASE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="$SCRIPT_DIR/serve_with_pt_profile.py"
SUMMARIZE="$SCRIPT_DIR/summarize_trace.py"
if [[ ! -f "$WRAPPER" ]]; then
  echo "ERROR: serve_with_pt_profile.py not found next to this script ($WRAPPER)"
  exit 1
fi

# vLLM reads VLLM_TORCH_PROFILER_DIR at engine init; export NOW so the child
# process sees it. The dir is a single shared output sink for all ranks.
export VLLM_TORCH_PROFILER_DIR="$RESULT_DIR/torch_traces"
export PT_PHASE
export PT_LABEL_EVERY_STEP

# ---- Environment ----
if [[ -z "${SETVARS_COMPLETED:-}" ]]; then
  if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh
  fi
else
  echo "oneAPI already sourced (SETVARS_COMPLETED=$SETVARS_COMPLETED) - skipping."
fi

# torch.profiler doesn't need ZE/PTI tracing layers; in fact, having them
# enabled with VTune-style configs sometimes breaks XPU activity capture.
for VAR in ZE_ENABLE_TRACING_LAYER ZE_LOADER_LAYERS_ENABLE \
           PTI_ENABLE_COLLECTION PTI_ENABLE_RUNTIME_TRACING; do
  unset "$VAR"
done

# ---- Pre-flight checks ----
preflight() {
  local fail=0

  # 1. python + torch importable
  if ! python -c 'import torch' 2>/dev/null; then
    echo "  [FAIL] torch not importable in current python"
    fail=1
  else
    TORCH_VER=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null)
    echo "  [ OK ] torch $TORCH_VER importable"
  fi

  # 2. GPU visible (XPU first; fall back to CUDA - we don't hard-fail either way)
  if python -c 'import torch,sys; sys.exit(0 if torch.xpu.is_available() and torch.xpu.device_count()>0 else 1)' 2>/dev/null; then
    XPU_NAME=$(python -c 'import torch; print(torch.xpu.get_device_name(0))' 2>/dev/null || echo "unknown")
    echo "  [ OK ] XPU detected: $XPU_NAME"
  elif python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    CUDA_NAME=$(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo "unknown")
    echo "  [ OK ] CUDA detected: $CUDA_NAME"
  else
    echo "  [WARN] No XPU or CUDA detected (CPU-only trace will result)"
  fi

  # 3. profiler dir is writable
  mkdir -p "$VLLM_TORCH_PROFILER_DIR" 2>/dev/null || true
  if [[ ! -w "$VLLM_TORCH_PROFILER_DIR" ]]; then
    echo "  [FAIL] $VLLM_TORCH_PROFILER_DIR not writable"
    fail=1
  else
    echo "  [ OK ] profiler dir writable: $VLLM_TORCH_PROFILER_DIR"
  fi

  # 4. Free disk
  local free_gb
  free_gb=$(df -PB1G "$RESULT_ROOT" 2>/dev/null | awk 'NR==2 {print $4+0}')
  if [[ -n "$free_gb" && "$free_gb" -ge "$MIN_FREE_DISK_GB" ]]; then
    echo "  [ OK ] Free disk: ${free_gb} GB (>= ${MIN_FREE_DISK_GB} GB)"
  else
    echo "  [FAIL] Free disk on $RESULT_ROOT: ${free_gb:-?} GB < ${MIN_FREE_DISK_GB} GB"
    fail=1
  fi

  # 5. vLLM v1 import works
  if python -c 'from vllm.v1.worker.gpu_worker import Worker' 2>/dev/null; then
    echo "  [ OK ] vllm.v1.worker.gpu_worker.Worker importable"
  else
    echo "  [WARN] vllm v1 Worker import failed; will fall back to v0 path"
  fi

  # 6. Nothing already on $PORT
  if curl -sf "http://$HOST:$PORT/health" > /dev/null 2>&1; then
    echo "  [FAIL] http://$HOST:$PORT already serving (stale vLLM?)"
    fail=1
  else
    echo "  [ OK ] Port $PORT free"
  fi

  return $fail
}

mkdir -p "$RESULT_ROOT" "$VLLM_TORCH_PROFILER_DIR"
echo "===== Pre-flight checks ====="
if ! preflight; then
  echo "Pre-flight checks FAILED. Refusing to start."
  exit 1
fi
echo

mkdir -p "$RESULT_DIR"
echo "Result dir: $RESULT_DIR"
echo "Phase: $PT_PHASE  every_step=$PT_LABEL_EVERY_STEP"
echo "Workload: input=$INPUT_LEN output=$OUTPUT_LEN prompts=$NUM_PROMPTS conc=$MAX_CONCURRENCY"

# Stamp metadata for future-you opening this result later.
cat > "$RESULT_DIR/metadata.json" <<META
{
  "model": "$MODEL",
  "dtype": "$DTYPE",
  "max_model_len": $MAX_MODEL_LEN,
  "gpu_memory_utilization": $GPU_MEM_UTIL,
  "enforce_eager": $ENFORCE_EAGER,
  "phase": "$PT_PHASE",
  "label_every_step": $PT_LABEL_EVERY_STEP,
  "input_len": $INPUT_LEN,
  "output_len": $OUTPUT_LEN,
  "num_prompts": $NUM_PROMPTS,
  "max_concurrency": $MAX_CONCURRENCY,
  "warmup_prompts": $WARMUP_PROMPTS,
  "drain_seconds": $DRAIN_SECONDS,
  "host": "$HOST",
  "port": $PORT,
  "torch_profiler_dir": "$VLLM_TORCH_PROFILER_DIR",
  "started_at": "$(date -Iseconds)"
}
META

# ---- Step 1: Launch vLLM (via wrapper) ----
EXTRA_ARGS=()
[[ "$ENFORCE_EAGER" == "1" ]] && EXTRA_ARGS+=(--enforce-eager)

python "$WRAPPER" \
      --model "$MODEL" \
      --host "$HOST" \
      --port "$PORT" \
      --dtype "$DTYPE" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      "${EXTRA_ARGS[@]}" \
      > "$RESULT_DIR/server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  echo "Cleaning up (server pid=$SERVER_PID)..."
  # Best-effort stop_profile in case we crashed mid-window.
  curl -sf -X POST "http://$HOST:$PORT/stop_profile" >/dev/null 2>&1 || true
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---- Step 2: Wait for /health ----
echo "Waiting for vLLM server on http://$HOST:$PORT ..."
for _ in $(seq 1 240); do
  if curl -sf "http://$HOST:$PORT/health" > /dev/null; then
    echo "Server ready."
    break
  fi
  sleep 2
done
if ! curl -sf "http://$HOST:$PORT/health" > /dev/null; then
  echo "Server failed to come up. See $RESULT_DIR/server.log"
  exit 1
fi

# ---- Step 3: Warmup (profiling NOT started) ----
echo "Warmup: $WARMUP_PROMPTS prompts (not profiled)..."
vllm bench serve \
  --backend openai \
  --base-url "http://$HOST:$PORT" \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$WARMUP_PROMPTS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  > "$RESULT_DIR/warmup.log" 2>&1
sleep "$DRAIN_SECONDS"

# ---- Step 4: /start_profile -> Benchmark -> /stop_profile ----
echo "Starting torch profiler ..."
if ! curl -sf -X POST "http://$HOST:$PORT/start_profile" > "$RESULT_DIR/start_profile.log" 2>&1; then
  echo "ERROR: /start_profile failed. Did VLLM_TORCH_PROFILER_DIR get set BEFORE server start?"
  echo "       See $RESULT_DIR/start_profile.log and $RESULT_DIR/server.log"
  exit 1
fi

BENCH_T0=$(date +%s)
echo "Benchmarking $NUM_PROMPTS prompts (profiled)..."
vllm bench serve \
  --backend openai \
  --base-url "http://$HOST:$PORT" \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --save-result \
  --result-dir "$RESULT_DIR" \
  | tee "$RESULT_DIR/bench.log"
BENCH_T1=$(date +%s)
BENCH_ELAPSED=$((BENCH_T1 - BENCH_T0))

echo "Stopping torch profiler ..."
# stop_profile flushes the trace to disk; this can take many seconds for a
# large ROI. Don't time it out aggressively.
curl --max-time 300 -sf -X POST "http://$HOST:$PORT/stop_profile" \
     > "$RESULT_DIR/stop_profile.log" 2>&1 \
  || echo "WARN: /stop_profile returned non-zero; check stop_profile.log"

# Give the server a beat to finish flushing before we kill it.
sleep 2

# ---- Step 5: Shut server down cleanly ----
trap - EXIT INT TERM
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

# ---- Step 6: Inventory + verification ----
echo
echo "===== Trace inventory ====="
shopt -s nullglob
TRACES=("$VLLM_TORCH_PROFILER_DIR"/*.pt.trace.json* "$VLLM_TORCH_PROFILER_DIR"/*.json.gz)
shopt -u nullglob
if [[ ${#TRACES[@]} -eq 0 ]]; then
  echo "  [FAIL] no trace files found in $VLLM_TORCH_PROFILER_DIR"
  echo "         Common causes:"
  echo "           - VLLM_TORCH_PROFILER_DIR set AFTER server start"
  echo "           - /start_profile returned 404 (endpoint missing on this build)"
  echo "           - Server crashed before /stop_profile flushed"
  echo "         See $RESULT_DIR/server.log and $RESULT_DIR/start_profile.log"
  exit 1
fi

for t in "${TRACES[@]}"; do
  size_mb=$(du -m "$t" | awk '{print $1}')
  echo "  $t  (${size_mb} MB)"
done

echo
echo "===== Quick summary ====="
if [[ -f "$SUMMARIZE" ]]; then
  python "$SUMMARIZE" "$VLLM_TORCH_PROFILER_DIR" \
       --bench-elapsed "$BENCH_ELAPSED" \
    | tee "$RESULT_DIR/summary.txt"
else
  echo "  summarize_trace.py not found - skipping quantitative summary"
fi

echo
echo "Done. Result: $RESULT_DIR"
echo "Open in Perfetto: https://ui.perfetto.dev  -> Open trace file -> ${TRACES[0]}"
