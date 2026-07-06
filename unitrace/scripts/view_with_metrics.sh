#!/usr/bin/env bash
# view_with_metrics.sh — one-command "open this unitrace run in Perfetto with
# per-kernel metric graphs".
#
# What it does:
#   1. Locates the newest EngineCore trace JSON in RESULT_DIR (or takes --trace).
#   2. Extracts the "=== Device #0 Metrics ===" CSV block out of unitrace.log
#      into metrics_<group>.csv (if not already there).
#   3. Picks the right BMG/PVC config from pti-gpu (unless --config given).
#   4. Starts uniview.py in the background. It binds:
#        127.0.0.1:9001 — one-shot trace-hosting server (Perfetto fetches from here)
#        127.0.0.1:8000 — long-lived per-kernel metric-graph server
#   5. Prints the Perfetto URL + an SSH-tunnel recipe so a browser on your
#      laptop can reach both ports.
#
# Usage:
#   ./view_with_metrics.sh <result_dir>
#   ./view_with_metrics.sh <result_dir> --group ComputeBasic --platform bmg
#   ./view_with_metrics.sh <result_dir> --stop           # kill background uniview
#
# Environment overrides:
#   UNITRACE_REPO   — path to pti-gpu/tools/unitrace checkout (holds uniview.py + configs)
#   UNIVIEW_LOG     — where uniview stdout/stderr goes (default: $RESULT_DIR/uniview.log)
#   UNIVIEW_PID     — where the pid gets written (default: $RESULT_DIR/uniview.pid)
#   BROWSER         — set to /bin/true to suppress webbrowser.open_new_tab in headless runs

set -uo pipefail

_die() { echo "view_with_metrics: $*" >&2; exit 1; }

RESULT_DIR=""
GROUP="ComputeBasic"
PLATFORM=""     # auto-detect if empty
TRACE=""
CONFIG=""
DO_STOP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group)    GROUP="$2"; shift 2 ;;
    --platform) PLATFORM="$2"; shift 2 ;;
    --trace)    TRACE="$2"; shift 2 ;;
    --config)   CONFIG="$2"; shift 2 ;;
    --stop)     DO_STOP=1; shift ;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0 ;;
    -*)
      _die "unknown flag: $1" ;;
    *)
      [[ -z "$RESULT_DIR" ]] && RESULT_DIR="$1" || _die "extra arg: $1"
      shift ;;
  esac
done

[[ -n "$RESULT_DIR" ]] || _die "missing <result_dir>. Try -h for help."
[[ -d "$RESULT_DIR" ]] || _die "not a directory: $RESULT_DIR"
RESULT_DIR="$(cd "$RESULT_DIR" && pwd)"

UNIVIEW_PID_FILE="${UNIVIEW_PID:-$RESULT_DIR/uniview.pid}"
UNIVIEW_LOG_FILE="${UNIVIEW_LOG:-$RESULT_DIR/uniview.log}"

# ---------------- --stop ----------------
if (( DO_STOP )); then
  if [[ -r "$UNIVIEW_PID_FILE" ]]; then
    pid="$(cat "$UNIVIEW_PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && echo "view_with_metrics: killed uniview pid=$pid"
    else
      echo "view_with_metrics: pid=$pid not running"
    fi
    rm -f "$UNIVIEW_PID_FILE"
  else
    echo "view_with_metrics: no pidfile at $UNIVIEW_PID_FILE"
  fi
  exit 0
fi

# ---------------- Locate uniview repo ----------------
if [[ -z "${UNITRACE_REPO:-}" ]]; then
  for cand in \
    /data/workspace/*/pti-gpu/tools/unitrace \
    /opt/pti-gpu/tools/unitrace \
    "$HOME/pti-gpu/tools/unitrace"; do
    [[ -f "$cand/scripts/uniview.py" ]] && { UNITRACE_REPO="$cand"; break; }
  done
fi
UNIVIEW="${UNITRACE_REPO:-}/scripts/uniview.py"
[[ -f "$UNIVIEW" ]] || _die "uniview.py not found. Set UNITRACE_REPO=<pti-gpu/tools/unitrace>."

# ---------------- Locate trace ----------------
if [[ -z "$TRACE" ]]; then
  # Prefer the EngineCore JSON (the one with gpu_op events → biggest python.*.json).
  TRACE="$(ls -1S "$RESULT_DIR"/python.*.json 2>/dev/null | head -1)"
fi
[[ -n "$TRACE" && -r "$TRACE" ]] || _die "no python.*.json trace in $RESULT_DIR. Pass --trace <file>."

# ---------------- Extract metric CSV ----------------
CSV="$RESULT_DIR/metrics_$(echo "$GROUP" | tr '[:upper:]' '[:lower:]').csv"
if [[ ! -s "$CSV" ]]; then
  ULOG="$RESULT_DIR/unitrace.log"
  [[ -r "$ULOG" ]] || _die "unitrace.log missing; can't extract $GROUP CSV."
  # unitrace prints "=== Device #<N> Metrics ===" then the CSV (with a header
  # containing GlobalInstanceId or IP[Address]), then blank line, then
  # "=== Device Timing Summary ===". Everything between is the CSV.
  awk '/^=== Device #.* Metrics ===$/{flag=1; next}
       /^=== Device Timing Summary ===$/{flag=0}
       flag && NF' "$ULOG" > "$CSV" || true
  if [[ ! -s "$CSV" ]]; then
    rm -f "$CSV"
    _die "no metric block found in $ULOG (was this captured with --metric-query?)"
  fi
  echo "view_with_metrics: extracted $(wc -l < "$CSV") rows to $CSV"
fi

# ---------------- Locate config ----------------
if [[ -z "$CONFIG" ]]; then
  if [[ -z "$PLATFORM" ]]; then
    # Try to infer from env_snapshot.json's gpu_info; fall back to bmg.
    if [[ -r "$RESULT_DIR/env_snapshot.json" ]]; then
      _gi=$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get('gpu_info','') or '')" "$RESULT_DIR/env_snapshot.json" 2>/dev/null)
      case "$_gi" in
        *"[0xe223]"*|*BMG*|*"Battlemage"*) PLATFORM=bmg ;;
        *"Data Center GPU Max"*|*PVC*)      PLATFORM=pvc ;;
        *) PLATFORM=bmg ;;
      esac
    else
      PLATFORM=bmg
    fi
  fi
  CONFIG="$UNITRACE_REPO/scripts/metrics/config/$PLATFORM/$GROUP.txt"
fi
[[ -r "$CONFIG" ]] || _die "config not found: $CONFIG (try --platform pvc or --config <path>)"

# ---------------- Check ports ----------------
for port in 8000 9001; do
  # /proc/net/tcp lists LISTEN sockets (state 0A). Convert :port hex to decimal to compare.
  hex=$(printf ':%04X$' "$port")
  if awk 'NR>1 && $4=="0A"{print $2}' /proc/net/tcp | grep -qE "$hex"; then
    _die "port $port is already bound. Run '$0 <dir> --stop' first, or free the port."
  fi
done

# ---------------- Start uniview ----------------
export BROWSER="${BROWSER:-/bin/true}"
: > "$UNIVIEW_LOG_FILE"
nohup python3 "$UNIVIEW" -t "$TRACE" -m "$CSV" -f "$CONFIG" \
  > "$UNIVIEW_LOG_FILE" 2>&1 &
echo $! > "$UNIVIEW_PID_FILE"
pid=$(cat "$UNIVIEW_PID_FILE")

# Give uniview a moment to parse the CSV and bind :9001. Metric-server :8000
# only binds AFTER Perfetto fetches the trace, so we can't wait for it here.
for _ in $(seq 1 30); do
  if awk 'NR>1 && $4=="0A"{print $2}' /proc/net/tcp | grep -qE ':2329$'; then
    break
  fi
  sleep 1
done

if ! awk 'NR>1 && $4=="0A"{print $2}' /proc/net/tcp | grep -qE ':2329$'; then
  echo "view_with_metrics: :9001 didn't come up in 30s — check $UNIVIEW_LOG_FILE" >&2
  tail -20 "$UNIVIEW_LOG_FILE" >&2 || true
  exit 3
fi

trace_name="$(basename "$TRACE")"
cat <<EOF

=== uniview started ===
  pid       : $pid  (pidfile: $UNIVIEW_PID_FILE)
  log       : $UNIVIEW_LOG_FILE
  trace     : $TRACE
  metrics   : $CSV
  config    : $CONFIG  (platform=$PLATFORM  group=$GROUP)

=== Open in a browser on the machine that has your desktop ===
If the browser is on a laptop, tunnel first:

    ssh -L 8000:127.0.0.1:8000 -L 9001:127.0.0.1:9001 <this-host>

Then open:

    https://ui.perfetto.dev/#!/?url=http://127.0.0.1:9001/$trace_name

Perfetto fetches the trace over :9001 (one-shot), then :8000 stays up to
serve per-kernel metric graphs when you click args.metrics on any kernel slice.

Stop uniview when done:

    $0 "$RESULT_DIR" --stop
EOF
