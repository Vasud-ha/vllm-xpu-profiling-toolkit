#!/usr/bin/env python3
"""
summarize_trace.py - Quantitative summary of a vLLM PyTorch-profiler trace.

Implements the "starting points" checklist from
~/.claude/skills/vllm-pytorch-profiler/references/perfetto-analysis.md §5:

  - Step cadence (median time between Worker.execute_model starts)
  - Forward fraction (forward kernel time / step duration)
  - Attention fraction (attention kernels / forward)
  - NCCL fraction (collectives / step duration, TP only)
  - Top kernels by total time
  - Per-phase breakdown (using vllm.<phase> spans emitted by
    serve_with_pt_profile.py)

Usage:
  python summarize_trace.py <trace.pt.trace.json>          # one file
  python summarize_trace.py <dir>                          # auto-pick newest
  python summarize_trace.py <dir> --bench-elapsed 17       # add wall-clock check

Notes:
  - We parse the raw chrome-trace JSON instead of pulling in torch.profiler
    just to load it; this keeps the tool runnable outside the vLLM env and
    avoids tying it to a specific torch version.
  - For huge traces (multi-GB), use Perfetto's trace_processor instead.
    This script tries to be linear-time on the events, but loading a 5GB
    JSON in Python is its own problem.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path


# ---------- File loading ----------

def _open_trace(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def load_trace(path: Path) -> dict:
    with _open_trace(path) as f:
        return json.load(f)


def pick_trace(arg: str) -> Path:
    p = Path(arg).expanduser()
    if p.is_file():
        return p
    if p.is_dir():
        candidates = sorted(
            list(p.glob("*.pt.trace.json"))
            + list(p.glob("*.pt.trace.json.gz"))
            + list(p.glob("*.json.gz")),
            key=lambda x: x.stat().st_mtime,
        )
        if not candidates:
            sys.exit(f"No trace files in {p}")
        # Newest first; warn if multiple ranks present.
        return candidates[-1]
    sys.exit(f"Not a file or directory: {arg}")


# ---------- Event filters ----------

# Heuristics; intentionally simple. If a name doesn't match here it lands in
# "other" and you can spot-check via the top-kernels table.
ATTN_HINTS = (
    "flash_attn", "flashattn", "paged_attn", "pagedattention",
    "attention", "scaled_dot_product", "sdpa",
    "xetla_attn", "ipex_attn",
)
GEMM_HINTS = (
    "gemm", "matmul", "addmm", "linear",
    "cublas", "cublaslt", "xetla_gemm",
)
NCCL_HINTS = (
    "ncclallreduce", "ncclbroadcast", "nccl",
    "allreduce", "ccl_", "xpu_ccl",
)


def is_kernel(ev: dict) -> bool:
    cat = (ev.get("cat") or "").lower()
    # Chrome-trace kernel events have ph="X" and category like "kernel" or
    # "gpu_op" depending on backend. Phase 'X' is "complete" duration event.
    if ev.get("ph") != "X":
        return False
    if "kernel" in cat or "gpu" in cat:
        return True
    # XPU traces sometimes use cat="xpu_runtime"; treat anything with a
    # device-side timestamp as a kernel candidate. Best-effort.
    return False


def name_matches(name: str, hints) -> bool:
    n = name.lower()
    return any(h in n for h in hints)


# ---------- Analyses ----------

def step_cadence(events) -> dict:
    """Times of Worker.execute_model starts (or our vllm.<phase> spans)."""
    step_starts = []
    phase_durations = defaultdict(list)

    for ev in events:
        if ev.get("ph") != "X":
            continue
        name = ev.get("name") or ""
        # vLLM's record_function spans we emit live as cpu_op; match on prefix.
        if name.startswith("vllm."):
            ts = ev.get("ts")
            dur = ev.get("dur")
            if ts is not None and dur is not None:
                step_starts.append(ts)
                # vllm.<phase> bs=N tok=M -> phase = first token after 'vllm.'
                phase = name.split()[0].split(".", 1)[-1]
                phase_durations[phase].append(dur)
        elif name.endswith("execute_model") and ev.get("cat", "").startswith("user_annotation"):
            ts = ev.get("ts")
            if ts is not None:
                step_starts.append(ts)

    step_starts.sort()
    cadences = [b - a for a, b in zip(step_starts, step_starts[1:])]
    return {
        "n_steps": len(step_starts),
        "cadence_us_p50": statistics.median(cadences) if cadences else None,
        "cadence_us_p90": _pct(cadences, 0.90),
        "cadence_us_p99": _pct(cadences, 0.99),
        "phase_count": {k: len(v) for k, v in phase_durations.items()},
        "phase_us_p50": {
            k: statistics.median(v) if v else None
            for k, v in phase_durations.items()
        },
    }


def _pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, int(p * len(s)))
    return s[i]


def kernel_breakdown(events) -> dict:
    total = 0
    by_name = defaultdict(int)
    attn = gemm = nccl = 0
    for ev in events:
        if not is_kernel(ev):
            continue
        name = ev.get("name") or ""
        dur = ev.get("dur") or 0
        total += dur
        by_name[name] += dur
        if name_matches(name, ATTN_HINTS):
            attn += dur
        elif name_matches(name, NCCL_HINTS):
            nccl += dur
        elif name_matches(name, GEMM_HINTS):
            gemm += dur

    top = sorted(by_name.items(), key=lambda kv: kv[1], reverse=True)[:15]
    return {
        "total_us": total,
        "attn_us": attn,
        "gemm_us": gemm,
        "nccl_us": nccl,
        "other_us": total - attn - gemm - nccl,
        "top": top,
    }


def trace_window_us(events) -> int | None:
    """Total wall-clock span of the trace in microseconds."""
    lo = hi = None
    for ev in events:
        if ev.get("ph") != "X":
            continue
        ts = ev.get("ts")
        dur = ev.get("dur") or 0
        if ts is None:
            continue
        if lo is None or ts < lo:
            lo = ts
        end = ts + dur
        if hi is None or end > hi:
            hi = end
    if lo is None or hi is None:
        return None
    return hi - lo


# ---------- Reporting ----------

def fmt_us(us):
    if us is None:
        return "n/a"
    if us < 1000:
        return f"{us:.0f} us"
    if us < 1_000_000:
        return f"{us/1000:.2f} ms"
    return f"{us/1_000_000:.2f} s"


def fmt_pct(num, denom):
    if not denom:
        return "n/a"
    return f"{100.0 * num / denom:.1f}%"


def report(path: Path, trace: dict, bench_elapsed: float | None):
    events = trace.get("traceEvents") or []
    print(f"# Trace: {path}")
    print(f"# events: {len(events)}")

    window = trace_window_us(events)
    print(f"# trace window: {fmt_us(window)}")
    if bench_elapsed and window:
        bench_us = bench_elapsed * 1_000_000
        ratio = window / bench_us
        print(f"# bench wall-clock: {bench_elapsed:.1f}s   (trace/bench ratio = {ratio:.2f})")
        if not 0.5 <= ratio <= 2.0:
            print("  [WARN] trace/bench ratio outside [0.5, 2.0]; ROI may not match the bench window")

    print()
    print("## Step cadence")
    cad = step_cadence(events)
    print(f"  steps observed: {cad['n_steps']}")
    print(f"  inter-step    : p50={fmt_us(cad['cadence_us_p50'])}  p90={fmt_us(cad['cadence_us_p90'])}  p99={fmt_us(cad['cadence_us_p99'])}")
    if cad["phase_count"]:
        print("  by phase:")
        for ph, n in sorted(cad["phase_count"].items()):
            print(f"    {ph:10s}  n={n:5d}  median_step={fmt_us(cad['phase_us_p50'].get(ph))}")
    else:
        print("  [WARN] no vllm.<phase> spans found - was serve_with_pt_profile.py used?")

    print()
    print("## Kernel breakdown (GPU/XPU)")
    kb = kernel_breakdown(events)
    total = kb["total_us"]
    if total == 0:
        print("  [FAIL] no kernel events found - GPU activity collection isn't capturing.")
        print("         CUDA: check CUPTI is loadable.")
        print("         XPU : check IPEX/torch versions; --enforce-eager recommended.")
    else:
        print(f"  total kernel time: {fmt_us(total)}")
        print(f"    attention      : {fmt_us(kb['attn_us'])}  ({fmt_pct(kb['attn_us'], total)})")
        print(f"    gemm/matmul    : {fmt_us(kb['gemm_us'])}  ({fmt_pct(kb['gemm_us'], total)})")
        print(f"    nccl/collective: {fmt_us(kb['nccl_us'])}  ({fmt_pct(kb['nccl_us'], total)})")
        print(f"    other          : {fmt_us(kb['other_us'])}  ({fmt_pct(kb['other_us'], total)})")

        if window:
            print(f"  forward fraction (kernels/window): {fmt_pct(total, window)}")
            if kb['attn_us']:
                print(f"  attention fraction (attn/kernels): {fmt_pct(kb['attn_us'], total)}")
            if kb['nccl_us']:
                print(f"  nccl fraction (nccl/window):       {fmt_pct(kb['nccl_us'], window)}")

        print("  top kernels:")
        for name, dur in kb["top"]:
            print(f"    {fmt_us(dur):>10s}  {fmt_pct(dur, total):>6s}  {name}")

    # Heuristic warnings (ties to perfetto-analysis.md §4 pathologies).
    print()
    print("## Heuristic checks")
    warned = False
    if total and window:
        fwd_frac = total / window
        if fwd_frac < 0.7:
            print(f"  [WARN] forward fraction {fwd_frac:.0%} < 70%: "
                  "CPU/scheduler likely the bottleneck (see perfetto-analysis.md §4.1)")
            warned = True
    if kb["nccl_us"] and window:
        nccl_frac = kb["nccl_us"] / window
        if nccl_frac > 0.2:
            print(f"  [WARN] NCCL fraction {nccl_frac:.0%} > 20%: "
                  "TP comm-bound (see perfetto-analysis.md §4.5)")
            warned = True
    if total and kb["attn_us"] / total > 0.8:
        print("  [INFO] attention dominates (>80% of kernels): "
              "long-context decode is normal here; check backend if context is short")
        warned = True
    if not warned:
        print("  no heuristic warnings")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("path", help="trace file or directory containing traces")
    ap.add_argument("--bench-elapsed", type=float, default=None,
                    help="benchmark wall-clock seconds (for ROI ratio check)")
    args = ap.parse_args()

    path = pick_trace(args.path)
    print(f"Loading {path} ...", file=sys.stderr)
    trace = load_trace(path)
    report(path, trace, args.bench_elapsed)


if __name__ == "__main__":
    main()
