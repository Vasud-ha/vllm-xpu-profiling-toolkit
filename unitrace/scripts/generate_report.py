#!/usr/bin/env python3
"""
generate_report.py - Build a self-contained HTML report from a unitrace result dir.

Usage:
    generate_report.py <result_dir>
    generate_report.py                    # uses $RESULT_DIR

Looks at the unitrace Chrome-trace JSONs (`python.*.json`, `sh.*.json`),
picks the EngineCore JSON (the one with gpu_op events), computes:

  - KPIs: total GPU op time, ROI wall span, kernel count, GEMM share
  - Iteration segmentation by inter-kernel gaps > 500 us
  - Per-iteration table (prefill = first dense iter; decode = the rest)
  - Top kernels overall and within prefill / decode

and writes `unitrace_vllm_report.html` into the result dir. The HTML embeds
its own CSS + SVG plots so it can be opened standalone.

The Setup / Scripts / Commands / ITT / Troubleshooting sections are static
prose drawn from the skill; the Summary / Trace files / Prefill-Decode /
Findings sections are recomputed for the run being reported.
"""

from __future__ import annotations

import collections
import datetime as _dt
import glob
import html as _html
import json
import os
import sys
from pathlib import Path

GAP_US = 500          # iteration boundary
DENSE_MIN_KERNELS = 20  # what counts as a "dense" iteration (filters scheduler ticks)


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------

def _load_trace(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    return data.get("traceEvents", data) if isinstance(data, dict) else data


def _trace_summary(path: str) -> dict:
    """Per-file headline numbers."""
    try:
        ev = _load_trace(path)
    except Exception as e:
        return {"path": path, "error": str(e), "events": 0, "gpu": 0, "span_ms": 0.0}
    gpu = [e for e in ev if e.get("cat") == "gpu_op" and e.get("ph") == "X"]
    span = 0.0
    if gpu:
        ts = [e["ts"] for e in gpu if "ts" in e]
        span = (max(ts) - min(ts)) / 1000.0
    return {
        "path": path,
        "size": os.path.getsize(path),
        "events": len(ev),
        "gpu": len(gpu),
        "span_ms": span,
    }


def _pick_engine_trace(result_dir: str) -> tuple[str | None, list[dict]]:
    """Return (engine_core_trace_path, [per-file summary dicts])."""
    summaries = []
    for p in sorted(glob.glob(os.path.join(result_dir, "python.*.json")) +
                    glob.glob(os.path.join(result_dir, "sh.*.json"))):
        summaries.append(_trace_summary(p))
    engine = max(
        (s for s in summaries if s.get("gpu", 0) > 0),
        key=lambda s: s["gpu"],
        default=None,
    )
    return (engine["path"] if engine else None), summaries


def _segment_iters(gpu_events: list) -> list[list]:
    """Split kernels into iterations using inter-kernel gaps > GAP_US."""
    if not gpu_events:
        return []
    gpu_events = sorted(gpu_events, key=lambda e: e.get("ts", 0))
    iters = [[gpu_events[0]]]
    for i in range(1, len(gpu_events)):
        prev = gpu_events[i - 1]
        cur = gpu_events[i]
        gap = cur["ts"] - (prev["ts"] + prev.get("dur", 0))
        if gap > GAP_US:
            iters.append([])
        iters[-1].append(cur)
    return iters


def _kernel_totals(events: list) -> list[tuple[str, int, float, float, float]]:
    """Returns [(name, calls, total_ms, avg_us, share_pct), ...] sorted desc."""
    tot = collections.defaultdict(lambda: [0, 0.0])
    for e in events:
        if e.get("cat") != "gpu_op" or e.get("ph") != "X":
            continue
        tot[e.get("name", "-")][0] += 1
        tot[e.get("name", "-")][1] += e.get("dur", 0)
    grand = sum(t[1] for t in tot.values()) or 1.0
    rows = []
    for n, (c, s) in tot.items():
        rows.append((n, c, s / 1000.0, s / max(c, 1), 100.0 * s / grand))
    rows.sort(key=lambda r: -r[2])
    return rows


def _kernel_detailed(events: list) -> list[dict]:
    """Per-kernel detailed timing stats (us): calls, total_ms, share, avg, min, p50, p95, max, stddev."""
    durs = collections.defaultdict(list)
    for e in events:
        if e.get("cat") != "gpu_op" or e.get("ph") != "X":
            continue
        durs[e.get("name", "-")].append(e.get("dur", 0))
    grand = sum(sum(v) for v in durs.values()) or 1.0
    out = []
    for name, ds in durs.items():
        ds_sorted = sorted(ds)
        n = len(ds_sorted)
        s = sum(ds_sorted)
        avg = s / n
        mn = ds_sorted[0]
        mx = ds_sorted[-1]
        p50 = ds_sorted[n // 2] if n else 0
        p95_idx = max(0, min(n - 1, int(0.95 * n)))
        p95 = ds_sorted[p95_idx]
        var = sum((d - avg) ** 2 for d in ds_sorted) / n if n else 0
        std = var ** 0.5
        out.append({
            "name": name,
            "calls": n,
            "total_ms": s / 1000.0,
            "share": 100.0 * s / grand,
            "avg_us": avg,
            "min_us": mn,
            "p50_us": p50,
            "p95_us": p95,
            "max_us": mx,
            "std_us": std,
        })
    out.sort(key=lambda r: -r["total_ms"])
    return out


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _esc(s) -> str:
    return _html.escape(str(s), quote=True)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _read_text(path: str) -> str:
    try:
        return open(path).read()
    except OSError:
        return f"# {path}: not found"


def _kpi_card(label: str, value: str) -> str:
    return f'<div class="card"><div class="l">{_esc(label)}</div><div class="v">{_esc(value)}</div></div>'


def _classify_kernel(name: str) -> str:
    """Group kernels into coarse categories for the timing breakdown."""
    n = name.lower()
    if "gemm_kernel" in n:
        return "GEMM"
    if "rms_norm" in n or "rmsnorm" in n:
        return "RMSNorm"
    if "rotary_embedding" in n:
        return "RoPE"
    if "act_and_mul" in n or "silu" in n or "gelu" in n:
        return "Activation"
    if "reshape_and_cache" in n or "kv_cache" in n:
        return "KV-cache"
    if "attention" in n or "flash_attn" in n or "cutlass" in n:
        return "Attention"
    if "elementwise" in n or "fillfunctor" in n or "copy" in n:
        return "Elementwise/Copy"
    if "sampler" in n or "topk" in n or "argmax" in n or "softmax" in n:
        return "Sampling/Softmax"
    return "Other"


def _kernel_detail_table(detail_rows, top: int = 15) -> str:
    """Render detailed kernel timing stats with min/p50/p95/max/stddev."""
    rows = ['<table><tr>'
            '<th>#</th><th>Kernel</th><th>Category</th>'
            '<th>Calls</th><th>Total (ms)</th><th>Share</th>'
            '<th>Avg (µs)</th><th>Min</th><th>P50</th><th>P95</th><th>Max</th><th>StdDev</th>'
            '</tr>']
    for i, r in enumerate(detail_rows[:top], 1):
        rows.append(
            f"<tr>"
            f"<td>{i}</td>"
            f"<td><code>{_esc(_truncate(r['name'], 100))}</code></td>"
            f"<td><span class='tag'>{_classify_kernel(r['name'])}</span></td>"
            f"<td>{r['calls']:,}</td>"
            f"<td>{r['total_ms']:.3f}</td>"
            f"<td>{r['share']:.2f}%</td>"
            f"<td>{r['avg_us']:.1f}</td>"
            f"<td>{r['min_us']:.1f}</td>"
            f"<td>{r['p50_us']:.1f}</td>"
            f"<td>{r['p95_us']:.1f}</td>"
            f"<td>{r['max_us']:.1f}</td>"
            f"<td>{r['std_us']:.1f}</td>"
            f"</tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _category_breakdown_table(detail_rows) -> str:
    """Aggregate kernels by category and render a summary table."""
    cat = collections.defaultdict(lambda: {"calls": 0, "total_ms": 0.0, "kinds": 0})
    for r in detail_rows:
        c = _classify_kernel(r["name"])
        cat[c]["calls"] += r["calls"]
        cat[c]["total_ms"] += r["total_ms"]
        cat[c]["kinds"] += 1
    grand = sum(v["total_ms"] for v in cat.values()) or 1.0
    rows = sorted(cat.items(), key=lambda kv: -kv[1]["total_ms"])
    out = ['<table><tr><th>Category</th><th>Distinct kernels</th><th>Calls</th>'
           '<th>Total (ms)</th><th>Share</th><th>Avg per call (µs)</th></tr>']
    for c, d in rows:
        avg = d["total_ms"] * 1000.0 / d["calls"] if d["calls"] else 0
        out.append(
            f"<tr><td><b>{c}</b></td><td>{d['kinds']}</td><td>{d['calls']:,}</td>"
            f"<td>{d['total_ms']:.3f}</td><td>{100*d['total_ms']/grand:.2f}%</td>"
            f"<td>{avg:.1f}</td></tr>"
        )
    out.append("</table>")
    return "\n".join(out)


def _kernel_distribution_svg(detail_rows, top_n: int = 8) -> str:
    """Render an avg/min/max range plot for the top-N kernels (log-ish scale)."""
    if not detail_rows:
        return ""
    rows = detail_rows[:top_n]
    max_us = max(r["max_us"] for r in rows) or 1.0
    X0, X1 = 200, 1060
    span = X1 - X0
    H = 30 * len(rows) + 80
    bars = []
    for i, r in enumerate(rows):
        y = 60 + i * 30
        x_min = X0 + (r["min_us"] / max_us) * span
        x_max = X0 + (r["max_us"] / max_us) * span
        x_p50 = X0 + (r["p50_us"] / max_us) * span
        x_p95 = X0 + (r["p95_us"] / max_us) * span
        x_avg = X0 + (r["avg_us"] / max_us) * span
        # min..max line
        bars.append(f'<line x1="{x_min:.1f}" y1="{y}" x2="{x_max:.1f}" y2="{y}" stroke="#30363d" stroke-width="1.5"/>')
        # p50..p95 band
        bars.append(f'<rect x="{x_p50:.1f}" y="{y-7}" width="{max(x_p95-x_p50,1):.1f}" height="14" fill="#79c0ff" opacity="0.45"/>')
        # min/max ticks
        bars.append(f'<line x1="{x_min:.1f}" y1="{y-5}" x2="{x_min:.1f}" y2="{y+5}" stroke="#8b949e" stroke-width="1.5"/>')
        bars.append(f'<line x1="{x_max:.1f}" y1="{y-5}" x2="{x_max:.1f}" y2="{y+5}" stroke="#8b949e" stroke-width="1.5"/>')
        # avg marker
        bars.append(f'<circle cx="{x_avg:.1f}" cy="{y}" r="4" fill="#d2a8ff" stroke="#0d1117" stroke-width="1"/>')
        # label
        label = _truncate(r["name"], 28)
        bars.append(f'<text x="190" y="{y+4}" text-anchor="end" fill="#e6edf3" font="11px monospace" font-family="monospace" font-size="11">{_esc(label)}</text>')
        bars.append(f'<text x="{X1+5}" y="{y+4}" fill="#8b949e" font-family="monospace" font-size="11">{r["max_us"]:.0f} µs</text>')
    # x-axis ticks
    ticks = []
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        tx = X0 + frac * span
        val = max_us * frac
        ticks.append(f'<line x1="{tx:.1f}" y1="40" x2="{tx:.1f}" y2="{H-30}" stroke="#21262d" stroke-dasharray="2,3"/>')
        ticks.append(f'<text x="{tx:.1f}" y="{H-12}" fill="#8b949e" text-anchor="middle" font-family="monospace" font-size="10">{val:.0f} µs</text>')
    return f"""<svg viewBox="0 0 1100 {H}" width="1100" height="{H}" xmlns="http://www.w3.org/2000/svg">
  <text x="20" y="22" fill="#fff" font-family="monospace" font-size="13">Per-call duration distribution (top {len(rows)} kernels)</text>
  <text x="20" y="38" fill="#8b949e" font-family="monospace" font-size="11">grey line: min—max  |  blue band: p50—p95  |  purple dot: avg</text>
  {chr(10).join(ticks)}
  {chr(10).join(bars)}
</svg>"""


def _kernel_table(rows, top: int = 10) -> str:
    out = ['<table><tr><th>#</th><th>Kernel</th><th>Calls</th><th>Total (ms)</th><th>Avg (µs)</th><th>Share</th></tr>']
    for i, (n, c, ms, avg, pct) in enumerate(rows[:top], 1):
        out.append(
            f"<tr><td>{i}</td><td><code>{_esc(_truncate(n, 90))}</code></td>"
            f"<td>{c}</td><td>{ms:.2f}</td><td>{avg:.1f}</td><td>{pct:.1f}%</td></tr>"
        )
    out.append("</table>")
    return "\n".join(out)


def _iter_table(iters_dense, iters_all) -> str:
    if not iters_dense:
        return '<div class="note bad">No dense iterations found in this trace.</div>'
    t0 = iters_dense[0][0]["ts"]
    rows = ['<table><tr><th>Iter</th><th>Kind</th><th>Start (ms)</th><th>End (ms)</th>'
            '<th>Span (ms)</th><th>GPU dur (ms)</th><th>Kernels</th><th>GPU util</th></tr>']
    for i, it in enumerate(iters_dense):
        start = (it[0]["ts"] - t0) / 1000.0
        end = (it[-1]["ts"] + it[-1].get("dur", 0) - t0) / 1000.0
        gpu_dur = sum(e.get("dur", 0) for e in it) / 1000.0
        span = max(end - start, 1e-6)
        util = 100.0 * gpu_dur / span
        kind = ("<span style='color:#d2a8ff'>prefill</span>" if i == 0 else "decode")
        rows.append(
            f"<tr><td>{i}</td><td>{kind}</td><td>{start:.2f}</td><td>{end:.2f}</td>"
            f"<td>{span:.2f}</td><td>{gpu_dur:.2f}</td><td>{len(it)}</td><td>{util:.1f}%</td></tr>"
        )
    rows.append(
        f"<tr><td colspan=8 style='color:#8b949e;font-size:12px'>"
        f"Detected {len(iters_all)} total iterations; {len(iters_dense)} were dense "
        f"(≥{DENSE_MIN_KERNELS} kernels). Boundaries: inter-kernel gap &gt; {GAP_US} µs."
        f"</td></tr>"
    )
    rows.append("</table>")
    return "\n".join(rows)


def _file_table(summaries) -> str:
    rows = ['<table><tr><th>File</th><th>Size</th><th>Events</th><th>GPU ops</th><th>Span</th></tr>']
    for s in summaries:
        if "error" in s:
            rows.append(f"<tr><td><code>{_esc(os.path.basename(s['path']))}</code></td>"
                        f"<td colspan=4 style='color:#f85149'>{_esc(s['error'])}</td></tr>")
            continue
        size_kb = s["size"] / 1024.0
        size_str = f"{size_kb/1024:.2f} MB" if size_kb > 1024 else f"{size_kb:.1f} KB"
        span = f"{s['span_ms']:.2f} ms" if s["span_ms"] else "—"
        rows.append(
            f"<tr><td><code>{_esc(os.path.basename(s['path']))}</code></td>"
            f"<td>{size_str}</td><td>{s['events']:,}</td><td>{s['gpu']:,}</td><td>{span}</td></tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _timeline_svg(iters_dense) -> str:
    """One bar per dense iteration, x-mapped from the iteration time range."""
    if not iters_dense:
        return ""
    t0 = iters_dense[0][0]["ts"]
    end = max(it[-1]["ts"] + it[-1].get("dur", 0) for it in iters_dense)
    total_ms = (end - t0) / 1000.0
    if total_ms <= 0:
        return ""
    X0, X1 = 80, 1060
    span = X1 - X0

    def x(ms):
        return X0 + ms / total_ms * span

    def grid_steps():
        for n in (10, 20, 25, 50, 100, 200, 250, 500, 1000):
            if total_ms / n <= 12:
                return n
        return 2000

    step = grid_steps()
    grid = []
    g_ms = 0.0
    while g_ms <= total_ms + 1e-6:
        gx = x(g_ms)
        grid.append(
            f'<line class="gr" x1="{gx:.1f}" y1="60" x2="{gx:.1f}" y2="160"/>'
            f'<text class="ax-l" x="{gx-8:.1f}" y="175">{int(round(g_ms))}</text>'
        )
        g_ms += step
    grid.append(f'<text class="ax-l" x="{X1+5}" y="175">ms</text>')

    bars = []
    max_dur = max(sum(e.get("dur", 0) for e in it) for it in iters_dense) / 1000.0
    for i, it in enumerate(iters_dense):
        s = (it[0]["ts"] - t0) / 1000.0
        e_ = (it[-1]["ts"] + it[-1].get("dur", 0) - t0) / 1000.0
        dur = sum(ev.get("dur", 0) for ev in it) / 1000.0
        cls = "pf" if i == 0 else "dc"
        x0 = x(s)
        w = max(x(e_) - x0, 1.5)
        h = 30 + 50 * (dur / max_dur if max_dur else 0)
        y = 140 - h
        label = ("prefill " if i == 0 else f"d{i} ") + f"{len(it)}k·{dur:.1f}ms"
        bars.append(f'<rect class="{cls}" x="{x0:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"/>')
        bars.append(f'<text class="lb" x="{x0:.1f}" y="{y-4:.1f}">{_esc(label)}</text>')

    bars_svg = "\n  ".join(bars)
    grid_svg = "\n  ".join(grid)
    return f"""<svg viewBox="0 0 1100 220" width="1100" height="220" xmlns="http://www.w3.org/2000/svg">
  <style>
    .ax{{stroke:#30363d;stroke-width:1}}
    .gr{{stroke:#21262d;stroke-width:1;stroke-dasharray:2,3}}
    .pf{{fill:#d2a8ff;opacity:0.85}}
    .dc{{fill:#3fb950;opacity:0.85}}
    .lb{{fill:#e6edf3;font:11px monospace}}
    .ttl{{fill:#fff;font:13px monospace}}
    .ax-l{{fill:#8b949e;font:10.5px monospace}}
  </style>
  <text class="ttl" x="80" y="20">GPU kernel timeline (ROI: 0 — {total_ms:.0f} ms)</text>
  <line class="ax" x1="80" y1="160" x2="1060" y2="160"/>
  {grid_svg}
  {bars_svg}
</svg>"""


def _stacked_bar_svg(rows, total_ms: float) -> str:
    """Top-kernel share stacked horizontally."""
    if not rows or total_ms <= 0:
        return ""
    palette = ["#79c0ff", "#a371f7", "#d2a8ff", "#56d364", "#f0883e",
               "#e34c26", "#f9826c", "#ffa657", "#6e7681"]
    X0, W = 20, 1040
    top = rows[:8]
    seen_pct = sum(r[4] for r in top)
    other_pct = max(0.0, 100.0 - seen_pct)
    segments = list(top) + ([("other", 0, 0.0, 0.0, other_pct)] if other_pct > 0.5 else [])
    bars = []
    cursor = X0
    for i, (n, c, ms, avg, pct) in enumerate(segments):
        w = pct / 100.0 * W
        col = palette[i % len(palette)]
        bars.append(f'<rect class="bar" x="{cursor:.1f}" y="50" width="{w:.1f}" height="50" fill="{col}"/>')
        if w > 38:
            label = f"{pct:.1f}% {_truncate(n, 16)}" if n != "other" else "other"
            bars.append(f'<text class="pct" x="{cursor+6:.1f}" y="80">{_esc(label)}</text>')
        cursor += w
    return f"""<svg viewBox="0 0 1100 140" width="1100" height="140" xmlns="http://www.w3.org/2000/svg">
  <style>
    .bar{{stroke:#0d1117;stroke-width:1}}
    .pct{{fill:#0d1117;font:11.5px monospace;font-weight:700}}
    .ttl{{fill:#fff;font:13px monospace}}
  </style>
  <text class="ttl" x="20" y="22">GPU time share — top kernels ({total_ms:.2f} ms total)</text>
  {chr(10).join(bars)}
</svg>"""


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

def _findings(total_gpu_ms, wall_span_ms, top_rows, iters_dense, has_itt) -> list[tuple[str, str, str]]:
    """Return [(severity, title, body_html), ...] - severity in {good,note,warn,bad}."""
    out = []
    util = 100.0 * total_gpu_ms / wall_span_ms if wall_span_ms else 0.0
    top_kernel = top_rows[0] if top_rows else None
    gemm_share = sum(r[4] for r in top_rows if "gemm_kernel" in r[0])

    if top_kernel and "gemm_kernel" in top_kernel[0]:
        out.append((
            "note",
            "F1 — GEMM-dominated workload",
            f"Top kernel <code>{_esc(_truncate(top_kernel[0], 80))}</code> is "
            f"{top_kernel[4]:.1f}% of GPU time over {top_kernel[1]} calls "
            f"(avg {top_kernel[3]:.1f} µs). All GEMM kernels combined: "
            f"<b>{gemm_share:.1f}%</b> of GPU. Decode-shaped GEMVs (M=1) here are "
            f"memory-bandwidth-bound — batch more tokens (continuous batching, spec decode) "
            f"to fix; the kernels themselves are already saturating bandwidth.",
        ))
    if util < 40:
        out.append((
            "warn",
            "F2 — Most ROI wall time is host-side gaps between iterations",
            f"GPU op time = {total_gpu_ms:.2f} ms / wall span = {wall_span_ms:.2f} ms "
            f"&rarr; <b>{util:.1f}%</b> GPU utilization across the ROI. "
            f"Likely culprits: Python decode loop, sampling, scheduler tick, and "
            f"<code>--enforce-eager</code> giving up CUDA-graph-equivalent benefits. "
            f"Within an iteration GPU utilization is typically much higher; the gap "
            f"between iterations is what to attack.",
        ))
    elif util < 75:
        out.append((
            "note",
            "F2 — Moderate inter-iteration host gaps",
            f"GPU utilization across ROI is {util:.1f}%. Acceptable for batched serving; "
            f"single-stream tail latency may still benefit from reducing host-side gaps.",
        ))
    else:
        out.append((
            "good",
            "F2 — Tight ROI — GPU is busy",
            f"GPU utilization across the captured ROI is {util:.1f}%. Inter-iteration "
            f"gaps are small relative to per-iteration GPU work.",
        ))

    if iters_dense:
        kc = [len(it) for it in iters_dense]
        durs = [sum(e.get("dur", 0) for e in it) / 1000.0 for it in iters_dense]
        if max(kc) > 1.5 * min(kc) or max(durs) > 1.5 * min(durs):
            out.append((
                "note",
                "F3 — Per-iteration GPU work varies",
                f"Iteration kernel counts range {min(kc)} → {max(kc)}; "
                f"per-iter GPU dur range {min(durs):.2f} → {max(durs):.2f} ms. "
                f"Variance is normal (KV-cache growth, sporadic sampler kernels). "
                f"For benchmark averages, drop the first iteration — it's the prefill.",
            ))

    if not has_itt:
        out.append((
            "bad",
            "F4 — No ITT events visible in the Chrome trace",
            f"No events with <code>cat=ITT</code> or names containing 'itt' were found "
            f"in this capture. Either <code>--chrome-itt-logging</code> wasn't set, the "
            f"unitrace build emits ITT under a different category, or ITT was used only "
            f"for paused-mode gating (not recorded as visible events). The GPU timestamps "
            f"are still authoritative for ROI bounds.",
        ))
    else:
        out.append((
            "good",
            "F4 — ITT markers present",
            "Resume/pause markers landed in the trace — you can see exactly which "
            "kernels fall inside the ROI in Perfetto.",
        ))

    out.append((
        "note",
        "F5 — The static <code>libittnotify.a</code> in pti-gpu is unused at runtime",
        "ITT symbols come from <code>libunitrace_tool.so</code> (statically links ittnotify). "
        "Don't build a private <code>libittnotify.so</code> from pti-gpu's ittapi tree — "
        "calls into it become silent no-ops because they connect to a second copy of ITT "
        "with no live collector.",
    ))
    out.append((
        "note",
        "F6 — <code>--enforce-eager</code> is required on intel/vllm-xpu",
        "torch.compile inside <code>intel/vllm:0.14.1-xpu</code> hits an llvm-foreach "
        "SEGV in SYCL JIT. <code>--enforce-eager</code> is mandatory and (as a side-effect) "
        "gives clearer kernel names. Trade-off: you lose graph-replay benefits, which is "
        "part of why host gaps are larger than they'd otherwise be.",
    ))
    return out


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

CSS = r"""
:root{
  --bg:#0e1116; --panel:#161b22; --panel2:#1f2630; --fg:#e6edf3;
  --muted:#8b949e; --accent:#79c0ff; --accent2:#d2a8ff; --good:#3fb950;
  --warn:#d29922; --bad:#f85149; --line:#30363d; --code:#0d1117;
}
*{box-sizing:border-box}
body{margin:0;padding:0;background:var(--bg);color:var(--fg);
  font:14px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
header{background:linear-gradient(135deg,#1f2630 0%, #2a1f3d 100%);
  padding:32px 48px;border-bottom:1px solid var(--line)}
h1{margin:0 0 8px;font-size:28px;color:#fff}
header .sub{color:var(--muted);font-size:14px}
nav{position:sticky;top:0;background:#0e1116ee;border-bottom:1px solid var(--line);
  padding:10px 48px;z-index:10;backdrop-filter:blur(8px)}
nav a{color:var(--accent);text-decoration:none;margin-right:14px;font-size:13px}
nav a:hover{text-decoration:underline}
main{max-width:1180px;margin:0 auto;padding:24px 48px 80px}
section{margin:36px 0;padding:24px;background:var(--panel);border:1px solid var(--line);
  border-radius:8px}
h2{margin:0 0 16px;font-size:20px;color:#fff;border-left:3px solid var(--accent);padding-left:10px}
h3{margin:22px 0 10px;font-size:16px;color:var(--accent2)}
h4{margin:18px 0 6px;font-size:14px;color:#fff}
p{margin:8px 0}
code{background:var(--code);padding:2px 6px;border-radius:3px;font:13px/1.4 Menlo,Consolas,monospace;color:#ffa657}
pre{background:var(--code);border:1px solid var(--line);border-radius:6px;padding:14px 16px;
  overflow-x:auto;font:12.5px/1.55 Menlo,Consolas,monospace;color:#c9d1d9}
pre.bash::before{content:"$ ";color:var(--good)}
.kbd{font:12px Menlo,monospace;background:#21262d;padding:1px 6px;border:1px solid #444;
  border-bottom-width:2px;border-radius:3px;color:#fff}
table{width:100%;border-collapse:collapse;margin:12px 0;font-size:13px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;background:var(--panel2)}
tr:hover td{background:#1c222b}
.kpi{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:12px 0}
.kpi .card{background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:14px}
.kpi .v{font-size:22px;font-weight:600;color:#fff}
.kpi .l{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.note{border-left:3px solid var(--accent);background:#0d2138;padding:10px 14px;margin:12px 0;
  border-radius:4px;font-size:13px}
.warn{border-left-color:var(--warn);background:#2a1f08}
.bad{border-left-color:var(--bad);background:#2a1010}
.good{border-left-color:var(--good);background:#0d2010}
.flow{display:flex;align-items:center;gap:6px;flex-wrap:wrap;font:13px Menlo,monospace;
  background:var(--code);padding:14px;border-radius:6px;border:1px solid var(--line)}
.flow .b{padding:6px 10px;background:#21262d;border:1px solid #444;border-radius:4px}
.flow .b.api{border-color:#79c0ff;color:#79c0ff}
.flow .b.itt{border-color:#d2a8ff;color:#d2a8ff}
.flow .b.gpu{border-color:#3fb950;color:#3fb950}
.flow .a{color:var(--muted)}
svg{display:block;max-width:100%;background:#0d1117;border:1px solid var(--line);
  border-radius:6px;margin-top:8px}
.legend{font-size:12px;color:var(--muted);margin-top:6px;display:flex;gap:16px;flex-wrap:wrap}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:2px}
details{background:var(--panel2);border:1px solid var(--line);border-radius:6px;
  padding:8px 14px;margin:8px 0}
details summary{cursor:pointer;color:var(--accent);font-weight:500}
details[open] summary{margin-bottom:8px}
.tag{display:inline-block;font-size:11px;background:#21262d;color:var(--muted);
  padding:2px 7px;border-radius:10px;margin-right:6px;border:1px solid var(--line)}
.tag.ok{color:var(--good);border-color:#1f4d2a}
hr{border:none;border-top:1px solid var(--line);margin:20px 0}
"""


CALL_GRAPH_SVG = r"""<svg viewBox="0 0 1100 360" width="1100" height="360" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#79c0ff"/>
    </marker>
  </defs>
  <style>
    .b{stroke:#30363d;stroke-width:1;rx:6;ry:6}
    .api{fill:#0d2138;stroke:#79c0ff}
    .itt{fill:#1d1130;stroke:#d2a8ff}
    .gpu{fill:#0d2010;stroke:#3fb950}
    .ext{fill:#1f2630}
    .lbl{fill:#e6edf3;font:13px monospace}
    .sub{fill:#8b949e;font:11px monospace}
    .arrow{stroke:#79c0ff;stroke-width:1.6;fill:none}
  </style>
  <rect class="b ext" x="20"  y="20"  width="180" height="50"/>
  <text class="lbl" x="36" y="42">curl / OpenAI SDK /</text>
  <text class="lbl" x="36" y="58">vllm bench serve</text>
  <rect class="b ext" x="240" y="10"  width="380" height="160"/>
  <text class="sub" x="248" y="26">API-server process</text>
  <rect class="b api" x="260" y="40"  width="160" height="40"/>
  <text class="lbl" x="270" y="64">FastAPI handler</text>
  <rect class="b itt" x="430" y="40"  width="180" height="40"/>
  <text class="lbl" x="438" y="64">__itt_resume()/_pause()</text>
  <rect class="b ext" x="260" y="100" width="350" height="50"/>
  <text class="sub" x="270" y="118">Worker.execute_model wrapper</text>
  <text class="lbl" x="270" y="138">checks roi_gate file each step</text>
  <rect class="b ext" x="660" y="10" width="420" height="160"/>
  <text class="sub" x="670" y="26">EngineCore subprocess</text>
  <rect class="b api" x="680" y="40" width="180" height="40"/>
  <text class="lbl" x="692" y="64">Worker.execute_model</text>
  <rect class="b itt" x="870" y="40" width="180" height="40"/>
  <text class="lbl" x="880" y="64">itt_resume / pause</text>
  <rect class="b gpu" x="680" y="100" width="370" height="50"/>
  <text class="sub" x="690" y="118">Level Zero kernel launches (gemm, rmsnorm, attn)</text>
  <text class="lbl" x="690" y="138">capture + flow events</text>
  <rect class="b ext" x="430" y="220" width="240" height="40"/>
  <text class="lbl" x="440" y="244">$RESULT_DIR/roi_gate (sentinel)</text>
  <rect class="b itt" x="40" y="220" width="320" height="120"/>
  <text class="sub" x="50" y="236">unitrace launcher (--start-paused)</text>
  <text class="lbl" x="50" y="258">- injects libunitrace_tool.so</text>
  <text class="lbl" x="50" y="278">- listens on __itt_resume / __itt_pause</text>
  <text class="lbl" x="50" y="298">- writes python.&lt;pid&gt;.json on exit</text>
  <path class="arrow" marker-end="url(#arr)" d="M200,45 L240,45"/>
  <path class="arrow" marker-end="url(#arr)" d="M420,60 L430,60"/>
  <path class="arrow" marker-end="url(#arr)" d="M610,125 L660,125"/>
  <path class="arrow" marker-end="url(#arr)" d="M860,60 L870,60"/>
  <path class="arrow" marker-end="url(#arr)" d="M860,80 L860,100 L870,100"/>
  <path class="arrow" marker-end="url(#arr)" d="M520,80 L520,220"/>
  <path class="arrow" marker-end="url(#arr)" d="M520,260 L435,260 L435,150"/>
  <path class="arrow" marker-end="url(#arr)" d="M200,300 L40,300"/>
  <path class="arrow" marker-end="url(#arr)" d="M340,220 L540,170"/>
  <text class="sub" x="208" y="40">/start_profile</text>
  <text class="sub" x="623" y="120">spawn (multiproc)</text>
  <text class="sub" x="540" y="200">create / remove gate</text>
  <text class="sub" x="200" y="318">ITT calls reach unitrace via injected .so</text>
</svg>
"""


def build_html(
    result_dir: str,
    file_summaries: list[dict],
    engine_path: str | None,
    gpu_events: list,
    iters_all: list[list],
    iters_dense: list[list],
    overall_top: list,
    prefill_top: list,
    decode_top: list,
    overall_detail: list,
    has_itt: bool,
    skill_dir: str,
) -> str:
    total_gpu_ms = sum(e.get("dur", 0) for e in gpu_events) / 1000.0
    wall_span_ms = (
        (max(e["ts"] + e.get("dur", 0) for e in gpu_events) -
         min(e["ts"] for e in gpu_events)) / 1000.0
        if gpu_events else 0.0
    )
    gemm_share = sum(r[4] for r in overall_top if "gemm_kernel" in r[0])
    util_pct = 100.0 * total_gpu_ms / wall_span_ms if wall_span_ms else 0.0
    rd_basename = os.path.basename(os.path.normpath(result_dir))

    findings_html = "\n".join(
        f'<div class="note {sev}"><h4 style="margin-top:0">{title}</h4>{body}</div>'
        for sev, title, body in _findings(total_gpu_ms, wall_span_ms, overall_top, iters_dense, has_itt)
    )

    run_sh = _read_text(os.path.join(skill_dir, "scripts", "run_unitrace_vllm.sh"))
    serve_py = _read_text(os.path.join(skill_dir, "scripts", "serve_with_unitrace.py"))

    timeline_svg = _timeline_svg(iters_dense)
    stacked_svg = _stacked_bar_svg(overall_top, total_gpu_ms)

    # Static prose blocks ----------------------------------------------------
    setup_table = """
<table>
<tr><th>Component</th><th>Version / Path</th></tr>
<tr><td>Container image</td><td><code>intel/vllm:0.14.1-xpu</code></td></tr>
<tr><td>vLLM</td><td>v1 API path (<code>VLLM_USE_V1=1</code>)</td></tr>
<tr><td>oneAPI</td><td>2025.3 (sourced via <code>/opt/intel/oneapi/setvars.sh --force</code>)</td></tr>
<tr><td>unitrace</td><td>pti-gpu 2.3.0; binary at <code>$UNITRACE_BIN</code></td></tr>
<tr><td>Result root</td><td><code>$RESULT_ROOT</code> (this run: <code>{rd}</code>)</td></tr>
</table>
""".replace("{rd}", _esc(result_dir))

    flags_table = """
<table>
<tr><th>Flag</th><th>Effect</th></tr>
<tr><td><code>--start-paused</code></td><td>Boot with collection paused; only ITT <code>__itt_resume</code> turns it on.</td></tr>
<tr><td><code>-d</code></td><td>Print device-timing summary to stdout at process exit.</td></tr>
<tr><td><code>--chrome-kernel-logging</code></td><td>Emit GPU kernels into the Chrome trace.</td></tr>
<tr><td><code>--chrome-sycl-logging</code></td><td>Emit SYCL/UR runtime calls.</td></tr>
<tr><td><code>--chrome-itt-logging</code></td><td>Surface <code>__itt_resume</code>/<code>__itt_pause</code> markers in the trace.</td></tr>
</table>
"""

    troubleshooting_table = """
<table>
<tr><th>Symptom</th><th>Cause</th><th>Fix</th></tr>
<tr><td><code>(uintptr_t)(val)</code> cast error during build</td><td>L0 headers in oneAPI 2025.3 made <code>ze_ipc_event_counter_based_handle_t</code> a struct.</td><td>Patch <code>scripts/gen_tracing_callbacks.py::gen_to_hex_string_functions</code> with the templated <code>to_hex_value_</code> helper; <code>rm -f *.gen && make -j</code>.</td></tr>
<tr><td><code>ITT available: False</code></td><td>Wrapper opened <code>libittnotify.so</code> first; pti-gpu builds it as <code>.a</code> only.</td><td>Wrapper falls back to <code>libunitrace_tool.so</code>. Verify <code>UNITRACE_TOOL_LIB</code> env or default path.</td></tr>
<tr><td><code>/start_profile</code> returns 404</td><td><code>runpy.run_module</code> would re-import api_server fresh and drop the monkey-patch.</td><td>Wrapper uses the <code>run_server(args)</code> direct call. Don't switch to runpy.</td></tr>
<tr><td><code>[ERROR] Failed to launch target: --</code></td><td><code>unitrace -- python ...</code> rejects the separator.</td><td>Drop the <code>--</code>: <code>unitrace &lt;flags&gt; python ...</code>.</td></tr>
<tr><td>GPU span ≈ entire server lifetime</td><td>ROI gate never closed (client crashed or stale gate from prior run).</td><td>Delete stale gate files. Launcher places them in <code>$RESULT_DIR/roi_gate</code>.</td></tr>
<tr><td><code>llvm-foreach SEGV</code> in SYCL JIT</td><td>torch.compile + intel/vllm-xpu interaction.</td><td>Always launch with <code>--enforce-eager</code> (already in <code>run_unitrace_vllm.sh</code>).</td></tr>
<tr><td>Truncated / missing JSONs</td><td><code>kill -9</code> on the launcher; unitrace writes during teardown.</td><td>Single Ctrl-C; wait for the device-timing summary to print.</td></tr>
</table>
"""

    # Assemble ---------------------------------------------------------------
    now = _dt.datetime.fromtimestamp(os.path.getmtime(engine_path)).strftime("%Y-%m-%d %H:%M:%S") \
        if engine_path else "unknown"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>unitrace + vLLM Profiling Report — {_esc(rd_basename)}</title>
<style>{CSS}</style>
</head>
<body>

<header>
<h1>unitrace + vLLM Profiling Report</h1>
<div class="sub">Result dir <code>{_esc(rd_basename)}</code> &middot; trace mtime {_esc(now)} &middot; auto-generated by <code>generate_report.py</code></div>
</header>

<nav>
<a href="#summary">Summary</a>
<a href="#arch">Architecture</a>
<a href="#setup">Setup</a>
<a href="#scripts">Scripts</a>
<a href="#commands">Commands</a>
<a href="#integration">vLLM integration</a>
<a href="#itt">ITT integration</a>
<a href="#trace">Trace files</a>
<a href="#perfetto">Perfetto guide</a>
<a href="#prefill-decode">Prefill / Decode</a>
<a href="#top-kernels">Top kernels</a>
<a href="#findings">Findings</a>
<a href="#troubleshoot">Troubleshooting</a>
</nav>

<main>

<section id="summary">
<h2>1. Summary</h2>
<p>Single end-to-end profiling capture of vLLM serving on an Intel Xe / Xe2 GPU under <a style="color:var(--accent)" href="https://github.com/intel/pti-gpu/tree/master/tools/unitrace">pti-gpu unitrace</a>. The capture is bounded to the request window with curl-driven <code>/start_profile</code>&nbsp;→&nbsp;<code>/stop_profile</code> hooks that fire ITT API calls.</p>
<div class="kpi">
  {_kpi_card("GPU op time", f"{total_gpu_ms:.2f} ms")}
  {_kpi_card("Wall span (ROI)", f"{wall_span_ms:.2f} ms")}
  {_kpi_card("Kernel launches", f"{len(gpu_events):,}")}
  {_kpi_card("GEMM share", f"{gemm_share:.1f}%")}
</div>
<div class="kpi">
  {_kpi_card("ROI GPU util", f"{util_pct:.1f}%")}
  {_kpi_card("Dense iterations", str(len(iters_dense)))}
  {_kpi_card("Total iterations", str(len(iters_all)))}
  {_kpi_card("ITT events", "yes" if has_itt else "no")}
</div>
</section>

<section id="arch">
<h2>2. How unitrace integrates with vLLM</h2>
<p>vLLM v1 spawns the model in a separate <code>EngineCore</code> subprocess. unitrace wraps the launcher with <code>--start-paused</code>; the wrapper script registers FastAPI routes that fire <code>__itt_resume()</code>/<code>__itt_pause()</code> from the API process and a sentinel-file gate the worker subprocess re-checks every <code>execute_model</code> step.</p>
<div class="flow">
  <span class="b">curl / OpenAI SDK / vllm bench</span>
  <span class="a">—POST /start_profile→</span>
  <span class="b api">FastAPI handler</span>
  <span class="a">→</span>
  <span class="b itt">__itt_resume()</span>
  <span class="a">+ touch roi_gate</span>
</div>
<div class="flow" style="margin-top:8px">
  <span class="a">EngineCore: every step,</span>
  <span class="b api">Worker.execute_model</span>
  <span class="a">→</span>
  <span class="b itt">if gate exists: itt_resume()</span>
  <span class="a">→</span>
  <span class="b gpu">forward</span>
  <span class="a">→</span>
  <span class="b itt">xpu.synchronize() + itt_pause()</span>
</div>
<div class="flow" style="margin-top:8px">
  <span class="b">curl POST /stop_profile</span>
  <span class="a">→</span>
  <span class="b api">FastAPI handler</span>
  <span class="a">→</span>
  <span class="b itt">__itt_pause() + rm roi_gate</span>
  <span class="a">→ unitrace flushes Chrome trace on Ctrl-C</span>
</div>
</section>

<section id="setup">
<h2>3. Environment</h2>
{setup_table}
</section>

<section id="scripts">
<h2>4. Scripts (full content)</h2>
<details><summary><code>scripts/run_unitrace_vllm.sh</code> — unitrace launcher</summary>
<pre>{_esc(run_sh)}</pre></details>
<details><summary><code>scripts/serve_with_unitrace.py</code> — vLLM wrapper with ITT-driven endpoints</summary>
<pre>{_esc(serve_py)}</pre></details>
<h3>Default unitrace flags</h3>
{flags_table}
</section>

<section id="commands">
<h2>5. Running a session</h2>
<h3>5.1 Launch the server (terminal A)</h3>
<pre class="bash">cd ~/.claude/skills/unitrace-vllm-profiling/scripts
bash run_unitrace_vllm.sh</pre>
<p>Confirm <code>ITT available: True</code> in the log before sending traffic.</p>
<h3>5.2 Drive the ROI (terminal B)</h3>
<pre class="bash">PORT=9090
curl -X POST http://localhost:$PORT/start_profile

curl -X POST http://localhost:$PORT/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{"model":"meta-llama/Llama-3.1-8B-Instruct",
        "messages":[{{"role":"user","content":"San Francisco is a"}}],
        "max_tokens":24}}'

curl -X POST http://localhost:$PORT/stop_profile</pre>
<h3>5.3 Stop the server cleanly</h3>
<p>Hit <span class="kbd">Ctrl-C</span> in terminal A once. unitrace flushes JSONs and prints the device summary; <code>generate_report.py</code> then writes this HTML report. <code>kill -9</code> truncates the trace and skips the report.</p>
</section>

<section id="integration">
<h2>6. Where unitrace plugs into vLLM</h2>
{CALL_GRAPH_SVG}
</section>

<section id="itt">
<h2>7. ITT integration</h2>
<p>unitrace links <code>ittnotify_static.c</code> into <code>libunitrace_tool.so</code>. When <code>unitrace</code> launches the target, it injects that .so so <code>__itt_resume</code>/<code>__itt_pause</code> are resolvable from the target's global symbol table — and they're the symbols unitrace's collector listens on.</p>
<pre>$ nm -D libunitrace_tool.so | grep -E '__itt_resume|__itt_pause'
00000000000fc940 T __itt_pause
00000000000fcaf0 T __itt_pause_scoped
00000000000fcb00 T __itt_resume
00000000000fccb0 T __itt_resume_scoped</pre>
<p>The wrapper resolves ITT through three sources, first match wins:</p>
<table>
<tr><th>#</th><th>Source</th><th>When it works</th></tr>
<tr><td>1</td><td><code>ctypes.CDLL(None)</code> (RTLD_DEFAULT)</td><td>Process is running under <code>unitrace</code> — symbols injected. <span class="tag ok">primary</span></td></tr>
<tr><td>2</td><td><code>libunitrace_tool.so</code> by absolute path with <code>RTLD_GLOBAL</code></td><td>Wrapper running outside unitrace (testing).</td></tr>
<tr><td>3</td><td><code>libittnotify.so</code> from oneAPI VTune install</td><td>Hybrid setups; works only when VTune is the collector.</td></tr>
</table>
<p>This run: ITT events present in the trace = <b>{"yes" if has_itt else "no"}</b>.</p>
</section>

<section id="trace">
<h2>8. Trace files</h2>
<p>Result dir: <code>{_esc(result_dir)}</code></p>
{_file_table(file_summaries)}
<p>EngineCore trace used for analysis below: <code>{_esc(os.path.basename(engine_path)) if engine_path else "(none found)"}</code></p>
</section>

<section id="perfetto">
<h2>9. Loading the trace in Perfetto</h2>
<ol>
<li>Open <a style="color:var(--accent)" href="https://ui.perfetto.dev/">https://ui.perfetto.dev/</a></li>
<li>Drop in <code>{_esc(os.path.basename(engine_path)) if engine_path else "python.&lt;pid&gt;.json"}</code> (the EngineCore JSON — <em>not</em> the API-server one)</li>
<li>Press <span class="kbd">/</span> to search; type a kernel name from the table below to jump to a specific kernel.</li>
<li>Press <span class="kbd">M</span> to mark and <span class="kbd">[</span>&nbsp;/&nbsp;<span class="kbd">]</span> to step kernel-by-kernel through one iteration.</li>
</ol>
<details><summary>Useful Perfetto SQL: top kernels by total time</summary>
<pre>SELECT name, COUNT(*) AS calls, SUM(dur)/1e6 AS total_ms,
       AVG(dur)/1e3 AS avg_us
FROM slice WHERE category = 'gpu_op'
GROUP BY name ORDER BY total_ms DESC LIMIT 10;</pre></details>
</section>

<section id="prefill-decode">
<h2>10. Prefill vs decode in this capture</h2>
{timeline_svg}
<div class="legend">
  <span><i style="background:#d2a8ff"></i>prefill iteration</span>
  <span><i style="background:#3fb950"></i>decode iteration</span>
  <span>bar height ∝ GPU work in iteration</span>
</div>
<h3>10.1 Iteration table</h3>
{_iter_table(iters_dense, iters_all)}
<h3>10.2 Top kernels overall</h3>
{_kernel_table(overall_top, top=12)}
<h3>10.3 Top kernels in prefill (iteration 0)</h3>
{_kernel_table(prefill_top, top=8) if prefill_top else "<p style='color:var(--muted)'>No prefill iteration detected.</p>"}
<h3>10.4 Top kernels across decode iterations</h3>
{_kernel_table(decode_top, top=8) if decode_top else "<p style='color:var(--muted)'>No decode iterations detected.</p>"}
<h3>10.5 GPU time share</h3>
{stacked_svg}
</section>

<section id="top-kernels">
<h2>11. Top kernels — detailed timing</h2>
<p>Per-kernel statistics across the captured ROI. <b>Total</b> and <b>Share</b> are
the headline budget figures; <b>Min/P50/P95/Max</b> show the per-call distribution
and surface kernels with bimodal or long-tail behavior; <b>StdDev</b> highlights
launch-time variance.</p>
{_kernel_detail_table(overall_detail, top=15)}
<h3>11.1 Per-call duration distribution</h3>
<p style="color:var(--muted);font-size:12px">Range plot: grey line spans min→max, blue band is the P50→P95 interquartile range, purple dot marks the mean. A wide grey line with a tight blue band = mostly fast, with occasional outliers; a wide blue band = consistent variability across calls.</p>
{_kernel_distribution_svg(overall_detail, top_n=8)}
<h3>11.2 Category breakdown</h3>
<p style="color:var(--muted);font-size:12px">Aggregated by coarse kernel role (GEMM / Attention / RMSNorm / RoPE / KV-cache / Sampling / Elementwise / Other). Useful as a sanity check: in steady-state decode, GEMM should dominate and KV-cache writes should be small.</p>
{_category_breakdown_table(overall_detail)}
</section>

<section id="findings">
<h2>12. Findings</h2>
{findings_html}
</section>

<section id="troubleshoot">
<h2>13. Troubleshooting reference</h2>
{troubleshooting_table}
<h3>Trace size mitigations</h3>
<ul>
<li>Tighten the ROI: short <code>/start_profile</code> &rarr; <code>/stop_profile</code> window.</li>
<li>Drop <code>--chrome-sycl-logging</code> if you only need kernel-level timing.</li>
<li>Use <code>--include-kernels=&lt;substr&gt;</code> to filter by kernel-name substring (Level Zero only).</li>
</ul>
<hr>
<p style="color:var(--muted);font-size:12px">Auto-generated from <code>{_esc(os.path.basename(engine_path)) if engine_path else "(none)"}</code>. Iteration boundaries: inter-kernel gap &gt; {GAP_US} µs. Dense-iter threshold: ≥ {DENSE_MIN_KERNELS} kernels.</p>
</main>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    if len(sys.argv) > 1:
        result_dir = sys.argv[1]
    else:
        result_dir = os.environ.get("RESULT_DIR", "")
    if not result_dir or not os.path.isdir(result_dir):
        print(f"generate_report.py: result_dir not found: {result_dir!r}", file=sys.stderr)
        return 2

    engine_path, file_summaries = _pick_engine_trace(result_dir)
    if not engine_path:
        # Still write a minimal report so the user knows the run produced no GPU events
        out = os.path.join(result_dir, "unitrace_vllm_report.html")
        with open(out, "w") as f:
            f.write(
                "<!doctype html><meta charset=utf-8><title>unitrace report</title>"
                f"<style>{CSS}</style><body><main>"
                "<section><h2>No GPU events in this capture</h2>"
                "<p>None of the <code>python.*.json</code> files in this result dir "
                "contain <code>gpu_op</code> events. Likely causes: ROI gate never opened, "
                "<code>ITT available: False</code> in the launcher log, or unitrace was "
                "killed with <code>kill -9</code>.</p>"
                f"<p>Result dir: <code>{_esc(result_dir)}</code></p>"
                f"{_file_table(file_summaries)}"
                "</section></main>"
            )
        print(f"generate_report.py: wrote {out} (no GPU events)")
        return 0

    ev = _load_trace(engine_path)
    gpu = [e for e in ev if e.get("cat") == "gpu_op" and e.get("ph") == "X"]
    has_itt = any(
        "itt" in str(e.get("cat", "")).lower() or "itt" in str(e.get("name", "")).lower()
        for e in ev
    )

    iters_all = _segment_iters(gpu)
    iters_dense = [it for it in iters_all if len(it) >= DENSE_MIN_KERNELS]

    overall_top = _kernel_totals(gpu)
    overall_detail = _kernel_detailed(gpu)
    prefill_top = _kernel_totals(iters_dense[0]) if iters_dense else []
    decode_top = _kernel_totals([e for it in iters_dense[1:] for e in it]) if len(iters_dense) > 1 else []

    skill_dir = Path(__file__).resolve().parent.parent.as_posix()
    html = build_html(
        result_dir=result_dir,
        file_summaries=file_summaries,
        engine_path=engine_path,
        gpu_events=gpu,
        iters_all=iters_all,
        iters_dense=iters_dense,
        overall_top=overall_top,
        prefill_top=prefill_top,
        decode_top=decode_top,
        overall_detail=overall_detail,
        has_itt=has_itt,
        skill_dir=skill_dir,
    )

    out = os.path.join(result_dir, "unitrace_vllm_report.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"generate_report.py: wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
