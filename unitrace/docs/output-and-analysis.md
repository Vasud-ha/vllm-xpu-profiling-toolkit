# unitrace output: what you get, what to do with it

## Files produced

After running `run_unitrace_vllm.sh` and stopping the server, the result directory contains:

```
unitrace_results/<timestamp>_<model>/
├── python.<api_server_pid>.json    # mostly idle FastAPI / asyncio
├── python.<engine_core_pid>.json   # this one has the GPU kernels
├── sh.<launcher_pid>.json          # tiny, the bash wrapper
└── roi_gate                        # absent if /stop_profile was called
```

unitrace writes one Chrome-trace JSON per process. With `--follow-child-process 1` (default), it traces both the API-server process and the spawned EngineCore subprocess — the latter is where vLLM v1 runs the model, so that's the file with kernels.

If `-d` was passed, the device-timing summary is printed to stdout at process exit:

```
=== Device Timing Summary ===

                Total Execution Time (ns):          475123456

                Kernel,                Calls,    Time (ns), Time (%), Average (ns), Min (ns), Max (ns)
                ---,                       ---,         ---,      ---,          ---,      ---,     ---
                ...
```

A summary printed for every traced process — interpret only the EngineCore one for GPU timing.

## Quick health check

```python
import json, collections, glob, os
d = sorted(glob.glob("unitrace_results/*/"))[-1]
for p in glob.glob(os.path.join(d, "python.*.json")):
    data = json.load(open(p))
    ev = data.get("traceEvents", data) if isinstance(data, dict) else data
    cats = collections.Counter(e.get("cat","-") for e in ev)
    gpu = [e for e in ev if e.get("cat")=="gpu_op" and e.get("ph")=="X"]
    span = 0
    if gpu:
        ts = sorted(e["ts"] for e in gpu if "ts" in e)
        span = (ts[-1]-ts[0])/1e6 if ts else 0
    print(f"{os.path.basename(p)}: {len(ev)} events, gpu={len(gpu)}, span={span:.2f}s")
```

What healthy output looks like for a tight ROI (one short chat completion):

```
python.7954.json: 200 events, gpu=0, span=0.00s          # API server, no kernels
python.8236.json: 29064 events, gpu=1164, span=0.48s     # EngineCore, kernels in tight window
```

If `span` is close to the server lifetime instead of the inference window, the ROI gate didn't engage — re-check the launch log for `ITT available: True` in BOTH processes.

If `gpu=0` in every file, unitrace didn't see Level Zero kernels at all — usually means the build was missing `BUILD_WITH_L0=1` (default is on) or the workload didn't actually exercise GPU.

## Top categories you'll see

- `gpu_op` — Level Zero kernel executions on the device. Names look like `submit`, the demangled SYCL kernel, or the L0 entry-point name depending on capture mode.
- `cpu_op` — host-side L0/SYCL/UR API calls (`urKernelSetArgValue`, `zeCommandListAppendLaunchKernel`, etc.). High frequency is normal.
- `Flow_H2D_*` — host → device copy correlation.
- `Flow_D2H_*` — device → host copy correlation.
- (with `--chrome-itt-logging`) ITT track shows resume/pause markers and any explicit ITT regions.

## Loading the trace

### Chrome / chromium

`chrome://tracing` → Load → pick one `python.*.json`. Multiple files = multiple loads / can't easily merge.

### Perfetto (recommended for unitrace)

[ui.perfetto.dev](https://ui.perfetto.dev/) → Open trace file → drop in any `python.*.json`. Perfetto handles unitrace's Chrome-trace dialect well and gives a much better Flow-event UI than Chrome.

For a multi-process view, Perfetto can open them sequentially in different tabs, but it cannot merge JSONs. To get a single view, concatenate by hand (events are in `traceEvents` with `pid`/`tid` fields, so dumb-concat usually works):

```python
import json, glob, os
out = {"traceEvents": []}
for p in sorted(glob.glob("unitrace_results/*/python.*.json"))[-2:]:
    out["traceEvents"].extend(json.load(open(p))["traceEvents"])
json.dump(out, open("merged.json","w"))
```

This is fragile if process IDs collide (they shouldn't here — different PIDs).

### Command-line summary scripts (pti-gpu)

`pti-gpu/tools/unitrace/scripts/summary/` ships post-processing helpers:

```bash
python pti-gpu/tools/unitrace/scripts/summary/<script>.py path/to/python.<pid>.json
```

(Inventory the directory — names change between releases.)

## What to look at first

1. **Where is GPU time spent?** — read the device-timing summary. Top kernels by % time tell you what to optimize.
2. **Are kernels back-to-back, or are there gaps?** — open the trace in Perfetto and zoom in on a few-millisecond span of the ROI. Visible gaps between kernels = host-bound overhead (sampling, scheduling, sync).
3. **Are there H2D/D2H copies you didn't expect?** — Flow_H2D events. In a steady-state decode, you should see almost none.
4. **Per-iteration shape.** — In the EngineCore trace, find one full `execute_model` cycle (between two ITT resume markers if `--chrome-itt-logging` is on). Its kernel sequence is the inner loop you're optimizing.

## Trace size warnings

`--chrome-kernel-logging` + `--chrome-sycl-logging` for a 30-second benchmark of a 7B model can hit hundreds of MB. Chrome refuses traces over ~256 MB; Perfetto handles >1 GB but slowly. Mitigations:

- Tighten the ROI: shorter `/start_profile` → `/stop_profile` window.
- Drop `--chrome-sycl-logging` if you only care about kernel-level timing.
- Use `--include-kernels=<comma-list>` to filter to a few kernel-name substrings (Level Zero only).

## Gotcha: trace files arrive AFTER the process exits

unitrace writes the JSONs during teardown. If you `kill -9` the launcher, you'll get truncated or missing JSONs. Always Ctrl-C the unitrace launcher (one Ctrl-C is enough — give it a few seconds to flush). The device-timing summary printing to stdout is a good sign that teardown completed.
