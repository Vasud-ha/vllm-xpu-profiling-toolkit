# Resolving `__itt_resume` / `__itt_pause` from a vLLM process under unitrace

## The wrong assumption: `libittnotify.so`

The first thing you'd try is opening `libittnotify.so` with ctypes (this is what `serve_with_vtune.py` does, because oneAPI VTune ships `/opt/intel/oneapi/vtune/latest/lib64/libittnotify.so`).

Inside the `intel/vllm:0.14.1-xpu` container, **there is no `libittnotify.so`**:

```
$ find /opt/intel /path/to/pti-gpu -name 'libittnotify*'
.../pti-gpu/tools/unitrace/build/ittapi/include/libittnotify.h
.../pti-gpu/tools/unitrace/build/ittapi/bin/libittnotify.a   # static
.../pti-gpu/tools/unitrace/build/libittnotify.a              # static
```

Only the static `.a` exists. If you build a private `libittnotify.so` from `pti-gpu/tools/unitrace/build/ittapi`, vLLM can dlopen it and call `__itt_resume`/`__itt_pause` — but those calls go to a SECOND copy of ITT that has no connection to whatever collector is running. They become silent no-ops.

## The right model: unitrace exports ITT itself

unitrace's CMake compiles `ittnotify_static.c` into `libunitrace_tool.so` and forwards `__itt_resume` / `__itt_pause` from there. When `unitrace` launches a target, it injects `libunitrace_tool.so` into the target process via `LD_PRELOAD` (or equivalent). After injection, the target process has those ITT symbols available in its global symbol table — they're the ones unitrace's collector listens on.

```
$ nm -D libunitrace_tool.so | grep -E '__itt_resume|__itt_pause'
00000000000fc940 T __itt_pause
00000000000fcaf0 T __itt_pause_scoped
00000000000fcb00 T __itt_resume
00000000000fccb0 T __itt_resume_scoped
```

## Resolution strategy used by `serve_with_unitrace.py`

The wrapper tries three sources in order:

```python
# 1. RTLD_DEFAULT — unitrace already injected the symbols
ctypes.CDLL(None)

# 2. libunitrace_tool.so by absolute path
#    (covers the rare case where the wrapper runs OUTSIDE unitrace, e.g.
#    while you're prototyping or running it under VTune instead)
ctypes.CDLL(os.environ.get(
    "UNITRACE_TOOL_LIB",
    "/opt/pti-gpu/tools/unitrace/build/libunitrace_tool.so"
), mode=ctypes.RTLD_GLOBAL)

# 3. find_library("ittnotify") — for VTune installs that DO have a .so
ctypes.CDLL(ctypes.util.find_library("ittnotify"))
```

For each candidate, we resolve `__itt_resume` and `__itt_pause` with `getattr(lib, name)`. The first lib that has both wins.

### Why option 2 uses `RTLD_GLOBAL`

If we open `libunitrace_tool.so` ourselves (option 2), we want its symbols to merge into the global namespace so any subsequent `dlsym(RTLD_DEFAULT, "__itt_resume")` finds them. With the default `RTLD_LOCAL`, the symbols stay scoped to that handle.

When unitrace itself injects (option 1), it already does the equivalent — that's why option 1 works without extra flags.

### Why we don't bother with `libittnotify.so` first

In the unitrace flow it's unhelpful: even if a VTune-shipped `libittnotify.so` exists, calling its `__itt_resume` does nothing useful unless VTune is the collector. Putting it last keeps it as a fallback for hybrid setups (e.g., running `serve_with_unitrace.py` outside unitrace for testing — RTLD_DEFAULT will fail; libunitrace_tool.so probably also fails to provide a working collector; libittnotify.so at least won't crash).

## Verifying ITT is wired up

The launcher prints:

```
2026-06-10 12:15:39 [unitrace.vllm] ITT available: True
```

You want **`True`** in the API-server process AND in the `EngineCore_DP0` subprocess. If only one says True, the worker process didn't inherit the injected lib (rare — would need `--follow-child-process 0`).

## What ITT integration unlocks

- `--chrome-itt-logging` lights up an "ITT" track in the Chrome trace, so `__itt_resume`/`__itt_pause` calls show up as visible markers framing the kernels.
- `--start-paused` makes unitrace observe `__itt_resume` to begin recording. Without `--start-paused`, ITT pause/resume just toggles whether the existing collection is "active" — it doesn't reduce the captured-data window for the WHOLE trace, just hides events from device-timing summaries while paused.

For this skill, **always combine `--start-paused` with the ITT calls** — that's what gives you a tight ROI rather than just a flag in an otherwise-full trace.
