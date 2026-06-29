#!/usr/bin/env python3
"""
serve_with_unitrace.py - Launch vLLM OpenAI server with ITT-driven unitrace ROI.

Adds /start_profile and /stop_profile FastAPI routes that fire ITT
__itt_resume() / __itt_pause(). When the server is wrapped in
`unitrace --start-paused`, those routes bracket the unitrace ROI.

Any HTTP client works: `curl`, the OpenAI SDK, `vllm bench serve`,
`vllm.entrypoints.cli.benchmark`, etc. As long as the client POSTs to
/start_profile before traffic and /stop_profile after, the captured trace
covers exactly that window.

Compatible with the same vLLM v1 path as serve_with_vtune.py — the ITT
calls run inside the EngineCore subprocess too because we register the
routes in the FastAPI app and patch the v1 Worker's execute_model so the
worker process honors the gate as well.
"""

import logging
import os
import sys
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [unitrace.vllm] %(message)s",
)
log = logging.getLogger("unitrace.vllm")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------- ITT resolution ----------
# unitrace links libittnotify statically into libunitrace_tool.so and injects
# that .so into the target process, so __itt_resume / __itt_pause are
# resolvable from RTLD_DEFAULT once unitrace launches us. We try, in order:
#   1. RTLD_DEFAULT (unitrace already injected the symbols)
#   2. libunitrace_tool.so by absolute path (fallback)
#   3. Standard libittnotify.so search (oneAPI VTune layout)
import ctypes
import ctypes.util


def _resolve_itt():
    candidates = []
    # 1. process's own symbol table
    try:
        candidates.append(ctypes.CDLL(None))
    except OSError:
        pass
    # 2. unitrace_tool.so
    unitrace_tool = os.environ.get(
        "UNITRACE_TOOL_LIB",
        "/opt/pti-gpu/tools/unitrace/build/libunitrace_tool.so",
    )
    if os.path.exists(unitrace_tool):
        try:
            candidates.append(ctypes.CDLL(unitrace_tool, mode=ctypes.RTLD_GLOBAL))
        except OSError as e:
            log.debug("CDLL(%s) failed: %s", unitrace_tool, e)
    # 3. libittnotify.so (VTune install)
    name = ctypes.util.find_library("ittnotify")
    if name:
        try:
            candidates.append(ctypes.CDLL(name))
        except OSError:
            pass
    for path in (
        "/opt/intel/oneapi/vtune/latest/lib64/libittnotify.so",
        "/opt/intel/vtune_profiler/lib64/libittnotify.so",
    ):
        if os.path.exists(path):
            try:
                candidates.append(ctypes.CDLL(path))
            except OSError:
                pass
    for lib in candidates:
        try:
            r = lib.__itt_resume
            p = lib.__itt_pause
            r.restype = None
            r.argtypes = []
            p.restype = None
            p.argtypes = []
            return r, p
        except AttributeError:
            continue
    return None, None


_itt_resume_fn, _itt_pause_fn = _resolve_itt()


def itt_available() -> bool:
    return _itt_resume_fn is not None


def itt_resume():
    if _itt_resume_fn:
        _itt_resume_fn()


def itt_pause():
    if _itt_pause_fn:
        _itt_pause_fn()


if itt_available():
    itt_pause()  # belt-and-suspenders: ensure paused at startup
log.info("ITT available: %s", itt_available())

# Sentinel file shared across the API-server process and EngineCore subprocess.
# /start_profile creates it; /stop_profile removes it. The Worker patch reads
# it to gate per-step itt_resume/pause inside the worker process.
ROI_GATE_PATH = os.environ.get(
    "UNITRACE_ROI_GATE", f"/tmp/unitrace_roi_gate.{os.getpid()}"
)
log.info("ROI gate path: %s", ROI_GATE_PATH)
# Re-export so the spawned worker sees the same path.
os.environ["UNITRACE_ROI_GATE"] = ROI_GATE_PATH

# Lock so concurrent /start_profile or /stop_profile calls don't race the
# gate file or the ITT counter.
_gate_lock = threading.Lock()


def _gate_open() -> bool:
    return os.path.exists(ROI_GATE_PATH)


def _open_gate():
    with _gate_lock:
        try:
            with open(ROI_GATE_PATH, "w") as f:
                f.write("1")
        except OSError as e:
            log.error("failed to create gate file: %s", e)


def _close_gate():
    with _gate_lock:
        try:
            os.unlink(ROI_GATE_PATH)
        except FileNotFoundError:
            pass
        except OSError as e:
            log.error("failed to remove gate file: %s", e)


# ---------- FastAPI route attachment ----------

def _build_profile_router():
    from fastapi import APIRouter
    from fastapi.responses import JSONResponse

    router = APIRouter()

    @router.post("/start_profile")
    async def start_profile():  # noqa: F811
        log.info("start_profile: opening ROI gate + itt_resume")
        _open_gate()
        itt_resume()
        return JSONResponse({"status": "profiling", "gate": ROI_GATE_PATH})

    @router.post("/stop_profile")
    async def stop_profile():  # noqa: F811
        log.info("stop_profile: itt_pause + closing ROI gate")
        # Best-effort device sync so the trace contains kernel completions.
        try:
            import torch
            if torch.xpu.is_available():
                torch.xpu.synchronize()
        except Exception as e:
            log.debug("xpu.synchronize skipped: %s", e)
        itt_pause()
        _close_gate()
        return JSONResponse({"status": "stopped"})

    return router


def _patch_build_app():
    """Wrap api_server.build_app so /start_profile and /stop_profile exist
    regardless of whether vLLM's --profiler flag is set."""
    from vllm.entrypoints.openai import api_server as _api

    orig_build_app = _api.build_app

    def wrapped(*args, **kwargs):
        app = orig_build_app(*args, **kwargs)
        app.include_router(_build_profile_router())
        log.info("Attached /start_profile and /stop_profile (ITT-driven)")
        return app

    _api.build_app = wrapped


# ---------- Worker patch (mirrors serve_with_vtune.py logic) ----------

def _patch_worker():
    """Inside the EngineCore subprocess, gate per-step ITT around
    Worker.execute_model so kernels outside the curl window aren't captured
    even if the outer ITT pause was missed (belt-and-suspenders)."""
    import torch
    try:
        from vllm.v1.worker.gpu_worker import Worker as WorkerV1
    except ImportError as e:
        log.warning("v1 worker not importable, skipping worker patch: %s", e)
        return

    orig = WorkerV1.execute_model

    def wrapped(self, *args, **kwargs):
        active = _gate_open()
        if active:
            itt_resume()
        try:
            return orig(self, *args, **kwargs)
        finally:
            if active:
                if torch.xpu.is_available():
                    torch.xpu.synchronize()
                itt_pause()

    WorkerV1.execute_model = wrapped
    log.info("Patched vllm.v1.worker.gpu_worker.Worker.execute_model")


_patch_build_app()
_patch_worker()


# ---------- Hand off to vLLM ----------
#
# We do NOT use runpy here because runpy re-executes the api_server module
# under run_name="__main__", which discards our build_app monkey-patch (the
# module gets re-loaded fresh in the new namespace). Instead we call
# run_server directly, after replicating api_server's __main__ block.

if __name__ == "__main__":
    import uvloop
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.utils import cli_env_setup

    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI API server with unitrace ROI hooks."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))
