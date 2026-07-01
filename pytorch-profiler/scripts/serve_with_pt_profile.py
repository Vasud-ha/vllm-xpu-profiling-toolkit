#!/usr/bin/env python3
"""
serve_with_pt_profile.py - Launch vLLM OpenAI server with phase-aware
PyTorch profiler annotations.

This is the PyTorch-profiler analogue of serve_with_vtune.py. vLLM's official
profiling endpoints (POST /start_profile, /stop_profile) drive the actual
torch.profiler instance; this wrapper only ADDS structured `record_function`
spans around Worker.execute_model so prefill / decode / mixed steps are
identifiable in Perfetto without forking vLLM.

Why not also drive torch.profiler from here?
  vLLM already does that when VLLM_TORCH_PROFILER_DIR is set. Running a second
  profiler on top of it would either fight for the CUPTI/XPU activity stream
  or double-trace. So we sit on top of vLLM's profiler and only contribute
  human-readable ROI labels.

Phase classification mirrors serve_with_vtune.py:
  prefill | decode | mixed | cache_hit | empty

Environment variables (read at process start):
  VLLM_TORCH_PROFILER_DIR  vLLM <= 0.14 profiler gate. Read at engine init.
                           Ignored on vLLM >= 0.17, which uses the
                           `--profiler-config` CLI arg with a ProfilerConfig
                           JSON blob (e.g. `{"profiler":"torch",
                           "torch_profiler_dir":"/abs/path"}`). Either mechanism
                           must be provided by the launcher (run_pt_profile_vllm.sh
                           sets both) or /start_profile 404s.
  PT_PHASE                 prefill | decode | mixed | both  (default: both)
                           Controls which phases get a record_function span.
                           Steps not in the selected phase still execute, they
                           just don't get an ROI annotation. (We never gate
                           the actual model call on the profiler.)
  PT_LABEL_EVERY_STEP      1 (default) -> annotate every selected step.
                           0 -> annotate first/last step of each contiguous
                           run only. Reduces label spam in long captures.

Launch (via run_pt_profile_vllm.sh):
  VLLM_TORCH_PROFILER_DIR=/path python serve_with_pt_profile.py \
        --model <id> --port 8000 --enforce-eager
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [pt.vllm] %(message)s",
)
log = logging.getLogger("pt.vllm")


# ---------- Environment surface ----------

PROFILE_DIR = os.environ.get("VLLM_TORCH_PROFILER_DIR", "").strip()
# vLLM >= 0.17 also accepts --profiler-config on the CLI; peek at argv so we
# don't emit a scary "not set" warning when the launcher used the new flag.
_has_profiler_config = any(
    a == "--profiler-config" or a.startswith("--profiler-config=")
    for a in sys.argv
)
if PROFILE_DIR:
    os.makedirs(PROFILE_DIR, exist_ok=True)
    log.info("VLLM_TORCH_PROFILER_DIR=%s (vLLM <= 0.14 gate)", PROFILE_DIR)
elif _has_profiler_config:
    log.info(
        "Using --profiler-config CLI arg (vLLM >= 0.17 gate); "
        "VLLM_TORCH_PROFILER_DIR unset is expected"
    )
else:
    log.warning(
        "Neither VLLM_TORCH_PROFILER_DIR nor --profiler-config is set; "
        "/start_profile and /stop_profile will 404. Pass one of them BEFORE "
        "launching this script."
    )

PHASE = os.environ.get("PT_PHASE", "both").strip().lower()
if PHASE not in {"prefill", "decode", "mixed", "both"}:
    log.warning("PT_PHASE=%r invalid; falling back to 'both'", PHASE)
    PHASE = "both"

LABEL_EVERY_STEP = os.environ.get("PT_LABEL_EVERY_STEP", "1") == "1"


# ---------- Phase classification (mirrors serve_with_vtune.py) ----------

def _classify_v1(scheduler_output) -> tuple:
    """v1 SchedulerOutput -> (phase, batch_size, scheduled_tokens).

    phase is one of: prefill | decode | mixed | cache_hit | empty.
    """
    new = getattr(scheduler_output, "scheduled_new_reqs", None) or []
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None) or []
    sched = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}

    def _toks_for(reqs):
        total = 0
        for r in reqs:
            req_id = getattr(r, "req_id", None)
            if req_id is not None:
                total += sched.get(req_id, 0)
        return total

    new_tok = _toks_for(new)
    cac_tok = _toks_for(cached)
    bs = len(new) + len(cached)
    tok = new_tok + cac_tok

    if not new and not cached:
        return ("empty", bs, tok)
    if new and new_tok == 0 and not cached:
        return ("cache_hit", bs, tok)
    if new and cached:
        return ("mixed", bs, tok)
    if new:
        return ("prefill", bs, tok)
    return ("decode", bs, tok)


def _classify_v0(req) -> tuple:
    md_list = getattr(req, "seq_group_metadata_list", None) or []
    bs = len(md_list)
    is_prefill = any(getattr(md, "is_prompt", False) for md in md_list)
    return ("prefill" if is_prefill else "decode" if bs else "empty", bs, 0)


def _phase_selected(phase: str) -> bool:
    if phase in ("cache_hit", "empty"):
        return False
    if PHASE == "both":
        return phase in ("prefill", "decode", "mixed")
    if PHASE == "mixed":
        return phase == "mixed"
    if PHASE == "prefill":
        return phase in ("prefill", "mixed")
    if PHASE == "decode":
        return phase == "decode"
    return False


def _step_label(phase: str, bs: int, tok: int) -> str:
    return f"{phase} bs={bs} tok={tok}"


# ---------- Patching ----------

_last_phase = None  # for LABEL_EVERY_STEP=0 transition detection


def _patch_worker(worker_cls, classify):
    """Wrap Worker.execute_model with a torch.profiler.record_function span.

    Spans are nested INSIDE whatever profiling window vLLM has opened via
    /start_profile. When the profiler is off, record_function is a cheap
    no-op (a few ns), so it's safe to leave the patch on permanently.
    """
    import torch
    from torch.profiler import record_function

    orig = worker_cls.execute_model

    def wrapped(self, *args, **kwargs):
        global _last_phase
        req = (args[0] if args
               else kwargs.get("execute_model_req")
               or kwargs.get("scheduler_output"))
        try:
            phase, bs, tok = classify(req) if req is not None else ("empty", 0, 0)
        except Exception as e:
            log.debug("phase classify failed: %s", e)
            phase, bs, tok = ("empty", 0, 0)

        annotate = _phase_selected(phase)
        if annotate and not LABEL_EVERY_STEP:
            # Only label phase boundaries: skip annotation if same as last.
            if phase == _last_phase:
                annotate = False

        if not annotate:
            _last_phase = phase
            return orig(self, *args, **kwargs)

        label = _step_label(phase, bs, tok)
        _last_phase = phase

        with record_function(f"vllm.{label}"):
            return orig(self, *args, **kwargs)

    worker_cls.execute_model = wrapped
    log.info(
        "Patched %s.execute_model (phase=%s every_step=%s)",
        worker_cls.__module__ + "." + worker_cls.__name__,
        PHASE, LABEL_EVERY_STEP,
    )


def _try_patch():
    """Patch the GPU Worker class. Prefer v1 (default in vLLM >= 0.10,
    including intel/vllm:0.14.1-xpu); fall back to v0 only if v1 is absent."""
    try:
        from vllm.v1.worker.gpu_worker import Worker as WorkerV1  # type: ignore
        _patch_worker(WorkerV1, _classify_v1)
        return
    except ImportError as e:
        log.debug("v1 worker not available: %s", e)

    try:
        from vllm.worker.worker import Worker as WorkerV0  # type: ignore
        _patch_worker(WorkerV0, _classify_v0)
        return
    except ImportError as e:
        log.debug("v0 worker not available: %s", e)

    log.warning(
        "Could not patch any vLLM Worker class - phase labels will be missing. "
        "Trace will still capture vLLM's default annotations."
    )


_try_patch()
log.info(
    "Phase annotation: phase=%s every_step=%s profile_dir=%s",
    PHASE, LABEL_EVERY_STEP, PROFILE_DIR or "<unset>",
)


# ---------- Hand off to vLLM's normal entrypoint ----------

if __name__ == "__main__":
    # Same runpy trick as serve_with_vtune.py: vLLM v1 doesn't export `main`
    # from api_server, so invoke __main__ via runpy.
    import runpy
    sys.argv[0] = "vllm.entrypoints.openai.api_server"
    runpy.run_module(
        "vllm.entrypoints.openai.api_server",
        run_name="__main__",
        alter_sys=True,
    )
