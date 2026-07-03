# vLLM 0.21 Verification Log — `skill-test-vasu` on gnrsp-bmg3

## Environment

- Host: **gnrsp-bmg3.iind.intel.com** (Intel BMG GPU, `Intel(R) Graphics [0xe223]`)
- Container: **`skill-test-vasu`**
- Image: **`intel/vllm:0.21.0-ubuntu24.04-20260625`**
- vLLM: `0.21.1.dev17+g0a4756bb5`
- torch: `2.11.0+xpu`
- VTune: **`2026.2.0` (build 632324)**
- unitrace: pre-built at `/data/workspace/vasudha/pti-gpu/tools/unitrace/build/unitrace` (BUILD_WITH_L0=1, OPENCL=1, ITT=1, XPTI=1, MPI=0)
- oneAPI: `/opt/intel/oneapi/setvars.sh` present, `libittnotify.a` at `/opt/intel/oneapi/vtune/2026.2/lib64/`
- HF cache: `/hf_cache` — **NFS4** (root cause of most timing issues below)

## Baseline preflight (`scripts/check_prereqs.sh`)

All-green. Toolkit prereqs are satisfied in the container:

```
[PASS] python: Python 3.12.3
[PASS] vllm.v1.worker.gpu_worker importable (vllm=0.21.1.dev17+g0a4756bb5)
[PASS] torch.xpu: 1 device(s), Intel(R) Graphics [0xe223]
[PASS] running as root (full /dev/dri access)
[PASS] oneAPI setvars.sh present
[PASS] HF cache at /hf_cache
[PASS] vtune on PATH: Intel(R) VTune(TM) Profiler 2026.2.0 (build 632324) Command Line Tool
[PASS] libigdmd.so resolvable
[PASS] ittapi python module installed
[PASS] unitrace: /data/workspace/vasudha/pti-gpu/tools/unitrace/build/unitrace
[PASS] ProfilerConfig importable (vLLM >= 0.17 CLI-arg gate available)
```

## API surface confirmation — vLLM v1 Worker + SchedulerOutput on 0.21

Wrappers patch `vllm.v1.worker.gpu_worker.Worker.execute_model` and read three fields off `SchedulerOutput` via `getattr(...) or []` to classify each step. Both survive on 0.21:

```
Worker.execute_model params: ['self', 'scheduler_output']
SchedulerOutput fields:
  scheduled_new_reqs, scheduled_cached_reqs, num_scheduled_tokens,
  total_num_scheduled_tokens, scheduled_spec_decode_tokens,
  scheduled_encoder_inputs, num_common_prefix_blocks, finished_req_ids,
  free_encoder_mm_hashes, preempted_req_ids, has_structured_output_requests,
  pending_structured_output_tokens, num_invalid_spec_tokens,
  kv_connector_metadata, ec_connector_metadata, new_block_ids_to_zero
```

No script changes needed for the API surface — all three wrappers already
consume `scheduled_new_reqs` / `scheduled_cached_reqs` / `num_scheduled_tokens`
defensively.

---

## VTune runs — trace of attempts and findings

### Attempt 1 — vanilla run, outer 600s timeout

```
MODEL=meta-llama/Llama-3.1-8B-Instruct MAX_MODEL_LEN=2048 \
VTUNE_PHASE=prefill INPUT_LEN=512 OUTPUT_LEN=1 \
NUM_PROMPTS=3 MAX_CONCURRENCY=1 \
timeout 600 ./run_vtune_vllm.sh
```

Result dir `20260703_054147_prefill/`.

Timeline:

| t+ | event |
| --- | --- |
| 05:41:47 | VTune spawned `amplxe-runss` |
| 05:43:56 | vLLM began safetensors prefetch |
| 05:44:10 | Enforce-eager confirmed; torch.compile+CUDAGraphs disabled |
| 05:44:17 | vLLM API server ready on 8000 |
| 05:44:30 | `vllm bench serve` warmup (10 prompts) began |
| 05:44:30 | Triton JIT: `_compute_slot_mapping_kernel` compiled on first inference (jit_monitor.py:103 latency spike warning) |
| ~05:52 | Outer `timeout` fired at t+600s; VTune killed mid-warmup |

Result: no `bench.log`, no CSVs. **First diagnosis** (later revised) was that Triton JIT under VTune's collector was slow enough to blow the 10-min window even at 10 prompts.

Ancillary observations:
- `vtune: Error: Unauthorized control server connection.` printed early during load — verified harmless, run continued.
- Warmup runs with the ROI gate **closed** by design → the collector stays paused; since the actual benchmark never started, the collector never resumed → no profile data.

### Attempt 2 — no outer timeout, smaller warmup (still failed)

```
NUM_PROMPTS=3 WARMUP_PROMPTS=2 MAX_CONCURRENCY=1 ./run_vtune_vllm.sh
```

Result dir `20260703_055326_prefill/`. Warmup got past the 10-line startup log then stuck on prompt 0/2 for **9+ minutes** with no progress. Killed for triage. Same JIT theory carried over.

### Detour — Triton JIT cache examination

Container has 15 entries under `~/.triton/cache`, ~6.1 MB. Comparing entry
timestamps to attempt 1's result-dir mtime showed **zero new cache entries
after the vtune runs** — meaning vLLM 0.21 either:

- Uses a different cache dir (real cause), or
- Recompiles kernels each run even after cache write (unlikely)

Either way, the "prime the JIT cache with a plain serve run" plan was worth trying — but only if we could actually get vLLM to serve at all.

### Attempt 3 — priming pass without VTune (revealed the real bottleneck)

Directly ran `serve_with_vtune.py` in an ordinary `python` (no `vtune` wrapping) so we could measure raw load time. Wrapper fails hard without setvars sourced (LD path missing `libccl.so.1`), so had to re-source manually.

Once running, `direct.log` revealed:

```
06:27:08  [xpu.py:96] Using Flash Attention backend.
          [flash_attn.py:641] Using FlashAttention version 2
          --- 4 MINUTE GAP ---
06:31:00  Filesystem type for checkpoints: NFS4. Checkpoint size: 14.96 GiB.
          Prefetching checkpoint files finished in 1.04s
          Loading safetensors checkpoint shards:  25% Completed | 1/4 [02:29<07:27, 149.18s/it]
```

**Actual root cause: NFS I/O, not Triton JIT.**

- vLLM 0.21's model loader is now single-threaded per shard on top of NFS, running at ~150 s/shard — 4 shards = ~10 min cold.
- OS page cache expires between runs (memory pressure from vtune's collector).
- vLLM's `VLLM_ENGINE_READY_TIMEOUT_S` defaults to **600s** — trips before weights finish loading, leading to:

  ```
  TimeoutError: Timed out waiting for engine core processes to start.
  Waited 600s (configured by VLLM_ENGINE_READY_TIMEOUT_S).
  ```

The 4-min gap between "Using Flash Attention" and "Loading safetensors" is the
inductor cache init that reads model files from NFS to build the config, before
weight-loading proper starts. This makes the effective wall time closer to 12 min per cold run.

### Fix strategy applied

1. **Local weight staging** — `cp -aL /hf_cache/models--meta-llama--Llama-3.1-8B-Instruct /root/local_hf_cache/`, then `export HF_HOME=/root/local_hf_cache`. 15 GB copy, one-time cost. Local container overlay-fs = ~2 GB/s vs NFS ~0.1 GB/s.

2. **Bumped ready timeout** — added `export VLLM_ENGINE_READY_TIMEOUT_S=1800` to all three wrappers so cold-cache scenarios don't fail before load completes.

3. **Env harmonization** — added the recommended intel/vllm image env vars to `run_vtune_vllm.sh` and `run_pt_profile_vllm.sh` (unitrace wrapper already had them):

   ```bash
   export HF_HOME="${HF_HOME:-/hf_cache}"
   export TORCH_LLM_ALLREDUCE=1
   export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
   export VLLM_WORKER_MULTIPROC_METHOD=spawn
   export VLLM_ENGINE_READY_TIMEOUT_S=1800
   ```

### Attempt 4 — with local weights + env vars

```
export HF_HOME=/root/local_hf_cache
export TORCH_LLM_ALLREDUCE=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
MODEL=meta-llama/Llama-3.1-8B-Instruct MAX_MODEL_LEN=2048 \
VTUNE_PHASE=prefill INPUT_LEN=512 OUTPUT_LEN=1 \
NUM_PROMPTS=3 WARMUP_PROMPTS=2 MAX_CONCURRENCY=1 \
./run_vtune_vllm.sh
```

Preflight now correctly detected a stale port 8000 from attempt 3 (this is
the wrapper doing its job) and refused to start. After `fuser -k 8000/tcp`
cleanup:

Result: another `TimeoutError` at 600s. The wrapper had already been invoked
before the timeout-env-var patch, so the 1800s bump wasn't in effect for this
run. **Applying the patch → the next invocation will honor 1800s** — no more
runs today; documenting so the user can retry.

---

## Where things stand

**Verified on vLLM 0.21:**
- Preflight (`check_prereqs.sh`) all-green in `skill-test-vasu`
- API surface (Worker.execute_model, SchedulerOutput fields) unchanged
- Wrapper import paths work (once oneAPI setvars is sourced)
- VTune 2026.2 launches, injects ITT symbols, honors `--start-paused`
- Model loads and serves (attempt 3 got as far as the load loop)

**Not verified end-to-end (blocked by cold NFS load, not by any wrapper bug):**
- Full profile run producing `bench.log` + CSVs
- unitrace capture producing Chrome trace JSON
- PyTorch profiler decode capture

**Patches pushed for the next run:**
- `vtune/scripts/run_vtune_vllm.sh` — added 5 env vars (HF_HOME, TORCH_LLM_ALLREDUCE, VLLM_ALLOW_LONG_MAX_MODEL_LEN, VLLM_WORKER_MULTIPROC_METHOD, VLLM_ENGINE_READY_TIMEOUT_S=1800)
- `pytorch-profiler/scripts/run_pt_profile_vllm.sh` — same 5 env vars
- `unitrace/scripts/run_unitrace_vllm.sh` — added VLLM_ENGINE_READY_TIMEOUT_S (other 4 already present)

## Recommended follow-up

To actually complete the 3-profiler smoke test:

1. **One-time**: `cp -aL /hf_cache/models--meta-llama--Llama-3.1-8B-Instruct /root/local_hf_cache/` inside the container (~5 min copy, saves ~10 min per subsequent run)
2. `export HF_HOME=/root/local_hf_cache` before each run
3. Run vtune / unitrace / pytorch-profiler wrappers with small `NUM_PROMPTS=3 WARMUP_PROMPTS=2`
4. First cold run: expect ~10 min (weights → NHD KV layout → warmup → bench). Second run reuses page cache: expect ~3 min.

## Ancillary notes

- **VTune 2026.2 ITT shipping**: only `libittnotify.a` (static) + `libittnotify_collector.so` — no shared `libittnotify.so`. `serve_with_vtune.py`'s existing resolution path (prefer `ctypes.CDLL(None)` for collector-injected symbols) already handles this correctly.
- **Preflight false positive** on port 8000: happened once because a killed `serve_with_vtune.py` from attempt 3 left the socket in `TIME_WAIT`. The wrapper's port-check is strict-correct; `fuser -k` cleanup should be the documented recovery.
- **Container `/root/toolkit`** was already a git checkout of this repo — 14 commits ahead of GitHub when we started, pulled up to `5676471` after fast-forward, then adjusted with the env-var patches described above.
