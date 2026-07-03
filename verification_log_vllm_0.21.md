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

---

## Session 2 — Fresh container on GPU BDF `0000:ba:00.0`

New pod spun up (`skill-test-vasu`, same image `intel/vllm:0.21.0-ubuntu24.04-20260625`), user manually installed **VTune 2026.0.0 (build 631999)**. Toolkit re-cloned at `d56f27f`. Preflight all-green.

### Attempt 5 — VTune E2E after clean install

```
export HF_HOME=/root/local_hf_cache
MODEL=meta-llama/Llama-3.1-8B-Instruct MAX_MODEL_LEN=2048 \
VTUNE_PHASE=prefill INPUT_LEN=512 OUTPUT_LEN=1 \
NUM_PROMPTS=3 WARMUP_PROMPTS=2 MAX_CONCURRENCY=1 \
./run_vtune_vllm.sh
```

**Weight-load went to fast path**:

```
08:01:23  Initializing V1 LLM engine (v0.21.1.dev17+g0a4756bb5)
08:01:27  Starting to load model
08:03:37  Loading weights took 2.56 seconds   [via local cache]
```

Weight load: **2.56 s**. NFS bottleneck fully resolved by local staging.

**Failure at KV-cache probe (this is the real blocker now):**

```
08:04:10  EngineCore failed to start.
          File .../triton/runtime/build.py:117 in _build
          subprocess.CalledProcessError:
            Command '['/opt/intel/oneapi/compiler/2025.3/bin/icpx',
                       '/tmp/tmp07quuk08/main.cpp', '-O3', '-shared',
                       ..., '-fsycl']' returned non-zero exit status 1.

          RuntimeError: llvm-foreach: Illegal instruction (core dumped)
          icpx: error: llvm-spirv command failed with exit code 254
          icpx: note: diagnostic msg: Error generating preprocessed source(s).

vtune: Error: [Instrumentation Engine]: ParentExecAppWithInjectorControl:
       Injector (6609) was terminated by a signal: 11
vtune: Collection failed.
```

### Root cause identified

**`llvm-foreach` (Intel SYCL compiler toolchain, called by `icpx` during SPIR-V generation) crashes with SIGILL under VTune's runtime environment.** Same signature previously seen on `intel/vllm:0.14.1-xpu` — see [[feedback-vllm-xpu-enforce-eager]].

Called from `_initialize_kv_caches → determine_available_memory → collective_rpc`, i.e. the KV-cache-size probe that vLLM 0.21 runs early in engine init. It JIT-compiles `spirv_utils.cpython-312-x86_64-linux-gnu.so` via `triton/runtime/build.py::_build` regardless of `--enforce-eager`, because the eager flag only disables torch.compile / Inductor / CUDAGraphs — it does not disable Triton XPU backend's own SYCL utility builds.

Sequence of failure:
1. Triton XPU backend calls `icpx` in a subprocess to build `spirv_utils.so`.
2. `icpx` internally invokes `llvm-spirv` → `llvm-foreach`.
3. Under VTune's ITT injection (`LD_PRELOAD=libittnotify_collector.so`), `llvm-foreach` hits **SIGILL** on some `pext/pdep`-family instruction. Prior memory record narrows this to a code path exercised only when the tracer's instrumentation ABI-mismatches with an old llvm binary.
4. Triton propagates the CalledProcessError up; EngineCore dies.
5. VTune injector cleaning up the SIGILL child gets SIGSEGV itself.

### Workaround options

1. **Pre-build `spirv_utils.so` outside VTune** — start vLLM once WITHOUT VTune, let Triton write `spirv_utils.so` into its cache dir, then run under VTune with `TRITON_CACHE_DIR=<primed dir>`. **Blocked**: Triton for vLLM 0.21 puts the compiled artifact in `/tmp/tmpXXXXX/` (per-run temp dir), not in `~/.triton/cache`, so it isn't reused across runs.

2. **Downgrade oneAPI compiler** — the vLLM 0.14.1-xpu image bundled oneAPI 2025.2 for which our older memory record noted `--enforce-eager` was sufficient. The 0.21 image ships **oneAPI 2025.3**; something in the 2025.3 `icpx`/`llvm-foreach` combo doesn't survive VTune injection. Would need a different vLLM image build.

3. **Pre-warm inside VTune by starting collection AFTER init** — use VTune `-start-paused` which the wrapper already does, PLUS defer the initial `vtune -command resume` until after the API server's `/health` is 200-OK. The wrapper already does this correctly (`vtune -command resume` is called just before the bench.log run, after warmup completes) — but the crash occurs during EngineCore init, BEFORE the API server is ever ready. So this workaround doesn't help.

4. **Use unitrace instead of VTune for this vLLM version** — unitrace uses `LD_PRELOAD=libunitrace_tool.so`, not VTune's collector — different injection ABI, may bypass the SIGILL. This is the most promising path.

5. **Use PyTorch profiler instead** — pt_profile does NOT rely on binary injection; it uses `torch.profiler` API calls emitted by the vLLM built-in `/start_profile` endpoint. Should be unaffected by the icpx crash.

### Status

- **VTune E2E: BLOCKED** on this container image. Root cause is upstream `oneAPI 2025.3 icpx / llvm-foreach` vs `VTune 2026.0`, not our wrappers. Wrappers are correct.
- **unitrace E2E: not yet tried in this session** — recommend running next.
- **PyTorch profiler E2E: not yet tried in this session** — recommend running next.

### Environment-var hygiene notes

- **HF_HOME=/root/local_hf_cache is critical** — without it every attempt spends 4-10 min on NFS reads. Once local weights exist the load time is <10 s.
- **VLLM_ENGINE_READY_TIMEOUT_S=1800** and wrapper `/health` poll bumped to 900 iterations × 2 s = 30 min. Both already pushed to `d56f27f`.
- **VLLM_WORKER_MULTIPROC_METHOD=spawn** is the intel/vllm-recommended default and does NOT cause the icpx crash — a run with `fork` would only differ in whether the child inherits Python interpreter state, not in whether icpx runs.

### Recommended next steps (for the user)

1. Try the unitrace wrapper — different injection mechanism, likely bypasses the SIGILL.
2. Try the pytorch-profiler wrapper — no injection at all.
3. If VTune is required, ask the intel/vllm team for a build that pairs vLLM 0.21 with oneAPI 2025.2 (or waits for oneAPI 2025.4 with a llvm-foreach fix).

---

## Image-level design shift: 0.17.0-xpu → 0.21.0-ubuntu24.04

Root cause of the naked `source setvars.sh` failures in Session 2, contributed by user:

**`intel/vllm:0.17.0-xpu`**
- No `ENTRYPOINT`; `Cmd = ['/bin/bash']`.
- All oneAPI env vars (`LD_LIBRARY_PATH`, `CCL_ROOT`, `PATH`, `MKLROOT`, `SETVARS_COMPLETED=1`, …) baked into image `Config.Env` at build time (Dockerfile `ENV` directives).
- Docker injects these into every process regardless of shell mode. Non-interactive `docker exec -d ... bash -c '…'` → works — `python -c "import torch"` prints `2.10.0+xpu`.
- Duplicate `source /opt/intel/oneapi/setvars.sh --force` in `/root/.bashrc:100–101` is belt-and-braces convenience for interactive users.

**`intel/vllm:0.21.0-ubuntu24.04-20260625`**
- `ENTRYPOINT = /bin/bash -c 'source /opt/intel/oneapi/setvars.sh --force && exec "$@"'`.
- Image `Config.Env` contains only `PATH` and `VLLM_VERSION`. No oneAPI vars.
- Env is populated at container start **by ENTRYPOINT, not baked into `Config.Env`**.
- **`docker exec` bypasses `ENTRYPOINT` entirely** — the exec'd process is a direct child of `dockerd`, not descended from the ENTRYPOINT process. So exec'd shells see only `Config.Env`, which lacks oneAPI.
- Interactive `docker exec -it bash` still works only because `.bashrc` redundantly sources setvars (guarded by `[ -z "$PS1" ] && return`).
- Non-interactive `docker exec -d bash -c '…'` skips `.bashrc` and gets no oneAPI env → `libccl.so.1 not found`.

**One-line summary**: 0.17 promotes oneAPI to image-level `ENV` (works everywhere); 0.21 demotes it to `ENTRYPOINT + interactive .bashrc` (works only in the paths the docs assume).

### Consequences for the toolkit

Every wrapper must source `setvars.sh` itself; it can't rely on the shell's ambient environment. The VTune wrapper (`run_vtune_vllm.sh`) already did this at commit `4cc95a7` — but with a subtlety: `setvars.sh` uses `unset`/`return` on nonempty variables under `set -euo pipefail`, which the script would then propagate as failure. **The fix is to drop pipefail/nounset around the source and re-enable after.**

Session 2 discovered the unitrace and pytorch-profiler wrappers had **not** been ported to this pattern. Session-2 patches:

```bash
# both wrappers now do:
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
  set +euo pipefail
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
  set -euo pipefail
fi
```

Failure signature in `/tmp/uni_log.txt` before the patch:

```
/opt/intel/oneapi/compiler/latest/env/vars.sh: line 258: OCL_ICD_FILENAMES: unbound variable
```

(setvars.sh unsets `OCL_ICD_FILENAMES` under `set -u`; wrapper dies immediately.)

---

## Session 2 (cont.) — E2E runs after setvars fix

### unitrace E2E — SUCCESS

Command:

```bash
cd /root/toolkit/unitrace/scripts
export HF_HOME=/root/local_hf_cache
MODEL=meta-llama/Llama-3.1-8B-Instruct MAX_MODEL_LEN=2048 \
UNITRACE_PRESET=lite ./run_unitrace_vllm.sh &
# Wait for /health 200, then:
curl -X POST http://127.0.0.1:9090/start_profile
vllm bench serve --backend openai --base-url http://127.0.0.1:9090 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random --random-input-len 512 --random-output-len 1 \
  --num-prompts 3 --max-concurrency 1
curl -X POST http://127.0.0.1:9090/stop_profile
kill -INT <serve_with_unitrace pid>  # graceful, so unitrace flushes
```

Result dir: `unitrace_results/20260703_094123_meta-llama_Llama-3.1-8B-Instruct/`

Artifacts:
- `python.9876.json` — 2.6 MB Chrome trace (Perfetto-compatible)
- `device_timing_summary.txt` — 254 KB per-kernel L0 timing
- `unitrace.log` — 514 KB full unitrace log with Device Timing Summary
- `unitrace_vllm_report.html` — 62 KB self-contained HTML report (auto-generated by wrapper's EXIT trap)

Load time: model weights loaded from local cache in <10 s; total wall time from launch to `/health 200` was ~90 s.

### PyTorch profiler E2E — SUCCESS

Command:

```bash
cd /root/toolkit/pytorch-profiler/scripts
export HF_HOME=/root/local_hf_cache
MODEL=meta-llama/Llama-3.1-8B-Instruct MAX_MODEL_LEN=2048 PT_PHASE=prefill \
INPUT_LEN=512 OUTPUT_LEN=1 NUM_PROMPTS=3 WARMUP_PROMPTS=2 MAX_CONCURRENCY=1 \
./run_pt_profile_vllm.sh
```

Result dir: `pt_results/20260703_095233_prefill/`

Artifacts:
- `torch_traces/1783072377385327975-rank-0.*.pt.trace.json.gz` — 1.3 MB compressed Chrome trace (worker/EngineCore process)
- `torch_traces/gnrsp-bmg3_12690.async_llm.*.pt.trace.json.gz` — 1.3 MB (async_llm/API server process)
- `torch_traces/profiler_out_0.txt` — 20 KB text summary
- `bench.log` — 3 prompts, **4622 tok/s** total token throughput, 111 ms mean TTFT
- `openai-infqps-*.json` — infqps benchmark JSON
- `summary.txt`, `warmup.log`, `server.log` — supporting metadata

Load time: model weights loaded in <10 s from local cache. Wrapper's own bench + profiling round-trip took ~1 min.

## Final verification status on vLLM 0.21

| Profiler | End-to-end capture | Status | Notes |
|---|---|---|---|
| VTune | ❌ | **BLOCKED — upstream** | `llvm-foreach` SIGILL in oneAPI 2025.3 icpx under VTune injection. Not a wrapper bug. Fix requires vLLM 0.21 image with oneAPI 2025.2 OR oneAPI ≥ 2025.4. |
| unitrace | ✅ | **PASS** | 2.6 MB Chrome trace + 254 KB device timing + HTML report. Uses `libunitrace_tool.so` LD_PRELOAD instead of VTune's collector — bypasses the SIGILL. |
| PyTorch profiler | ✅ | **PASS** | Two 1.3 MB compressed Chrome traces (worker + async_llm) + text summary. No binary injection at all. |

## Combined patch set pushed to `origin/main` this session

- `cb6f12e` — align wrappers with intel/vllm:0.21 defaults (5 env vars) + doc
- `d56f27f` — /health poll bumped to 30 min in vtune + pytorch-profiler
- `6b0b3b7` — session-2 verification log with VTune llvm-foreach root cause
- `2b2e4b7` — unitrace + pytorch-profiler: isolate setvars.sh from set -euo pipefail (0.17→0.21 image env shift)

## Reproducibility recipe

```bash
# One-time inside container (~10 s on warm page cache):
mkdir -p /root/local_hf_cache
cp -aL /hf_cache/models--meta-llama--Llama-3.1-8B-Instruct /root/local_hf_cache/

# Every run:
cd /root/toolkit && git pull
export HF_HOME=/root/local_hf_cache

# unitrace:
cd unitrace/scripts
MODEL=meta-llama/Llama-3.1-8B-Instruct MAX_MODEL_LEN=2048 \
UNITRACE_PRESET=lite ./run_unitrace_vllm.sh &
# ... drive ROI via curl /start_profile + bench + /stop_profile ...

# pytorch-profiler:
cd pytorch-profiler/scripts
MODEL=meta-llama/Llama-3.1-8B-Instruct MAX_MODEL_LEN=2048 \
PT_PHASE=prefill INPUT_LEN=512 OUTPUT_LEN=1 \
NUM_PROMPTS=3 WARMUP_PROMPTS=2 MAX_CONCURRENCY=1 \
./run_pt_profile_vllm.sh
```
