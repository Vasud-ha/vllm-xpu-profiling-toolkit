# vLLM PyTorch Profiler — Troubleshooting

Common failure modes, ordered by frequency.

---

## 1. Trace directory is empty after `stop_profile`

Checks, in order:

1. `echo $VLLM_TORCH_PROFILER_DIR` in the **server's** environment
   (not just your shell). Containers often miss it.
2. The dir exists and the server uid can write to it
   (`ls -ld $VLLM_TORCH_PROFILER_DIR`).
3. `start_profile` actually returned 200/None (curl `-i` to see).
4. Did you call `stop_profile`? The trace flush happens on stop.
5. Did the process exit cleanly? A `SIGKILL` mid-profile loses the
   buffer. Use `SIGTERM` or the HTTP endpoint.

If env var was set late (after the engine started), restart the
server. There's no runtime way to point the profiler at a directory
the engine never knew about.

---

## 2. Trace opens but GPU lane is empty

The PyTorch profiler captured CPU events only; GPU activity collection
is failing.

CUDA:

- CUPTI not loadable. `python -c "import torch; print(torch.cuda.is_available()); torch.cuda.profiler.start(); torch.cuda.profiler.stop()"` should not error.
- On locked-down hosts: Nvidia driver "perf counter" permission may be
  disabled (`/proc/driver/nvidia/params` → `RmProfilingAdminOnly`). Run
  as root or relax the kmod param.
- Inside Docker: pass `--cap-add=SYS_ADMIN` and ensure CUPTI lib is
  visible (`LD_LIBRARY_PATH`).

Intel XPU:

- Wrong torch / IPEX combo. The profiler hooks are loaded from the
  XPU extension; mismatched versions silently produce CPU-only traces.
- For intel/vllm:0.14.1-xpu specifically, prefer `--enforce-eager`
  ([[feedback_vllm_xpu_enforce_eager]]) — SYCL JIT under graph capture
  has produced incomplete kernel traces.
- Verify with a small standalone script:

  ```python
  import torch, intel_extension_for_pytorch as ipex
  from torch.profiler import profile, ProfilerActivity
  x = torch.randn(1024, 1024, device="xpu")
  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as p:
      (x @ x).sum().item()
  print(p.key_averages().table(sort_by="xpu_time_total", row_limit=5))
  ```

  If this shows zero XPU rows, the platform side is broken — fix
  before involving vLLM.

---

## 3. Trace is huge / Perfetto won't load it

Thresholds (rule of thumb):

- < 500 MB: `ui.perfetto.dev` directly.
- 500 MB – 2 GB: still works in Perfetto but slow; close other tabs.
- \> 2 GB: use the local trace_processor.

Local trace_processor flow:

```bash
# Install once
curl -L https://get.perfetto.dev/trace_processor -o trace_processor
chmod +x trace_processor

# Run as a server
./trace_processor --httpd big.pt.trace.json

# In ui.perfetto.dev, choose "Connect to a trace processor instance"
```

To shrink a trace at capture time:

- Reduce ROI (fewer prompts / shorter outputs).
- `--enforce-eager` — graph capture inflates the trace.
- Disable `record_shapes`, `with_stack`, `profile_memory` if they
  were forced on.
- For TP: open one rank at a time, don't merge unnecessarily.

---

## 4. `start_profile` 404 / endpoint missing

Either:

- `VLLM_TORCH_PROFILER_DIR` was unset → endpoints are not registered.
- vLLM build is too old / too new — endpoint path may have moved.
  `vllm serve --help` and check the API spec; older builds used
  `/v1/profile/start` style paths.

Workaround: switch to programmatic `LLM.start_profile()` /
`AsyncLLMEngine.start_profile()` from a sidecar script, or
upgrade/downgrade vLLM.

---

## 5. Profile starts, then process OOMs

Profiling itself adds memory pressure (event buffers, optional shape
tracking). If the server runs at `--gpu-memory-utilization 0.95`, the
extra MB can OOM.

- Drop `--gpu-memory-utilization` to 0.85.
- Shorten ROI.
- Disable `profile_memory=True` if you turned it on.

---

## 6. Kernel names are unreadable (`Memcpy HtoD`, `void at::native::elementwise_kernel<...>`)

Either:

- CUDA graphs / `torch.compile` is on — fused regions appear as
  `CUDAGraphLaunch` or generic compiled blobs. Re-capture with
  `--enforce-eager`.
- Stripped symbols on a custom-built torch — usually nothing to do
  except match kernels by signature.

---

## 7. Multi-rank traces don't line up in Perfetto

Each rank's trace has its own clock origin. Perfetto syncs *if* the
clock-sync metadata events are present (PyTorch emits them by default).
If they're missing:

- Open ranks in separate tabs and align by NVTX/ITT-marked landmarks
  you control (e.g. wrap the same `record_function("anchor")` on all
  ranks).
- Or merge with `hta` for offline analysis instead of Perfetto.

---

## 8. "It worked yesterday"

Most likely culprits:

- Container image rebuilt — env var no longer in the new image.
- vLLM upgraded — endpoint path or flag renamed.
- Driver / CUPTI / IPEX mismatched after an apt/yum upgrade.

Cheap diagnostic: capture a ~100-prompt cold run *without* `--profile`,
confirm it works, then add profiling back. Isolates "profiler broken"
from "vLLM broken".
