# pti-gpu unitrace build issues seen on BMG / Xe2 + oneAPI 2025.3

## 1. `(uintptr_t)(val)` cast on `ze_ipc_event_counter_based_handle_t`

### Error

```
[ 91%] Building CXX object CMakeFiles/unitrace_tool.dir/src/tracer.cc.o
.../tools/unitrace/build/tracing.gen: error: invalid cast from type
    'ze_ipc_event_counter_based_handle_t' {aka '_ze_ipc_event_counter_based_handle_t'}
    to type 'uintptr_t' {aka 'long unsigned int'}
    7 |     std::sprintf(buffer, "0x%lx", (uintptr_t)(val)); \
      |                                   ^~~~~~~~~~~~~~~~
.../tracing.gen:18854:7: note: in expansion of macro 'TO_HEX_STRING'
18854 |       TO_HEX_STRING(str, **(params->pphIpc));
```

### Cause

Newer Level Zero loader headers (those carrying `ze_event_counter_based_*`) make `ze_ipc_event_counter_based_handle_t` a struct (typically `struct { char data[N]; }` or similar). The pti-gpu generator (`scripts/gen_tracing_callbacks.py`) emits a single C-style `(uintptr_t)(val)` cast inside the `TO_HEX_STRING` macro, which is valid for pointers and integers but **not** for struct types.

This generator predates the new IPC handle type and hasn't been updated to handle structs.

### Fix

Replace the macro emission with a templated helper that dispatches on type traits:

```diff
--- a/tools/unitrace/scripts/gen_tracing_callbacks.py
+++ b/tools/unitrace/scripts/gen_tracing_callbacks.py
@@ def gen_to_hex_string_functions(f):
     f.write("#include <string>\n")
     f.write("#include <cstdio>\n")
     f.write("#include <cstdint>\n")
     f.write("#include <cstring>\n")
+    f.write("#include <type_traits>\n")
+    f.write("template <typename T>\n")
+    f.write("static inline uintptr_t to_hex_value_(const T& v) {\n")
+    f.write("  if constexpr (std::is_pointer_v<T>) {\n")
+    f.write("    return reinterpret_cast<uintptr_t>(v);\n")
+    f.write("  } else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {\n")
+    f.write("    return static_cast<uintptr_t>(v);\n")
+    f.write("  } else {\n")
+    f.write("    uintptr_t out = 0;\n")
+    f.write("    std::memcpy(&out, &v, sizeof(v) < sizeof(out) ? sizeof(v) : sizeof(out));\n")
+    f.write("    return out;\n")
+    f.write("  }\n")
+    f.write("}\n")
     f.write("#define TO_HEX_STRING(str, val) \\\n")
     f.write("    {char buffer[32]; \\\n")
-    if (sys.platform == 'win32'):
-        f.write("    sprintf_s(buffer, sizeof(buffer), \"0x%lx\", (uintptr_t)(val)); \\\n")
-    else:
-        f.write("    std::sprintf(buffer, \"0x%lx\", (uintptr_t)(val)); \\\n")
+    if (sys.platform == 'win32'):
+        f.write("    sprintf_s(buffer, sizeof(buffer), \"0x%lx\", (unsigned long)to_hex_value_(val)); \\\n")
+    else:
+        f.write("    std::sprintf(buffer, \"0x%lx\", (unsigned long)to_hex_value_(val)); \\\n")
     f.write("    str += std::string(buffer); \\\n")
     f.write("    }\n")
```

After patching, regenerate and rebuild:

```bash
cd build
rm -f tracing.gen common_header.gen l0_loader.gen
make -j$(nproc)
```

The `rm -f *.gen` is necessary — CMake's regen rule sometimes doesn't pick up Python-script changes alone.

### Why this works

- `std::is_pointer_v<T>` — original behavior preserved for L0 handles that ARE pointers (most of the API)
- `std::is_integral_v<T> || std::is_enum_v<T>` — sizes, indices, enum tags
- `else` — struct handles. We `memcpy` the first 8 bytes into a `uintptr_t`. The output isn't necessarily a meaningful address, but it's a stable, printable identifier — which is all the original tracing intended for IPC handles anyway.

### Upstream

Not yet fixed upstream as of pti-gpu master at the time of this skill's writing. If you `git pull`, re-apply this patch or check whether [intel/pti-gpu](https://github.com/intel/pti-gpu) has merged a fix.

---

## 2. NFS-squashed empty directory blocks `git clone`

### Symptom

```
$ git clone --depth 1 https://github.com/intel/pti-gpu.git
fatal: could not create work tree dir 'pti-gpu': File exists
$ ls -ld pti-gpu
drwxr-xr-x 2 nobody nogroup 6 ... pti-gpu
```

### Cause

NFS root-squashing on `/mnt/data` made an unrelated `pti-gpu` directory unowned (`nobody:nogroup`). It's empty but blocks the clone target.

### Fix

```bash
rmdir /mnt/data/.../pti-gpu       # works because it's empty
git clone --depth 1 https://github.com/intel/pti-gpu.git
```

If `rmdir` fails (non-empty / permission), `sudo rmdir` or pick a different parent dir.

---

## 3. Build artifacts owned by `nobody:nogroup`

Cosmetic, not a build failure: when the build runs as root inside a container against a root-squashed bind-mount, files end up owned by `nobody:nogroup` from the host's view but are perfectly usable from inside the container. Don't try to chown — you can't.

---

## 4. Detected dubious ownership in `/path/to/repo`

```
fatal: detected dubious ownership in repository at '/path/to/repo'
```

Harmless during build (it just skips embedding the git commit hash into the binary). To silence:

```bash
git config --global --add safe.directory '*'
```

(Inside the container; doesn't affect the host's git config.)

---

## 5. Modification time clock skew

```
make[2]: Warning: File 'level_zero/layers/zel_tracing_api.h' has modification time 91 s in the future
```

Container clock vs. NFS mtime mismatch. Build still produces correct binaries. Ignore unless you see a `Clock skew detected. Your build may be incomplete.` followed by an actual missing artifact.
