# PyObject-relative tensor `cdata` implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store Torch's tensor `cdata` offset relative to `PyObject_HEAD` and let each generated module add its compiled `sizeof(PyObject)` once at load time.

**Architecture:** The TOML row, Python `TensorABI`, and setter argument carry a relative offset. The Python and C++ detectors independently subtract their own CPython header sizes from an absolute measurement. The generated C setter validates and saves an absolute `uint16_t` offset; the launch reader remains unchanged.

**Tech Stack:** Python 3.8–3.14, C for the generated extension, C++20 for `cpp_detect`, Jinja2, TOML, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-24-pyobject-relative-cdata-design.md`

## Global constraints

- The recorded table covers Torch 2.2–2.14 on x86-64. This plan does not add an architecture or a CPython access mode.
- The supported CPython builds are GIL 3.8–3.14 and free-threaded 3.13t/3.14t. A rendered binary targets one exact CPython ABI, even in Torch `RUNTIME_SHIM` mode.
- `cdata` in TOML, `TensorABI`, Python probe output, C++ `detect()` output, and `set_torch_version()` input is relative to the end of `PyObject_HEAD`. `intj_torch_abi.cdata` in C module state is absolute from `PyObject *`.
- Keep the seven-offset plus itemsize tuple shape, `uint16_t` C fields, exact dtype-set validation, and `layout_for() -> None` refusal for unavailable layouts. Reject an effective `cdata` above 65535 before writing module state.
- Preserve `INTERPRETER` and `STATIC_COMPILE` behavior, the CPython integer/float readers, and the per-launch `object + st->abi.cdata` read.
- Do not add a dependency. Do not claim CI covers the table; the repository has no CI job for its detectors.

## File map

| File | Responsibility in this change |
| --- | --- |
| `intj/torch_intf/torch_abi.toml`, `torch_abi.py` | Recorded relative rows and early eligibility/refusal. |
| `intj/torch_intf/abi_detect.py`, `cpp_detect.py` | Independent relative measurements; retain absolute addresses only while probing. |
| `intj/runtime/entry.c.jinja`, `intj/launcher.py` | Convert the setter argument to a checked absolute module-state offset at load; explain the call site. |
| `tests/test_runtime.py`, `tests/test_launcher.py` | Boundary, detector parity, CPU stub launch, and GPU specialization checks. |
| `intj/python_intf/{README.md,AGENTS.md}`, `intj/torch_intf/README.md`, `CONCETPS.md`, `docs/Usage.md` | Explain the new origin and the separate CPython/Torch responsibilities. |

---

### Task 1: Convert the offset contract end to end

**Files:**
- Modify: `intj/torch_intf/torch_abi.py:41-64,190-195,253-274`
- Modify: `intj/torch_intf/torch_abi.toml:1-21` and the other 12 `cdata` rows
- Modify: `intj/torch_intf/abi_detect.py:16-25,97-153,156-180,197-223`
- Modify: `intj/torch_intf/cpp_detect.py:162-177`
- Modify: `intj/runtime/entry.c.jinja:363-425`
- Modify: `intj/launcher.py:301-309,377-382`
- Test: `tests/test_runtime.py:149-230`, `tests/test_launcher.py:952-1027,1195-1206`

**Interfaces:**
- Consumes: `python_intf.cpython_abi.pyobject_size() -> int`, `TensorABI.as_args()`, and the existing `set_torch_version(version, layout, dtype_index)` call.
- Produces: `probe_layout(header_size: int) -> TensorABI | None`, `_read(layout: TensorABI, tensor: Any, header_size: int) -> tuple[int, int, int]`, `cpp_detect.detect() -> (relative_offsets, sizes)`, `layout_for() -> relative TensorABI | None`, and an absolute `st->abi.cdata` after successful installation. `cpp_detect.measure()` keeps its raw absolute output.

- [ ] **Step 1: Make the contract fail in focused tests.** In `tests/test_runtime.py`, replace the header parametrization and its `layout.cdata` expectation with:

```python
@pytest.mark.parametrize("head, version, expected", [
    (16, (2, 9), 8),
    (32, (2, 9), 8),
    (16, (2, 10), 0),
    (32, (2, 10), 0),
    (65535, (2, 9), None),
    (65535, (2, 10), 0),
    (65536, (2, 10), None),
])
def test_runtime_shim_cdata_is_header_relative(monkeypatch, head, version, expected):
    from intj.torch_intf import torch_abi as abi

    monkeypatch.setattr(abi, "pyobject_size", lambda: head)
    monkeypatch.setattr(abi, "itemsize_table", lambda _version: b"\x01" * NDTYPES)
    abi.layout_for.cache_clear()
    try:
        layout = abi.layout_for(version)
        assert (layout.cdata if layout else None) == expected
    finally:
        abi.layout_for.cache_clear()
```

In `test_runtime_shim_layout_matches_live_torch`, replace its probe and direct read with:

```python
assert abi_detect.probe_layout(cpython_abi.pyobject_size()) == layout
tensor = torch.arange(8, dtype=torch.float32)
absolute = cpython_abi.pyobject_size() + layout.cdata
assert ctypes.c_void_p.from_address(id(tensor) + absolute).value == tensor._cdata
```

Add these two tests beside it. The first exercises the detector's negative-offset guard; the second proves that a valid relative `uint16_t` can become an invalid absolute one, without damaging an installed module:

```python
def test_runtime_shim_probe_refuses_slot_before_header():
    from intj.torch_intf.abi_detect import probe_layout

    assert probe_layout(torch.Tensor.__basicsize__ + 1) is None


def test_runtime_shim_rejects_rebased_cdata_overflow(built):
    module, stub, _ = built
    if module.__name__ != "rt_runtime_shim":
        pytest.skip("only RUNTIME_SHIM installs cdata")
    layout = layout_for()
    assert layout is not None
    x = torch.arange(8, dtype=torch.float32)
    args = (0, 0, 1, x, 5, 0, 0.0, False, 64)
    module.entry(*args)
    assert stub.last(5)[3][0] == x.data_ptr()

    relative = (1 << 16) - cpython_abi.pyobject_size()
    assert 0 <= relative <= 65535
    bad = (relative, *layout.as_args()[1:])
    index = bytes(c if c < 32 else 0xFF for c in range(NDTYPES))
    with pytest.raises(ValueError, match="16-bit"):
        module.set_torch_version(torch_version(), bad, index)

    module.entry(*args)
    assert stub.last(5)[3][0] == x.data_ptr()
```

In `tests/test_launcher.py`, use `id(x) + cpython_abi.pyobject_size() + layout.cdata` in `test_layout_probe_reproduces_torch`; call `probe_layout(cpython_abi.pyobject_size())` in `test_hardcoded_layout_matches_this_torch`; pass `cpython_abi.pyobject_size()` as the third `_read()` argument in `test_custom_sizes_policy_tensors_are_a_known_limitation`. Keep the existing C++ parity assertion: once `cpp_detect.detect()` changes, both sides of that assertion will be relative.

- [ ] **Step 2: Confirm red.** Run:

```sh
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_runtime.py -q -k 'cdata_is_header_relative or probe_refuses_slot_before_header or rebased_cdata_overflow or layout_matches_live_torch'
```

Expected: at least one failure: old `layout_for()` returns 24/16 rather than 8/0, and old `probe_layout()` takes no argument. No full-suite run is needed while the contract is half-migrated.

- [ ] **Step 3: Make the Python table representation relative.** In `torch_abi.py`, change the `TensorABI.cdata` comment to `# bytes after PyObject_HEAD to the TensorImpl* slot`, delete `MEASURED_PYOBJECT_SIZE`, and replace `layout_for()`'s conversion and return with:

```python
    # Preflight with the interpreter's reported size; the compiled setter repeats
    # the bound check using its own sizeof(PyObject) before saving the offset.
    absolute = pyobject_size() + offsets["cdata"]
    if not 0 <= absolute < 1 << 16:
        return None
    return TensorABI(itemsize=items, **offsets)
```

Keep the existing pointer-width and dtype-match checks above this block. Update the docstring to say `cdata` is relative and the C setter reconstructs the absolute offset. In `torch_abi.toml`, replace its two-line `cdata` header comment with:

```toml
#   cdata: byte offset from the end of PyObject_HEAD to the TensorImpl* slot;
#          set_torch_version adds its compiled sizeof(PyObject) at module load
```

Leave the numeric rows unchanged until Step 7 regenerates them with the updated detector. The setter, detector, and table must be committed together.

- [ ] **Step 4: Reconstruct in the compiled setter.** In the `runtime_shim` branch of `set_torch_version()` in `entry.c.jinja`, after parsing all seven offsets and the itemsize table but before `if (st->abi.ready)`, add:

```c
  size_t absolute_cdata = (size_t)off[0] + sizeof(PyObject);
  if (absolute_cdata > UINT16_MAX) {
    PyErr_SetString(PyExc_ValueError,
                    "intj: tensor layout cdata plus PyObject header must fit a 16-bit offset");
    return NULL;
  }
```

Change only the commit assignment to `st->abi.cdata = (uint16_t)absolute_cdata;`. Leave `intj_read_tensor()` unchanged. In `launcher.py`, change the `_loaded_module()` comment to say it passes a relative offset and the compiled setter saves the absolute one; remove “on a default-GIL build” from `_unverified_torch_message()` because the generator will support both builds.

- [ ] **Step 5: Normalize the Python detector.** Give `probe_layout()` a required `header_size: int` argument. Replace its `head = pyobject_size()` with `head = header_size` and reject `header_size <= 0`. After the unique offsets are pinned, but before constructing `TensorABI`, add:

```python
    absolute_cdata = pinned["cdata"]
    if absolute_cdata < header_size:
        return None
    pinned["cdata"] = absolute_cdata - header_size
```

Pass `header_size` through `_selfcheck(layout, header_size)` and `_read(layout, tensor, header_size)`. In `_read()`, change the first read to:

```python
    impl = ctypes.c_size_t.from_address(id(t) + header_size + layout.cdata).value
```

Make `_selfcheck()` call `_read(layout, t, header_size)`. In `_main()`, set `header_size = pyobject_size()`, call `probe_layout(header_size)`, and delete the `MEASURED_PYOBJECT_SIZE` import and the 16-byte generation guard. Leave the bounded object scan and all other live-pointer reads intact.

- [ ] **Step 6: Normalize the independent C++ result.** Keep `cpp_detect.measure()` unchanged: its `facts["pyobject"]` and `offsets["cdata"]` are raw compiled facts. At the end of `cpp_detect.detect()`, after the existing `intj_THPVariable` check, add:

```python
    if offsets["cdata"] < head["pyobject"]:
        raise RuntimeError("intj: torch's cdata pointer slot precedes PyObject_HEAD")
    offsets["cdata"] -= head["pyobject"]
    return offsets, sizes
```

Remove the old `return offsets, sizes`. Update `detect()`'s docstring to say it returns the relative offset, while `measure()` stays raw. The C++ source already computes the **final** pointer slot, including `MaybeOwned<Tensor>` before Torch 2.10; do not subtract from `thpvariable_cdata` alone.

- [ ] **Step 7: Regenerate all 13 table rows from the updated detector.** The checked one-off script below uses the installed Torch 2.2–2.13 CPU environments and the active Torch 2.14 environment. It compares the generated row with the old one before writing: only `cdata` may change numerically, and it must change by exactly the documented 16-byte header. It installs each detector-produced stanza verbatim, satisfying `torch_intf/AGENTS.md`'s no-hand-edited-offset rule.

```sh
/tmp/gb2/bin/python - <<'PY'
from pathlib import Path
import os
import subprocess

from intj.torch_intf.torch_abi import _parse_abi

path = Path("intj/torch_intf/torch_abi.toml")
source = path.read_text()
old = _parse_abi(source)
assert len(old) == 13
envs = [Path(f"/tmp/intj_torch/2.{minor}") for minor in range(2, 14)]
envs.append(Path("/tmp/gb2"))
generated = {}
for env_dir in envs:
    interpreter = env_dir / "bin/python"
    assert interpreter.is_file(), interpreter
    entry = subprocess.check_output(
        [str(interpreter), "-m", "intj.torch_intf.abi_detect"],
        env={**os.environ, "PYTHONPATH": str(Path.cwd())}, text=True,
    )
    rows = _parse_abi(entry)
    assert len(rows) == 1
    version = next(iter(rows))
    assert version not in generated and version in old
    new_offsets, new_dtypes = rows[version]
    old_offsets, old_dtypes = old[version]
    assert new_dtypes == old_dtypes
    assert {k: v for k, v in new_offsets.items() if k != "cdata"} == {
        k: v for k, v in old_offsets.items() if k != "cdata"
    }
    assert old_offsets["cdata"] in (16, 24)
    assert new_offsets["cdata"] == old_offsets["cdata"] - 16
    generated[version] = entry.rstrip()
assert set(generated) == set(old)
header = source[:source.index("# torch 2.2")].rstrip()
path.write_text(header + "\n\n" + "\n\n".join(
    entry for _, entry in sorted(generated.items())
) + "\n")
PY
```

- [ ] **Step 8: Run focused checks, then commit this atomic contract change.** Run these commands once after all code above is in place:

```sh
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_runtime.py -q
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_launcher.py -q -k 'layout_probe or hardcoded_layout or every_mode_matches_triton_specialization or spec_key_is_never_coarser_than_triton or custom_sizes_policy'
PYTHONPATH=$PWD /tmp/gb2/bin/python -m intj.torch_intf.abi_detect
PYTHONPATH=$PWD /tmp/gb2/bin/python -m intj.torch_intf.cpp_detect
git diff --check
```

Expected: tests pass (existing skips remain). On the active Torch 2.14/CPython
3.12 GIL build, the Python detector prints `cdata = 0` and the C++ detector
prints `cdata=0`. The pre-change CPU-stub baseline is 60 passed, 5 skipped;
do not use that count as a post-change assertion. The generation script has
already asserted that no other numeric offsets or dtypes changed. Review any
updated generator provenance comments, then commit this atomic change:

```sh
git add intj/torch_intf/torch_abi.py intj/torch_intf/torch_abi.toml intj/torch_intf/abi_detect.py intj/torch_intf/cpp_detect.py intj/runtime/entry.c.jinja intj/launcher.py tests/test_runtime.py tests/test_launcher.py
git commit -m 'Record tensor cdata relative to PyObject header'
```

### Task 2: Update the interface contract and verify build variants

**Files:**
- Modify: `intj/python_intf/README.md:39-53`, `intj/python_intf/AGENTS.md:38-42,116-118`
- Modify: `intj/torch_intf/README.md:14-25`, `CONCETPS.md:29-32`, `docs/Usage.md:132-161`
- Delete after implementation: `intj/python_intf/TODO.md` (its only item is complete)
- Delete if the false CI claim is corrected: `intj/torch_intf/TODO.md` (its only item tracks that claim)
- Verify: `tests/run_python_matrix.sh`, `tests/test_launcher.py`, `intj/python_intf/check.py`

**Interfaces:**
- Consumes: Task 1's relative table and detector contract, and its compiled setter's checked absolute state.
- Produces: Current documentation and GIL/free-threaded evidence. No Python or C API changes.

- [ ] **Step 1: Replace the obsolete 16-byte descriptions.** In `python_intf/README.md`, replace the `RUNTIME_SHIM` sentences under “PyObject Header Size” with:

```markdown
In `RUNTIME_SHIM`, the recorded `cdata` offset starts after `PyObject_HEAD`.
`cpython_abi.pyobject_size()` (`object.__basicsize__`) supplies the detecting
interpreter's header size. At module load, the compiled setter adds its own
`sizeof(PyObject)` and saves the absolute offset before the first launch.
```

Keep its statement that supported x86-64 GIL/free-threaded headers are 16/32 bytes. In `python_intf/AGENTS.md`, replace its old Python-side adjustment sentences in both the “Every CPython internal lives here” and “Free-threaded modules” sections with this contract:

```markdown
`pyobject_size()` supplies the detecting interpreter's header size and lets
`layout_for()` refuse an unrepresentable offset early. For `RUNTIME_SHIM`, the
generated C setter adds its compiled `sizeof(PyObject)` to the table's
post-header `cdata` offset and saves the absolute result before any launch.
```

In `torch_intf/README.md`, move the misplaced `cdata` paragraph below its detector bullets, replace its old wording and the false CI claim with:

```markdown
The table stores `cdata` relative to the end of `PyObject_HEAD`. The runtime
detector subtracts `python_intf.cpython_abi.pyobject_size()` from the measured
pointer-slot offset; the independent C++ detector subtracts its compiled
`sizeof(PyObject)`. The generated module adds its own compiled size once when
installing the layout. Run `python -m intj.torch_intf.abi_detect` to generate an
entry and `python -m intj.torch_intf.cpp_detect` to check it independently.
```

In `CONCETPS.md`, replace its final paragraph with:

```markdown
For `RUNTIME_SHIM`, `torch_abi.toml` stores `cdata` relative to the end of
`PyObject_HEAD`. `layout_for()` selects the verified Torch row, and the
generated module's `set_torch_version()` adds its compiled `sizeof(PyObject)`
and saves the absolute offset before the first launch.
```

In `docs/Usage.md`, replace lines 132–136 and 154–161 with these two paragraphs, leaving its existing refusal and dtype-validation explanations in place:

```markdown
`RUNTIME_SHIM` reads offsets from a table of verified torch versions
(`intj/torch_intf/torch_abi.toml`), installed at load by `set_torch_version`.
The recorded `cdata` starts after `PyObject_HEAD`; the generated module adds
its compiled `sizeof(PyObject)` once and saves the absolute offset before any
launch.

To add a version, run `python -m intj.torch_intf.abi_detect` on a supported
GIL or free-threaded CPython build with that Torch and paste the entry it
prints. The detector subtracts that interpreter's reported header size from
the measured pointer-slot offset. `python -m intj.torch_intf.cpp_detect`
derives the relative offset independently from Torch's C++ headers.
The test suite compares the table entry with both detectors on the Torch
version it runs against.
```

Keep the existing `CONCETPS.md` filename.

- [ ] **Step 2: Close only resolved TODOs.** Delete `intj/python_intf/TODO.md` after all Task 1 checks pass. The corrected local-verification wording in `torch_intf/README.md` also resolves the sole item in `intj/torch_intf/TODO.md`, so delete that file. Leave the root `TODO.md` intact: its non-x86-64 and broader interface-spec items are outside this refactor.

- [ ] **Step 3: Check all recorded Torch rows against both detectors.** The local `/tmp/intj_torch/2.2` through `2.13` CPU environments contain the historical versions. Run this one-off check from the repository root; the active `/tmp/gb2` environment covers 2.14:

```bash
for env_dir in /tmp/intj_torch/2.{2..13} /tmp/gb2; do
  test -x "$env_dir/bin/python" || exit 1
  PYTHONPATH=$PWD "$env_dir/bin/python" - <<'PY' || exit 1
from intj.python_intf.cpython_abi import pyobject_size
from intj.torch_intf.abi_detect import probe_layout
from intj.torch_intf.cpp_detect import detect
from intj.torch_intf.torch_abi import NDTYPES, OFFSETS, layout_for

table = layout_for()
assert table is not None
assert probe_layout(pyobject_size()) == table
offsets, sizes = detect()
assert offsets == {name: getattr(table, name) for name in OFFSETS}
assert {code: size for code, size in sizes.items() if size} == {
    code: table.itemsize[code] for code in range(NDTYPES) if table.itemsize[code]
}
PY
done
```

Expected: all 13 interpreters exit zero. These environments are a local convenience verified in this workspace, not a repository dependency. Do not use `/tmp/intj_torch/run.sh` unchanged: it still compares the two detectors' old absolute outputs.

- [ ] **Step 4: Verify GIL and free-threaded modules.** Run the supported Python matrix, the compiled CPython checker, the C++ detector on a free-threaded build, and the full AMD launcher suite:

```sh
bash tests/run_python_matrix.sh
PYTHONPATH=$PWD /tmp/gb2/bin/python -m intj.python_intf.check
PYTHONPATH=$PWD uv run -q --no-project --python cpython-3.14t --index https://download.pytorch.org/whl/cpu --with torch --with setuptools python -c 'from intj.python_intf.cpython_abi import pyobject_size; from intj.torch_intf.abi_detect import probe_layout; from intj.torch_intf.cpp_detect import detect; from intj.torch_intf.torch_abi import OFFSETS, layout_for; table = layout_for(); assert table is not None; assert probe_layout(pyobject_size()) == table; assert detect()[0] == {name: getattr(table, name) for name in OFFSETS}'
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests -q
pyright --pythonpath /tmp/gb2/bin/python
```

Expected: zero exit status for each command; `check` prints `ok`; the matrix exercises both free-threaded variants without enabling the GIL. The free-threaded command explicitly compares `cpp_detect` with the table because the matrix does not call it. The GPU suite includes all three Torch modes and `test_spec_key_is_never_coarser_than_triton`; report any skips accurately.

- [ ] **Step 5: Prove the new contract test catches the old table.** In the disposable implementation worktree, temporarily change the active Torch 2.14 row from relative `cdata = 0` back to its old absolute `cdata = 16`:

```sh
/tmp/gb2/bin/python - <<'PY'
from pathlib import Path
import os
import subprocess

path = Path("intj/torch_intf/torch_abi.toml")
source = path.read_text()
old = '["2.14"]\ncdata = 0\n'
assert source.count(old) == 1
try:
    path.write_text(source.replace(old, '["2.14"]\ncdata = 16\n', 1))
    result = subprocess.run(
        ["/tmp/gb2/bin/python", "-m", "pytest", "-q",
         "tests/test_runtime.py::test_runtime_shim_layout_matches_live_torch"],
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(result.stdout[-4000:])
    assert result.returncode == 1
    assert "assert abi_detect.probe_layout" in result.stdout
finally:
    path.write_text(source)
PY
git diff --exit-code -- intj/torch_intf/torch_abi.toml
```

Expected: the subprocess fails at `probe_layout(...) == layout` before reading a wrong pointer. The `finally` block restores the table even if the output is unexpected.

- [ ] **Step 6: Prove the specialization-key invariant test is sensitive.** In the same disposable worktree, change only the pointer-alignment condition inside `INTJ_DECODE`; replacing its expression preserves the macro's trailing `\` line continuations:

```sh
/tmp/gb2/bin/python - <<'PY'
from pathlib import Path
import os
import subprocess

path = Path("intj/runtime/intj_runtime.h")
source = path.read_text()
condition = "(((uintptr_t)_p & 15u) == 0)"
assert source.count(condition) == 1
try:
    path.write_text(source.replace(condition, "0", 1))
    result = subprocess.run(
        ["/tmp/gb2/bin/python", "-m", "pytest", "-q",
         "tests/test_launcher.py::test_spec_key_is_never_coarser_than_triton"],
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(result.stdout[-4000:])
    assert result.returncode == 1
    assert "intj key collides" in result.stdout
finally:
    path.write_text(source)
PY
git diff --exit-code -- intj/runtime/intj_runtime.h
```

Expected: the subprocess fails because aligned and unaligned pointers share an intj key but have different Triton specializations. The `finally` block restores the header; `intj_runtime.h` is otherwise untouched by this plan. Neither mutation is committed.

- [ ] **Step 7: Finish and commit documentation.** Search active sources for obsolete claims, excluding historical Superpowers specs, then check the diff and commit only the Task 2 docs/TODO removals:

```sh
rg -n 'MEASURED_PYOBJECT_SIZE|recorded 16-byte|default-GIL build of this torch|replaces the 16-byte|checked using both method in our CI' intj CONCETPS.md docs/Usage.md
git diff --check
git status --short
```

Expected: the `rg` command finds no matches (its exit code 1 means no matches), `git diff --check` is clean, and the status contains only the intended docs/TODO edits. Commit and confirm a clean worktree:

```sh
git add CONCETPS.md docs/Usage.md intj/python_intf/README.md intj/python_intf/AGENTS.md intj/python_intf/TODO.md intj/torch_intf/README.md intj/torch_intf/TODO.md
git commit -m 'Document relative tensor cdata contract'
git status --short
```
