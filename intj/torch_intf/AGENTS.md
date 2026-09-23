# torch_intf

Read `README.md` first for what this directory is. The rules below are what keep it
correct.

## Never guess an offset

A wrong offset cannot raise: it reads whatever sits at that address and hands the
kernel a pointer built from it. So:

- Never hand-edit a number in the ABI table. Regenerate the entry with a detector
  and paste its output verbatim.
- An entry's layout and dtypes are verified together or not at all. If the running
  torch's dtypes differ from the recorded ones, there is no layout.
- No verified layout found is an error: raise, naming the torch version and how to
  add an entry. Never fall back silently, and never fill a gap with a nearby
  version's entry.

## The generator checks what CXX assumes

CXX mode needs no table, but it declares `intj_THPVariable` itself
(`intj/runtime/intj_thpvariable.h`) instead of including `python_variable.h`, which
drags in pybind11, and does not check that at load. `cpp_detect` compiles it beside
torch's own and refuses a torch where they differ. Any new layout assumption in C++
code gets the same treatment: checked in `cpp_detect`, not at load.

## The ABI table is a trust boundary

It is pasted by hand, so its parser rejects malformed input with a `ValueError`
naming the file, never skips it.

## Import torch and triton lazily

Inside the function that needs them, never at module top: importing `intj` must not
import torch, and probing must not initialize the GPU.
