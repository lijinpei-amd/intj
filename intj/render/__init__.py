"""Turn a kernel's `RenderKey` into a loaded extension module.

`module._build_module` assembles a kernel's `RenderKey` and requests its module
from its process cache. `key.RenderKey` is everything the module depends on;
`context.RenderContext` derives from it every value `entry.c.jinja` prints or
branches on; `build` compiles the rendered source and loads the result.
"""
