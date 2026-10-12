"""Turn a kernel's `RenderKey` into a loaded extension module.

`key.RenderKey` is everything the module depends on; `context.RenderContext`
derives from it every value `entry.c.jinja` prints or branches on; `build`
compiles the rendered source and loads the result.
"""
