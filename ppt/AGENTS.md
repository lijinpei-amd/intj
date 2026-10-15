This dir contains work for creating ppt representing intj.

generate 2 ppt of the same content with different themes, one use the ppt theme @~/blank.potx, the other not, and use some light theme.

No cover (divider) page at the beginning of a chapter: a chapter starts directly with its first content page.

Call the step that decodes a launch's Python arguments into its specialization key "intj args -> spec-key", spelled exactly so, everywhere in the ppt (charts, tables, text, notes). Never "decode + key", "decode args" or "spec key" for that step. It is what `launcher.__self__._C_spec_key(...)` times: decode plus key, no cache lookup, grid or driver launch.

## Structure
files mentioned in this parts are relative to dir of this AGENTS.md.

@ppt.md is the outline of the ppt. It structures the whole ppt, each section and each page. detailed instruction for each page is contained in parts dir, and is refered to in ppt.md.
@parts dir contains details of each page.
@data dir contains performance data, migration statistics as source of data.
@results dir contains regenrated results.
