# Callable grid source survey

Study completed on 2026-09-24. This records the observed Python language needs
that informed the callable-grid API. Most source grids use integer arithmetic,
`triton.cdiv`, and a fixed tuple/list result. A hypothetical compiler with
scalar captures and Triton-style metadata access would fit 136 of the 145
identified definitions structurally. That number is **not** a compatibility
rate for the implemented `grid_cpp` API: it uses explicit annotated parameters
and refuses free variables. `grid_py` handles existing Python grid callables.
These counts also do not imply that intj can launch every underlying kernel.

## 1. Scope and counts

| Checkout | Revision | Scope examined |
|---|---|---|
| `aiter` | `874d253dd30770a00565398f901eca577a4478d1` | All 1,314 tracked Python files, including production kernels, tests and benchmarks |
| `triton` | `fe5a423a1e7c754c011205cc35ad0dfd94e3d6fa` | 31 tutorial Python files and 138 test Python files; runtime binding/launch code |
| This intj worktree | `4a3df7f359093f717f181fa1e7d684e53baa706f` | Launcher, rendered entry point, runtime helpers, annotations and usage documentation |

All three worktrees were clean before the study. Aiter and Triton were detached
at the revisions above. Their files were not changed or executed.

| Corpus | Callable definitions | Callable launch sites | Fits proposed scalar subset |
|---|---:|---:|---:|
| Aiter production, `aiter/ops/triton/` | 82 | 92 | 79 |
| Aiter tests/benchmarks | 3 | 3 | 3 |
| Triton tutorials, including Gluon and Proton | 23 | 24 | 21 |
| Triton tests, including AMD/Proton and embedded scripts | 37 | 51 | 33 |
| **Total** | **145** | **170** | **136** |

Definitions are source locations, not unique algorithms. Sites are syntactic
launch expressions, not runtime launch counts. Repeated formulas, autotuning,
parameterized tests and kernels that fail before launching are not normalized.
The combined number describes this corpus, not usage frequency or ecosystem
coverage.

The tutorial split is 18 definitions in standard tutorials, two in Gluon and
three in Proton. Test scanning covered `python/test/`,
`python/triton_kernels/tests/`, `third_party/amd/python/test/`,
`third_party/proton/test/` and `test/`. Of the 37 test definitions, 35 are visible
in the outer AST; two occur inside generated-script strings in
`python/test/backend/test_mir_stage.py`.

Method: parse tracked files with stdlib `ast`; locate launch subscriptions and
explicit `grid=` calls; resolve local names and returned closures; inspect the
callable bodies and surrounding callers. Cross-check with text searches for
lambdas, grid definitions, aliases and forwarding wrappers. Four Aiter sites
initially matched by a flow-insensitive resolver actually passed tuples from a
different branch and were removed. This is static analysis with manual checks,
not a proof about arbitrary imported wrappers or generated/reflected Python.

## 2. Language required by the common cases

| Feature | Actual evidence | Proposed first implementation |
|---|---|---|
| Lambdas and nested named functions | Aiter: 53 lambdas and 29 nested defs; Triton tutorials: 16 and 7 | Accept source-backed Python functions; one supplied metadata argument, with supported scalar defaults for additional parameters |
| Fixed tuple/list result | Aiter: 42 one-tuples, 22 two-tuples, 16 three-tuples and two one-element lists | Lower literal tuple/list syntax directly to 1–3 native dimensions |
| Metadata and scalar captures | Literal string keys; captured sizes, block counts and device counts | Resolve metadata against logical kernel parameters; read actual scalar closure/global/default bindings |
| Integer calculation | Aiter uses `+`, `-`, `*`, `//`, `triton.cdiv`; Triton adds two-argument builtin `min` | Checked integer arithmetic and recognized intrinsic calls |
| Simple statements and conditional expression | Local assignments in persistent matmul; one Aiter `a if flag else b` | Straight-line local assignments followed by return; lazy conditional-expression evaluation |

Typical source, from [Aiter GEMM][a-gemm]:

```python
grid = lambda META: (
    META["NUM_KSPLIT"]
    * triton.cdiv(M, META["BLOCK_SIZE_M"])
    * triton.cdiv(N, META["BLOCK_SIZE_N"]),
)
```

[HSTU backward][a-conditional] demonstrates the only directly observed
conditional expression:

```python
grid = lambda meta: (
    Z * H,
    triton.cdiv(N, meta["BLOCK_N"]) if meta["SEQUENCE_PARALLEL"] else 1,
)
```

The predicate is already a metadata value. No explicit comparison or boolean
operator is needed for this example, and the unselected branch must not run.

[Persistent matmul][t-locals] demonstrates ordinary local assignment:

```python
def grid(META):
    BLOCK_M = META["BLOCK_SIZE_M"]
    BLOCK_N = META["BLOCK_SIZE_N"]
    return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
```

Additional details that affect unchanged-source compatibility:

1. [RMSNorm][a-list] returns a list. It needs no native list allocation: its
   fixed elements can become output dimensions exactly like a tuple.
2. [Flash KDA][a-default] uses `lambda meta, _w=W: (...)`. The default captures
   `W` when the lambda is created; other names use closure cells. Rejecting every
   function with more than one declared parameter would exclude it needlessly.
3. [Convolution][a-factory] has four factories returning ordinary nested grid
   functions, reused at 11 launch sites. Python executes the factory before
   passing its result to the launcher; the grid compiler need only accept that
   resulting function.
4. [Persistent matmul][t-nonlocal] declares `nonlocal a_desc, b_desc, c_desc`
   without reading or writing those names. Accepting that unused declaration
   accounts for one of the 21 supported tutorial definitions. Capture discovery
   should follow actual expression loads, not require support for every name in
   `co_freevars`.

Across Aiter's 82 production bodies, 74 read metadata and 73 call `triton.cdiv`.
The only other calls are `max` and `len` in two sequence-dependent grids. No
production grid body reads tensor shape, stride, numel, dtype or device. Those
operations generally happen in the enclosing Python wrapper before the grid
function is constructed.

## 3. The nine outlier definitions

| Extension | Definitions | Source and exact additional requirement |
|---|---:|---|
| Captured tensor metadata | 5 | [Attention][t-shape] reads `q.shape[0]`, `[1]`, `[2]`; four test grids call captured `x.numel()`, including [matmul][t-numel] |
| Captured CPU integer sequence | 2 | [Causal convolution][a-sequence] and [its prefill variant][a-sequence-prefill] call `max(seq_lens_cpu)` and `len(seq_lens_cpu)` |
| Captured configuration dictionary | 1 | [FP4 GEMM][a-config] reads `config["NUM_KSPLIT"]`, independently of its `META` argument |
| User helper on nested constexpr data | 1 | [Gluon multicta][t-helper-call] calls `get_split_dim(meta["CGA_LAYOUT"], 0)` |

The Gluon [helper body][t-helper] is:

```python
return 1 << sum(b[dim] != 0 for b in cga_layout)
```

Literal helper translation would add iteration, `sum`, comparison, tuple
indexing and a shift. Its input is a kernel constexpr, so a future explicitly
supported constant-evaluation path could handle it without general runtime
loop support. The first implementation can refuse this helper.

Tensor metadata support should be a separate decision about accepted tensor
types and live reads. In three `numel()` tests, the captured tensor is the
original `x`, while the kernel receives `triton.reinterpret(x, dtype)`. Attention
may pass a tensor descriptor while capturing the original tensor. The capture
cannot generally be replaced with the corresponding kernel pointer argument.
The existing intj tensor ABI records `numel`, but does not expose shape/stride
accessors for grid calculations.

For sequence and dictionary captures, freezing contents silently would change
Python semantics. Either read and validate the supported container on each
call, or require users to capture precomputed scalar values. Empty-sequence
errors from `max` also need a defined policy if that extension is implemented.

There is no observed need in the common subset for statement-level `if`, loops,
recursion, comprehensions, arbitrary method calls, exceptions, mutation, float
arithmetic, `%`, `next_power_of_2`, dynamic metadata keys, callable classes,
bound methods or `functools.partial`. The Gluon helper above is the explicit
exception for generator/comparison/shift syntax when following callees.

Several apparent requirements disappeared after tracing call sites:

1. Aiter `_gmm_grid` helpers compute tuples before launch. Their GPU reductions
   and `.item()` calls are outside callable-grid evaluation.
2. Triton tests build some tuple grids with generator expressions and `zip`;
   the launcher receives the finished tuple.
3. Persistent-matmul descriptor mutations occur in a separate autotune pre-hook,
   not in its grid. Device-property queries likewise run before constructing
   scalar-capturing grids.
4. Benchmark runner lambdas and `@triton.heuristics` callbacks are separate from
   launch grids. Their syntax should not inflate this compiler's scope.

## 4. Semantics the native implementation must preserve

1. **Metadata means bound kernel arguments.** Triton's [binder][t-binder]
   constructs a dictionary of formal parameter names and their values,
   including defaults; extra launch options are separate. [JITFunction.run][t-run]
   calls `grid(bound_args)` on each non-warmup launch after obtaining a kernel,
   including kernel-cache hits. Autotuning supplies the selected configuration
   before this binding. A metadata key is not restricted to uppercase block
   sizes or inferred constexpr names. Intj must also account for baked/bound
   parameters omitted from its public call signature.
2. **Compiled code and capture values have different lifetimes.** Two closures
   can have the same code object and different sizes. A live cell or global can
   also change while the callable object stays the same. Preserve those reads,
   or refuse the unsupported case; do not cache a previous environment as if
   it were part of the function's immutable code. [An autotuner test][t-capture]
   captures `N` even though no kernel parameter is named `N`. Name matching
   cannot substitute kernel arguments for captures. Aiter's [fast launcher][a-fast]
   also evaluates the current grid with current arguments on cache hits and
   documents stale-shape/tensor-retention problems from keeping old bound maps.
3. **Python integer semantics need a stated native domain.** Recommend exact
   `int`/`bool` scalar inputs within a checked signed-64-bit arithmetic domain
   for v1; reject unsupported values/intermediate overflow. Python `//` floors
   rather than truncates, and division by zero must raise before launch.
   [Triton's cdiv][t-cdiv] is `(x + (y - 1)) // y`; lower that helper's actual
   semantics. Final dimensions retain intj's existing `[0, 2**32)` validation.
   The corpus does not itself test negative inputs or overflow, so new
   differential tests are necessary.
4. **Native evaluation must remain separate from Python dispatch.** Resolve
   `triton.cdiv` and builtin `min` by the supported callable's identity, not
   merely an AST spelling that could name an unrelated function. Unknown syntax,
   calls, objects, or unavailable source should raise `UnsupportedKernel` with
   a source location. Do not silently execute the Python grid on the hot path.
5. **Grid identity is not kernel specialization.** A different grid expression
   need not compile a different GPU binary. Include every baked grid-code
   dependency in the native artifact identity, and keep the GPU specialization
   key at the scope of the existing invariant. Retain immutable render/key
   fields, the full `EXT_SUFFIX`, and per-launcher state ownership.

## 5. Resulting API and checks

`grid_cpp` compiles a source-backed, annotated `def` with stdlib `ast` and an
explicit allowlist. Positional parameter names map to JIT parameters;
keyword-only names are extra launcher inputs. It refuses captures and global
data. Its checked C evaluator runs before the zero-grid decision, while kernel
argument packing and specialization remain independent of grid controls.

Existing metadata-style lambdas belong in `grid_py`: the extension builds a
fresh dict of all JIT values and calls Python on every launch. This preserves
live captures and callback behavior without pretending those functions can be
compiled. The legacy int/tuple/list grid call remains available.

Tests cover AMD 1D/2D/3D output, compiled arithmetic and refusals, current
Python metadata on cache hits, bound and baked values, and module ownership.
All three new template modes are compile-checked with CUDA driver branches.
Autotuning, heuristics, Gluon kernels and tensor descriptors remain separate
launcher limitations. This survey makes no launch-overhead claim for the new
grid modes.

[a-gemm]: /mnt/nvme2/jinpli/workspace/home/jinpli/development/aiter/aiter/ops/triton/gemm/basic/gemm_a16w16.py:501
[a-conditional]: /mnt/nvme2/jinpli/workspace/home/jinpli/development/aiter/aiter/ops/triton/attention/hstu_attention.py:214
[a-list]: /mnt/nvme2/jinpli/workspace/home/jinpli/development/aiter/aiter/ops/triton/normalization/rmsnorm.py:173
[a-default]: /mnt/nvme2/jinpli/workspace/home/jinpli/development/aiter/aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_kda.py:1063
[a-factory]: /mnt/nvme2/jinpli/workspace/home/jinpli/development/aiter/aiter/ops/triton/conv/_launch.py:59
[a-sequence]: /mnt/nvme2/jinpli/workspace/home/jinpli/development/aiter/aiter/ops/triton/conv/causal_conv1d.py:167
[a-sequence-prefill]: /mnt/nvme2/jinpli/workspace/home/jinpli/development/aiter/aiter/ops/triton/gated_delta_net/causal_conv1d_prefill.py:85
[a-config]: /mnt/nvme2/jinpli/workspace/home/jinpli/development/aiter/aiter/ops/triton/gemm/basic/gemm_a8wfp4.py:106
[a-fast]: /mnt/nvme2/jinpli/workspace/home/jinpli/development/aiter/aiter/ops/triton/_triton_kernels/chunk_delta_attn/fast_launch.py:112
[t-locals]: /mnt/nvme2/jinpli/workspace/home/jinpli/development/triton/python/tutorials/09-persistent-matmul.py:265
[t-nonlocal]: /mnt/nvme2/jinpli/workspace/home/jinpli/development/triton/python/tutorials/09-persistent-matmul.py:560
[t-shape]: /mnt/nvme2/jinpli/workspace/home/jinpli/development/triton/python/tutorials/06-fused-attention.py:550
[t-numel]: /mnt/nvme2/jinpli/workspace/home/jinpli/development/triton/python/test/unit/language/test_matmul.py:24
[t-helper-call]: /mnt/nvme2/jinpli/workspace/home/jinpli/development/triton/python/tutorials/gluon/14-multicta.py:1112
[t-helper]: /mnt/nvme2/jinpli/workspace/home/jinpli/development/triton/python/tutorials/gluon/14-multicta.py:663
[t-binder]: /mnt/nvme2/jinpli/workspace/home/jinpli/development/triton/python/triton/runtime/jit.py:443
[t-run]: /mnt/nvme2/jinpli/workspace/home/jinpli/development/triton/python/triton/runtime/jit.py:749
[t-capture]: /mnt/nvme2/jinpli/workspace/home/jinpli/development/triton/python/test/unit/runtime/test_autotuner.py:37
[t-cdiv]: /mnt/nvme2/jinpli/workspace/home/jinpli/development/triton/python/triton/__init__.py:69
