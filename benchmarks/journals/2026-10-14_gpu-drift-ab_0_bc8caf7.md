# GPU enqueue rows: `76bef91` vs `bc8caf7`, same day (index 0)

[`2026-10-14_full-suite_0_bc8caf7.md`](2026-10-14_full-suite_0_bc8caf7.md) read every `launch_gpu` row 4-9 %
slower than the 2026-09-29 record at `76bef91`. This A/B runs both revisions in one session to separate the 51
commits from the machine.

## Environment

- Baseline `76bef91` as a detached worktree at `/tmp/ab-76bef91` (removed afterwards); candidate `bc8caf7`, the
  clean `develop` checkout. Every raw file records its `commit=` and `/proc/loadavg`.
- Same machine and software as the full-suite record: Xeon Platinum 8480C on `taskset -c 0`, MI308X (`gfx942`)
  GPU 0, uv venv with Python 3.12.3, Torch `2.14.0.dev20260804+rocm7.1`, Triton `3.8.0`, apache-tvm-ffi
  `0.1.14.post1`, default `TRITON_HOME`. Load average about 0.6 at the start.
- 2026-10-14, about 16:47 to 16:52 +08:00. CUDA: untested.

## Command

`/tmp/intj-bench/run_ab.sh ab-gpu`: 3 rounds; each case runs both revisions back to back, `76bef91` first in
rounds 0 and 2 and `bc8caf7` first in round 1. Every process:
`cd <tree>; PYTHONPATH=$PWD taskset -c 0 <venv>/bin/python <args>` with `HIP_VISIBLE_DEVICES=0`.

```
launch_gpu      benchmarks/bench_launch.py --iters 1000 --batches 9
launch_readme   benchmarks/bench_launch.py --readme --iters 1000 --batches 9
ffi_compare     benchmarks/bench_ffi_compare.py --iters 1000 --batches 9
hip_module      benchmarks/bench_hip_module_launch.py --iters 1000 --batches 9
launch_dynamic  benchmarks/bench_launch.py --dynamic --iters 20000 --batches 9
```

Compared with `/tmp/intj-bench/bench_compare.py` restricted to these 5 cases (`CASES` overridden), median of the
3 process medians. All 30 files have `exit_status=0`.

## Result

**No regression; 0 of 40 rows flagged.** `bc8caf7` against `76bef91`, same session:

| case | rows | Δ % range |
|---|---:|---|
| `launch_gpu` | 12 | -2.5 .. +0.4 |
| `ffi_compare` (INTJ rows) | 4 | -0.1 .. +0.9 |
| `hip_module` | 3 | -1.9 .. +0.2 |
| `launch_readme` | 7 | -3.5 .. +1.5 |
| `launch_dynamic` | 8 | -7.7 .. -0.3 |

`launch_gpu` baked read 2.88 us at `76bef91` on 2026-09-29, and 3.13 us (`76bef91`) and 3.05 us (`bc8caf7`)
here: both revisions moved together, so the full-suite record's GPU-row increase is a machine shift (GPU, driver
or host state), not code. `launch_dynamic` launch read 3.0-3.1 us in all 6 processes, confirming the 3.8 us
rounds in the full-suite record were noise.

## Raw output

### Comparison (`76bef91` = baseline, `bc8caf7` = new)

```
ffi_compare      median           FFI empty kernel                 value    baseline=3.3163 new=3.3537 delta=0.0374 pct=+1.1% (us)
ffi_compare      median           FFI mixed kernel                 value    baseline=3.2883 new=3.3463 delta=0.0580 pct=+1.8% (us)
ffi_compare      median           FFI packed nop                   value    baseline=0.1463 new=0.1443 delta=-0.0020 pct=-1.4% (us)
ffi_compare      median           FFI packed nop mixed             value    baseline=0.1730 new=0.1714 delta=-0.0016 pct=-0.9% (us)
ffi_compare      median           FFI typed nop                    value    baseline=0.1515 new=0.1504 delta=-0.0011 pct=-0.7% (us)
ffi_compare      median           FFI typed nop mixed              value    baseline=0.1775 new=0.1774 delta=-0.0001 pct=-0.1% (us)
ffi_compare      median           INTJ empty kernel                value    baseline=2.9582 new=2.9644 delta=0.0062 pct=+0.2% (us)
ffi_compare      median           INTJ fixed mixed kernel          value    baseline=2.9689 new=2.9960 delta=0.0271 pct=+0.9% (us)
ffi_compare      median           INTJ fixed-device kernel         value    baseline=2.9636 new=2.9613 delta=-0.0023 pct=-0.1% (us)
ffi_compare      median           INTJ mixed kernel                value    baseline=2.9387 new=2.9609 delta=0.0222 pct=+0.8% (us)
hip_module       median           INTJ same function               value    baseline=3.0128 new=3.0071 delta=-0.0057 pct=-0.2% (us)
hip_module       median           TVM FFI HIP HSACO                value    baseline=3.2003 new=3.2070 delta=0.0067 pct=+0.2% (us)
hip_module       median           Triton same HSACO                value    baseline=15.8066 new=15.5114 delta=-0.2952 pct=-1.9% (us)
launch_dynamic   dynamic          2 dynamic options                launch   baseline=3129.0 new=3118.9 delta=-10.1000 pct=-0.3% (ns)
launch_dynamic   dynamic          2 dynamic options                spec_key baseline=99.8000 new=92.1000 delta=-7.7000 pct=-7.7% (ns)
launch_dynamic   dynamic          int constexpr                    launch   baseline=3026.6 new=3016.1 delta=-10.5000 pct=-0.3% (ns)
launch_dynamic   dynamic          int constexpr                    spec_key baseline=88.4000 new=87.3000 delta=-1.1000 pct=-1.2% (ns)
launch_dynamic   dynamic          lazy plain                       launch   baseline=3123.7 new=3099.8 delta=-23.9000 pct=-0.8% (ns)
launch_dynamic   dynamic          lazy plain                       spec_key baseline=94.5000 new=94.8000 delta=0.3000 pct=+0.3% (ns)
launch_dynamic   dynamic          str constexpr                    launch   baseline=3036.1 new=3019.1 delta=-17.0000 pct=-0.6% (ns)
launch_dynamic   dynamic          str constexpr                    spec_key baseline=92.0000 new=88.4000 delta=-3.6000 pct=-3.9% (ns)
launch_gpu       launch           auto map                         ns/call  baseline=3315.7 new=3248.1 delta=-67.6000 pct=-2.0% (ns)
launch_gpu       launch           baked                            ns/call  baseline=3129.3 new=3051.3 delta=-78.0000 pct=-2.5% (ns)
launch_gpu       launch           bound pointer                    ns/call  baseline=3117.9 new=3064.0 delta=-53.9000 pct=-1.7% (ns)
launch_gpu       launch           bound tensor                     ns/call  baseline=3143.9 new=3079.3 delta=-64.6000 pct=-2.1% (ns)
launch_gpu       launch           fixed device map                 ns/call  baseline=3144.5 new=3064.6 delta=-79.9000 pct=-2.5% (ns)
launch_gpu       launch           fixed device no-map              ns/call  baseline=3151.6 new=3074.5 delta=-77.1000 pct=-2.4% (ns)
launch_gpu       launch           generic                          ns/call  baseline=3181.3 new=3178.8 delta=-2.5000 pct=-0.1% (ns)
launch_gpu       launch           generic tensor ann               ns/call  baseline=3166.6 new=3171.1 delta=4.5000 pct=+0.1% (ns)
launch_gpu       launch           reduced key                      ns/call  baseline=3082.8 new=3094.8 delta=12.0000 pct=+0.4% (ns)
launch_gpu       launch           tensor annotated                 ns/call  baseline=3337.4 new=3310.9 delta=-26.5000 pct=-0.8% (ns)
launch_gpu       launch           verify off                       ns/call  baseline=3077.6 new=3087.8 delta=10.2000 pct=+0.3% (ns)
launch_gpu       launch           verify on                        ns/call  baseline=3092.1 new=3082.9 delta=-9.2000 pct=-0.3% (ns)
launch_readme    readme_path      grid=(0,) intj                   us       baseline=0.0300 new=0.0300 delta=0.0000 pct=+0.0% (us)
launch_readme    readme_path      grid=(0,) triton                 us       baseline=12.6700 new=12.6300 delta=-0.0400 pct=-0.3% (us)
launch_readme    readme_path      grid=(1,) intj                   us       baseline=3.3300 new=3.3400 delta=0.0100 pct=+0.3% (us)
launch_readme    readme_path      grid=(1,) triton                 us       baseline=16.3100 new=16.5500 delta=0.2400 pct=+1.5% (us)
launch_readme    readme_torch_access interpreter                      decode ns baseline=901.8 new=890.5 delta=-11.3000 pct=-1.3% (ns)
launch_readme    readme_torch_access runtime_shim                     decode ns baseline=92.7000 new=90.5000 delta=-2.2000 pct=-2.4% (ns)
launch_readme    readme_torch_access static_compile                   decode ns baseline=96.6000 new=93.2000 delta=-3.4000 pct=-3.5% (ns)
```

### `/tmp/intj-bench/ab-gpu/old`

#### `old/round0_ffi_compare.txt`

```
commit=76bef910d3e5dd69fd5cb681bdcd89ccaf1066d7
loadavg=0.77 1.82 1.74 1/7279 1283172
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1441 samples=[0.147048, 0.154284, 0.14280199999999998, 0.143923, 0.144622, 0.143461, 0.144149, 0.14575, 0.143927]
FFI typed nop              median=0.1515 samples=[0.15220699999999998, 0.15148599999999998, 0.148477, 0.152784, 0.147788, 0.149141, 0.152501, 0.152847, 0.150808]
FFI empty kernel           median=3.3163 samples=[3.614282, 3.526063, 3.5530790000000003, 3.3162510000000003, 3.3500189999999996, 3.285559, 3.290078, 3.2709960000000002, 3.29625]
INTJ empty kernel          median=2.9582 samples=[3.452351, 3.133104, 3.153369, 2.958184, 2.965344, 2.886318, 2.928014, 2.872632, 2.9033339999999996]
INTJ fixed-device kernel   median=2.9650 samples=[3.4325259999999997, 3.126739, 3.147676, 2.972449, 2.964968, 2.9161219999999997, 2.920711, 2.905124, 2.924398]
FFI packed nop mixed       median=0.1715 samples=[0.17193, 0.17097, 0.17134100000000002, 0.170883, 0.174074, 0.174176, 0.171517, 0.171275, 0.175329]
FFI typed nop mixed        median=0.1775 samples=[0.176544, 0.178744, 0.180925, 0.178732, 0.173989, 0.177466, 0.17882900000000002, 0.174466, 0.176719]
FFI mixed kernel           median=3.2883 samples=[3.481237, 3.465297, 3.532318, 3.363439, 3.288312, 3.2782970000000002, 3.276094, 3.2620120000000004, 3.272268]
INTJ mixed kernel          median=2.9279 samples=[3.106853, 3.09259, 3.145899, 2.9259299999999997, 2.924204, 2.851066, 2.927899, 2.842391, 2.932339]
INTJ fixed mixed kernel    median=2.9689 samples=[3.1578000000000004, 3.1589810000000003, 3.0319380000000002, 2.971435, 2.968893, 2.8946199999999997, 2.9511860000000003, 2.897869, 2.950994]
exit_status=0
```

#### `old/round0_hip_module.txt`

```
commit=76bef910d3e5dd69fd5cb681bdcd89ccaf1066d7
loadavg=1.01 1.76 1.72 1/7274 1283248
hsaco_sha256=03409dde23b75ed425f9f7cc5341892b5f85c3c81874562ccaec070852969125 kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.4499 samples=[3.596653, 3.449922, 3.458526, 3.470609, 3.461826, 3.323764, 3.3205270000000002, 3.290864, 3.269853]
Triton same HSACO         median=15.8451 samples=[15.973621999999999, 15.872475, 15.875895, 15.905757, 15.845139999999999, 15.785027, 15.745379999999999, 15.772694, 15.676144]
INTJ same function        median=3.2166 samples=[3.216647, 3.243403, 3.252152, 3.251461, 3.2239079999999998, 3.105592, 3.0949319999999996, 3.099186, 3.043399]
exit_status=0
```

#### `old/round0_launch_dynamic.txt`

```
commit=76bef910d3e5dd69fd5cb681bdcd89ccaf1066d7
loadavg=1.08 1.73 1.71 1/7286 1283340
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
         lazy plain     3136.6       93.3
  2 dynamic options     3136.6       99.8
      int constexpr     3026.6       88.4
      str constexpr     3036.1       90.7
exit_status=0
```

#### `old/round0_launch_gpu.txt`

```
commit=76bef910d3e5dd69fd5cb681bdcd89ccaf1066d7
loadavg=0.55 1.94 1.78 3/7282 1283108
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3339.1       +0.0      +0.0      77.44         -
   tensor annotated     3368.6      +29.5      +0.9       3.39         -
            generic     3271.5      -67.6      -2.0       2.96         -
 generic tensor ann     3166.6     -172.6      -5.2       3.03         -
        reduced key     3082.8     -256.3      -7.7       2.86         -
         verify off     3118.1     -221.0      -6.6       2.88         -
          verify on     3092.1     -247.0      -7.4       2.81         -
              baked     3173.9     -165.2      -4.9       2.80         -
       bound tensor     3165.8     -173.3      -5.2       2.68      0.12
      bound pointer     3134.6     -204.6      -6.1       2.68      0.12
   fixed device map     3191.1     -148.0      -4.4       2.62      0.13
fixed device no-map     3160.0     -179.1      -5.4       2.66      0.11
exit_status=0
```

#### `old/round0_launch_readme.txt`

```
commit=76bef910d3e5dd69fd5cb681bdcd89ccaf1066d7
loadavg=0.68 1.88 1.76 1/7278 1283134
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.37       3.33      4.9x
   grid=(0,)       12.72       0.03    471.4x

torch_access_mode   decode ns    build s
     runtime_shim        91.7       0.02
   static_compile        96.0       0.06
      interpreter       896.0       0.00
exit_status=0
```

#### `old/round1_ffi_compare.txt`

```
commit=76bef910d3e5dd69fd5cb681bdcd89ccaf1066d7
loadavg=1.21 1.62 1.68 1/7276 1283515
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1505 samples=[0.154798, 0.150522, 0.14961000000000002, 0.150846, 0.15, 0.14990299999999998, 0.15175, 0.149571, 0.151874]
FFI typed nop              median=0.1515 samples=[0.153174, 0.151454, 0.151205, 0.15334299999999998, 0.15146, 0.15073699999999998, 0.152356, 0.151906, 0.15047]
FFI empty kernel           median=3.2643 samples=[3.7817109999999996, 3.476817, 3.4988989999999998, 3.264289, 3.29074, 3.22281, 3.2487399999999997, 3.222875, 3.248425]
INTJ empty kernel          median=2.9153 samples=[3.381193, 3.112168, 3.156879, 2.9439499999999996, 2.915306, 2.892733, 2.8973, 2.8815999999999997, 2.911548]
INTJ fixed-device kernel   median=2.9271 samples=[3.047275, 3.126768, 3.147502, 2.9888879999999998, 2.927133, 2.918973, 2.913665, 2.9159029999999997, 2.905103]
FFI packed nop mixed       median=0.1769 samples=[0.17579499999999998, 0.178677, 0.17683000000000001, 0.17558, 0.179392, 0.17722200000000002, 0.17803200000000002, 0.176891, 0.176601]
FFI typed nop mixed        median=0.1791 samples=[0.179347, 0.177865, 0.179308, 0.184468, 0.179343, 0.179107, 0.178066, 0.178455, 0.178907]
FFI mixed kernel           median=3.2730 samples=[3.4204369999999997, 3.446627, 3.344645, 3.304598, 3.27298, 3.2518290000000003, 3.261881, 3.252618, 3.261012]
INTJ mixed kernel          median=2.9387 samples=[3.157626, 3.086332, 3.000212, 2.9141179999999998, 2.938677, 2.840697, 2.940868, 2.8541190000000003, 2.935873]
INTJ fixed mixed kernel    median=2.9495 samples=[3.1045279999999997, 3.1126329999999998, 3.019598, 2.964102, 2.949542, 2.8915059999999997, 2.9442399999999997, 2.882082, 2.9475749999999996]
exit_status=0
```

#### `old/round1_hip_module.txt`

```
commit=76bef910d3e5dd69fd5cb681bdcd89ccaf1066d7
loadavg=1.15 1.58 1.66 1/7267 1283544
hsaco_sha256=03409dde23b75ed425f9f7cc5341892b5f85c3c81874562ccaec070852969125 kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.2003 samples=[3.348322, 3.182194, 3.2003429999999997, 3.2646610000000003, 3.242248, 3.20504, 3.176437, 3.182674, 3.161378]
Triton same HSACO         median=15.8066 samples=[15.815205, 15.764707, 15.792803, 15.903003, 15.934239999999999, 15.781972, 15.806567999999999, 15.774269, 15.845333]
INTJ same function        median=3.0128 samples=[3.081233, 3.037886, 3.1151060000000004, 3.124075, 3.01276, 2.971166, 2.9658200000000003, 2.985643, 2.9436419999999996]
exit_status=0
```

#### `old/round1_launch_dynamic.txt`

```
commit=76bef910d3e5dd69fd5cb681bdcd89ccaf1066d7
loadavg=1.18 1.55 1.65 1/7263 1283585
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
         lazy plain     3100.3       94.5
  2 dynamic options     3078.9       99.1
      int constexpr     2990.0       87.3
      str constexpr     2980.4       92.0
exit_status=0
```

#### `old/round1_launch_gpu.txt`

```
commit=76bef910d3e5dd69fd5cb681bdcd89ccaf1066d7
loadavg=1.21 1.68 1.70 1/7271 1283433
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3224.5       +0.0      +0.0      77.49         -
   tensor annotated     3319.5      +95.0      +2.9       3.34         -
            generic     3167.5      -57.0      -1.8       2.91         -
 generic tensor ann     3130.9      -93.6      -2.9       2.96         -
        reduced key     3051.4     -173.1      -5.4       2.79         -
         verify off     3062.0     -162.5      -5.0       2.82         -
          verify on     3046.7     -177.8      -5.5       2.81         -
              baked     3106.1     -118.4      -3.7       2.82         -
       bound tensor     3143.9      -80.6      -2.5       2.66      0.12
      bound pointer     3117.9     -106.6      -3.3       2.72      0.11
   fixed device map     3144.5      -80.0      -2.5       2.66      0.13
fixed device no-map     3151.6      -73.0      -2.3       2.73      0.11
exit_status=0
```

#### `old/round1_launch_readme.txt`

```
commit=76bef910d3e5dd69fd5cb681bdcd89ccaf1066d7
loadavg=1.30 1.67 1.69 1/7270 1283474
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.30       3.33      4.9x
   grid=(0,)       12.64       0.03    468.0x

torch_access_mode   decode ns    build s
     runtime_shim        93.5       0.02
   static_compile        96.6       0.06
      interpreter       901.8       0.00
exit_status=0
```

#### `old/round2_ffi_compare.txt`

```
commit=76bef910d3e5dd69fd5cb681bdcd89ccaf1066d7
loadavg=1.16 1.50 1.62 1/7282 1283688
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1463 samples=[0.151499, 0.146285, 0.144371, 0.146217, 0.146505, 0.144926, 0.148141, 0.14619300000000002, 0.147581]
FFI typed nop              median=0.1515 samples=[0.152728, 0.152106, 0.148221, 0.152479, 0.150347, 0.151498, 0.14641200000000001, 0.15357300000000002, 0.14934999999999998]
FFI empty kernel           median=3.3645 samples=[3.797581, 3.57138, 3.621044, 3.364524, 3.3834989999999996, 3.3343960000000004, 3.35352, 3.323436, 3.3499749999999997]
INTJ empty kernel          median=2.9888 samples=[3.277815, 3.184836, 3.2142429999999997, 2.9954989999999997, 2.9887550000000003, 2.940779, 2.971349, 2.9347869999999996, 2.9666390000000002]
INTJ fixed-device kernel   median=2.9636 samples=[3.0153499999999998, 3.161324, 3.182031, 2.989947, 2.952792, 2.959398, 2.946184, 2.9498939999999996, 2.96359]
FFI packed nop mixed       median=0.1730 samples=[0.17296199999999998, 0.171663, 0.171826, 0.172964, 0.17300100000000002, 0.173922, 0.172977, 0.173226, 0.171642]
FFI typed nop mixed        median=0.1770 samples=[0.177035, 0.17793, 0.176641, 0.178014, 0.17396, 0.177118, 0.174373, 0.17882599999999998, 0.174567]
FFI mixed kernel           median=3.3860 samples=[3.474013, 3.54096, 3.45727, 3.4204499999999998, 3.385986, 3.339512, 3.350383, 3.341978, 3.3497779999999997]
INTJ mixed kernel          median=2.9907 samples=[3.2071799999999997, 3.148962, 3.024864, 2.9513649999999996, 2.976339, 2.891902, 3.001376, 2.891241, 2.99068]
INTJ fixed mixed kernel    median=2.9957 samples=[3.157969, 3.1559589999999997, 3.097134, 3.014955, 2.995682, 2.920599, 2.972543, 2.921877, 2.983797]
exit_status=0
```

#### `old/round2_hip_module.txt`

```
commit=76bef910d3e5dd69fd5cb681bdcd89ccaf1066d7
loadavg=1.18 1.48 1.62 1/7287 1283734
hsaco_sha256=03409dde23b75ed425f9f7cc5341892b5f85c3c81874562ccaec070852969125 kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.2003 samples=[4.763203000000001, 3.19418, 3.185554, 3.2925590000000002, 3.2056, 3.200275, 3.203605, 3.16147, 3.168098]
Triton same HSACO         median=15.4891 samples=[15.552829, 15.489139, 15.529027, 15.577129999999999, 15.406739, 15.411618, 15.377133, 15.376549, 15.620880999999999]
INTJ same function        median=2.9853 samples=[3.1059650000000003, 3.03725, 3.1040010000000002, 3.118263, 2.982527, 2.985346, 2.960967, 2.958699, 2.936871]
exit_status=0
```

#### `old/round2_launch_dynamic.txt`

```
commit=76bef910d3e5dd69fd5cb681bdcd89ccaf1066d7
loadavg=1.20 1.46 1.60 1/7281 1283768
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
         lazy plain     3123.7       94.7
  2 dynamic options     3129.0      100.0
      int constexpr     3091.8       90.9
      str constexpr     3077.2       92.9
exit_status=0
```

#### `old/round2_launch_gpu.txt`

```
commit=76bef910d3e5dd69fd5cb681bdcd89ccaf1066d7
loadavg=1.23 1.55 1.65 1/7272 1283613
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3315.7       +0.0      +0.0      77.84         -
   tensor annotated     3337.4      +21.8      +0.7       3.39         -
            generic     3181.3     -134.4      -4.1       2.95         -
 generic tensor ann     3209.8     -105.9      -3.2       3.01         -
        reduced key     3105.3     -210.3      -6.3       2.84         -
         verify off     3077.6     -238.0      -7.2       2.80         -
          verify on     3129.8     -185.9      -5.6       2.78         -
              baked     3129.3     -186.3      -5.6       2.80         -
       bound tensor     3074.0     -241.6      -7.3       2.69      0.12
      bound pointer     3048.7     -267.0      -8.1       2.67      0.12
   fixed device map     3013.5     -302.2      -9.1       2.60      0.13
fixed device no-map     3099.4     -216.2      -6.5       2.73      0.11
exit_status=0
```

#### `old/round2_launch_readme.txt`

```
commit=76bef910d3e5dd69fd5cb681bdcd89ccaf1066d7
loadavg=1.23 1.53 1.64 1/7280 1283647
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.31       3.40      4.8x
   grid=(0,)       12.67       0.03    470.7x

torch_access_mode   decode ns    build s
     runtime_shim        92.7       0.02
   static_compile        97.2       0.06
      interpreter       911.2       0.00
exit_status=0
```

### `/tmp/intj-bench/ab-gpu/new`

#### `new/round0_ffi_compare.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
loadavg=0.92 1.77 1.72 1/7281 1283234
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1439 samples=[0.15074, 0.143184, 0.14435900000000002, 0.1459, 0.145316, 0.143924, 0.14343199999999998, 0.14313, 0.14322200000000002]
FFI typed nop              median=0.1498 samples=[0.151243, 0.152314, 0.146274, 0.14875899999999997, 0.146861, 0.151949, 0.149801, 0.155545, 0.149042]
FFI empty kernel           median=3.3537 samples=[3.788164, 3.52967, 3.573373, 3.34664, 3.353701, 3.284801, 3.3195279999999996, 3.527681, 3.3228519999999997]
INTJ empty kernel          median=2.9488 samples=[3.124381, 3.1456109999999997, 3.187335, 2.970956, 2.947741, 2.911188, 2.943698, 2.9136979999999997, 2.9487840000000003]
INTJ fixed-device kernel   median=2.9992 samples=[3.027037, 3.2817730000000003, 3.140782, 3.007103, 2.944241, 2.936533, 2.9363989999999998, 2.943134, 2.999216]
FFI packed nop mixed       median=0.1712 samples=[0.17246899999999998, 0.170382, 0.17178, 0.171227, 0.17137, 0.1711, 0.170807, 0.172363, 0.17053800000000002]
FFI typed nop mixed        median=0.1755 samples=[0.175763, 0.17929, 0.17521, 0.174391, 0.17451, 0.175927, 0.175196, 0.178679, 0.175486]
FFI mixed kernel           median=3.3221 samples=[3.446291, 3.508791, 3.3894349999999998, 3.39368, 3.31787, 3.3196790000000003, 3.312292, 3.306822, 3.3221239999999996]
INTJ mixed kernel          median=2.9607 samples=[3.117145, 3.105853, 3.034752, 2.9521529999999996, 2.956424, 2.881011, 2.967558, 2.884902, 2.960688]
INTJ fixed mixed kernel    median=2.9960 samples=[3.127949, 3.142366, 3.069799, 2.998092, 2.972432, 2.90814, 2.996019, 2.9143440000000003, 2.98099]
exit_status=0
```

#### `new/round0_hip_module.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
loadavg=1.09 1.75 1.72 1/7280 1283275
hsaco_sha256=03409dde23b75ed425f9f7cc5341892b5f85c3c81874562ccaec070852969125 kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.2070 samples=[3.453916, 3.207009, 3.2130639999999997, 3.3011179999999998, 3.260583, 3.195393, 3.170815, 3.170236, 3.1774839999999998]
Triton same HSACO         median=15.4836 samples=[15.554343999999999, 15.561549999999999, 15.588093, 15.675184, 15.483630999999999, 15.377784, 15.398515999999999, 15.385435, 15.37334]
INTJ same function        median=3.0163 samples=[3.084072, 3.016254, 3.1022420000000004, 3.117778, 3.019664, 2.966757, 2.953777, 2.9444369999999997, 2.945687]
exit_status=0
```

#### `new/round0_launch_dynamic.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
loadavg=1.22 1.72 1.71 1/7287 1283409
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
         lazy plain     3090.4       93.6
  2 dynamic options     3128.5       90.0
      int constexpr     2984.1       87.3
      str constexpr     2989.2       88.5
exit_status=0
```

#### `new/round0_launch_gpu.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
loadavg=0.62 1.91 1.77 1/7279 1283123
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3248.1       +0.0      +0.0      18.36         -
   tensor annotated     3300.5      +52.3      +1.6       3.60         -
            generic     3178.8      -69.4      -2.1       2.94         -
 generic tensor ann     3119.9     -128.3      -3.9       2.86         -
        reduced key     3094.8     -153.4      -4.7       3.04         -
         verify off     3081.7     -166.5      -5.1       2.86         -
          verify on     3082.9     -165.3      -5.1       2.79         -
              baked     3051.3     -196.9      -6.1       2.92         -
       bound tensor     3067.5     -180.7      -5.6       2.81      0.03
      bound pointer     3064.0     -184.1      -5.7       2.79      0.03
   fixed device map     3064.6     -183.6      -5.7       2.75      0.02
fixed device no-map     3060.5     -187.7      -5.8       2.79      0.02
exit_status=0
```

#### `new/round0_launch_readme.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
loadavg=0.73 1.85 1.75 3/7289 1283159
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.20       3.36      4.8x
   grid=(0,)       12.69       0.03    435.1x

torch_access_mode   decode ns    build s
     runtime_shim        91.9       0.02
   static_compile        93.2       0.00
      interpreter       889.7       0.00
exit_status=0
```

#### `new/round1_ffi_compare.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
loadavg=1.25 1.65 1.69 1/7269 1283493
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1443 samples=[0.151157, 0.147386, 0.143739, 0.144303, 0.14376, 0.144121, 0.146148, 0.14306, 0.14896399999999999]
FFI typed nop              median=0.1504 samples=[0.14937799999999998, 0.151394, 0.150719, 0.15123599999999998, 0.15041300000000002, 0.15075, 0.146644, 0.150118, 0.149808]
FFI empty kernel           median=3.3573 samples=[3.792026, 3.582579, 3.601445, 3.371238, 3.357277, 3.316552, 3.327187, 3.326462, 3.33286]
INTJ empty kernel          median=2.9713 samples=[3.110474, 3.182835, 3.213896, 2.970203, 2.977395, 2.922286, 2.971257, 2.9279349999999997, 2.958355]
INTJ fixed-device kernel   median=2.9613 samples=[3.033828, 3.150709, 3.182377, 2.996145, 2.958758, 2.96131, 2.957259, 2.940364, 2.958484]
FFI packed nop mixed       median=0.1757 samples=[0.17590199999999998, 0.173142, 0.17617, 0.174177, 0.17334200000000002, 0.174487, 0.175656, 0.176536, 0.177607]
FFI typed nop mixed        median=0.1774 samples=[0.176998, 0.17819200000000002, 0.177393, 0.17772900000000003, 0.176321, 0.179125, 0.176636, 0.178781, 0.176974]
FFI mixed kernel           median=3.3778 samples=[3.481281, 3.5319540000000003, 3.453375, 3.425312, 3.377833, 3.338326, 3.3651489999999997, 3.328683, 3.3562469999999998]
INTJ mixed kernel          median=2.9901 samples=[3.0818209999999997, 3.143837, 3.0418429999999996, 2.965038, 2.995467, 2.921349, 2.990126, 2.906994, 2.986236]
INTJ fixed mixed kernel    median=3.0105 samples=[3.149511, 3.162632, 3.089944, 3.017668, 3.0015880000000004, 2.939715, 3.0104520000000003, 2.923166, 2.982205]
exit_status=0
```

#### `new/round1_hip_module.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
loadavg=1.18 1.60 1.67 1/7268 1283530
hsaco_sha256=03409dde23b75ed425f9f7cc5341892b5f85c3c81874562ccaec070852969125 kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.2164 samples=[3.31804, 3.2250199999999998, 3.212013, 3.315543, 3.282256, 3.2163649999999997, 3.210095, 3.2159989999999996, 3.2105129999999997]
Triton same HSACO         median=16.0350 samples=[16.117440000000002, 16.026154000000002, 16.049404, 16.188648999999998, 16.043297, 15.981804, 16.035044, 15.936807, 15.892843000000001]
INTJ same function        median=3.0071 samples=[3.0489499999999996, 3.0583780000000003, 3.149644, 3.1424450000000004, 3.007062, 2.995692, 2.986587, 2.9963200000000003, 3.000015]
exit_status=0
```

#### `new/round1_launch_dynamic.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
loadavg=1.13 1.56 1.66 1/7268 1283568
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
         lazy plain     3099.8       94.8
  2 dynamic options     3108.2       92.1
      int constexpr     3016.1       88.2
      str constexpr     3035.3       88.4
exit_status=0
```

#### `new/round1_launch_gpu.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
loadavg=1.25 1.70 1.71 1/7274 1283421
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3239.0       +0.0      +0.0      18.28         -
   tensor annotated     3310.9      +71.9      +2.2       3.60         -
            generic     3159.9      -79.2      -2.4       2.94         -
 generic tensor ann     3178.1      -61.0      -1.9       2.88         -
        reduced key     3095.6     -143.4      -4.4       2.99         -
         verify off     3123.3     -115.8      -3.6       2.88         -
          verify on     3095.3     -143.7      -4.4       2.87         -
              baked     3083.6     -155.4      -4.8       2.93         -
       bound tensor     3101.8     -137.2      -4.2       2.83      0.03
      bound pointer     3187.8      -51.2      -1.6       2.79      0.03
   fixed device map     3038.9     -200.1      -6.2       2.76      0.02
fixed device no-map     3118.5     -120.6      -3.7       2.70      0.02
exit_status=0
```

#### `new/round1_launch_readme.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
loadavg=1.18 1.66 1.69 1/7282 1283461
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.62       3.33      5.0x
   grid=(0,)       12.63       0.03    465.0x

torch_access_mode   decode ns    build s
     runtime_shim        87.1       0.02
   static_compile        93.3       0.00
      interpreter       890.5       0.00
exit_status=0
```

#### `new/round2_ffi_compare.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
loadavg=1.22 1.50 1.62 1/7282 1283709
iters=1000 batches=9 unit=us/call (host-only, no timed sync)
FFI packed nop             median=0.1452 samples=[0.151532, 0.146003, 0.147121, 0.145281, 0.143121, 0.14522900000000002, 0.144809, 0.143818, 0.141847]
FFI typed nop              median=0.1515 samples=[0.148982, 0.151701, 0.148478, 0.152257, 0.146582, 0.152302, 0.14959, 0.151527, 0.151809]
FFI empty kernel           median=3.3418 samples=[3.80448, 3.53848, 3.575181, 3.341837, 3.347789, 3.279335, 3.3198220000000003, 3.304268, 3.3219830000000004]
INTJ empty kernel          median=2.9644 samples=[3.116682, 3.1665970000000003, 3.1965630000000003, 2.964407, 2.979583, 2.927464, 2.9541350000000004, 2.924701, 2.9431930000000004]
INTJ fixed-device kernel   median=2.9579 samples=[3.042544, 3.1524490000000003, 3.176144, 3.010124, 2.957917, 2.950665, 2.950888, 2.942478, 2.9541239999999998]
FFI packed nop mixed       median=0.1714 samples=[0.17141, 0.172268, 0.171593, 0.174383, 0.17207, 0.169889, 0.171428, 0.169902, 0.170505]
FFI typed nop mixed        median=0.1778 samples=[0.177386, 0.17829599999999998, 0.179085, 0.179267, 0.17610800000000001, 0.177839, 0.176214, 0.180576, 0.1747]
FFI mixed kernel           median=3.3463 samples=[3.459574, 3.515175, 3.380748, 3.3857150000000003, 3.322087, 3.314242, 3.346317, 3.329719, 3.336239]
INTJ mixed kernel          median=2.9609 samples=[3.1443290000000004, 3.1260470000000002, 3.0235079999999996, 2.951989, 2.9608719999999997, 2.881769, 2.96042, 2.88831, 2.962252]
INTJ fixed mixed kernel    median=2.9899 samples=[3.154887, 3.1458180000000002, 3.0668539999999997, 2.989936, 2.978047, 2.912754, 2.9928980000000003, 2.9143980000000003, 2.988406]
exit_status=0
```

#### `new/round2_hip_module.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
loadavg=1.15 1.46 1.61 1/7284 1283753
hsaco_sha256=03409dde23b75ed425f9f7cc5341892b5f85c3c81874562ccaec070852969125 kernel=empty_kernel threads=256 shared=0 callback_count=1
iters=1000 batches=9 unit=us/call (host enqueue, no timed sync)
TVM FFI HIP HSACO         median=3.2047 samples=[3.3042979999999997, 3.2047339999999997, 3.229551, 3.2880320000000003, 3.2223640000000002, 3.188236, 3.162276, 3.181968, 3.169165]
Triton same HSACO         median=15.5114 samples=[15.601175, 15.461885, 15.504137, 15.617497, 15.543552, 15.406637, 15.511389999999999, 15.461804, 15.678598]
INTJ same function        median=2.9699 samples=[3.022411, 3.025942, 3.105991, 3.197737, 2.96992, 2.945398, 2.946147, 2.944285, 2.9668870000000003]
exit_status=0
```

#### `new/round2_launch_dynamic.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
loadavg=1.25 1.46 1.60 1/7287 1283789
mode=dynamic; 20000 calls × 9 batches; median ns/call
               path     launch   spec_key
         lazy plain     3104.0       95.0
  2 dynamic options     3118.9       95.1
      int constexpr     3146.2       87.1
      str constexpr     3019.1       86.3
exit_status=0
```

#### `new/round2_launch_gpu.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
loadavg=1.27 1.55 1.65 2/7269 1283626
mode=gpu; 1000 calls × 9 batches; median ns/call
               path    ns/call       Δ ns       Δ %   build ms   bind ms
           auto map     3273.6       +0.0      +0.0      18.20         -
   tensor annotated     3316.5      +42.9      +1.3       3.62         -
            generic     3178.9      -94.7      -2.9       2.95         -
 generic tensor ann     3171.1     -102.4      -3.1       2.86         -
        reduced key     3089.5     -184.1      -5.6       3.03         -
         verify off     3087.8     -185.7      -5.7       2.83         -
          verify on     3076.8     -196.8      -6.0       2.78         -
              baked     3042.6     -230.9      -7.1       2.91         -
       bound tensor     3079.3     -194.3      -5.9       2.71      0.03
      bound pointer     3038.1     -235.5      -7.2       2.76      0.03
   fixed device map     3070.7     -202.8      -6.2       2.76      0.02
fixed device no-map     3074.5     -199.1      -6.1       2.75      0.02
exit_status=0
```

#### `new/round2_launch_readme.txt`

```
commit=bc8caf71f9323fb05dc2458f1e12ed8dd9ede5f8
loadavg=1.19 1.51 1.63 1/7280 1283666
mode=readme; 1000 calls × 9 batches; median
        path   triton us    intj us   speedup
   grid=(1,)       16.55       3.34      5.0x
   grid=(0,)       12.52       0.03    465.6x

torch_access_mode   decode ns    build s
     runtime_shim        90.5       0.02
   static_compile        92.2       0.00
      interpreter       916.8       0.00
exit_status=0
```
