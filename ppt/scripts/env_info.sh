#!/bin/bash
# usage: env_info.sh <cpu> <hip-index>   prints the machine and software state
echo "date: $(date -Iseconds)"
echo "host: $(hostname)"
lscpu | grep -E "^Model name|^CPU\(s\)|^Socket|^NUMA node[0-9]|MHz"
echo "pinned cpu $1 governor: $(cat /sys/devices/system/cpu/cpu$1/cpufreq/scaling_governor 2>/dev/null)"
echo "loadavg: $(cat /proc/loadavg)"
echo "kernel: $(uname -r)"
"$PY" - <<'PY'
import sys, torch, triton
print("python", sys.version.split()[0])
print("torch", torch.__version__, "hip", torch.version.hip, "cuda", torch.version.cuda)
print("triton", triton.__version__)
try:
    import tvm_ffi; print("apache-tvm-ffi", tvm_ffi.__version__)
except ImportError:
    print("apache-tvm-ffi absent")
d = torch.cuda.current_device(); p = torch.cuda.get_device_properties(d)
arch = p.gcnArchName if torch.version.hip else "sm_%d%d" % (p.major, p.minor)
print("gpu", torch.cuda.get_device_name(d), arch, "CUs/SMs", p.multi_processor_count)
PY
