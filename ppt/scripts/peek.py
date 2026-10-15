import json, sys

for line in sys.stdin:
    if not line.startswith("{"):
        continue
    d = json.loads(line)
    if d.get("header"):
        continue
    keys = ("host_us", "eager_us", "do_bench_us", "graph_us")
    vals = " ".join(f"{k[:-3]}={d[k]:.2f}" for k in keys if k in d)
    err = d.get("error") or d.get("graph_error") or ""
    print(d["op"], d["side"], d["variant"], d["shape"], d["sha1"][:8], vals, err[:150])
