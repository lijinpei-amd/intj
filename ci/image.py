"""Builds and publishes the CI images described by ci/matrix.json.

    python3 ci/image.py matrix                          # writes ci/out/matrix.json
    CELL=<id> [FLAVOR=cpu|rocm] python3 ci/image.py tag|build|push

The tag hashes every file under ci/dockerfiles/, the cell and the pins, so any
change to what goes into an image publishes a new one.
"""

import hashlib
import json
import os
import pathlib
import subprocess
import sys

CI = pathlib.Path(__file__).resolve().parent
DOCKERFILES = CI / "dockerfiles"
REPO = os.environ.get("INTJ_IMAGE_REPO", "ghcr.io/lijinpei-amd/intj-ci")
PINS = {
    "triton": "3.8.0",
    "torch_index": {
        "cpu": "https://download.pytorch.org/whl/cpu",
        "rocm": "https://download.pytorch.org/whl/rocm7.2",
    },
}


def die(message):
    sys.exit("ci: " + message)


def load():
    return json.loads((CI / "matrix.json").read_text())


def tag(cell, flavor):
    digest = hashlib.sha256()
    for path in sorted(DOCKERFILES.iterdir()):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    digest.update(json.dumps([cell, flavor, PINS], sort_keys=True).encode())
    return "%s:%s-%s-%s" % (REPO, cell["id"], flavor, digest.hexdigest()[:12])


def selected():
    flavor = os.environ.get("FLAVOR", "cpu")
    if flavor not in PINS["torch_index"]:
        die("FLAVOR must be cpu or rocm, got %r" % flavor)
    cell_id = os.environ.get("CELL", "")
    cells = {cell["id"]: cell for cell in load()["cells"]}
    if cell_id not in cells:
        die("unknown CELL %r; ids are in ci/matrix.json" % cell_id)
    cell = cells[cell_id]
    if flavor == "rocm" and not cell["gpu"]:
        die(
            "CELL %s has no rocm image: triton %s has no wheel for its python"
            % (cell_id, PINS["triton"])
        )
    return cell, flavor


def build(cell, flavor):
    args = {
        "BASE": cell["base"],
        "PY_PKGS": cell["py_pkgs"],
        "PY_ENSURE": cell["py_ensure"],
        "PY_BIN": cell["py_bin"],
        "TORCH_INDEX": PINS["torch_index"][flavor],
        "TRITON": PINS["triton"] if cell["gpu"] else "",
    }
    command = [
        "docker",
        "build",
        "-f",
        str(DOCKERFILES / cell["dockerfile"]),
        "-t",
        tag(cell, flavor),
    ]
    for key, value in args.items():
        command += ["--build-arg", "%s=%s" % (key, value)]
    subprocess.run(command + [str(DOCKERFILES)], check=True)


def push(cell, flavor):
    name = tag(cell, flavor)
    exists = subprocess.run(
        ["docker", "manifest", "inspect", name], capture_output=True
    )
    if exists.returncode == 0:
        print("ci: %s already published" % name)
        return
    build(cell, flavor)
    subprocess.run(["docker", "push", name], check=True)


def matrix():
    data = load()
    cells = data["cells"]
    out = {
        "primary": tag(next(c for c in cells if c["id"] == data["primary"]), "cpu"),
        "cpu": [{"id": c["id"], "image": tag(c, "cpu")} for c in cells],
        "rocm": [{"id": c["id"], "image": tag(c, "rocm")} for c in cells if c["gpu"]],
    }
    path = CI / "out" / "matrix.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(
        "ci: wrote %s (%d cpu, %d rocm cells)"
        % (path, len(out["cpu"]), len(out["rocm"]))
    )


def main():
    command = sys.argv[1] if len(sys.argv) == 2 else ""
    if command == "matrix":
        matrix()
    elif command == "tag":
        print(tag(*selected()))
    elif command == "build":
        build(*selected())
    elif command == "push":
        push(*selected())
    else:
        die("usage: image.py matrix|tag|build|push")


if __name__ == "__main__":
    main()
