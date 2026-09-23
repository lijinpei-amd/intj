# TODO

- **The README claims CI that does not exist.** `README.md` says the table is
  "checked using both method in our CI", and the repo has no CI. Either drop the
  line, or add a CI job that runs both detectors against every torch in
  `torch_abi.toml`. The per-version script used to generate the 2.2–2.13 entries
  lives outside the repo, at `/tmp/intj_torch/run.sh`: for one torch minor it makes
  a torch-only CPU venv, runs `abi_detect` and `cpp_detect`, and compares them.
