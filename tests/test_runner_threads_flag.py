"""--threads on run_classical and run_quantum: pins the pools and n_jobs; absent, nothing changes.

Each case imports the runner in a fresh interpreter (the flag acts before numpy is imported),
with the thread variables removed from its environment.
"""

import json
import os
import subprocess
import sys

import pytest

from qgridbench.utils.run_tracking import REPO_ROOT

VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")
PROBE = """
import json, os, sys
sys.argv = ["runner"] + {argv!r}
import experiments.{module} as m
out = {{"THREADS": m.THREADS, "env": {{v: os.environ.get(v) for v in {vars!r}}}}}
if hasattr(m, "build_pinned"):
    out["n_jobs"] = {{n: m.build_pinned(n, {{}}, 0).get_params().get("n_jobs")
                     for n in ("rf", "xgb", "lgbm")}}
print(json.dumps(out))
"""


def _probe(module: str, argv: list[str]) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in VARS}
    env["PYTHONPATH"] = str(REPO_ROOT)
    code = PROBE.format(argv=argv, module=module, vars=VARS)
    res = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr[-2000:]
    return json.loads(res.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("module", ["run_classical", "run_quantum"])
def test_absent_flag_changes_nothing(module):
    out = _probe(module, ["--variant", "binary"])
    assert out["THREADS"] is None
    assert all(v is None for v in out["env"].values())
    if module == "run_classical":  # the zoo's n_jobs=-1, as every published run
        assert out["n_jobs"] == {"rf": -1, "xgb": -1, "lgbm": -1}


@pytest.mark.parametrize("module", ["run_classical", "run_quantum"])
def test_flag_pins_pools_and_n_jobs(module):
    out = _probe(module, ["--variant", "binary", "--threads", "3"])
    assert out["THREADS"] == 3
    assert all(v == "3" for v in out["env"].values())
    if module == "run_classical":
        assert out["n_jobs"] == {"rf": 3, "xgb": 3, "lgbm": 3}
