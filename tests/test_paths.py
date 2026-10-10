from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _probe_demo_mode(env: dict) -> tuple[bool, str, str]:
    code = (
        "import src.paths as p;"
        "print(int(p.DEMO_MODE));"
        "print(p.DATASET_ROOT);"
        "print(p.RUNTIME_ROOT)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = proc.stdout.strip().splitlines()
    return bool(int(lines[0])), lines[1], lines[2]


def test_paths_demo_mode_on_and_off():
    env_on = {**os.environ, "DEMO_MODE": "1"}
    demo_mode, dataset_root, runtime_root = _probe_demo_mode(env_on)
    assert demo_mode is True
    assert "demo_data/dataset" in dataset_root
    assert "demo_data/runtime" in runtime_root

    env_off = dict(os.environ)
    env_off.pop("DEMO_MODE", None)
    demo_mode_off, dataset_root_off, runtime_root_off = _probe_demo_mode(env_off)
    assert demo_mode_off is False
    assert "demo_data" not in dataset_root_off
    assert "demo_data/runtime" not in runtime_root_off
