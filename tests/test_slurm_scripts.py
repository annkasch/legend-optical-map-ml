"""Job scripts pass their arguments on unchanged (stub srun, no Slurm needed)."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/slurm/train_surrogate.sbatch"


def _run(tmp_path, *args):
    stub = tmp_path / "bin"
    stub.mkdir(exist_ok=True)
    srun = stub / "srun"
    srun.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$SRUN_ARGS"\n')
    srun.chmod(0o755)
    record = tmp_path / "args.txt"
    env = {
        **os.environ,
        "PATH": f"{stub}:{os.environ['PATH']}",
        "SLURM_SUBMIT_DIR": str(SCRIPT.parents[2]),
        "SLURM_JOB_ID": "0",
        "RESUM_PYTHON": "/bin/true",
        "SRUN_ARGS": str(record),
    }
    result = subprocess.run(
        ["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, check=False
    )
    return result, record.read_text().split("\n")[:-1] if record.exists() else None


@pytest.mark.parametrize(
    "args,stage",
    [
        (["c.yaml"], ["train", "c.yaml"]),
        (["c.yaml", "score"], ["score", "c.yaml"]),
        (["c.yaml", "train", "--run-seed", "2"], ["train", "c.yaml", "--run-seed", "2"]),
    ],
)
def test_stage_arguments_reach_the_cli(tmp_path, args, stage):
    result, forwarded = _run(tmp_path, *args)
    assert result.returncode == 0, result.stderr
    assert forwarded == ["--cpu-bind=none", "/bin/true", "-u", "-m", "core.surrogates", *stage]


def test_unknown_stage_is_refused(tmp_path):
    result, forwarded = _run(tmp_path, "c.yaml", "fit")
    assert result.returncode == 2 and forwarded is None
