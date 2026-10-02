"""
Device selection and output-path safety (src/device.py, src/train.py,
experiments/run_context_length_sweep.py). Guards two properties the
cross-hardware comparison depends on: a run labelled with a device really ran
there, and new runs can never replace the historical MPS results/*.json.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import run_context_length_sweep as sweep  # noqa: E402
from device import resolve_device  # noqa: E402

HISTORICAL = sorted((ROOT / "results").glob("*_bs*.json"))


def _set_available(monkeypatch, cuda, mps):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


@pytest.mark.parametrize("cuda,mps,expected", [
    (True, True, "cuda"),
    (True, False, "cuda"),
    (False, True, "mps"),
    (False, False, "cpu"),
])
def test_auto_order_is_cuda_mps_cpu(monkeypatch, cuda, mps, expected):
    _set_available(monkeypatch, cuda, mps)
    assert resolve_device("auto") == expected


def test_explicit_cpu_always_available(monkeypatch):
    _set_available(monkeypatch, True, True)
    assert resolve_device("cpu") == "cpu"


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_explicit_unavailable_device_fails(monkeypatch, device):
    _set_available(monkeypatch, False, False)
    with pytest.raises(RuntimeError, match=f"'{device}' was requested"):
        resolve_device(device)


def test_unknown_device_rejected():
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_sweep_defaults_to_per_device_subdirectory():
    for device in ("cpu", "mps", "cuda"):
        out_dir = sweep.resolve_out_dir(None, device)
        assert out_dir == ROOT / "results" / device
        for mixer in ("attn", "s4d"):
            out, cmd = sweep.build_command(mixer, 64, device, out_dir, overwrite=False)
            assert out.parent == out_dir
            assert cmd[cmd.index("--device") + 1] == device
            assert "--overwrite" not in cmd


def test_sweep_refuses_historical_directory():
    with pytest.raises(SystemExit):
        sweep.resolve_out_dir(str(ROOT / "results"), "mps")


def test_sweep_refuses_existing_files_without_overwrite(tmp_path):
    (tmp_path / "attn_bs64.json").write_text("{}")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        sweep.main(["--device", "cpu", "--out_dir", str(tmp_path), "--dry_run"])


@pytest.mark.parametrize("extra", [[], ["--overwrite"]])
def test_train_refuses_historical_file(extra):
    assert HISTORICAL, "historical results/*_bs*.json missing"
    target = HISTORICAL[0]
    before = target.read_bytes()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "src" / "train.py"), "--mixer", "s4d", "--device", "cpu",
         "--out", str(target), *extra],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "historical" in proc.stderr
    assert target.read_bytes() == before


def test_train_short_cpu_run_records_device(tmp_path):
    out = tmp_path / "s4d_smoke.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "src" / "train.py"), "--mixer", "s4d", "--device", "cpu",
         "--block_size", "16", "--n_layer", "1", "--n_embd", "16", "--s4d_state", "8",
         "--batch_size", "2", "--max_iters", "3", "--eval_interval", "3", "--eval_iters", "1",
         "--data_dir", str(ROOT / "data"), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    r = json.loads(out.read_text())
    assert r["device"] == "cpu"
    assert r["device_info"]["requested"] == "cpu"
    assert r["device_info"]["resolved"] == "cpu"
    assert r["memory"]["device_memory_api"] is None
    assert r["memory"]["cuda_max_memory_allocated_bytes"] is None
    assert r["timing"]["train_step_tokens_per_second"] > 0

    # A second write to the same path needs --overwrite.
    proc = subprocess.run(
        [sys.executable, str(ROOT / "src" / "train.py"), "--mixer", "s4d", "--device", "cpu",
         "--out", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0 and "--overwrite" in proc.stderr
