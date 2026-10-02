"""
Controlled experiment: train both mixers (attn, s4d) across a range of
context lengths (block_size), holding every other hyperparameter fixed
(same n_layer, n_embd, batch_size, seed, optimizer, LR schedule), and log
loss/perplexity, timing, and memory for each run. See architecture.md
Section 7 for exactly what is and isn't held constant, and why.

max_iters is reduced at block_size=1024 purely for wall-clock reasons on a
single laptop GPU (MPS) -- documented here rather than silently changed --
that run is used for the timing/memory scaling curve, not as an additional
fully-converged loss comparison point.

Outputs go to results/<device>/ by default (e.g. results/cuda/attn_bs512.json).
The top-level results/*.json files are the historical Apple MPS dataset and
are never written by this script.

    python experiments/run_context_length_sweep.py                  # auto device
    python experiments/run_context_length_sweep.py --device cuda    # fails if no CUDA
    python experiments/run_context_length_sweep.py --device mps --out_dir results/mps_rerun
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "src" / "train.py"
RESULTS_DIR = ROOT / "results"

sys.path.insert(0, str(ROOT / "src"))
from device import DEVICE_CHOICES, resolve_device  # noqa: E402

COMMON = dict(
    n_layer=4,
    n_embd=128,
    n_head=4,
    s4d_state=64,
    batch_size=32,
    eval_interval=200,
    eval_iters=40,
    seed=42,
)

BLOCK_SIZES = [64, 128, 256, 512, 1024]
MAX_ITERS_BY_BLOCK_SIZE = {64: 1200, 128: 1200, 256: 1200, 512: 1200, 1024: 400}


def resolve_out_dir(out_dir: str | None, device: str) -> Path:
    path = Path(out_dir).resolve() if out_dir else RESULTS_DIR / device
    if path == RESULTS_DIR:
        raise SystemExit(
            f"error: {RESULTS_DIR} holds the historical MPS results; choose a subdirectory "
            "such as results/<device>/"
        )
    return path


def build_command(mixer: str, block_size: int, device: str, out_dir: Path, overwrite: bool):
    max_iters = MAX_ITERS_BY_BLOCK_SIZE[block_size]
    out_path = out_dir / f"{mixer}_bs{block_size}.json"
    cmd = [
        sys.executable, str(TRAIN),
        "--mixer", mixer,
        "--block_size", str(block_size),
        "--max_iters", str(max_iters),
        "--out", str(out_path),
        "--tag", "context_length_sweep",
        "--device", device,
    ]
    for k, v in COMMON.items():
        cmd += [f"--{k}", str(v)]
    if overwrite:
        cmd.append("--overwrite")
    return out_path, cmd


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    p.add_argument("--out_dir", type=str, default=None, help="default: results/<resolved device>/")
    p.add_argument("--overwrite", action="store_true", help="replace existing files in --out_dir")
    p.add_argument("--dry_run", action="store_true", help="print the commands without running them")
    args = p.parse_args(argv)

    try:
        device = resolve_device(args.device)  # resolved once so every run uses the same device
    except RuntimeError as e:
        raise SystemExit(f"error: {e}")
    out_dir = resolve_out_dir(args.out_dir, device)

    jobs = [build_command(mixer, bs, device, out_dir, args.overwrite)
            for bs in BLOCK_SIZES for mixer in ("attn", "s4d")]
    existing = [str(out) for out, _ in jobs if out.exists()]
    if existing and not args.overwrite:
        raise SystemExit("error: refusing to overwrite existing results (pass --overwrite):\n  "
                         + "\n  ".join(existing))

    print(f"device={device} out_dir={out_dir}", flush=True)
    for out_path, cmd in jobs:
        print(f"\n=== RUN: {' '.join(cmd[2:8])} -> {out_path} ===", flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    print("\nSweep complete." if not args.dry_run else "\nDry run complete.")


if __name__ == "__main__":
    main()
