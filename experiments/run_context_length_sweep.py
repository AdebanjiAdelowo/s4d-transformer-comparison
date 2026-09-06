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
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "src" / "train.py"
RESULTS_DIR = ROOT / "results"

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


def run(mixer: str, block_size: int):
    max_iters = MAX_ITERS_BY_BLOCK_SIZE[block_size]
    out_path = RESULTS_DIR / f"{mixer}_bs{block_size}.json"
    cmd = [
        sys.executable, str(TRAIN),
        "--mixer", mixer,
        "--block_size", str(block_size),
        "--max_iters", str(max_iters),
        "--out", str(out_path),
        "--tag", "context_length_sweep",
    ]
    for k, v in COMMON.items():
        cmd += [f"--{k}", str(v)]
    print(f"\n=== RUN: mixer={mixer} block_size={block_size} max_iters={max_iters} ===", flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    for bs in BLOCK_SIZES:
        for mixer in ("attn", "s4d"):
            run(mixer, bs)
    print("\nSweep complete.")
