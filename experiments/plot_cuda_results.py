"""
Plots the CUDA sweep in results/cuda/ using the CUDA-specific metrics that
make_plots.py does not show: both throughput definitions and the exact CUDA
allocator peaks (allocated and reserved), instead of whole-process RSS, which
on CUDA is dominated by runtime overhead.

    python experiments/plot_cuda_results.py   # -> results/cuda/cuda_scaling.png

Reads only numbers present in the JSON files.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CUDA_DIR = ROOT / "results" / "cuda"
COLORS = {"attn": "#1f77b4", "s4d": "#ff7f0e"}  # same entity colors as figures/
MIB = 2 ** 20


def load():
    runs = {}
    for f in sorted(CUDA_DIR.glob("*_bs*.json")):
        r = json.loads(f.read_text())
        if r.get("tag") == "context_length_sweep" and r["device"] == "cuda":
            runs.setdefault(r["mixer"], []).append(r)
    for rs in runs.values():
        rs.sort(key=lambda r: r["config"]["block_size"])
    return runs


def series(rs, get):
    return [r["config"]["block_size"] for r in rs], [get(r) for r in rs]


def label_end(ax, xs, ys, text):
    ax.annotate(text, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color="#333333")


def main():
    runs = load()
    if not runs:
        raise SystemExit(f"No CUDA sweep results in {CUDA_DIR}")
    gpu = next(iter(runs.values()))[0]["device_info"]["gpu_name"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    panels = [
        (axes[0], "final validation loss", [("final_val_loss", "-", "")]),
        (axes[1], "training tokens / second", [
            ("tokens_per_second", "-", " incl. eval"),
            ("train_step_tokens_per_second", "--", " step only")]),
        (axes[2], "peak CUDA memory (MiB, log scale)", [
            ("cuda_max_memory_allocated_bytes", "-", " allocated"),
            ("cuda_max_memory_reserved_bytes", "--", " reserved")]),
    ]
    for ax, ylabel, metrics in panels:
        for mixer, rs in runs.items():
            for key, ls, suffix in metrics:
                if key in ("tokens_per_second", "train_step_tokens_per_second"):
                    get = lambda r, k=key: r["timing"][k]
                elif key.startswith("cuda_"):
                    get = lambda r, k=key: r["memory"][k] / MIB
                else:
                    get = lambda r, k=key: r[k]
                xs, ys = series(rs, get)
                ax.plot(xs, ys, ls, marker="o", markersize=5, linewidth=2,
                        color=COLORS[mixer], label=f"{mixer}{suffix}")
                if ls == "-":
                    label_end(ax, xs, ys, mixer)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("context length (block_size)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    axes[2].set_yscale("log", base=2)
    axes[2].set_yticks([128, 256, 512, 1024, 2048, 4096, 8192])
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.suptitle(f"CUDA sweep on one {gpu} (seed 42, one run per configuration); "
                 "block_size 1024 uses 400 steps, others 1200")
    fig.tight_layout()
    out = CUDA_DIR / "cuda_scaling.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
