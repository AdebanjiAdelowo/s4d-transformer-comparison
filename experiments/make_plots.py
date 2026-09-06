"""
Reads every results/*_bs*.json produced by run_context_length_sweep.py and
produces:
  figures/loss_curves.png       -- train/val loss vs step, one panel per block_size
  figures/scaling.png           -- 4-panel: val loss, val perplexity, tokens/sec, peak memory vs block_size
  results/summary_table.md      -- markdown table, one row per (mixer, block_size) run

This script only reads numbers that are actually in the JSON files -- it does
not compute or assume anything about results that were not logged.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_sweep_results():
    runs = []
    for f in sorted(RESULTS_DIR.glob("*_bs*.json")):
        with open(f) as fh:
            data = json.load(fh)
        if data.get("tag") == "context_length_sweep":
            runs.append(data)
    return runs


def plot_loss_curves(runs):
    block_sizes = sorted({r["config"]["block_size"] for r in runs})
    fig, axes = plt.subplots(1, len(block_sizes), figsize=(4 * len(block_sizes), 3.5), sharey=True)
    if len(block_sizes) == 1:
        axes = [axes]
    for ax, bs in zip(axes, block_sizes):
        for r in runs:
            if r["config"]["block_size"] != bs:
                continue
            steps = r["loss_curve"]["steps"]
            ax.plot(steps, r["loss_curve"]["val_loss"], label=f"{r['mixer']} (val)", linewidth=2)
        ax.set_title(f"block_size={bs}")
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].legend()
    fig.suptitle("Validation loss vs. training step, attn vs. S4D, by context length")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "loss_curves.png", dpi=150)
    print(f"Wrote {FIGURES_DIR / 'loss_curves.png'}")


def plot_scaling(runs):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    metrics = [
        ("final_val_loss", "final val loss", axes[0, 0]),
        ("final_val_perplexity", "final val perplexity", axes[0, 1]),
    ]
    for key, label, ax in metrics:
        for mixer in ("attn", "s4d"):
            xs, ys = [], []
            for r in sorted(runs, key=lambda r: r["config"]["block_size"]):
                if r["mixer"] != mixer:
                    continue
                xs.append(r["config"]["block_size"])
                ys.append(r[key])
            ax.plot(xs, ys, marker="o", label=mixer)
        ax.set_xlabel("context length (block_size)")
        ax.set_ylabel(label)
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)
        ax.legend()

    ax = axes[1, 0]
    for mixer in ("attn", "s4d"):
        xs, ys = [], []
        for r in sorted(runs, key=lambda r: r["config"]["block_size"]):
            if r["mixer"] != mixer or r["timing"]["tokens_per_second"] is None:
                continue
            xs.append(r["config"]["block_size"])
            ys.append(r["timing"]["tokens_per_second"])
        ax.plot(xs, ys, marker="o", label=mixer)
    ax.set_xlabel("context length (block_size)")
    ax.set_ylabel("tokens / second (training)")
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    for mixer in ("attn", "s4d"):
        xs, ys = [], []
        for r in sorted(runs, key=lambda r: r["config"]["block_size"]):
            if r["mixer"] != mixer:
                continue
            xs.append(r["config"]["block_size"])
            ys.append(r["memory"]["peak_process_rss_bytes"] / 1e6)
        ax.plot(xs, ys, marker="o", label=mixer)
    ax.set_xlabel("context length (block_size)")
    ax.set_ylabel("peak process RSS (MB)")
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle("Attn vs. S4D: quality and compute cost vs. context length (Tiny Shakespeare, char-level)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "scaling.png", dpi=150)
    print(f"Wrote {FIGURES_DIR / 'scaling.png'}")


def write_summary_table(runs):
    lines = [
        "| mixer | block_size | max_iters | params (excl. pos. embed) | final val loss | val ppl | "
        "tokens/s | peak RSS (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(runs, key=lambda r: (r["config"]["block_size"], r["mixer"])):
        c = r["config"]
        tps = r["timing"]["tokens_per_second"]
        rss_mb = r["memory"]["peak_process_rss_bytes"] / 1e6
        lines.append(
            f"| {r['mixer']} | {c['block_size']} | {c['max_iters']} | "
            f"{r['params']['excl_embeddings']/1e6:.3f}M | {r['final_val_loss']:.4f} | "
            f"{r['final_val_perplexity']:.2f} | {tps:.0f} | {rss_mb:.1f} |"
        )
    out_path = RESULTS_DIR / "summary_table.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    runs = load_sweep_results()
    if not runs:
        raise SystemExit("No sweep results found in results/*_bs*.json -- run experiments/run_context_length_sweep.py first.")
    print(f"Loaded {len(runs)} runs.")
    plot_loss_curves(runs)
    plot_scaling(runs)
    write_summary_table(runs)
