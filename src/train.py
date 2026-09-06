"""
Shared, instrumented training loop for both the attention and S4D backbones.
Everything (optimizer, LR schedule, batch construction, eval protocol) is
held identical between the two mixer types being compared -- see
architecture.md, Section 7. Produces one JSON result file per run under
results/, consumed by experiments/make_plots.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
import warnings
from pathlib import Path

import torch

# Harmless PyTorch-internal autograd-buffer resize warning triggered by repeated rfft/irfft
# calls under grad in the S4D path; verified separately (tests/test_s4d_numerics.py) that this
# does not affect correctness. Filtered here only to keep training logs legible.
warnings.filterwarnings("ignore", message=".*resized since it had shape.*")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import get_batch, load_tinyshakespeare  # noqa: E402
from model import BackboneConfig, SequenceBackbone  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mixer", choices=["attn", "s4d"], required=True)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--s4d_state", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_min", type=float, default=1e-4)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--out", type=str, required=True, help="path to write result JSON")
    p.add_argument("--tag", type=str, default="", help="free-text label stored in the result JSON")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    ds = load_tinyshakespeare(args.data_dir)

    config = BackboneConfig(
        block_size=args.block_size,
        vocab_size=ds["vocab_size"],
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        dropout=args.dropout,
        mixer=args.mixer,
        n_head=args.n_head,
        s4d_state=args.s4d_state,
    )
    model = SequenceBackbone(config).to(device)
    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_no_embed = model.num_parameters(exclude_embeddings=True)

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    def cosine_lr(step):
        if step < args.warmup_iters:
            return args.lr * step / max(1, args.warmup_iters)
        t = (step - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1.0 + math.cos(math.pi * t))

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                x, y = get_batch(ds[split], args.block_size, args.batch_size, device)
                _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    def mps_current_bytes():
        if device == "mps":
            try:
                return torch.mps.current_allocated_memory()
            except Exception:
                return None
        return None

    train_losses, val_losses, loss_steps = [], [], []
    mps_peak_sampled = 0  # sampled only at eval_interval -- documented limitation, see README
    step_times = []  # wall-clock seconds per optimizer step (forward+backward+update)

    t_start = time.time()
    for step in range(args.max_iters + 1):
        lr = cosine_lr(step)
        for pg in optimiser.param_groups:
            pg["lr"] = lr

        if step % args.eval_interval == 0:
            ev = estimate_loss()
            train_losses.append(ev["train"])
            val_losses.append(ev["val"])
            loss_steps.append(step)
            mb = mps_current_bytes()
            if mb is not None:
                mps_peak_sampled = max(mps_peak_sampled, mb)
            elapsed = time.time() - t_start
            print(
                f"[{args.mixer}|bs{args.block_size}] step {step:5d}/{args.max_iters} | "
                f"train {ev['train']:.4f} | val {ev['val']:.4f} | lr {lr:.2e} | {elapsed:6.1f}s"
            )

        if step == args.max_iters:
            break

        x, y = get_batch(ds["train"], args.block_size, args.batch_size, device)
        t0 = time.time()
        _, loss = model(x, y)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimiser.step()
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.time() - t0)

    total_time = time.time() - t_start
    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1024 if sys.platform != "darwin" else 1  # macOS reports bytes, Linux reports KB
    )

    final_val_loss = val_losses[-1]
    # Perplexity is mathematically appropriate here: the training objective IS token-level
    # cross-entropy (see model.py's F.cross_entropy call), so exp(loss) is exactly the standard
    # LM perplexity definition, not an approximation.
    final_val_ppl = math.exp(final_val_loss)

    result = {
        "tag": args.tag,
        "mixer": args.mixer,
        "device": device,
        "config": {
            "block_size": args.block_size,
            "n_layer": args.n_layer,
            "n_embd": args.n_embd,
            "n_head": args.n_head if args.mixer == "attn" else None,
            "s4d_state": args.s4d_state if args.mixer == "s4d" else None,
            "batch_size": args.batch_size,
            "max_iters": args.max_iters,
            "lr": args.lr,
            "seed": args.seed,
        },
        "params": {
            "total": n_params_total,
            "excl_embeddings": n_params_no_embed,
        },
        "loss_curve": {
            "steps": loss_steps,
            "train_loss": train_losses,
            "val_loss": val_losses,
        },
        "final_train_loss": train_losses[-1],
        "final_val_loss": final_val_loss,
        "final_val_perplexity": final_val_ppl,
        "timing": {
            "total_seconds": total_time,
            "mean_step_seconds": sum(step_times) / len(step_times) if step_times else None,
            "tokens_per_second": (args.batch_size * args.block_size * len(step_times)) / total_time
            if step_times else None,
        },
        "memory": {
            "peak_process_rss_bytes": peak_rss_bytes,
            "mps_current_allocated_bytes_sampled_max": mps_peak_sampled or None,
            "note": (
                "peak_process_rss_bytes is whole-process peak resident memory (coarse but "
                "device-independent). mps_current_allocated_bytes_sampled_max is sampled only at "
                "eval_interval checkpoints (not every step), so it is a LOWER BOUND on true peak "
                "MPS tensor allocation, not an exact peak -- torch's MPS backend does not expose "
                "a max_memory_allocated()-style true-peak counter the way CUDA does. Documented "
                "here rather than silently presented as an exact figure."
            ),
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.out}")
    print(
        f"  params(excl.embed)={n_params_no_embed/1e6:.3f}M | "
        f"val_loss={final_val_loss:.4f} | val_ppl={final_val_ppl:.2f} | "
        f"total_time={total_time:.1f}s"
    )


if __name__ == "__main__":
    main()
