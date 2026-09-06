"""
Shared sequence-model backbone with a pluggable token-mixing sub-layer
("attn" = causal self-attention, "s4d" = diagonal state space model), so
that a fair, controlled comparison isolates the mixer as the only
architectural difference (see architecture.md, Section 7).

The attention path (CausalSelfAttention, MLP, Block skeleton) is adapted
from the author's own nano-gpt repository (an engineering-reuse
decision, not a source for any S4D theory -- see architecture.md's source
list). The S4D path uses S4DLayer from s4d.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from s4d import S4DLayer


@dataclass
class BackboneConfig:
    block_size: int = 128
    vocab_size: int = 65
    n_layer: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True
    mixer: str = "attn"       # "attn" or "s4d"
    n_head: int = 4           # only used when mixer == "attn"
    s4d_state: int = 64       # only used when mixer == "s4d" (N in architecture.md)


class CausalSelfAttention(nn.Module):
    """Adapted from nano-gpt/model.py (the author's own repository)."""

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.causal_mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.c_proj(y))


class S4DMixer(nn.Module):
    """Thin wrapper so S4DLayer presents the same (B, L, C) -> (B, L, C) interface as attention,
    plus an output projection + dropout to match the attention path's c_proj/resid_drop (so any
    parameter-count difference comes from the mixer's internal mechanism, not from a missing
    output projection on one side)."""

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.ssm = S4DLayer(h_dim=config.n_embd, n_state=config.s4d_state)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ssm(x)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """LayerNorm -> mixer -> residual, LayerNorm -> MLP -> residual (pre-norm, as in nano-gpt).
    `mixer` is the only thing that differs between the two configurations under comparison."""

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.mixer = CausalSelfAttention(config) if config.mixer == "attn" else S4DMixer(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SequenceBackbone(nn.Module):
    """Causal language model: token embedding (+ positional embedding, attention path only,
    per architecture.md Section 7) -> n_layer x Block -> final LayerNorm -> LM head."""

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd) if config.mixer == "attn" else None
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying, as in nano-gpt

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        b, t = idx.shape
        assert t <= self.config.block_size

        x = self.wte(idx)
        if self.wpe is not None:
            pos = torch.arange(t, dtype=torch.long, device=idx.device)
            x = x + self.wpe(pos)
        x = self.drop(x)

        for block in self.h:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def num_parameters(self, exclude_embeddings: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings and self.wpe is not None:
            n -= self.wpe.weight.numel()
        return n
