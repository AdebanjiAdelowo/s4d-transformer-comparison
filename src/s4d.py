"""
Diagonal structured state space (S4D) layer.

Implements the S4D-Lin variant from:
    Gu, Gupta, Goel, Re. "On the Parameterization and Initialization of
    Diagonal State Space Models." arXiv:2206.11893v2 (NeurIPS 2022).

Every non-obvious formula below cites the equation/section it comes from in
that paper (see architecture.md for the full derivation and the literature
vs. implementation-choice distinction). This module follows the paper's own
reference S4D-Lin NumPy implementation (Listing 1) for parameterization and
initialization, but deviates from it in two documented ways: it uses ZOH
discretization instead of Listing 1's bilinear rule, and it fully
materializes the Vandermonde tensor rather than the paper's own more
memory-efficient naive-summation fallback (see architecture.md Sections 2-3
for both). It does not use the faster Cauchy-kernel algorithm from the
original S4/state-spaces codebase either -- the goal is a scientifically
legible layer, not a maximally optimized one.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class S4DLayer(nn.Module):
    """
    H independent single-input single-output diagonal SSMs, one per channel
    ("given multiple input channels, independent S4D layers are broadcast
    over them" -- S4D paper, Section 3.4, "The full model").

    Input/output shape: (batch, length, H). Causal (the recurrence and the
    kernel are both one-directional), so this is safe to use as a drop-in
    causal sequence-mixing layer inside an autoregressive LM, in the same
    structural position a CausalSelfAttention block would occupy.
    """

    def __init__(self, h_dim: int, n_state: int = 64, dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        assert n_state % 2 == 0, "n_state must be even (stores N/2 conjugate pairs, S4D Sec. 3.3)"
        self.h_dim = h_dim
        self.n_state = n_state
        n_c = n_state // 2  # number of stored complex conjugate-pair states per channel

        # -- Initialization (S4D paper, Listing 1) --------------------------------------------
        # Per-channel log step size, log-uniform in [dt_min, dt_max].
        log_dt = torch.rand(h_dim) * (torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min))) \
            + torch.log(torch.tensor(dt_min))
        self.log_dt = nn.Parameter(log_dt)

        # S4D-Lin initialization (S4D eq. 9): Lambda_n = -1/2 + i*pi*n, n = 0..N/2-1,
        # broadcast identically across all H channels at init (they diverge under training).
        n_range = torch.arange(n_c, dtype=torch.float32)
        A_imag_init = torch.pi * n_range.unsqueeze(0).repeat(h_dim, 1)          # (H, n_c)
        # Real part is parameterized as -exp(A_real_raw) (S4D Sec. 3.3, "Parameterization of A")
        # so that Re(Lambda) < 0 is enforced for every value of A_real_raw -- this guarantees the
        # BIBO-stability condition the paper requires (K(t) = C e^{tA} B would blow up otherwise).
        # exp(A_real_raw) = 1/2  =>  A_real_raw = log(1/2), matching S4D-Lin's Re(Lambda_n) = -1/2.
        A_real_raw_init = torch.log(torch.full((h_dim, n_c), 0.5))
        self.A_real_raw = nn.Parameter(A_real_raw_init)
        self.A_imag = nn.Parameter(A_imag_init)

        # B fixed to 1 + 0i at init, but trainable (S4D Sec. 3.3: "we show that training B gives
        # a minor but consistent improvement in performance").
        self.B_real = nn.Parameter(torch.ones(h_dim, n_c))
        self.B_imag = nn.Parameter(torch.zeros(h_dim, n_c))

        # C: standard-normal real/imaginary parts, matching S4's variance-preserving
        # initialization noted in the paper (Sec. 3.3, "As described in [10], S4 initializes C
        # randomly with standard deviation 1").
        self.C_real = nn.Parameter(torch.randn(h_dim, n_c))
        self.C_imag = nn.Parameter(torch.randn(h_dim, n_c))

    # -- shared discretization -----------------------------------------------------------------

    def _discretize(self):
        """Return (A, dt, Abar, Bbar) each shape (H, n_c) complex, per S4D eq. 6 (ZOH rule)."""
        A = -torch.exp(self.A_real_raw) + 1j * self.A_imag           # (H, n_c) complex
        dt = torch.exp(self.log_dt).unsqueeze(-1)                    # (H, 1) real
        B = self.B_real + 1j * self.B_imag                           # (H, n_c) complex
        dtA = dt * A                                                 # (H, n_c)
        Abar = torch.exp(dtA)                                        # ZOH: Abar = exp(dt*A)
        # ZOH Bbar = (dt*A)^-1 (exp(dt*A) - 1) * dt*B = (exp(dt*A)-1)/A * B   (S4D eq. 6, Fig. 1)
        Bbar = (Abar - 1) / A * B
        return A, dt, Abar, Bbar

    # -- convolutional view (S4D eq. 7, used for training) --------------------------------------

    def kernel(self, length: int) -> torch.Tensor:
        """Materialize the length-`length` causal convolution kernel, shape (H, length), real."""
        _, _, Abar, Bbar = self._discretize()
        C = self.C_real + 1j * self.C_imag                           # (H, n_c)
        dtA = torch.log(Abar)                                        # = dt*A; recovers exponent
        # Vandermonde matrix V_L(Abar)[n, l] = Abar_n^l, computed stably as exp(l * dt*A)
        # (identical value to Abar**l for integer l -- no branch-cut ambiguity -- but avoids
        # amplifying rounding error from repeated complex multiplication). S4D eq. 7.
        l_range = torch.arange(length, device=Abar.device, dtype=Abar.real.dtype)
        V = torch.exp(dtA.unsqueeze(-1) * l_range.view(1, 1, -1))    # (H, n_c, L)
        weighted = (Bbar * C).unsqueeze(-1)                          # (H, n_c, 1)
        # Conjugate-pair doubling: output is 2*Re(sum over stored half of the spectrum),
        # recovering the true real kernel from the truncated conjugate-pair parameterization
        # (S4D Listing 1: "Return twice the real part - same as adding conjugate pairs").
        k = 2.0 * torch.real((weighted * V).sum(dim=1))               # (H, L)
        return k

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, L, H) real -> y: (B, L, H) real, causal linear convolution with self.kernel(L)."""
        b, length, h = u.shape
        assert h == self.h_dim
        k = self.kernel(length)                                       # (H, L)
        n_fft = 2 * length  # zero-pad to avoid wraparound (linear, not circular, convolution)
        u_f = torch.fft.rfft(u.transpose(1, 2), n=n_fft, dim=-1)       # (B, H, n_fft//2+1)
        k_f = torch.fft.rfft(k, n=n_fft, dim=-1)                       # (H, n_fft//2+1)
        y_f = u_f * k_f.unsqueeze(0)
        y = torch.fft.irfft(y_f, n=n_fft, dim=-1)[..., :length]        # (B, H, L)
        return y.transpose(1, 2)                                       # (B, L, H)

    # -- recurrent view (used only for the numerical equivalence test, and available for -------
    # -- step-by-step generation, mirroring S4's dual-view design; not used in training) --------

    def step(self, u_t: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One recurrence step: state_t = Abar*state_{t-1} + Bbar*u_t, y_t = 2*Re(C . state_t).
        u_t: (B, H) real. state: (B, H, n_c) complex (zeros at t=0). Returns (y_t, new_state).
        """
        _, _, Abar, Bbar = self._discretize()
        C = self.C_real + 1j * self.C_imag
        new_state = Abar.unsqueeze(0) * state + Bbar.unsqueeze(0) * u_t.unsqueeze(-1)
        y_t = 2.0 * torch.real((new_state * C.unsqueeze(0)).sum(dim=-1))
        return y_t, new_state

    def init_state(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.h_dim, self.n_state // 2, dtype=torch.complex64, device=device)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
