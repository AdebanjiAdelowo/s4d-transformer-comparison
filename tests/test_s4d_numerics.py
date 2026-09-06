"""
Numerical validation of the S4D layer (src/s4d.py) against the two properties
the S4D paper's entire "dual view" architecture depends on:

1. The convolutional form (kernel materialized via the Vandermonde formula,
   S4D eq. 7) and the discrete-time recurrence (state_t = Abar*state_{t-1} +
   Bbar*u_t) must compute the SAME function on the same input -- this is not
   an S4D-specific claim, it's a basic LTI-system fact the whole kernel-based
   training trick relies on, and it is exactly what Fig. 1 of the paper
   ("S4D Recurrent View" / "S4D Convolution Kernel") asserts are equivalent.

2. The eigenvalue real-part constraint (Re(Lambda) < 0, S4D Sec. 3.3) must
   actually hold after the exp-parameterization, at init and after a
   gradient step -- otherwise the kernel is not guaranteed BIBO-stable.

Run with: pytest tests/test_s4d_numerics.py -v
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s4d import S4DLayer  # noqa: E402


@pytest.mark.parametrize("h_dim,n_state,length,batch,seed", [
    (3, 8, 16, 2, 0),
    (1, 4, 32, 1, 1),
    (8, 16, 64, 4, 42),
])
def test_recurrent_matches_convolutional(h_dim, n_state, length, batch, seed):
    """Kernel-convolution output must match step-by-step recurrence output."""
    torch.manual_seed(seed)
    layer = S4DLayer(h_dim=h_dim, n_state=n_state)
    u = torch.randn(batch, length, h_dim)

    with torch.no_grad():
        y_conv = layer.forward(u)

        state = layer.init_state(batch)
        ys = []
        for t in range(length):
            y_t, state = layer.step(u[:, t, :], state)
            ys.append(y_t)
        y_rec = torch.stack(ys, dim=1)

    max_abs_diff = (y_conv - y_rec).abs().max().item()
    # float32 arithmetic through an FFT (conv path) vs. direct recurrence (step path)
    # accumulate rounding differently; 1e-4 is a generous but still meaningful bound
    # given the recurrence involves L sequential multiplications by Abar.
    assert max_abs_diff < 1e-4, f"recurrent/convolutional mismatch: max abs diff = {max_abs_diff}"


def test_recurrent_matches_convolutional_after_training_step():
    """The equivalence must still hold once A, B, C, dt have been updated by an optimizer step
    (not just at initialization) -- otherwise the two views could silently diverge during training."""
    torch.manual_seed(0)
    h_dim, n_state, length, batch = 4, 8, 20, 3
    layer = S4DLayer(h_dim=h_dim, n_state=n_state)
    opt = torch.optim.Adam(layer.parameters(), lr=1e-2)

    u = torch.randn(batch, length, h_dim)
    target = torch.randn(batch, length, h_dim)
    y = layer(u)
    loss = ((y - target) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

    with torch.no_grad():
        y_conv = layer.forward(u)
        state = layer.init_state(batch)
        ys = []
        for t in range(length):
            y_t, state = layer.step(u[:, t, :], state)
            ys.append(y_t)
        y_rec = torch.stack(ys, dim=1)

    max_abs_diff = (y_conv - y_rec).abs().max().item()
    assert max_abs_diff < 1e-4, f"post-training-step mismatch: max abs diff = {max_abs_diff}"


def test_s4d_lin_initialization_matches_eq9():
    """S4D-Lin init (S4D eq. 9): Lambda_n = -1/2 + i*pi*n for n = 0..N/2-1."""
    torch.manual_seed(0)
    h_dim, n_state = 2, 8
    layer = S4DLayer(h_dim=h_dim, n_state=n_state)
    A = -torch.exp(layer.A_real_raw) + 1j * layer.A_imag

    real_part = A.real
    assert torch.allclose(real_part, torch.full_like(real_part, -0.5), atol=1e-6), \
        f"Re(Lambda) should be exactly -0.5 at init, got {real_part}"

    expected_imag = math.pi * torch.arange(n_state // 2, dtype=torch.float32)
    for h in range(h_dim):
        assert torch.allclose(A.imag[h], expected_imag, atol=1e-6), \
            f"Im(Lambda) channel {h} does not match eq. 9's pi*n spacing"


def test_real_part_stays_negative_after_training_step():
    """S4D Sec. 3.3: Re(A) = -exp(A_real_raw) must be < 0 for every real A_real_raw -- this is
    an algebraic guarantee, not an empirical one, but we check it survives an actual optimizer
    step (i.e. the parameterization, not just the initial values, enforces it)."""
    torch.manual_seed(0)
    layer = S4DLayer(h_dim=4, n_state=16)
    opt = torch.optim.Adam(layer.parameters(), lr=0.5)  # deliberately large LR, stress test

    for _ in range(20):
        u = torch.randn(2, 10, 4)
        y = layer(u)
        loss = (y ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    A_real = -torch.exp(layer.A_real_raw)
    assert (A_real < 0).all(), "Re(Lambda) went non-negative after training -- stability guarantee broken"


def test_kernel_is_finite_and_decays():
    """Sanity check on eq. 3 / eq. 7: since Re(Lambda) < 0, |K_n(t)| = |e^{t*Lambda_n}| decays as
    t grows, so the materialized kernel should be finite everywhere and (in an L2 sense) should
    not be growing in magnitude at the far (oldest-context) end relative to the near end."""
    torch.manual_seed(0)
    layer = S4DLayer(h_dim=2, n_state=16)
    k = layer.kernel(length=200)
    assert torch.isfinite(k).all()

    early_energy = k[:, :20].abs().mean()
    late_energy = k[:, -20:].abs().mean()
    assert late_energy <= early_energy + 1e-6, (
        f"kernel should decay (Re(Lambda)<0 envelope e^-t/2), "
        f"but late-window mean |K|={late_energy} > early-window mean |K|={early_energy}"
    )


if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", __file__, "-v"])
