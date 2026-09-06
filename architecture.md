# Architecture Specification

This document separates three categories of statement, per the project's own methodology:

- **[LIT]** — established in the cited literature, quoted/paraphrased with equation numbers.
- **[IMPL]** — a choice this repository makes among several literature-valid options.
- **[HYP]** — a hypothesis this project's experiments are designed to test (not yet established).

Primary sources used (fetched and read directly from arXiv, not from any pre-existing local notes):

1. **[S4D]** Gu, Gupta, Goel, Ré. *"On the Parameterization and Initialization of Diagonal State
   Space Models."* arXiv:2206.11893v2 (NeurIPS 2022). This is the paper actually implemented here.
2. **[S4]** Gu, Goel, Ré. *"Efficiently Modeling Long Sequences with Structured State Spaces."*
   ICLR 2022 (arXiv:2111.00396) — cited by [S4D] as the source of the HiPPO-LegS matrix and the
   original DPLR (diagonal-plus-low-rank) construction; referenced here for context only, not
   implemented.
3. Karpathy-style nanoGPT — the Transformer baseline architecture already implemented in this
   candidate's own `nano-gpt` repository (attention mechanism, not SSM theory — that repo is not
   used as a source for any S4D claim, per the task constraint).

---

## 1. Continuous-time state space model **[LIT: S4D eq. 1–2]**

A single-input single-output linear time-invariant SSM is the pair of equations

```
x'(t) = A x(t) + B u(t)      (S4D eq. 1)
y(t)  = C x(t)
```

with `A ∈ C^{N×N}`, `B ∈ C^{N×1}`, `C ∈ C^{1×N}`. Equivalently, as a convolution:

```
K(t) = C e^{tA} B ,   y(t) = (K * u)(t)      (S4D eq. 2)
```

For a **diagonal** SSM, `A` is diagonal, and per-basis-function form **[LIT: S4D eq. 3]**:

```
K(t) = Σ_{n=0}^{N-1} C_n K_n(t),   K_n(t) := e_n^T e^{tA} B = e^{t A_n} B_n
```

i.e. the kernel is a linear combination of N independent scalar exponential modes.

## 2. Discretization **[LIT: S4D eq. 6, Section 3.1]**

The paper gives two literature-valid discretization rules for turning continuous `(A, B)` into
discrete `(Ā, B̄)` at step size `Δ`:

```
Bilinear:  Ā = (I − Δ/2·A)⁻¹(I + Δ/2·A)     B̄ = (I − Δ/2·A)⁻¹ · Δ·B
ZOH:       Ā = exp(ΔA)                       B̄ = (ΔA)⁻¹ (exp(ΔA) − I) · ΔB
```

with discrete-time output `y = u * K̄`, `K̄ = (CB̄, CĀB̄, …, CĀ^{L−1}B̄)` **[LIT: S4D eq. 6]**.

For a diagonal `A`, both formulas reduce to elementwise scalar operations on the `N` eigenvalues
— no matrix inversion or exponential is required.

**[IMPL]** This project uses the **ZOH** rule, following the `s4d_kernel` reference formula
`S4D` gives in Fig. 1 (`(exp(dt*A) - 1)/A` — this is exactly the ZOH `B̄` formula above, since
`(ΔA)⁻¹(exp(ΔA) − I)·Δ = (exp(ΔA) − 1)/A`). The paper states "S4D disentangles the discretization
method from the kernel computation... any discretization can be used" (Table 1), so this is a
literature-sanctioned choice, not a deviation. Bilinear (used in the paper's own Listing 1) is
algebraically equivalent up to O(Δ²) discretization error and is noted here as the alternative we
did **not** implement.

## 3. Convolution kernel via Vandermonde structure **[LIT: S4D eq. 7]**

```
K̄_l = Σ_{n=0}^{N-1} C_n Ā_n^l B̄_n  =  (B̄ᵀ ∘ C) · V_L(Ā),   V_L(Ā)_{n,ℓ} = Ā_n^ℓ
```

`V_L(Ā)` is an `N×L` Vandermonde matrix; computing it naively costs `O(NL)` time/space, which the
paper explicitly says is an acceptable simple algorithm on parallel hardware (Section 3.2,
"Time and Space Complexity" — the sub-quadratic `Õ(N+L)` algorithm is noted as a further
optimization the paper does *not* require you to implement to get a correct S4D).

**[IMPL]** This project computes `V_L(Ā)` by fully materializing the `(H, n_c, L)` tensor and
summing (`src/s4d.py`'s `kernel()`), which is `O(H·N·L)` **time and space**. This is *not* quite
the paper's own stated GPU-friendly naive fallback: the paper's fallback gets `O(NL)` time while
explicitly avoiding materializing the Vandermonde matrix, using `O(N+L)` space (Section 3.2:
"a simple fast algorithm is to compute (7) with naive summation (using `O(NL)` operations), but
*without materializing* the Vandermonde matrix (using `O(N+L)` space)"). This project's version is
a further simplification beyond that — more memory-hungry than even the paper's "simple" version —
made for code legibility (a single dense tensor is easy to read and to unit-test), not because it
matches the paper's stated fallback. Not the asymptotically faster Cauchy-kernel/FFT trick used in
the original `state-spaces/s4` codebase either. A documented simplicity-over-speed-and-memory
choice, consistent with the task's requirement to prioritize a "scientifically defensible"
implementation over performance engineering — but the complexity claim here should not be
overstated as matching the paper's own naive algorithm.

## 4. Parameterization and conjugate symmetry **[LIT: S4D Section 3.3]**

`A` is constrained to have negative real part (for BIBO stability of the autoregressive kernel,
since `K(t) = Ce^{tA}B → ∞` as `t → ∞` if any eigenvalue has positive real part) by writing
`A = −exp(A_re) + i·A_im` and optimizing the unconstrained `A_re`, `A_im` **[LIT: S4D Section 3.3,
"Parameterization of A"]**.

Because the underlying system is real (real input `u`, real output `y`), the diagonalized complex
eigenvalues occur in conjugate pairs; the paper's own Listing 1 keeps only `N/2` of them and
recovers the true kernel by taking **twice the real part** of the truncated sum
(`return 2 * (...).real`) **[LIT: S4D Listing 1]**. This project follows that convention exactly:
`N` below always refers to the number of *stored* complex conjugate-pair parameters (i.e. state
size `2N` real dimensions), matching the paper's own convention.

## 5. Initialization: S4D-Lin **[LIT: S4D eq. 9]**

```
Λ_n = −1/2 + i·π·n,   n = 0, …, N−1
```

This is the initialization implemented here. **[LIT]** justification: Re(Λ_n) = −1/2 is a
"good default that bounds the basis functions" via the envelope `e^{−t/2}` (Section 4,
"General Diagonal SSM Basis Functions"); the imaginary parts are spread linearly. The paper is
explicit that this specific linear scaling law is proposed as "an approximation of S4-FouT"
(Section 4, "S4D-Lin"), **not** as an approximation of dense HiPPO-LegS. **Correction from an
earlier draft of this document:** the HiPPO-LegS convergence guarantee (Theorem 3) and the
imaginary-part asymptotics it motivates (Conjecture 5) are explicitly the derivation behind
**S4D-Inv**, not S4D-Lin ("Based on Conjecture 5, we propose the initialization S4D-Inv...",
Section 4) — S4D-Inv is the variant with the HiPPO-convergence justification, and it is the one
this project does *not* implement (see below). S4D-Lin's own justification is simplicity and the
Fourier/S4-FouT connection, not a HiPPO-convergence proof.

**[IMPL]** S4D-Inv (`Λ_n = −1/2 + i(N/π)(N/(2n+1) − 1)`, eq. 8) is the paper's other proposed
initialization, closer to true HiPPO-LegS at finite `N` and the one with the stronger theoretical
justification (Theorem 3/Conjecture 5, as corrected above); **not implemented here** in the MVP
because S4D-Lin is simpler to verify numerically. **[LIT, verified directly against Table 4 and
Table 5 of the paper]** the two initializations are close on the paper's smaller ablation datasets
(Table 4: sCIFAR/Speech-Commands/BIDMC, within ~1-3 points of each other) but **not** comparable on
the paper's main benchmark: on the full Long Range Arena suite (Table 5), S4D-Inv averages 85.50%
vs. S4D-Lin's 78.39% — a ~7-point gap, with S4D-Lin failing entirely on Path-X (marked `✗`) where
S4D-Inv reaches 92.80%. Choosing S4D-Lin here for implementation simplicity is a legitimate,
documented trade-off, but it should not be described as choosing between two "comparable" options —
S4D-Inv is the paper's stronger-performing (and better-justified) variant on its headline benchmark.

## 6. Recurrent vs. convolutional equivalence — what we test numerically

**[LIT]** The paper's entire premise (Section 2, "S4: Structured State Spaces") is that the
convolutional form (§1 above, `y = u * K̄`) and the discrete-time linear recurrence
`x_t = Ā x_{t-1} + B̄ u_t`, `y_t = C x_t` compute *the same function* — this is a basic fact about
LTI systems, not an S4D-specific claim, but it is the property the whole "dual view" architecture
diagram (Fig. 1, "S4D Recurrent View" / "S4D Convolution Kernel") depends on.

**[HYP → verified numerically, see `tests/test_s4d_numerics.py`]** This project does not take that
equivalence on faith: `tests/test_s4d_numerics.py` computes the same diagonal SSM's output two
ways — (a) by materializing the length-`L` kernel `K̄` via the Vandermonde formula (eq. 7) and
convolving, and (b) by unrolling the discrete recurrence step-by-step for `L` steps — on small
random `(Ā, B̄, C)` and a short random input, and asserts the two outputs agree to float32
tolerance. This is the "validate equivalent formulations against each other on small synthetic
inputs" check the project plan requires, and it is the numerical evidence that our kernel
implementation is not silently wrong before it's ever used inside a trained model.

## 7. What is an S4D layer vs. what is "the model"

**[LIT]** S4D (like S4) defines a *sequence-to-sequence layer*, not a full language model. The
paper's own experiments (Section 5) plug S4D/DSS/S4 layers into task-specific stacks (bidirectional
classifiers for sCIFAR, etc.) — it does not specify a causal-LM head, tokenizer, or training recipe.

**[IMPL]** To make a **controlled** comparison against a Transformer, this project wraps the S4D
layer in a backbone that is *structurally identical* to the Transformer backbone in every respect
*except* the token-mixing sub-layer:

```
tokens
  │
[Token Embedding] (+ [Positional Embedding], attention path only — see below)
  │
  ▼ × n_layer
┌────────────────────────────────────────────┐
│ LayerNorm → [MIXER] → residual add          │   MIXER ∈ {CausalSelfAttention, S4DLayer}
│ LayerNorm → MLP (GELU, 4×)  → residual add  │
└────────────────────────────────────────────┘
  │
[LayerNorm] → [Linear head → logits]
```

**[IMPL]** The S4D-backbone omits the learned positional embedding table that the Transformer
needs (attention is permutation-invariant and requires one; the SSM recurrence is inherently
sequential/causal and does not). This is a real, literature-motivated architectural difference
between the two families, not an oversight — and it is called out explicitly in `experiments.md`
as a source of the (small) parameter-count mismatch between the two configurations, rather than
papered over.

Everything else — embedding dimension, number of layers, MLP width/activation, LayerNorm placement
(pre-norm), dropout, weight tying of `wte`/`lm_head`, optimizer (AdamW), LR schedule (warmup +
cosine decay), batch size, and the character-level Tiny Shakespeare data pipeline — is held
identical between the two configurations being compared, and is reused directly from the
`nano-gpt` repository's `train.py`/`model.py` (engineering reuse, not an SSM-theory source).
