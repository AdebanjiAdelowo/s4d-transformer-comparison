"""
Basic shape/sanity checks on the shared SequenceBackbone (src/model.py) for
both mixer types -- guards against silent bugs in the shared training
pipeline (embeddings, block stacking, weight tying, loss computation) that
the S4D-specific numerical tests in test_s4d_numerics.py do not cover.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import BackboneConfig, SequenceBackbone  # noqa: E402


@pytest.mark.parametrize("mixer", ["attn", "s4d"])
def test_forward_shapes_and_finite_loss(mixer):
    torch.manual_seed(0)
    config = BackboneConfig(
        block_size=16, vocab_size=20, n_layer=2, n_embd=16,
        n_head=2, s4d_state=8, mixer=mixer,
    )
    model = SequenceBackbone(config)
    idx = torch.randint(0, config.vocab_size, (3, 16))
    targets = torch.randint(0, config.vocab_size, (3, 16))

    logits, loss = model(idx, targets)
    assert logits.shape == (3, 16, config.vocab_size)
    assert loss.dim() == 0
    assert torch.isfinite(loss)

    logits_no_targets, loss_none = model(idx)
    assert logits_no_targets.shape == (3, 1, config.vocab_size)
    assert loss_none is None


def test_weight_tying_shared_across_mixers():
    for mixer in ("attn", "s4d"):
        config = BackboneConfig(block_size=8, vocab_size=10, n_layer=1, n_embd=8, n_head=2,
                                 s4d_state=4, mixer=mixer)
        model = SequenceBackbone(config)
        assert model.wte.weight is model.lm_head.weight


def test_attn_has_positional_embedding_s4d_does_not():
    """Documents the one intentional structural asymmetry between the two backbones
    (architecture.md Section 7): attention needs a positional embedding, the SSM recurrence
    does not."""
    attn_cfg = BackboneConfig(block_size=8, vocab_size=10, n_layer=1, n_embd=8, mixer="attn")
    s4d_cfg = BackboneConfig(block_size=8, vocab_size=10, n_layer=1, n_embd=8, mixer="s4d", s4d_state=4)
    assert SequenceBackbone(attn_cfg).wpe is not None
    assert SequenceBackbone(s4d_cfg).wpe is None
