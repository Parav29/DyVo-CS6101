"""Entity id 0 is a real entity (e.g. "United States"), and the collator also pads with id 0 (masked).
A padded slot must get weight exactly 0, so it can never create a spurious match with a real entity 0."""
import torch

from lsr.losses.entity_distil_kl_loss import sparse_dot_product
from conftest import VOCAB, encode_inputs


def test_padding_slot_with_id_zero_contributes_nothing(encoders, tokenizer):
    q, d = encoders
    q.eval(); d.eval()
    q_in = encode_inputs(tokenizer, ["is the us a member of the world health organization ?"], [[0]])
    # doc A really contains entity 0; doc B has no entities -> one padded slot with id 0 (+30522), mask 0
    dA = encode_inputs(tokenizer, ["the united states is a member"], [[0]])
    dB = encode_inputs(tokenizer, ["the united states is a member"], [[]])
    assert dB["entity_ids"].tolist() == [[VOCAB]] and dB["entity_masks"].tolist() == [[0]]
    with torch.no_grad():
        q_ids, q_w, _ = q(**q_in)
        a_ids, a_w, _ = d(**dA)
        b_ids, b_w, _ = d(**dB)
    assert q_w[0, -1] > 0, "query entity 0 should be active for this test to be meaningful"
    assert a_w[0, -1] > 0
    assert b_w[0, -1] == 0.0, "padded entity slot must have weight exactly 0"

    def ent_only(ids, w):
        return ids, w * (ids >= VOCAB)

    score_a = sparse_dot_product(*ent_only(q_ids, q_w), a_ids, a_w)
    score_b = sparse_dot_product(*ent_only(q_ids, q_w), b_ids, b_w)
    assert score_a.item() > 0
    assert score_b.item() == 0.0
    # identical text => the full scores differ exactly by the entity-0 contribution
    full_a = sparse_dot_product(q_ids, q_w, a_ids, a_w)
    full_b = sparse_dot_product(q_ids, q_w, b_ids, b_w)
    assert torch.allclose(full_a - full_b, score_a, atol=1e-5)
