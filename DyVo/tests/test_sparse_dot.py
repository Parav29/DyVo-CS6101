import torch

from lsr.losses.entity_distil_kl_loss import sparse_dot_product
from conftest import VOCAB

TOTAL = VOCAB + 200


def dense_reference(q_ids, q_w, d_ids, d_w):
    B = q_ids.shape[0]
    qd = torch.zeros(B, TOTAL).scatter_add_(1, q_ids, q_w)
    dd = torch.zeros(B, TOTAL).scatter_add_(1, d_ids, d_w)
    return (qd * dd).sum(dim=1)


def random_rep(B, n_words, n_ents, dup=True):
    words = torch.randint(0, 50 if dup else VOCAB, (B, n_words))  # small range -> duplicates like MLP queries
    ents = torch.randint(VOCAB, TOTAL, (B, n_ents))
    ids = torch.cat([words, ents], dim=1)
    w = torch.rand(B, n_words + n_ents)
    return ids, w


def test_matches_dense_reference():
    for _ in range(5):
        q_ids, q_w = random_rep(4, 12, 5)
        d_ids, d_w = random_rep(4, 40, 30)
        got = sparse_dot_product(q_ids, q_w, d_ids, d_w)
        ref = dense_reference(q_ids, q_w, d_ids, d_w)
        assert torch.allclose(got, ref, atol=1e-4), (got, ref)


def test_on_real_encoder_outputs(encoders, sample_inputs):
    q, d = encoders
    with torch.no_grad():
        q_ids, q_w, _ = q(**sample_inputs)
        d_ids, d_w, _ = d(**sample_inputs)
    got = sparse_dot_product(q_ids, q_w, d_ids, d_w)
    ref = dense_reference(q_ids, q_w, d_ids, d_w)
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_no_word_entity_collision():
    # word id 5 and entity id 5 (-> 30527) must never match each other
    q_ids = torch.tensor([[5, VOCAB + 5]])
    d_ids = torch.tensor([[5, 7]])  # words 5 and 7 only, no entities
    q_w = torch.tensor([[0.0, 1.0]])  # only the entity has weight
    d_w = torch.tensor([[1.0, 1.0]])
    assert sparse_dot_product(q_ids, q_w, d_ids, d_w).item() == 0.0
    q_w = torch.tensor([[1.0, 0.0]])  # only the word has weight
    assert sparse_dot_product(q_ids, q_w, d_ids, d_w).item() == 1.0
