import torch

from lsr.losses import EntityDistilKLLoss
from lsr.losses.regularizer import L1
from lsr.models import DualSparseEncoder
from conftest import encode_inputs, make_encoders


def make_loss(normalize="std"):
    # same shape as the real configs: no query reg, L1 on doc words only (reg_ent=False)
    return EntityDistilKLLoss(q_regularizer=L1(weight=0.0, T=50000), d_regularizer=L1(weight=1e-5, T=2),
                              normalize=normalize, reg_ent=False)


def batch(tokenizer):
    queries = encode_inputs(tokenizer, ["bitcoin and nft investment", "french support during the war"], [[1, 2], [3]])
    pos = encode_inputs(tokenizer, ["the bitcoin nft market", "french army and american navy"], [[1, 5], [3, 7]])
    neg = encode_inputs(tokenizer, ["gold standard crisis", "electric car battery"], [[10], []])
    # non-negative teacher scores: with normalize="std" (L2-normalize, see NOTES) negative scores give NaN KL
    labels = torch.tensor([[5.0, 1.0], [4.0, 0.5]])
    return queries, [pos, neg], labels


def test_loss_finite_and_grads_reach_everything(tokenizer):
    q, d = make_encoders(entity_dim=16)  # 16 -> projection path too
    model = DualSparseEncoder(q, d)
    loss_fn = make_loss()
    queries, groups, labels = batch(tokenizer)
    kl, q_reg, d_reg, log = model(loss_fn, queries, groups, labels=labels)
    total = kl + q_reg + d_reg
    assert total.dim() == 0 and torch.isfinite(total)
    assert "query #entities" in log and "doc #entities" in log
    total.backward()
    for name, enc in (("query", model.query_encoder), ("doc", model.doc_encoder)):
        assert enc.ent_weight.grad is not None and enc.ent_weight.grad.abs() > 0, name
        assert enc.entity_proj.weight.grad is not None, name
        backbone_grads = [p.grad for p in enc.model.parameters() if p.grad is not None]
        assert backbone_grads and sum(g.abs().sum() for g in backbone_grads) > 0, name
    assert model.query_encoder.linear.weight.grad is not None


def test_softmax_normalize_also_finite(tokenizer):
    q, d = make_encoders()
    queries, groups, labels = batch(tokenizer)
    kl, *_ = DualSparseEncoder(q, d)(make_loss("softmax"), queries, groups, labels=labels)
    assert torch.isfinite(kl) and kl >= 0


def test_l1_skips_entities_when_reg_ent_false():
    reg = L1(weight=1.0, T=1)
    reg.step()
    ids = torch.tensor([[5, 30522 + 3]])
    w = torch.tensor([[2.0, 7.0]])
    assert reg(w, tok_ids=ids).item() == 2.0
    assert reg(w).item() == 9.0


def test_std_normalize_requires_nonnegative_teacher_scores():
    """Documents a pitfall of the official loss: normalize='std' L2-normalizes raw teacher scores and feeds them to
    KLDivLoss as probabilities, so any negative teacher score (e.g. MonoT5 log-probs) makes the loss NaN.
    02_check_data.py checks the real *_monot5_3b_scores.json files for this on the big PC."""
    student = {"ids": torch.tensor([[1, 2]]), "w": torch.tensor([[1.0, 1.0]], requires_grad=True)}
    reps = (student["ids"], student["w"], [])
    loss = EntityDistilKLLoss(normalize="std")
    ok, *_ = loss(reps, reps, reps, labels=torch.tensor([[2.0, 1.0]]))
    bad, *_ = loss(reps, reps, reps, labels=torch.tensor([[-0.1, -3.0]]))
    assert torch.isfinite(ok)
    assert torch.isnan(bad)
