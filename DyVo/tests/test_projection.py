import torch

from conftest import make_encoders


def test_projection_created_and_trainable_when_dims_differ(sample_inputs):
    for enc in make_encoders(entity_dim=16):
        assert hasattr(enc, "entity_proj")
        assert enc.entity_proj.in_features == 16 and enc.entity_proj.out_features == 32
        assert all(p.requires_grad for p in enc.entity_proj.parameters())
        ids, w, _ = enc(**sample_inputs)
        n_ent = sample_inputs["entity_ids"].shape[1]
        w[:, -n_ent:].sum().backward()
        assert enc.entity_proj.weight.grad is not None
        assert enc.entity_proj.weight.grad.abs().sum() > 0


def test_no_projection_when_dims_match():
    for enc in make_encoders(entity_dim=32):
        assert not hasattr(enc, "entity_proj")
        assert not any("entity_proj" in n for n, _ in enc.named_parameters())
