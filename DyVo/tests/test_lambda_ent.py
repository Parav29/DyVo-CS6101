import torch
from torch import nn

from conftest import make_encoders


def test_ent_weight_is_parameter_initialized_from_config(sample_inputs):
    for lam in (0.05, 0.3):
        for enc in make_encoders(entity_weight=lam):
            assert isinstance(enc.ent_weight, nn.Parameter)
            assert enc.ent_weight.requires_grad
            assert abs(enc.ent_weight.item() - lam) < 1e-7


def test_ent_weight_gets_gradient_and_entities_nonzero_at_init(sample_inputs):
    n_ent = sample_inputs["entity_ids"].shape[1]
    for enc in make_encoders(entity_weight=0.05):
        _, w, _ = enc(**sample_inputs)
        ent_w = w[:, -n_ent:]
        assert (ent_w > 0).sum() > 0, "some entity weights must be non-zero at init"
        ent_w.sum().backward()
        assert enc.ent_weight.grad is not None and enc.ent_weight.grad.abs() > 0
