import torch

from conftest import ASSETS, VOCAB, make_entity_model


def test_rows_for_offset_ids_and_shape():
    emb = make_entity_model(32)
    table = torch.load(ASSETS / "entity_embs_tiny32.pt")
    ids = torch.tensor([[0, 5, 199], [3, 3, 42]]) + VOCAB
    out = emb(ids)
    assert out.shape == (2, 3, 32)
    for b in range(2):
        for k in range(3):
            assert torch.equal(out[b, k], table[ids[b, k] - VOCAB])


def test_table_is_frozen_and_no_grad():
    emb = make_entity_model(32)
    assert len(list(emb.parameters())) == 0, "entity table must not be a trainable parameter"
    assert "entity_emb" not in emb.state_dict(), "entity table must not be saved in checkpoints"
    out = emb(torch.tensor([[VOCAB + 1]]))
    assert not out.requires_grad
    assert not emb.entity_emb.requires_grad
