"""DualSparseEncoder loads the init checkpoints with strict=False; make sure the important keys really load."""
import torch
from safetensors.torch import load_file

from lsr.models import DualSparseEncoder
from conftest import ASSETS, make_encoders

Q_CKPT = str(ASSETS / "dyvo_init" / "query_encoder" / "model.safetensors")
D_CKPT = str(ASSETS / "dyvo_init" / "doc_encoder" / "model.safetensors")


def test_init_checkpoints_load(capsys):
    q, d = make_encoders()
    assert not torch.allclose(q.linear.bias, torch.tensor(0.123))  # fresh random init
    model = DualSparseEncoder(q, d, query_encoder_checkpoint=Q_CKPT, doc_encoder_checkpoint=D_CKPT)

    for enc, path in ((model.query_encoder, Q_CKPT), (model.doc_encoder, D_CKPT)):
        sd = load_file(path)
        res = enc.load_state_dict(sd, strict=False)
        print(f"{path}: missing={res.missing_keys} unexpected={res.unexpected_keys}")
        # only the tied MLM output projection may be absent (HF de-duplicates tied weights on save)
        assert set(res.missing_keys) <= {"model.vocab_projector.weight"}, res.missing_keys
        assert res.unexpected_keys == []
        own = enc.state_dict()
        for k, v in sd.items():
            assert torch.equal(own[k], v), k
        assert "ent_weight" in sd

    # marker written by build_assets.py proves the checkpoint (not the random init) is in use
    assert torch.allclose(model.query_encoder.linear.bias, torch.tensor(0.123))
    # tied weights stay tied after loading
    mlm = model.doc_encoder.model
    assert mlm.vocab_projector.weight.data_ptr() == mlm.distilbert.embeddings.word_embeddings.weight.data_ptr()


def test_lsrw_encoders_load_dyvo_init_ignoring_entity_keys():
    """The LSR-w configs load the same dyvo_init files into plain MLP/MLM encoders (extra keys ignored)."""
    from lsr.models.mlp import TransformerMLPConfig, TransformerMLPSparseEncoder
    from lsr.models.mlm import TransformerMLMConfig, TransformerMLMSparseEncoder
    base = str(ASSETS / "tiny-distilbert")
    q = TransformerMLPSparseEncoder(TransformerMLPConfig(tf_base_model_name_or_dir=base))
    d = TransformerMLMSparseEncoder(TransformerMLMConfig(tf_base_model_name_or_dir=base))
    model = DualSparseEncoder(q, d, query_encoder_checkpoint=Q_CKPT, doc_encoder_checkpoint=D_CKPT)
    res = model.query_encoder.load_state_dict(load_file(Q_CKPT), strict=False)
    assert res.missing_keys == []
    assert set(res.unexpected_keys) == {"ent_weight", "tok_weight"}
    assert torch.allclose(model.query_encoder.linear.bias, torch.tensor(0.123))
