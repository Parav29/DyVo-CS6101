"""Build tiny, randomly initialized assets for offline sanity checks (no downloads).

Outputs (under DyVo/smoke_assets/):
  tiny-tokenizer/            DistilBertTokenizerFast with exactly 30522 entries
  tiny-distilbert/           DistilBertForMaskedLM(dim=32, 1 layer), random init
  entity_embs_tiny32.pt      200 x 32 random entity table (= hidden size, no projection)
  entity_embs_tiny16.pt      200 x 16 random entity table (exercises the projection path)
  entity_names.json          {id: name} for ids 0..199 (fake + a few realistic names)
  dyvo_init/{query,doc}_encoder/model.safetensors   tiny DyVo dual-encoder init checkpoints

Usage: python smoke_assets/build_assets.py   (run from DyVo/, finishes in a few seconds)
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from words import all_words  # noqa: E402

VOCAB_SIZE = 30522  # BertEntityEmbedding hardcodes this offset; must match
HIDDEN = 32
NUM_ENTITIES = 200
SEED = 0

REALISTIC_NAMES = {
    0: "United States",
    1: "Non-fungible token",
    2: "Bitcoin",
    3: "American Revolutionary War",
    4: "Cryptocurrency",
    5: "Ethereum",
    6: "Blockchain",
    7: "France in the American Revolutionary War",
    8: "Electric car",
    9: "Lithium-ion battery",
    10: "Exchange rate",
    11: "Inflation",
    12: "World Health Organization",
    13: "United Nations",
    14: "Digital art",
}


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_tokenizer(out_dir: Path):
    from transformers import DistilBertTokenizerFast

    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    words = all_words()
    vocab = special + words
    i = 0
    while len(vocab) < VOCAB_SIZE:
        vocab.append(f"[unused{i}]")
        i += 1
    assert len(vocab) == VOCAB_SIZE and len(set(vocab)) == VOCAB_SIZE
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_file = out_dir / "vocab.txt"
    vocab_file.write_text("\n".join(vocab) + "\n")
    tok = DistilBertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True, model_max_length=128)
    tok.save_pretrained(str(out_dir))
    assert tok.vocab_size == VOCAB_SIZE, tok.vocab_size
    assert "[UNK]" not in tok.tokenize("how vital was french support during the american revolutionary war ?")
    return tok


def build_model(out_dir: Path):
    from transformers import DistilBertConfig, DistilBertForMaskedLM

    cfg = DistilBertConfig(vocab_size=VOCAB_SIZE, dim=HIDDEN, hidden_dim=64, n_layers=1, n_heads=2,
                           max_position_embeddings=128, pad_token_id=0)
    model = DistilBertForMaskedLM(cfg)
    model.save_pretrained(str(out_dir))
    assert model.config.vocab_size == VOCAB_SIZE
    return model


def build_entities(out: Path):
    g = torch.Generator().manual_seed(SEED)
    torch.save(torch.randn(NUM_ENTITIES, HIDDEN, generator=g), out / "entity_embs_tiny32.pt")
    torch.save(torch.randn(NUM_ENTITIES, 16, generator=g), out / "entity_embs_tiny16.pt")
    names = {i: REALISTIC_NAMES.get(i, f"Entity_{i}") for i in range(NUM_ENTITIES)}
    (out / "entity_names.json").write_text(json.dumps(names, indent=0))


def build_init_checkpoints(out: Path):
    """Save a tiny DyVo dual encoder the same way HF save_pretrained writes model.safetensors."""
    from lsr.models.entity_emb import BertEntityEmbedding
    from lsr.models.mlp_emlm import TransformerMLPeMLMConfig, TransformerMLPeMLMSparseEncoder
    from lsr.models.mlm_emlm import TransformerMLMeMLMConfig, TransformerMLMeMLMSparseEncoder

    base = str(out / "tiny-distilbert")
    ent = BertEntityEmbedding(entity_emb_path=str(out / "entity_embs_tiny32.pt"))
    q = TransformerMLPeMLMSparseEncoder(
        TransformerMLPeMLMConfig(tf_base_model_name_or_dir=base, entity_dim=HIDDEN, entity_weight=0.05), ent_emb_model=ent)
    d = TransformerMLMeMLMSparseEncoder(
        TransformerMLMeMLMConfig(tf_base_model_name_or_dir=base, entity_dim=HIDDEN, entity_weight=0.05), ent_emb_model=ent)
    # Mark the init checkpoint so tests can verify it (not the random default) was loaded.
    with torch.no_grad():
        q.linear.bias.fill_(0.123)
        d.model.vocab_projector.bias.fill_(0.0)
    q.save_pretrained(str(out / "dyvo_init" / "query_encoder"))
    d.save_pretrained(str(out / "dyvo_init" / "doc_encoder"))
    for sub in ("query_encoder", "doc_encoder"):
        assert (out / "dyvo_init" / sub / "model.safetensors").exists()


def main():
    t0 = time.time()
    seed_all()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tok = build_tokenizer(HERE / "tiny-tokenizer")
    build_model(HERE / "tiny-distilbert")
    build_entities(HERE)
    build_init_checkpoints(HERE)
    size = sum(f.stat().st_size for f in HERE.rglob("*") if f.is_file())
    print(f"vocab_size == {tok.vocab_size}")
    print(f"built smoke assets in {time.time() - t0:.1f}s, total {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
