"""Shared fixtures: tiny random DyVo encoders built from smoke_assets/ (offline, CPU)."""
import os
import random
import sys
from pathlib import Path

for _k, _v in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
               "WANDB_MODE": "disabled", "CUDA_VISIBLE_DEVICES": ""}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "smoke_assets"
sys.path.insert(0, str(REPO))

VOCAB = 30522
HIDDEN = 32


def _ensure_assets():
    if not (ASSETS / "dyvo_init" / "doc_encoder" / "model.safetensors").exists():
        import subprocess
        subprocess.run([sys.executable, str(ASSETS / "build_assets.py")], check=True)


_ensure_assets()


@pytest.fixture(autouse=True)
def _seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def tokenizer():
    from lsr.tokenizer import HFTokenizer
    tok = HFTokenizer(str(ASSETS / "tiny-tokenizer"))
    assert tok.tokenizer.vocab_size == VOCAB
    return tok


def make_entity_model(dim=32):
    from lsr.models.entity_emb import BertEntityEmbedding
    return BertEntityEmbedding(entity_emb_path=str(ASSETS / f"entity_embs_tiny{dim}.pt"))


def make_encoders(entity_dim=32, entity_weight=0.05):
    from lsr.models.mlp_emlm import TransformerMLPeMLMConfig, TransformerMLPeMLMSparseEncoder
    from lsr.models.mlm_emlm import TransformerMLMeMLMConfig, TransformerMLMeMLMSparseEncoder
    base = str(ASSETS / "tiny-distilbert")
    ent = make_entity_model(entity_dim)
    q = TransformerMLPeMLMSparseEncoder(TransformerMLPeMLMConfig(
        tf_base_model_name_or_dir=base, entity_dim=entity_dim, entity_weight=entity_weight), ent_emb_model=ent)
    d = TransformerMLMeMLMSparseEncoder(TransformerMLMeMLMConfig(
        tf_base_model_name_or_dir=base, entity_dim=entity_dim, entity_weight=entity_weight), ent_emb_model=ent)
    return q, d


@pytest.fixture
def encoders():
    return make_encoders()


def encode_inputs(tokenizer, texts, entity_lists, max_length=64):
    """Tokenize texts and build padded entity ids (offset by 30522) + masks, exactly like DynamicDataCollator."""
    enc = dict(tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                         return_tensors="pt", return_special_tokens_mask=True))
    n = max(1, max(len(e) for e in entity_lists))
    ids = [e + [0] * (n - len(e)) for e in entity_lists]
    masks = [[1] * len(e) + [0] * (n - len(e)) for e in entity_lists]
    enc["entity_ids"] = torch.tensor(ids) + VOCAB
    enc["entity_masks"] = torch.tensor(masks)
    return enc


@pytest.fixture
def sample_inputs(tokenizer):
    texts = ["why are many commentators arguing nft are the next big investment category ?",
             "how vital was french support during the american revolutionary war ?",
             "the price of gold"]
    ents = [[1, 2, 4, 5], [3, 7], []]
    return encode_inputs(tokenizer, texts, ents)
