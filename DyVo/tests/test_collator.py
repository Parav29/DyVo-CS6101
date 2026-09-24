import json

import torch

from lsr.datasets.text_entity_collator import DynamicDataCollator
from lsr.entity.bert_candidates import BERTCandidate
from conftest import VOCAB


def make_collator(tmp_path, tokenizer):
    (tmp_path / "q.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"id": "q1", "entities": [1, 2, 4]},
        {"id": "q2", "entities": []},
    ]) + "\n")
    (tmp_path / "d.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"id": "d1", "entities": [0, 5]},
        {"id": "d2", "entities": []},
        {"id": "d3", "entities": [7, 8, 9, 10, 11]},
    ]) + "\n")
    retr = BERTCandidate(query_ent_path=str(tmp_path / "q.jsonl"), doc_ent_path=str(tmp_path / "d.jsonl"))
    return DynamicDataCollator(tokenizer, q_max_length=16, d_max_length=32, candidate_retriever=retr, entity_dim=32)


def test_training_batch_padding_masks_and_offset(tmp_path, tokenizer):
    col = make_collator(tmp_path, tokenizer)
    batch = [
        {"query_id": "q1", "query_text": "bitcoin and nft", "doc_id": ["d1", "d2"],
         "doc_text": ["the bitcoin price", "gold standard"], "score": [3.0, 1.0]},
        {"query_id": "q2", "query_text": "the euro", "doc_id": ["d3", "d2"],
         "doc_text": ["euro currency crisis", "gold standard"], "score": [2.0, 0.5]},
    ]
    out = col(batch)
    q = out["queries"]
    assert q["entity_ids"].shape == (2, 3)
    assert q["entity_masks"].tolist() == [[1, 1, 1], [0, 0, 0]]
    assert (q["entity_ids"] >= VOCAB).all()
    assert sorted((q["entity_ids"][0] - VOCAB).tolist()) == [1, 2, 4]
    assert (q["entity_ids"][1] == VOCAB).all()  # padding id 0 offset by vocab size
    # two doc groups (positives, negatives), each padded separately
    assert len(out["doc_groups"]) == 2
    pos, neg = out["doc_groups"]
    assert pos["entity_ids"].shape == (2, 5)
    assert pos["entity_masks"].tolist() == [[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]]
    assert neg["entity_ids"].shape == (2, 1)  # zero-entity group still gets one masked slot
    assert neg["entity_masks"].tolist() == [[0], [0]]
    assert out["labels"].tolist() == [[3.0, 1.0], [2.0, 0.5]]
    for grp in (q, pos, neg):
        assert {"input_ids", "attention_mask", "special_tokens_mask", "entity_ids", "entity_masks"} <= set(grp)


def test_offset_uses_tokenizer_vocab_size(tmp_path, tokenizer):
    col = make_collator(tmp_path, tokenizer)
    assert col.tokenizer.tokenizer.vocab_size == VOCAB  # BertEntityEmbedding hardcodes 30522: must agree


def test_prediction_batches(tmp_path, tokenizer):
    col = make_collator(tmp_path, tokenizer)
    out = col([{"doc_id": "d1", "doc_text": "the bitcoin price"}, {"doc_id": "d2", "doc_text": "gold"}])
    assert out["doc_ids"] == ["d1", "d2"]
    assert out["doc_groups"]["entity_masks"].tolist() == [[1, 1], [0, 0]]
    assert out["queries"] == {} and out["labels"] is None
    out = col([{"query_id": "q2", "query_text": "the euro"}])
    assert out["queries"]["entity_masks"].tolist() == [[0]]
    assert isinstance(out["queries"]["input_ids"], torch.Tensor)
