import json

from lsr.entity.bert_candidates import BERTCandidate


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def make_retriever(tmp_path, query_rows=None, doc_rows=None):
    q = write_jsonl(tmp_path / "q.jsonl", query_rows or [
        {"id": "q1", "entities": [3, 1, 3, 2]},
        {"id": 7, "entities": []},
    ])
    d = write_jsonl(tmp_path / "d.jsonl", doc_rows or [
        {"id": "d1", "entities": [5, 5, 0]},
        {"id": "d2", "entities": []},
    ])
    return BERTCandidate(query_ent_path=q, doc_ent_path=d)


def test_reads_dedups_and_returns_ints_and_masks(tmp_path):
    r = make_retriever(tmp_path)
    ids, masks = r.retrieve_candidates("text", "q1", type="query")
    assert sorted(ids) == [1, 2, 3]
    assert all(isinstance(i, int) for i in ids)
    assert masks == [1, 1, 1]
    ids, masks = r.retrieve_candidates("text", "d1", type="doc")
    assert sorted(ids) == [0, 5] and masks == [1, 1]


def test_integer_ids_are_stringified(tmp_path):
    r = make_retriever(tmp_path)
    assert r.retrieve_candidates("t", "7", type="query") == ([], [])


def test_zero_entities_and_unknown_ids_return_empty(tmp_path):
    r = make_retriever(tmp_path)
    assert r.retrieve_candidates("t", "d2", type="doc") == ([], [])
    assert r.retrieve_candidates("t", "does-not-exist", type="query") == ([], [])
    assert r.retrieve_candidates("t", "does-not-exist", type="doc") == ([], [])


def test_reads_authors_llm_file_format(tmp_path):
    """Authors' GPT-4/Mixtral files carry extra `text` and `entity_names` fields; only id/entities are used."""
    row = {"id": "101", "text": "Why are many commentators arguing NFTs are the next big investment category?",
           "entity_names": json.dumps(["Non-fungible token", "Cryptocurrency", "Bitcoin", "Made Up Entity"]),
           "entities": [1, 4, 2]}
    r = make_retriever(tmp_path, query_rows=[row])
    ids, masks = r.retrieve_candidates(row["text"], "101", type="query")
    assert sorted(ids) == [1, 2, 4] and masks == [1, 1, 1]
