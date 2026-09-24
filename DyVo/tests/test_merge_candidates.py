import json
from pathlib import Path

import pytest

from lsr.entity.bert_candidates import BERTCandidate
from lsr.entity.merge_candidates import main, merge

REPO = Path(__file__).resolve().parents[1]
REAL = REPO / "dyvo_data"


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


@pytest.fixture
def files(tmp_path):
    rel = write(tmp_path / "rel.jsonl", [
        {"id": "mistral_1", "entities": [1, 2]},      # train
        {"id": "mistral_2", "entities": []},          # train, no entities
        {"id": "t1", "entities": [5]},                # test
        {"id": "t2", "entities": [6, 7]},             # test, missing from LLM file -> fallback
        {"id": 3, "entities": [9]},                   # test, integer id in file
    ])
    llm = write(tmp_path / "llm.jsonl", [
        {"id": "t1", "text": "q", "entity_names": json.dumps(["A", "B", "OOV"]), "entities": [0, 11]},
        {"id": "3", "text": "q", "entity_names": ["C"], "entities": [12]},
    ])
    test_ids = write(tmp_path / "test_ids.jsonl", [{"id": "t1", "entities": []}, {"id": "t2", "entities": []},
                                                  {"id": "3", "entities": []}])
    return tmp_path, rel, llm, test_ids


def test_train_keep_rel_test_take_source_missing_fall_back(files):
    tmp, rel, llm, test_ids = files
    out = str(tmp / "out.jsonl")
    stats = merge(rel, llm, test_ids, out)
    rows = {json.loads(l)["id"]: json.loads(l)["entities"] for l in open(out)}
    assert len(rows) == 5 and stats["lines"] == 5
    assert rows["mistral_1"] == [1, 2] and rows["mistral_2"] == []
    assert rows["t1"] == [0, 11]
    assert rows["3"] == [12]
    assert rows["t2"] == [6, 7]
    assert stats["test_ids_fallback_to_rel"] == 1 and stats["fallback_ids"] == ["t2"]
    assert stats["train_ids"] == 2 and stats["test_ids_from_source"] == 2


def test_output_reads_through_bertcandidate(files):
    tmp, rel, llm, test_ids = files
    out = str(tmp / "out.jsonl")
    merge(rel, llm, test_ids, out)
    retr = BERTCandidate(query_ent_path=out, doc_ent_path=write(tmp / "docs.jsonl", [{"id": "d", "entities": [1]}]))
    ids, masks = retr.retrieve_candidates("", "t1", type="query")
    assert sorted(ids) == [0, 11] and masks == [1, 1]
    assert retr.retrieve_candidates("", "mistral_2", type="query") == ([], [])


def test_refuses_to_overwrite_input(files):
    tmp, rel, llm, test_ids = files
    with pytest.raises(ValueError, match="overwrite"):
        merge(rel, llm, test_ids, llm)


def test_unknown_test_id_is_an_error(files):
    tmp, rel, llm, _ = files
    bad = write(tmp / "bad_test_ids.jsonl", [{"id": "nope", "entities": []}])
    with pytest.raises(ValueError, match="not in the REL file"):
        merge(rel, llm, bad, str(tmp / "o.jsonl"))


@pytest.mark.parametrize("dataset", ["codec", "robust04", "wapo"])
def test_real_small_files(dataset, tmp_path):
    """Runs against the authors' small query-entity files downloaded in S5.0 (tracked in the repo)."""
    base = REAL / dataset
    assert (base / "queries_entities_gpt4.jsonl").exists(), "run S5.0 downloads first"
    out = tmp_path / "merged.jsonl"
    assert main(["--dataset", dataset, "--source", "gpt4", "--data-dir", str(REAL), "--output", str(out)]) == 0
    rel = [json.loads(l) for l in open(base / "queries_test_train_entities_rel.jsonl")]
    llm = {str(json.loads(l)["id"]): json.loads(l)["entities"] for l in open(base / "queries_entities_gpt4.jsonl")}
    merged = {json.loads(l)["id"]: json.loads(l)["entities"] for l in open(out)}
    assert len(merged) == len(rel)
    assert all(merged[q] == e for q, e in llm.items())
    assert all(merged[str(r["id"])] == r["entities"] for r in rel if str(r["id"]) not in llm)


def test_real_codec_human_merge_reproduces_authors_file(tmp_path):
    out = tmp_path / "human.jsonl"
    main(["--dataset", "codec", "--source", "human", "--data-dir", str(REAL), "--output", str(out)])
    authors = {str(json.loads(l)["id"]): json.loads(l)["entities"] for l in open(REAL / "codec/queries_test_train_entities_human.jsonl")}
    merged = {json.loads(l)["id"]: json.loads(l)["entities"] for l in open(out)}
    assert merged == authors
