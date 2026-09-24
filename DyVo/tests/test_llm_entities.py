"""S5.5: generative entity retrieval with the mock backend (no network, no API keys)."""
import json
from pathlib import Path

import pytest

from lsr.entity.bert_candidates import BERTCandidate
from lsr.entity import generate_entities_llm as g

REPO = Path(__file__).resolve().parents[1]
NAMES = REPO / "smoke_assets" / "entity_names.json"
REAL = REPO / "dyvo_data"


def test_prompt_is_paper_prompt_a2():
    p = g.build_prompt("  Why are many commentators arguing NFTs are the next big investment category? ")
    assert p.startswith("Identify Wikipedia entities that are helpful to retrieve documents relevant to a web search "
                        "query. Please return a list of entity names only:\nExample 1:\n")
    assert '"Mining industry of the Democratic Republic of the Congo"' in p
    assert '"Original sin (economics)", "Exchange-rate regime"]\n' in p
    assert p.count("Query:") == 3
    assert p.endswith("Query: Why are many commentators arguing NFTs are the next big investment category?\nEntities:")


@pytest.mark.parametrize("raw", [
    '["Bitcoin", "Ethereum", "Non-fungible token"]',
    'Sure! Here are the entities:\n["Bitcoin", "Ethereum", "Non-fungible token"]\nHope this helps.',
    '```json\n["Bitcoin", "Ethereum", "Non-fungible token"]\n```',
    "['Bitcoin', 'Ethereum', 'Non-fungible token']",
    '["Bitcoin", "Ethereum", "Non-fungible token"',             # truncated (no closing bracket)
    "- Bitcoin\n- Ethereum\n- Non-fungible token",
    "1. Bitcoin\n2. Ethereum\n3. Non-fungible token",
    json.dumps(json.dumps(["Bitcoin", "Ethereum", "Non-fungible token"])),  # double-encoded
    ["Bitcoin", "Ethereum", "Non-fungible token"],
])
def test_parser_variants(raw):
    assert g.parse_entity_list(raw) == ["Bitcoin", "Ethereum", "Non-fungible token"]


def test_parser_duplicates_case_and_whitespace():
    assert g.parse_entity_list('["Bitcoin", "bitcoin", " Ethereum ", "BITCOIN", "Digital  art"]') == \
        ["Bitcoin", "Ethereum", "Digital art"]


def test_parser_malformed_and_empty():
    assert g.parse_entity_list("") == []
    assert g.parse_entity_list(None) == []
    assert g.parse_entity_list("[]") == []
    # prose without any list is kept as a single "name"; it is then dropped as OOV by the name->id mapping
    assert g.parse_entity_list("I cannot help with that.") == ["I cannot help with that."]
    out = g.parse_entity_list('Entities: ["Bitcoin", "Eth')
    assert out[0] == "Bitcoin"


def test_mapping_case_insensitive_and_oov():
    mapper = g.NameMapper(g.load_entity_vocab(NAMES))
    ids, oov = g.map_names(["bitcoin", "Non-fungible token", "Totally Unknown", "BITCOIN", "united_states"], mapper)
    assert ids == [2, 1, 0]
    assert oov == ["Totally Unknown"]


def test_vocab_shapes(tmp_path):
    as_list = tmp_path / "l.json"
    as_list.write_text(json.dumps(["United States", "Bitcoin"]))
    as_n2i = tmp_path / "n.json"
    as_n2i.write_text(json.dumps({"United States": 0, "Bitcoin": 1}))
    assert g.load_entity_vocab(as_list) == g.load_entity_vocab(as_n2i) == {"United States": 0, "Bitcoin": 1}
    bad = tmp_path / "b.json"
    bad.write_text(json.dumps({"a": [1, 2]}))
    with pytest.raises(ValueError, match="unsupported"):
        g.load_entity_vocab(bad)


def test_mock_end_to_end_output_format_and_bertcandidate(tmp_path):
    queries = tmp_path / "q.tsv"
    queries.write_text("q1\tWhy are many commentators arguing NFTs are the next big investment category?\n"
                       "q2\tHow vital was French support during the American Revolutionary War?\n")
    vocab = g.load_entity_vocab(NAMES)
    backend = g.MockBackend(responses={
        "Why are many commentators arguing NFTs are the next big investment category?":
            'Here you go: ["Non-fungible token", "Cryptocurrency", "Bitcoin", "bitcoin", "Madeup Thing"]',
        "How vital was French support during the American Revolutionary War?":
            '```json\n["American Revolutionary War", "France in the American Revolutionary War", "United States"]\n```',
    })
    out = tmp_path / "out.jsonl"
    stats = g.generate(g.read_queries(queries), backend, g.NameMapper(vocab), out)
    assert stats["written"] == 2 and stats["oov"] == 1 and abs(stats["oov_rate"] - 1 / 7) < 1e-9
    rows = [json.loads(l) for l in open(out)]
    assert set(rows[0]) == {"id", "text", "entity_names", "entities"}
    assert isinstance(rows[0]["entity_names"], str)  # JSON-encoded, like the authors' GPT-4 files
    assert json.loads(rows[0]["entity_names"]) == ["Non-fungible token", "Cryptocurrency", "Bitcoin", "Madeup Thing"]
    assert rows[0]["entities"] == [1, 4, 2]
    assert rows[1]["entities"] == [3, 7, 0]  # entity id 0 is a real entity
    retr = BERTCandidate(query_ent_path=str(out), doc_ent_path=str(out))
    ids, masks = retr.retrieve_candidates("", "q2", type="query")
    assert sorted(ids) == [0, 3, 7] and masks == [1, 1, 1]
    # resume: re-running skips done ids
    assert g.generate(g.read_queries(queries), backend, g.NameMapper(vocab), out)["skipped"] == 2


def test_cli_mock_and_dry_run(tmp_path, capsys):
    q = tmp_path / "q.tsv"
    q.write_text("a\tbitcoin and ethereum prices\n")
    out = tmp_path / "o.jsonl"
    assert g.main(["--backend", "mock", "--queries", str(q), "--entity-list", str(NAMES), "--output", str(out),
                   "--dry-run"]) == 0
    assert not out.exists()
    assert "first prompt" in capsys.readouterr().out
    assert g.main(["--backend", "mock", "--queries", str(q), "--entity-list", str(NAMES), "--output", str(out)]) == 0
    row = json.loads(out.read_text())
    assert 2 in row["entities"] and 5 in row["entities"]


def test_openai_backend_refuses_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        g.make_backend("openai")


@pytest.mark.parametrize("dataset,source", [("codec", "gpt4"), ("robust04", "gpt4"), ("wapo", "gpt4"),
                                            ("codec", "mistral"), ("robust04", "mistral"), ("wapo", "mistral")])
def test_parser_recovers_authors_entity_names(dataset, source):
    """Our parser must recover exactly the name list stored in the authors' files (both encodings)."""
    path = REAL / dataset / f"queries_entities_{source}.jsonl"
    for line in open(path):
        row = json.loads(line)
        raw = row["entity_names"]
        ref = raw if isinstance(raw, list) else json.loads(raw)
        ref_dedup = []
        for n in ref:
            n2 = " ".join(n.strip().strip("\"'").split())
            if n2 and n2.casefold() not in {x.casefold() for x in ref_dedup}:
                ref_dedup.append(n2)
        assert g.parse_entity_list(raw) == ref_dedup, row["id"]


def test_remap_mode_uses_existing_names(tmp_path):
    src = tmp_path / "authors.jsonl"
    src.write_text(json.dumps({"id": "301", "text": "q", "entity_names": json.dumps(["Bitcoin", "Nope", "Ethereum"]),
                               "entities": [999999, 999999]}) + "\n")
    out = tmp_path / "remapped.jsonl"
    assert g.main(["--backend", "remap", "--input", str(src), "--entity-list", str(NAMES), "--output", str(out)]) == 0
    row = json.loads(out.read_text())
    assert row["entities"] == [2, 5] and json.loads(row["entity_names"]) == ["Bitcoin", "Nope", "Ethereum"]
