"""Build synthetic smoke data in the exact formats the official loaders read (see NOTES.md "Data formats").

  docs.tsv                          doc_id \t text                (lsr.utils.dataset_utils.read_collection, local path)
  train_queries.tsv                 qid \t text                   (read_queries, local path)  -- ids like the Mixtral ones: mistral_<n>
  train_qrels.json                  {qid: {doc_id: rel}}          (read_qrels, local path)
  train_monot5_scores.json          {qid: {doc_id: score >= 0}}   (read_ce_score, *.json) -- MonoT5-style teacher scores
  test_queries.tsv / test_qrels.json                              (PredictionDataset with local paths)
  docs_entities.jsonl               {"id": doc_id, "entities": [int, ...]}   (BERTCandidate)
  queries_entities.jsonl            {"id": qid, "entities": [int, ...]}      (BERTCandidate; train + test ids, REL-style)

All words come from smoke_assets/words.py so the tiny tokenizer never emits [UNK].
Deterministic (seed 0). Usage (from DyVo/): python smoke_data/build_data.py
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "smoke_assets"))
from words import FILLER_WORDS, TOPIC_WORDS  # noqa: E402

SEED = 0
TOPICS = list(TOPIC_WORDS)  # 5 topics
# entity ids per topic (ids < 200 = rows of the tiny entity tables); 0 is a real entity ("United States")
TOPIC_ENTITIES = {"crypto": [1, 2, 4, 5, 6, 14], "war": [3, 7, 20, 21, 0], "cars": [8, 9, 30, 31],
                  "money": [10, 11, 40, 41, 42], "health": [12, 13, 0, 50]}
DOCS_PER_TOPIC = 4
FILLER = [w for w in FILLER_WORDS if w not in {"?", ".", ","}]


def sentence(rng, topic, n_topic, n_filler):
    words = rng.sample(TOPIC_WORDS[topic], n_topic) + rng.choices(FILLER, k=n_filler)
    rng.shuffle(words)
    return " ".join(words)


def main():
    rng = random.Random(SEED)
    docs, doc_topic, doc_ents = {}, {}, {}
    for t in TOPICS:
        for i in range(DOCS_PER_TOPIC):
            did = f"{t}_doc{i}"
            docs[did] = sentence(rng, t, 6, 10) + " . " + sentence(rng, t, 4, 8)
            doc_topic[did] = t
            # doc i==3 of each topic has no entities (exercises the zero-entity path)
            doc_ents[did] = [] if i == 3 else rng.sample(TOPIC_ENTITIES[t], k=min(3, len(TOPIC_ENTITIES[t])))
    doc_ids = list(docs)

    def make_queries(prefix, n, first_index=0):
        qs = {}
        for k in range(n):
            t = TOPICS[k % len(TOPICS)]
            qs[f"{prefix}{k + first_index}"] = ("how " if k % 2 else "why ") + sentence(rng, t, 3, 3) + " ?"
        return qs

    train_q = make_queries("mistral_", 10)
    test_q = {f"{t}-1": q for t, q in zip(TOPICS, make_queries("tmp", 5).values())}

    def topic_of(qid, qs):
        return TOPICS[list(qs).index(qid) % len(TOPICS)]

    train_qrels, ce = {}, {}
    for qid in train_q:
        t = topic_of(qid, train_q)
        rel = [d for d in doc_ids if doc_topic[d] == t]
        train_qrels[qid] = {rel[0]: 1}
        negs = rng.sample([d for d in doc_ids if doc_topic[d] != t], 4)  # "BM25 negatives"
        # MonoT5-style non-negative scores (see NOTES: normalize="std" needs non-negative teacher scores)
        ce[qid] = {d: round(rng.uniform(0.6, 0.99), 4) for d in rel}
        ce[qid].update({d: round(rng.uniform(0.001, 0.3), 4) for d in negs})
    test_qrels = {}
    for qid in test_q:
        t = qid.split("-")[0]
        rel = [d for d in doc_ids if doc_topic[d] == t]
        test_qrels[qid] = {d: (2 if j == 0 else 1) for j, d in enumerate(rel)}
        test_qrels[qid][rng.choice([d for d in doc_ids if doc_topic[d] != t])] = 0

    q_ents = {}
    for qs in (train_q, test_q):
        for j, qid in enumerate(qs):
            t = topic_of(qid, qs)
            q_ents[qid] = [] if j == 2 else rng.sample(TOPIC_ENTITIES[t], k=2)  # one query per split w/o entities

    def write_tsv(path, rows):
        for k, v in rows.items():
            assert "\t" not in k and "\t" not in v and "\n" not in v
        (HERE / path).write_text("".join(f"{k}\t{v}\n" for k, v in rows.items()))

    write_tsv("docs.tsv", docs)
    write_tsv("train_queries.tsv", train_q)
    write_tsv("test_queries.tsv", test_q)
    for name, obj in (("train_qrels.json", train_qrels), ("train_monot5_scores.json", ce), ("test_qrels.json", test_qrels)):
        (HERE / name).write_text(json.dumps(obj, indent=1))
    (HERE / "docs_entities.jsonl").write_text("".join(json.dumps({"id": d, "entities": e}) + "\n" for d, e in doc_ents.items()))
    (HERE / "queries_entities.jsonl").write_text("".join(json.dumps({"id": q, "entities": e}) + "\n" for q, e in q_ents.items()))
    print(f"docs={len(docs)} train_queries={len(train_q)} test_queries={len(test_q)} "
          f"docs_without_entities={sum(1 for e in doc_ents.values() if not e)} "
          f"queries_without_entities={sum(1 for e in q_ents.values() if not e)}")


if __name__ == "__main__":
    main()
