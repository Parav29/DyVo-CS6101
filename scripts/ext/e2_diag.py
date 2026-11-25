"""Extension 2 headroom diagnostic (plan step 3 / council exit gate "no missing-support headroom -> stop").

For (query entity e, document d) pairs where d is a positive/relevant document:
  linked      e is in d's REL-linked list (DyVo can match it today)
  missing     e is NOT linked but IS explicitly mentioned in d's visible text (what grounded repair can fix)
  absent      neither
Training side (decision-relevant): 9,000 training queries, their single positive, query entities = REL list
and the E1 pool's LaQue top-5. Test side (analysis only): REL / GPT-4 / Mixtral candidates vs judged documents.
Output: scratch/ext/e2/diag.json
"""
import json

import numpy as np

from common import EXT, CODEC, codec_test, load_jsonl_entities, log


def rates(pairs):
    n = len(pairs)
    if n == 0:
        return {"pairs": 0}
    a = np.array(pairs)
    return {"pairs": n, "linked": float((a == 0).mean()), "missing_but_mentioned": float((a == 1).mean()),
            "absent": float((a == 2).mean())}


def main():
    d = EXT / "e2"
    linked = {k: set(map(int, v)) for k, v in load_jsonl_entities(CODEC / "docs_entities.jsonl").items()}
    ment = {}
    for l in open(d / "mentions.jsonl"):
        r = json.loads(l)
        ment[r["id"]] = {m[0] for m in r["m"]}
    status = lambda e, doc: 0 if e in linked.get(doc, ()) else (1 if e in ment.get(doc, ()) else 2)
    out = {}
    # training side
    qrels = json.load(open(CODEC / "codec_mistral_qrels.json"))
    rel = load_jsonl_entities(CODEC / "queries_test_train_entities_rel.jsonl")
    pool = {}
    for l in open(EXT / "e1" / "pool_train.jsonl"):
        r = json.loads(l)
        pool[r["id"]] = [x["e"] for x in sorted(r["cands"], key=lambda x: x["laque_rank"]) if x["laque_rank"] >= 0][:5]
    for name, src in (("train REL vs positive", rel), ("train LaQue-top5 vs positive", pool)):
        pairs = [status(int(e), p) for q, ps in qrels.items() for p in ps for e in set(src.get(q, []))]
        out[name] = rates(pairs)
    # test side (analysis only)
    _, tq = codec_test()
    for s in ("rel", "gpt4", "mistral"):
        src = rel if s == "rel" else load_jsonl_entities(CODEC / f"queries_entities_{s}.jsonl")
        for lab, cond in (("relevant (rel>=1)", lambda r: r >= 1), ("non-relevant (rel=0)", lambda r: r == 0)):
            pairs = [status(int(e), doc) for q, docs in tq.items() for doc, r in docs.items() if cond(r)
                     for e in set(src.get(q, []))]
            out[f"test {s} vs {lab}"] = rates(pairs)
    json.dump(out, open(d / "diag.json", "w"), indent=1)
    for k, v in out.items():
        log(f"{k:40s} {json.dumps(v)}")


if __name__ == "__main__":
    main()
