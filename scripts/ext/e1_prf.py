"""Extension 1 v2: add a corpus-feedback (PRF) entity source to the candidate pool.

PRF(q): run DyVo's WORD branch alone over the full corpus (no entities, no relevance judgments), take the top-10
documents, score each linked entity by sum_d s_e(d) * word_score(q, d), keep the top-20 (fixed a priori, same K as
the BM25/LaQue generators). Every PRF candidate has postings by construction, and it is "document-utility shaped":
it proposes entities that co-occur with the query's best lexical matches.

Input  : scratch/ext/e1/pool_{train,test}.jsonl
Output : scratch/ext/e1/pool_{train,test}_prf.jsonl  (same records + PRF candidates; every candidate gains
         prf (score), prf_rank (-1 if not in PRF top-20))
"""
import json
import time

import numpy as np
import torch

from common import EXT, CODEC, KB, WORD_VOCAB, DocIndex, compose_conf, load_model, encode_queries, log
from e1_pool import TitleBM25, LaqueDense, DyVoDense

K_DOCS, K_ENTS = 10, 20


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index_linked")
    ap.add_argument("--tag", default="", help="output pool suffix after _prf, e.g. _g2 for the repaired index")
    args = ap.parse_args()
    d = EXT / "e1"
    titles = json.load(open(KB / "entity_list.json"))
    allowed = np.load(d / "df_linked.npy") > 0
    bm25 = TitleBM25(titles, allowed)
    idx = DocIndex(EXT / args.index, keep_csr=True)
    conf = compose_conf(); model = load_model(conf)
    dense = DyVoDense(model, conf, None)
    laque = LaqueDense(dense.qe.ent_emb_model.entity_emb, allowed)
    for split in ("test", "train"):
        t0 = time.time()
        recs = [json.loads(l) for l in open(d / f"pool_{split}.jsonl")]
        qv = encode_queries(model, conf, [(r["id"], r["text"]) for r in recs], {}, batch_size=1)  # word-only
        n_new = 0
        with open(d / f"pool_{split}_prf{args.tag}.jsonl", "w") as f:
            for n, r in enumerate(recs):
                ids, w = qv[r["id"]]
                s = np.asarray(idx.csc[:, ids] @ w).ravel()
                top = np.argpartition(-s, K_DOCS)[:K_DOCS]
                acc = {}
                for j in top:
                    for e, we in idx.doc_entity_weights(j).items():
                        acc[e] = acc.get(e, 0.0) + we * float(s[j])
                prf = sorted(acc, key=lambda e: (-acc[e], e))[:K_ENTS]
                prf_rank = {e: i for i, e in enumerate(prf)}
                have = {x["e"] for x in r["cands"]}
                new = [e for e in prf if e not in have]
                if new:
                    h = dense.hidden(r["text"]); sq, raw = dense.score(h, new)
                    v = laque.embed(r["text"]); ls = laque.score(v, new); bs = bm25.score(r["text"], new)
                    r["cands"] += [{"e": int(e), "sq": float(sq[j]), "raw": float(raw[j]), "laque": float(ls[j]),
                                    "bm25": float(bs[j]), "laque_rank": -1, "bm25_rank": -1, "in_rel": 0}
                                   for j, e in enumerate(new)]
                    n_new += len(new)
                for x in r["cands"]:
                    x["prf"] = float(acc.get(x["e"], 0.0)); x["prf_rank"] = prf_rank.get(x["e"], -1)
                f.write(json.dumps(r) + "\n")
                if n % 1000 == 0:
                    log(f"{split} {n}/{len(recs)} ({time.time()-t0:.0f}s)")
        log(f"{split}: {len(recs)} queries, {n_new/len(recs):.1f} new PRF candidates/query ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
