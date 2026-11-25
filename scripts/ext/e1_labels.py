"""Extension 1, step 2: features + supervision for every (training query, pool candidate).

Uses ONLY training data: the 9,000 synthetic CODEC training queries, their positive document (codec_mistral_qrels)
and the MonoT5-3B teacher scores over ~100 documents per query. Test qrels are never read here.

For query q, docs D_q (teacher-scored), frozen DyVo full100k, fixed linked-entity index:
  word(q,d)      word-piece part of the DyVo score (from the dumped index)
  c_e(d)         = s_e(q) * s_e(d)        entity e's contribution (s_e(d) = 0 if d has no posting for e)
Labels (one column each, same features for all):
  imit   e is in the REL list of q                 (teacher-list imitation; the only list teacher we have)
  occ    e has a posting in the positive document  (DREQ-style document-occurrence label)
  pair   s_e(q) * (s_e(d+) - mean_{d in D_q, teacher<0.5} s_e(d))   (analytic pairwise margin contribution)
  loo    sum_d t(d) [log softmax(S)(d) - log softmax(S - c_e)(d)],  t = teacher scores normalised to sum 1,
         S = word + all pool entity contributions  (listwise leave-one-out utility, multi-document)
Output: scratch/ext/e1/train_feats.npz  (+ test_feats.npz with features only)
"""
import json
import re
import time

import numpy as np

from common import EXT, CODEC, KB, WORD_VOCAB, DocIndex, compose_conf, load_model, encode_queries, log

TOK = re.compile(r"[a-z0-9]+")
FEATS = ["sq", "raw", "laque", "bm25", "laque_rank", "bm25_rank", "in_rel", "in_laque", "in_bm25", "n_src",
         "log_df", "idf", "log_eidx", "title_overlap", "title_len", "sq_rank", "laque_rel", "bm25_rel", "pool_size"]
PRF_FEATS = ["log_prf", "prf_rank", "in_prf", "prf_rel"]  # appended when the pool has the PRF source (E1 v2)


def featurize(rec, df, titles, n_docs):
    q_toks = set(TOK.findall(rec["text"].lower()))
    c = rec["cands"]
    sq = np.array([x["sq"] for x in c]); lq = np.array([x["laque"] for x in c]); bm = np.array([x["bm25"] for x in c])
    sq_rank = np.argsort(np.argsort(-sq)) / max(len(c), 1)
    has_prf = "prf" in c[0] if c else False
    pr_max = max([np.log1p(x["prf"]) for x in c], default=0.0) if has_prf else 0.0
    rows = []
    for j, x in enumerate(c):
        e = x["e"]
        tt = TOK.findall(titles[e].lower())
        in_l, in_b = int(x["laque_rank"] >= 0), int(x["bm25_rank"] >= 0)
        in_p = int(x.get("prf_rank", -1) >= 0)
        row = [x["sq"], x["raw"], x["laque"], x["bm25"],
               (x["laque_rank"] if in_l else 20) / 20, (x["bm25_rank"] if in_b else 20) / 20,
               x["in_rel"], in_l, in_b, x["in_rel"] + in_l + in_b + in_p,
               np.log1p(df[e]), np.log((n_docs + 1) / (df[e] + 1)), np.log1p(e),
               (sum(t in q_toks for t in tt) / len(tt)) if tt else 0.0, len(tt),
               sq_rank[j], x["laque"] - lq.max(), x["bm25"] - bm.max(), len(c)]
        if has_prf:
            row += [np.log1p(x["prf"]), (x["prf_rank"] if in_p else 20) / 20, in_p, np.log1p(x["prf"]) - pr_max]
        rows.append(row)
    return np.array(rows, np.float32)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index_linked", help="dumped index under scratch/ext/")
    ap.add_argument("--tag", default="", help="output suffix, e.g. _grounded for the E1+E2 recomputation")
    ap.add_argument("--max-train", type=int, default=-1)
    ap.add_argument("--pool", default="", help="pool file suffix, e.g. _prf for E1 v2")
    args = ap.parse_args()
    out = EXT / "e1"
    idx = DocIndex(EXT / args.index, keep_csr=True)
    df = idx.entity_df()
    n_docs = len(idx.doc_ids)
    titles = json.load(open(KB / "entity_list.json"))
    teacher = json.load(open(CODEC / "codec_monot5_3b_scores.json"))
    qrels = json.load(open(CODEC / "codec_mistral_qrels.json"))
    conf = compose_conf()
    model = load_model(conf)

    for split in ("test", "train"):
        recs = [json.loads(l) for l in open(out / f"pool_{split}{args.pool}.jsonl")]
        if split == "train" and args.max_train > 0:
            recs = recs[:args.max_train]
        X, qid_of, ent_of = [], [], []
        L = {k: [] for k in ("imit", "occ", "pair", "loo")}
        t0 = time.time()
        if split == "train":
            qv = encode_queries(model, conf, [(r["id"], r["text"]) for r in recs],
                                {r["id"]: [x["e"] for x in r["cands"]] for r in recs}, batch_size=1)
            log(f"encoded {len(recs)} training queries with their pools ({time.time()-t0:.0f}s)")
        for n, r in enumerate(recs):
            F = featurize(r, df, titles, n_docs)
            X.append(F); qid_of += [r["id"]] * len(F); ent_of += [x["e"] for x in r["cands"]]
            if split != "train":
                continue
            ents = np.array([x["e"] for x in r["cands"]])
            ids, w = qv[r["id"]]
            wmask = ids < WORD_VOCAB
            s_q = dict(zip((ids[~wmask] - WORD_VOCAB).tolist(), w[~wmask].tolist()))
            sq = np.array([s_q.get(int(e), 0.0) for e in ents])
            docs = [d for d in teacher.get(r["id"], {}) if d in idx.did2row]
            pos = list(qrels.get(r["id"], {}))
            if not docs or not pos or pos[0] not in idx.did2row:
                for k in L:
                    L[k].append(np.full(len(ents), np.nan, np.float32))
                continue
            rows = np.array([idx.did2row[d] for d in docs])
            sub = idx.csr[rows]
            word = np.asarray(sub[:, :WORD_VOCAB] @ np.bincount(ids[wmask], weights=w[wmask], minlength=WORD_VOCAB)).ravel()
            sd = sub[:, WORD_VOCAB + ents].toarray()  # [n_docs, n_cands] s_e(d)
            C = sd * sq[None, :]
            t = np.array([teacher[r["id"]][d] for d in docs]); tn = t / t.sum()
            p = docs.index(pos[0])
            neg = t < 0.5
            neg[p] = False
            S = word + C.sum(1)
            lse = lambda z: np.log(np.exp(z - z.max()).sum()) + z.max()
            logp = S - lse(S)
            loo = np.array([(tn * (logp - ((S - C[:, j]) - lse(S - C[:, j])))).sum() for j in range(len(ents))])
            L["loo"].append(loo.astype(np.float32))
            L["pair"].append((sq * (sd[p] - (sd[neg].mean(0) if neg.any() else 0))).astype(np.float32))
            L["occ"].append((sd[p] > 0).astype(np.float32))
            L["imit"].append(np.array([x["in_rel"] for x in r["cands"]], np.float32))
            if n % 1000 == 0:
                log(f"{split} {n}/{len(recs)} ({time.time()-t0:.0f}s)")
        names = FEATS + (PRF_FEATS if recs and "prf" in recs[0]["cands"][0] else [])
        save = {"X": np.concatenate(X), "qid": np.array(qid_of), "ent": np.array(ent_of), "feat_names": np.array(names)}
        if split == "train":
            save.update({k: np.concatenate(v) for k, v in L.items()})
        np.savez(out / f"{split}_feats{args.tag}.npz", **save)
        log(f"{split}: {len(save['X'])} (query, candidate) rows saved")


if __name__ == "__main__":
    main()
