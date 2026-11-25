"""Extension 1, step 3: train selectors, choose K on training-dev queries, evaluate on the 42 CODEC test queries.

Exactness: with batch size 1, DyVo's query word weights and each entity weight s_e(q) do not depend on the other
candidates (entity logits are per-entity, max-pooled over tokens). So each query is encoded ONCE with the union of
all candidates it can ever receive, and any selected set S scores documents as
    score(d) = word(q, d) + sum_{e in S} s_e(q) * s_e(d)
which equals re-encoding the query with candidates S.

Selection data split: 9,000 training queries -> 8,000 fit / 1,000 dev (seed 0). Dev = full-corpus retrieval with
the single training positive (binary qrels). K and every other choice are made on dev; test qrels only score.
Outputs: scratch/ext/e1/results/*.json, candidate files queries_e1_<method>_k<K>.jsonl
"""
import argparse
import json
import pickle
import time

import numpy as np
import torch

from common import (EXT, CODEC, WORD_VOCAB, DocIndex, compose_conf, load_model, encode_queries, codec_test, evaluate,
                    paired_tests, holm, log, load_jsonl_entities)

LABELS = ["imit", "occ", "pair", "loo"]
KS = [1, 3, 5, 10, 20]


# ------------------------------------------------------------------ selector
class MLP(torch.nn.Module):  # input width = number of features (19 for v1, 23 with PRF)
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                                       torch.nn.ReLU(), torch.nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def group(qids):
    """start/end offsets of consecutive rows per query."""
    cut = np.flatnonzero(qids[1:] != qids[:-1]) + 1
    starts = np.concatenate([[0], cut]); ends = np.concatenate([cut, [len(qids)]])
    return list(zip(starts, ends))


def train_selector(X, y, groups, epochs=8, seed=0, dev="cuda"):
    """ListNet: target = y+ normalised within the query (binary -> uniform over positives). Same for every label."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = MLP(X.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = torch.tensor(X, device=dev)
    yp = np.clip(np.nan_to_num(y, nan=0.0), 0, None)
    gs = [(a, b) for a, b in groups if yp[a:b].sum() > 0]
    tgt = torch.tensor(yp, device=dev)
    for ep in range(epochs):
        order = rng.permutation(len(gs)); tot = 0.0
        for i in range(0, len(order), 64):
            loss = 0.0
            for g in order[i:i + 64]:
                a, b = gs[g]
                t = tgt[a:b] / tgt[a:b].sum()
                loss = loss - (t * torch.log_softmax(model(Xt[a:b]), 0)).sum()
            loss = loss / len(order[i:i + 64])
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
    model.eval()
    return model, len(gs)


# ------------------------------------------------------------------ retrieval with cached word scores
class Scorer:
    def __init__(self, idx, qvecs):
        self.idx, self.qv, self.word = idx, qvecs, {}

    def word_scores(self, qid):
        if qid not in self.word:
            ids, w = self.qv[qid]
            m = ids < WORD_VOCAB
            self.word[qid] = np.asarray(self.idx.csc[:, ids[m]] @ w[m]).ravel().astype(np.float32)
        return self.word[qid]

    def run(self, qids, sel, topk=1000):
        run = {}
        for qid in qids:
            s = self.word_scores(qid).copy()
            ids, w = self.qv[qid]
            sq = dict(zip((ids[ids >= WORD_VOCAB] - WORD_VOCAB).tolist(), w[ids >= WORD_VOCAB].tolist()))
            ents = [e for e in dict.fromkeys(sel.get(qid, [])) if sq.get(e, 0) > 0]
            if ents:
                cols = np.array(ents) + WORD_VOCAB
                s += np.asarray(self.idx.csc[:, cols] @ np.array([sq[e] for e in ents], np.float32)).ravel()
            k = min(topk, len(s))
            top = np.argpartition(-s, k - 1)[:k]
            top = top[np.argsort(-s[top], kind="stable")]
            run[qid] = {self.idx.doc_ids[j]: float(s[j]) for j in top}
        return run


def rank_select(pool, scores, k):
    """pool: {qid: [ents]}, scores: {qid: array aligned with pool} -> top-k entities per query."""
    out = {}
    for q, ents in pool.items():
        o = np.argsort(-scores[q], kind="stable")[:k]
        out[q] = [ents[i] for i in o]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-n", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--index", default="index_linked")
    ap.add_argument("--tag", default="")
    ap.add_argument("--pool", default="", help="pool file suffix (_prf for E1 v2)")
    args = ap.parse_args()
    d = EXT / "e1"
    res_dir = d / f"results{args.tag}"; res_dir.mkdir(exist_ok=True)
    t0 = time.time()

    # dict(...) materialises every array once: indexing an NpzFile re-reads the whole array on EVERY access, and a
    # stored slice of it pins that full copy (this caused the 115 GB RSS / OOM kills, see extension1.md)
    tr = dict(np.load(d / f"train_feats{args.tag}.npz")); te = dict(np.load(d / f"test_feats{args.tag}.npz"))
    pools = {}
    for split in ("train", "test"):
        for l in open(d / f"pool_{split}{args.pool}.jsonl"):
            r = json.loads(l); pools[r["id"]] = [x["e"] for x in r["cands"]]
    qids_tr = tr["qid"]
    uq = np.array(list(dict.fromkeys(qids_tr.tolist())))
    rng = np.random.default_rng(0)
    dev_q = set(rng.choice(uq, args.dev_n, replace=False).tolist())
    fit_mask = np.array([q not in dev_q for q in qids_tr])
    mu, sd = tr["X"][fit_mask].mean(0), tr["X"][fit_mask].std(0) + 1e-6
    norm = lambda X: ((X - mu) / sd).astype(np.float32)

    # ---- selectors (identical architecture / data / epochs; only the label differs)
    Xf, qf = norm(tr["X"][fit_mask]), qids_tr[fit_mask]
    sel_scores = {}  # method -> {qid: scores aligned with pool}
    info = {}
    cache = res_dir / "selector_scores.pkl"
    if cache.exists():
        sel_scores, info = pickle.load(open(cache, "rb"))
        log(f"loaded cached selector scores {list(sel_scores)}")
    for lab in [l for l in LABELS if f"E1-{l}" not in sel_scores]:
        m, n_used = train_selector(Xf, tr[lab][fit_mask], group(qf), epochs=args.epochs)  # 19 or 23 features
        info[lab] = {"queries_with_positive_label": n_used}
        with torch.no_grad():
            for split, F in (("train", tr), ("test", te)):
                pr = m(torch.tensor(norm(F["X"]), device="cuda")).cpu().numpy()
                for a, b in group(F["qid"]):
                    sel_scores.setdefault(f"E1-{lab}", {})[F["qid"][a]] = pr[a:b].copy()
        log(f"selector {lab} trained on {n_used} queries ({time.time()-t0:.0f}s)")
        torch.save(m.state_dict(), res_dir / f"selector_{lab}.pt")
        pickle.dump((sel_scores, info), open(cache, "wb"))

    # ---- simple controls over the same pool
    fn = list(tr["feat_names"])
    for split, F in (("train", tr), ("test", te)):
        for a, b in group(F["qid"]):
            q, X = F["qid"][a], F["X"][a:b].copy()
            sel_scores.setdefault("C-dyvo-weight", {})[q] = X[:, fn.index("sq")]
            sel_scores.setdefault("C-laque-score", {})[q] = X[:, fn.index("laque")] - 1e3 * (1 - X[:, fn.index("in_laque")])
            sel_scores.setdefault("C-bm25-score", {})[q] = X[:, fn.index("bm25")] - 1e3 * (1 - X[:, fn.index("in_bm25")])
            sel_scores.setdefault("C-rarest-first", {})[q] = X[:, fn.index("idf")]  # DF-based ranking (rarest first)
            sel_scores.setdefault("C-random", {})[q] = np.random.default_rng(abs(hash(q)) % 2**32).random(b - a)
            sel_scores.setdefault("C-rel-first", {})[q] = X[:, fn.index("in_rel")] * 10 + X[:, fn.index("sq")]
            if "log_prf" in fn:  # E1 v2: rank the pool by the corpus-feedback score alone
                sel_scores.setdefault("C-prf-score", {})[q] = X[:, fn.index("log_prf")] - 1e3 * (1 - X[:, fn.index("in_prf")])

    # ---- query vectors: dev from e1_labels cache, test encoded with pool U released lists
    conf = compose_conf(); model = load_model(conf)
    idx = DocIndex(EXT / args.index, keep_csr=False)
    log(f"index loaded ({time.time()-t0:.0f}s)")
    released = {s: load_jsonl_entities(CODEC / f"queries_entities_{s}.jsonl") for s in ("gpt4", "mistral", "bm25", "laque")}
    released["rel"] = load_jsonl_entities(CODEC / "queries_test_train_entities_rel.jsonl")
    test_q, test_qrels = codec_test()
    test_ids = [q for q, _ in test_q]
    union_test = {q: list(dict.fromkeys(pools[q] + sum([released[s].get(q, []) for s in released], []))) for q in test_ids}
    qv_test = encode_queries(model, conf, test_q, union_test, batch_size=1)
    dev_ids = sorted(dev_q)
    train_text = dict(tuple(l.rstrip("\n").split("\t", 1)) for l in open(CODEC / "codec_mistral_topics.tsv"))
    qv_dev = encode_queries(model, conf, [(q, train_text[q]) for q in dev_ids], {q: pools[q] for q in dev_ids}, batch_size=1)
    dev_qrels = {q: {p: 1 for p in v} for q, v in json.load(open(CODEC / "codec_mistral_qrels.json")).items() if q in dev_q}
    sc_dev, sc_test = Scorer(idx, qv_dev), Scorer(idx, qv_test)
    json.dump({"mu": mu.tolist(), "sd": sd.tolist()}, open(res_dir / "feature_norm.json", "w"))
    log(f"queries encoded ({time.time()-t0:.0f}s)")

    # ---- dev sweep: every method x K
    dev_res = {}
    for meth, S in sel_scores.items():
        for k in KS:
            sel = rank_select({q: pools[q] for q in dev_ids}, S, k)
            agg, _ = evaluate(sc_dev.run(dev_ids, sel), dev_qrels)
            dev_res[f"{meth}|{k}"] = agg
        log(f"dev {meth}: " + " ".join(f"K{k}={dev_res[f'{meth}|{k}']['nDCG@10']:.4f}" for k in KS))
    agg, _ = evaluate(sc_dev.run(dev_ids, {}), dev_qrels); dev_res["word-only|0"] = agg
    rel_dev = {q: released["rel"].get(q, []) for q in dev_ids}
    agg, _ = evaluate(sc_dev.run(dev_ids, rel_dev), dev_qrels); dev_res["REL|native"] = agg
    agg, _ = evaluate(sc_dev.run(dev_ids, {q: pools[q] for q in dev_ids}), dev_qrels); dev_res["pool-all|all"] = agg
    json.dump(dev_res, open(res_dir / "dev.json", "w"), indent=1)

    # ---- choose K per method on dev; also report the fixed-K grid on test
    best_k = {m: max(KS, key=lambda k: (dev_res[f"{m}|{k}"]["nDCG@10"], -k)) for m in sel_scores}
    test_res, per_q = {}, {}

    def add(name, sel):
        agg, per = evaluate(sc_test.run(test_ids, sel, topk=10000), test_qrels)
        test_res[name] = agg; per_q[name] = per

    add("B0 word-only (DyVo word branch)", {})
    for s, lab in (("rel", "B1 REL"), ("gpt4", "B2 GPT-4 (released)"), ("mistral", "B2 Mixtral (released)"),
                   ("bm25", "B3 BM25 (released)"), ("laque", "B4 LaQue (released)")):
        add(lab, {q: released[s].get(q, []) for q in test_ids})
    add("C0 pool-all", {q: pools[q] for q in test_ids})
    for m, S in sel_scores.items():
        add(f"{m} K={best_k[m]} (dev)", rank_select({q: pools[q] for q in test_ids}, S, best_k[m]))
        for k in KS:
            sel = rank_select({q: pools[q] for q in test_ids}, S, k)
            agg, per = evaluate(sc_test.run(test_ids, sel, topk=10000), test_qrels)
            test_res[f"grid {m} K={k}"] = agg; per_q[f"grid {m} K={k}"] = per
            if m.startswith("E1"):
                with open(res_dir / f"queries_{m}_k{k}.jsonl", "w") as f:
                    for q in test_ids:
                        f.write(json.dumps({"id": q, "entities": sel[q]}) + "\n")

    # ---- statistics: predeclared family = E1-loo (dev K) vs B1 REL, teacher imitation, occurrence, best dev control
    ctrl = [m for m in sel_scores if m.startswith("C-")]
    best_ctrl = max(ctrl, key=lambda m: dev_res[f"{m}|{best_k[m]}"]["nDCG@10"])
    main_name = f"E1-loo K={best_k['E1-loo']} (dev)"
    comps = {"vs B1 REL": "B1 REL", "vs E1-imit": f"E1-imit K={best_k['E1-imit']} (dev)",
             "vs E1-occ": f"E1-occ K={best_k['E1-occ']} (dev)", "vs E1-pair": f"E1-pair K={best_k['E1-pair']} (dev)",
             f"vs best dev control ({best_ctrl})": f"{best_ctrl} K={best_k[best_ctrl]} (dev)",
             "vs B2 GPT-4": "B2 GPT-4 (released)"}
    stats = {c: paired_tests(per_q[b]["nDCG@10"], per_q[main_name]["nDCG@10"]) for c, b in comps.items()}
    adj = holm({c: v["p_rand"] for c, v in stats.items()})
    for c in stats:
        stats[c]["p_holm"] = adj[c]
    out = {"best_k": best_k, "best_dev_control": best_ctrl, "selector_info": info, "test": test_res, "stats_nDCG@10": stats,
           "n_dev": len(dev_ids), "n_fit_queries": int(len(set(qf.tolist())))}
    json.dump(out, open(res_dir / "test.json", "w"), indent=1)
    pickle.dump(per_q, open(res_dir / "test_per_query.pkl", "wb"))
    for k, v in test_res.items():
        if not k.startswith("grid"):
            log(f"TEST {k:40s} nDCG@10 {v['nDCG@10']:.4f}  nDCG@20 {v['nDCG@20']:.4f}  R@1000 {v['R@1000']:.4f}")
    log(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
