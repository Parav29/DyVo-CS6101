"""Extension 1: uncached per-query latency by stage (batch 1, 42 test queries, random order, 3 repeats).

Pipelines timed:
  REL replay   : DyVo query encode (REL ids) + full-corpus scoring          (REL linking time itself not available)
  E1           : BM25-title + LaQue pool + features + selector MLP + DyVo encode (top-K) + scoring
  GPT-4 replay : DyVo encode (released ids) + scoring; the live LLM call cannot be timed from cached files
Scoring = exact term-at-a-time over the dumped index with scipy on CPU (not an optimised inverted-index engine),
so absolute retrieval times are an upper bound; stage ratios are the useful output.
Output: scratch/ext/e1/results/latency.json
"""
import json
import time

import numpy as np
import torch

from common import EXT, CODEC, KB, DocIndex, compose_conf, load_model, encode_queries, codec_test, log, load_jsonl_entities
from e1_pool import TitleBM25, LaqueDense, DyVoDense
from e1_labels import featurize
from e1_run import MLP


def sync():
    torch.cuda.synchronize()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="_prf: E1 v2 pipeline (adds the word-only first pass + PRF)")
    args = ap.parse_args()
    prf = args.tag == "_prf"
    d = EXT / "e1"
    titles = json.load(open(KB / "entity_list.json"))
    allowed = np.load(d / "df_linked.npy") > 0
    bm25 = TitleBM25(titles, allowed)
    conf = compose_conf(); model = load_model(conf)
    dense = DyVoDense(model, conf, None)
    laque = LaqueDense(dense.qe.ent_emb_model.entity_emb, allowed)
    idx = DocIndex(EXT / "index_linked", keep_csr=True)
    df = idx.entity_df(); n_docs = len(idx.doc_ids)
    sel = MLP(23 if prf else 19).cuda().eval()
    test_q, _ = codec_test()
    rel = load_jsonl_entities(CODEC / "queries_test_train_entities_rel.jsonl")
    gpt = load_jsonl_entities(CODEC / "queries_entities_gpt4.jsonl")
    K = json.load(open(d / f"results{args.tag}" / "test.json"))["best_k"]["E1-loo"]
    # warm-up (model weights, CUDA kernels, OS file cache) -- results/LLM caches are NOT used
    for q, t in test_q[:3]:
        encode_queries(model, conf, [(q, t)], {q: rel.get(q, [])})
    T = {k: [] for k in ("bm25", "laque", "prf_first_pass", "dyvo_score_pool", "features_selector", "encode", "score",
                         "E1_total", "REL_total", "GPT4_total")}
    rng = np.random.default_rng(0)
    for rep in range(3):
        for i in rng.permutation(len(test_q)):
            q, text = test_q[i]
            # ---- E1
            t0 = time.perf_counter()
            b_ids, _ = bm25.search(text, 20)
            t1 = time.perf_counter()
            v = laque.embed(text); l_ids, _ = laque.search(v, 20); sync()
            t2 = time.perf_counter()
            r_ids = [int(e) for e in dict.fromkeys(rel.get(q, []))]
            pool = list(dict.fromkeys(r_ids + l_ids.tolist() + b_ids.tolist()))
            acc, prf_ids = {}, []
            if prf:  # word-only first pass over the full corpus + entity aggregation over its top-10 docs
                wv = encode_queries(model, conf, [(q, text)], {}); wi, ww = wv[q]
                sc = np.asarray(idx.csc[:, wi] @ ww).ravel()
                for j in np.argpartition(-sc, 10)[:10]:
                    for e, we in idx.doc_entity_weights(j).items():
                        acc[e] = acc.get(e, 0.0) + we * float(sc[j])
                prf_ids = sorted(acc, key=lambda e: (-acc[e], e))[:20]
                pool = list(dict.fromkeys(pool + prf_ids))
            t2b = time.perf_counter()
            h = dense.hidden(text); sq, raw = dense.score(h, pool); ls = laque.score(v, pool); bs = bm25.score(text, pool); sync()
            t3 = time.perf_counter()
            br = {int(e): r for r, e in enumerate(b_ids)}; lr = {int(e): r for r, e in enumerate(l_ids)}
            rec = {"text": text, "cands": [{"e": int(e), "sq": float(sq[j]), "raw": float(raw[j]), "bm25": float(bs[j]),
                                            "laque": float(ls[j]), "bm25_rank": br.get(int(e), -1),
                                            "laque_rank": lr.get(int(e), -1), "in_rel": int(e in r_ids)}
                                           for j, e in enumerate(pool)]}
            if prf:
                pr = {e: i for i, e in enumerate(prf_ids)}
                for x in rec["cands"]:
                    x["prf"] = float(acc.get(x["e"], 0.0)); x["prf_rank"] = pr.get(x["e"], -1)
            X = featurize(rec, df, titles, n_docs)
            with torch.no_grad():
                s = sel(torch.tensor(X, device="cuda")).cpu().numpy()
            chosen = [pool[j] for j in np.argsort(-s)[:K]]
            t4 = time.perf_counter()
            qv = encode_queries(model, conf, [(q, text)], {q: chosen}); sync()
            t5 = time.perf_counter()
            idx.score([q], qv, topk=1000)
            t6 = time.perf_counter()
            T["bm25"].append(t1 - t0); T["laque"].append(t2 - t1); T["prf_first_pass"].append(t2b - t2)
            T["dyvo_score_pool"].append(t3 - t2b)
            T["features_selector"].append(t4 - t3); T["encode"].append(t5 - t4); T["score"].append(t6 - t5)
            T["E1_total"].append(t6 - t0)
            # ---- replays
            for name, src in (("REL_total", rel), ("GPT4_total", gpt)):
                ta = time.perf_counter()
                qv = encode_queries(model, conf, [(q, text)], {q: src.get(q, [])}); sync()
                idx.score([q], qv, topk=1000)
                T[name].append(time.perf_counter() - ta)
    out = {k: {"p50_ms": 1000 * float(np.median(v)), "p95_ms": 1000 * float(np.percentile(v, 95)), "n": len(v)}
           for k, v in T.items()}
    out["K"] = K
    out["peak_gpu_mem_MB"] = torch.cuda.max_memory_allocated() / 2**20
    out["note"] = "scoring is scipy CSC on CPU; REL linking and live GPT-4 generation not timed (replay of cached lists)"
    json.dump(out, open(d / f"results{args.tag}" / "latency.json", "w"), indent=1)
    log(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
