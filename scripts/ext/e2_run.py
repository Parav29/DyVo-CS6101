"""Extension 2, step 5: evaluate document variants and the query x document design on CODEC test.

Exactness: a document's entity weight s_e(d) depends only on its own text and on e (per-entity logits,
max-pooled over tokens), so ONE dump with the union candidate list gives every variant's postings exactly:
  variant index = word postings (identical to the baseline dump) + linked-entity postings + selected additions.
Check: the linked-only entity postings taken from the union dump must equal the baseline dump's.

Memory-safe: word scores come from the memory-mapped baseline CSC; only the ~22M entity postings of the union dump
are materialised. Budget curve: per-document cap c in {1,2,3,5} (a variant's top-c additions are a prefix of top-5).
Output: scratch/ext/e2/results/test.json (+ per-query pickle, blinded review sample)
"""
import argparse
import json
import pickle
import time

import numpy as np
import scipy.sparse as sp

from common import (EXT, CODEC, KB, WORD_VOCAB, N_ENT, DocIndex, compose_conf, load_model, encode_queries, codec_test,
                    evaluate, paired_tests, holm, log, load_jsonl_entities)


def entity_postings(path):
    """(row, entity, weight) for every positive entity posting of a dump."""
    ids = np.load(path / "ids.npy", mmap_mode="r"); w = np.load(path / "w.npy", mmap_mode="r")
    ip = np.load(path / "indptr.npy")
    rows, ents, ws = [], [], []
    step = 50_000
    for a in range(0, len(ip) - 1, step):
        b = min(len(ip) - 1, a + step)
        i0, i1 = ip[a], ip[b]
        blk = np.asarray(ids[i0:i1]); m = blk >= WORD_VOCAB
        r = np.repeat(np.arange(a, b), np.diff(ip[a:b + 1]))
        rows.append(r[m]); ents.append(blk[m] - WORD_VOCAB); ws.append(np.asarray(w[i0:i1])[m].astype(np.float32))
    return np.concatenate(rows), np.concatenate(ents).astype(np.int64), np.concatenate(ws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", nargs="*", default=[], help="name=path of E1 candidate files for extra query sides")
    ap.add_argument("--caps", default="1,2,3,5")
    ap.add_argument("--v2", action="store_true", help="exploratory v2 variants (+ delta dump of new pairs)")
    args = ap.parse_args()
    sfx = "_v2" if args.v2 else ""
    d = EXT / "e2"; res = d / f"results{sfx}"; res.mkdir(exist_ok=True)
    t0 = time.time()
    base = DocIndex(EXT / "index_linked", keep_csr=False)
    order = base.doc_ids; N = len(order)
    br, be, bw = entity_postings(EXT / "index_linked")
    ur, ue, uw = entity_postings(EXT / "index_union")
    if args.v2:  # new (doc, entity) pairs encoded separately; rows re-mapped to the full-corpus order
        rowmap0 = {x: i for i, x in enumerate(order)}
        dorder = json.load(open(EXT / "index_delta_v2" / "doc_ids.json"))
        dr, de, dw = entity_postings(EXT / "index_delta_v2")
        dr = np.array([rowmap0[dorder[r]] for r in dr], dtype=ur.dtype)
        ur, ue, uw = np.concatenate([ur, dr]), np.concatenate([ue, de]), np.concatenate([uw, dw])
    log(f"entity postings: baseline {len(br)}, union {len(ur)} ({time.time()-t0:.0f}s)")
    ukey = ur.astype(np.int64) * N_ENT + ue
    bkey = br.astype(np.int64) * N_ENT + be

    # ---- linked postings from the union dump must equal the baseline dump
    linked = load_jsonl_entities(CODEC / "docs_entities.jsonl")
    lkey = np.unique(np.array([r * N_ENT + int(e) for r, x in enumerate(order) for e in linked.get(x, [])], np.int64))
    in_l = np.isin(ukey, lkey)
    o1, o2 = np.argsort(ukey[in_l]), np.argsort(bkey)
    same_keys = bool(np.array_equal(ukey[in_l][o1], bkey[o2]))
    max_diff = float(np.abs(uw[in_l][o1] - bw[o2]).max()) if same_keys else None
    check = {"linked_postings_equal_keys": same_keys, "max_abs_weight_diff": max_diff,
             "baseline_entity_postings": int(len(bkey)), "union_entity_postings": int(len(ukey))}
    log(f"union-vs-baseline check: {check}")

    # ---- variant entity matrices (and budget curve)
    V = dict(np.load(d / f"variants{sfx}.npz"))
    caps = [int(c) for c in args.caps.split(",")]
    mats, storage, unsupported = {}, {}, {}
    ment = {}
    for l in open(d / f"mentions{sfx}.jsonl"):
        r = json.loads(l); ment[r["id"]] = {m[0] for m in r["m"]}
    ent_mat = lambda m: sp.csc_matrix((uw[m], (ur[m], ue[m])), shape=(N, N_ENT))
    mats["linked"] = ent_mat(in_l)
    storage["linked"] = {"entity_postings": int(in_l.sum()), "added_candidates": 0, "added_positive_postings": 0}
    for v, sel in V.items():
        for c in caps:
            s = sel[sel[:, 2] < c]
            if c == max(caps):
                s = sel
            vkey = s[:, 0] * N_ENT + s[:, 1]
            m_add = np.isin(ukey, vkey) & ~in_l
            name = f"{v}@c{c}"
            mats[name] = ent_mat(in_l | m_add)
            storage[name] = {"entity_postings": int((in_l | m_add).sum()), "added_candidates": int(len(s)),
                             "added_positive_postings": int(m_add.sum()),
                             "index_bytes_entity_part": int((in_l | m_add).sum()) * 6}
        unsupported[v] = float(np.mean([int(e) not in ment.get(order[r], ()) for r, e, _ in sel[:200000]]))
    del ukey
    log(f"variants built ({time.time()-t0:.0f}s): {json.dumps(storage)}")
    log(f"unsupported-addition rate (no explicit mention in visible text): {unsupported}")

    # ---- query sides
    test_q, qrels = codec_test()
    qids = [q for q, _ in test_q]
    sides = {"REL": load_jsonl_entities(CODEC / "queries_test_train_entities_rel.jsonl"),
             "GPT-4": load_jsonl_entities(CODEC / "queries_entities_gpt4.jsonl"),
             "Mixtral": load_jsonl_entities(CODEC / "queries_entities_mistral.jsonl")}
    for spec in args.e1:
        name, path = spec.split("=", 1)
        sides[name] = load_jsonl_entities(path)
    union = {q: list(dict.fromkeys(sum([s.get(q, []) for s in sides.values()], []))) for q in qids}
    conf = compose_conf(); model = load_model(conf)
    qv = encode_queries(model, conf, test_q, union, batch_size=1)
    word = {}
    for q in qids:
        ids, w = qv[q]; m = ids < WORD_VOCAB
        word[q] = np.asarray(base.csc[:, ids[m]] @ w[m]).ravel()

    results, per_q, lat = {}, {}, {}
    for v, C in mats.items():
        for sname, side in sides.items():
            run, tt = {}, []
            for q in qids:
                ta = time.perf_counter()
                ids, w = qv[q]
                sq = dict(zip((ids[ids >= WORD_VOCAB] - WORD_VOCAB).tolist(), w[ids >= WORD_VOCAB].tolist()))
                ents = [e for e in dict.fromkeys(side.get(q, [])) if sq.get(e, 0) > 0]
                s = word[q].copy()
                if ents:
                    s += np.asarray(C[:, ents] @ np.array([sq[e] for e in ents], np.float32)).ravel()
                top = np.argpartition(-s, 9999)[:10000]
                top = top[np.argsort(-s[top], kind="stable")]
                tt.append(time.perf_counter() - ta)
                run[q] = {order[j]: float(s[j]) for j in top}
            agg, per = evaluate(run, qrels)
            name = f"{sname} x {v}"
            results[name] = agg; per_q[name] = per
            lat[name] = {"entity_score_ms_p50": 1000 * float(np.median(tt)), "p95": 1000 * float(np.percentile(tt, 95))}
            log(f"TEST {name:28s} nDCG@10 {agg['nDCG@10']:.4f} nDCG@20 {agg['nDCG@20']:.4f} R@1000 {agg['R@1000']:.4f}")

    # ---- mechanism diagnostics (test qrels used for ANALYSIS only; nothing is tuned on them)
    rowmap = {x: i for i, x in enumerate(order)}
    diag = {}
    for v, C in mats.items():
        Cr = C.tocsr()
        for sname, side in sides.items():
            acc = {"rel": [0, 0], "non": [0, 0]}
            for q in qids:
                ents = list(dict.fromkeys(side.get(q, [])))
                if not ents:
                    continue
                for x, r in qrels.get(q, {}).items():
                    if x in rowmap:
                        k = "rel" if r > 0 else "non"
                        acc[k][0] += int((Cr[rowmap[x], ents] > 0).sum()); acc[k][1] += 1
            diag[f"{sname} x {v}"] = {"entity_matches_per_relevant_doc": acc["rel"][0] / max(acc["rel"][1], 1),
                                      "entity_matches_per_nonrelevant_doc": acc["non"][0] / max(acc["non"][1], 1)}
    # ---- statistics: grounded vs each document control at the full budget, per query side (Holm within side)
    cmax = max(caps)
    stats = {}
    for sname in sides:
        g = f"{sname} x grounded@c{cmax}"
        fam = {f"grounded vs {c}": paired_tests(per_q[f"{sname} x {c}"]["nDCG@10"], per_q[g]["nDCG@10"])
               for c in ("linked", f"alias@c{cmax}", f"unfiltered@c{cmax}")}
        adj = holm({k: x["p_rand"] for k, x in fam.items()})
        for k in fam:
            fam[k]["p_holm"] = adj[k]
        stats[sname] = fam
    # ---- blinded review sample: 40 random additions per variant, variant label hidden in the main file
    titles = json.load(open(KB / "entity_list.json"))
    import ir_datasets
    rng = np.random.default_rng(0)
    picks = []
    for v, sel in V.items():
        for i in rng.choice(len(sel), 40, replace=False):
            picks.append((v, int(sel[i, 0]), int(sel[i, 1])))
    rng.shuffle(picks)
    want = {order[r] for _, r, _ in picks}
    texts = {}
    for doc in ir_datasets.load("codec").docs_iter():
        if doc.doc_id in want:
            texts[doc.doc_id] = (doc.title or "") + " " + (doc.text or "")[:1500]
    with open(res / "review_sample_blinded.jsonl", "w") as f, open(res / "review_sample_key.jsonl", "w") as k:
        for i, (v, r, e) in enumerate(picks):
            f.write(json.dumps({"item": i, "entity": titles[e], "doc_start": texts.get(order[r], "")}) + "\n")
            k.write(json.dumps({"item": i, "variant": v, "doc": order[r], "entity_idx": e}) + "\n")
    json.dump({"check": check, "storage": storage, "unsupported_addition_rate": unsupported, "test": results,
               "diagnostics": diag, "latency": lat, "stats_nDCG@10": stats}, open(res / "test.json", "w"), indent=1)
    pickle.dump(per_q, open(res / "test_per_query.pkl", "wb"))
    log(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
