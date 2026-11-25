"""Follow-ups F1-F3 (design: scratch/ext/followup_design.md, declared before running).

F1: base query list (REL / GPT-4 / Mixtral) + top-K E1-occ v2 entities.
F2: document score interpolation S_linked + alpha * (S_repaired - S_linked), repaired = E2 v2 grounded c=5.
F3: paired tests on nDCG@10 and R@1000 with Holm correction.
K and alpha are chosen by nested topic-disjoint CV (as e1_cvk.py); test qrels are never used to pick a setting
that is then scored on the same topics.
Output: scratch/ext/followup/{grid.json, per_query.pkl, results.json}
"""
import itertools
import json
import pickle

import numpy as np

from common import (EXT, CODEC, WORD_VOCAB, DocIndex, compose_conf, load_model, encode_queries, codec_test, evaluate,
                    paired_tests, holm, log, load_jsonl_entities)

KS = [0, 1, 3, 5, 10, 20]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
BASES = ["REL", "GPT-4", "Mixtral", "none"]
METRICS = ("nDCG@10", "nDCG@20", "R@1000")
SPLITS = 50


def topk_run(s, doc_ids, k=1000):
    top = np.argpartition(-s, k - 1)[:k]
    top = top[np.argsort(-s[top], kind="stable")]
    return {doc_ids[j]: float(s[j]) for j in top}


def main():
    out = EXT / "followup"; out.mkdir(exist_ok=True)
    test_q, qrels = codec_test()
    qids = [q for q, _ in test_q]
    base = {"REL": load_jsonl_entities(CODEC / "queries_test_train_entities_rel.jsonl"),
            "GPT-4": load_jsonl_entities(CODEC / "queries_entities_gpt4.jsonl"),
            "Mixtral": load_jsonl_entities(CODEC / "queries_entities_mistral.jsonl")}
    e1 = {k: load_jsonl_entities(EXT / "e1" / "results_prf" / f"queries_E1-occ_k{k}.jsonl") for k in KS if k}
    pool = {}
    for l in open(EXT / "e1" / "pool_test_prf.jsonl"):
        r = json.loads(l); pool[r["id"]] = [x["e"] for x in r["cands"]]
    union = {q: list(dict.fromkeys(pool[q] + sum((base[b].get(q, []) for b in base), []))) for q in qids}
    for q in qids:
        assert all(e in set(union[q]) for k in e1 for e in e1[k][q])

    conf = compose_conf(); model = load_model(conf)
    qv = encode_queries(model, conf, test_q, union, batch_size=1)
    del model
    lin = DocIndex(EXT / "index_linked", keep_csr=False)
    rep = DocIndex(EXT / "index_grounded_v2", keep_csr=False)
    assert lin.doc_ids == rep.doc_ids
    log("queries encoded, indexes loaded")

    configs = [(b, k, a) for b in BASES for k in KS for a in ALPHAS if not (b == "none" and k == 0)]
    runs = {c: {} for c in configs}
    for i, q in enumerate(qids):
        ids, w = qv[q]
        wm = ids < WORD_VOCAB
        word = [np.asarray(ix.csc[:, ids[wm]] @ w[wm]).ravel() for ix in (lin, rep)]
        sq = dict(zip((ids[~wm] - WORD_VOCAB).tolist(), w[~wm].tolist()))
        cols = {}
        for b, k, a in configs:
            ents = [e for e in dict.fromkeys((base[b].get(q, []) if b != "none" else []) + (e1[k][q] if k else []))
                    if sq.get(e, 0) > 0]
            for e in ents:
                if e not in cols:
                    cols[e] = [np.asarray(ix.csc[:, [e + WORD_VOCAB]].toarray()).ravel() * sq[e] for ix in (lin, rep)]
            s = [word[j] + (sum(cols[e][j] for e in ents) if ents else 0) for j in (0, 1)]
            runs[(b, k, a)][q] = topk_run((1 - a) * s[0] + a * s[1], lin.doc_ids)
        if i % 10 == 0:
            log(f"scored {i}/{len(qids)}")
    per, grid = {}, {}
    for c in configs:
        agg, pq = evaluate(runs[c], qrels)
        per[c], grid["|".join(map(str, c))] = pq, agg
    pickle.dump(per, open(out / "per_query.pkl", "wb"))
    json.dump(grid, open(out / "grid.json", "w"), indent=1)
    for c in [("REL", 0, 0.0), ("GPT-4", 0, 0.0), ("Mixtral", 0, 0.0), ("REL", 0, 1.0), ("none", 3, 0.0)]:
        log(f"sanity {c}: " + " ".join(f"{m} {grid['|'.join(map(str, c))][m]:.4f}" for m in METRICS))

    # ---------------- systems: allowed configs; CV picks one per fold
    def space(b, ks, als):
        return [(b, k, a) for k in ks for a in als]
    systems = {"DyVo baseline (REL, linked)": space("REL", [0], [0.0]),
               "E1 alone (E1-occ v2, CV-K)": space("none", KS[1:], [0.0]),
               "E2 alone (REL, grounded v2 c=5)": space("REL", [0], [1.0])}
    bases_fixed = {}
    for b in ("REL", "GPT-4", "Mixtral"):
        if b != "REL":
            systems[f"{b} (linked)"] = space(b, [0], [0.0])
        bases_fixed[b] = "DyVo baseline (REL, linked)" if b == "REL" else f"{b} (linked)"
        systems[f"F1 {b} + E1 (CV-K)"] = space(b, KS, [0.0])
        systems[f"F2 {b}, interpolated E2 (CV-alpha)"] = space(b, [0], ALPHAS)
        systems[f"F1+F2 {b} (CV-K x alpha)"] = space(b, KS, ALPHAS)

    def cv(cands, rng):
        perm = rng.permutation(qids); folds = [perm[:len(perm) // 2], perm[len(perm) // 2:]]
        sc, chosen = {}, []
        for a, b in ((0, 1), (1, 0)):
            best = max(cands, key=lambda c: (np.mean([per[c]["nDCG@10"][q] for q in folds[a]]), -c[1], -c[2]))
            chosen.append(best)
            for q in folds[b]:
                sc[q] = {m: per[best][m][q] for m in METRICS}
        return sc, chosen

    res, split0, allsplits = {}, {}, {}
    for name, cands in systems.items():
        vals, ch, sp = [], [], []
        for s in range(SPLITS):
            sc, c = cv(cands, np.random.default_rng(s))
            sp.append(sc); ch += c
            vals.append({m: np.mean([sc[q][m] for q in qids]) for m in METRICS})
        split0[name], allsplits[name] = sp[0], sp
        res[name] = {m: {"mean": float(np.mean([v[m] for v in vals])), "sd": float(np.std([v[m] for v in vals]))}
                     for m in METRICS}
        res[name]["chosen"] = {"|".join(map(str, c)): ch.count(c) for c in sorted(set(ch))}
        log(f"{name:42s} nDCG@10 {res[name]['nDCG@10']['mean']:.4f}±{res[name]['nDCG@10']['sd']:.4f} "
            f"R@1000 {res[name]['R@1000']['mean']:.4f}")

    def family(pairs):
        fam = {}
        for metric in ("nDCG@10", "R@1000"):
            t = {}
            for sysn, ref in pairs:
                r = paired_tests({q: split0[ref][q][metric] for q in qids}, {q: split0[sysn][q][metric] for q in qids})
                diffs = [np.mean([allsplits[sysn][s][q][metric] - allsplits[ref][s][q][metric] for q in qids])
                         for s in range(SPLITS)]
                r["frac_splits_better"] = float(np.mean(np.array(diffs) > 0))
                t[f"{sysn} vs {ref}"] = r
            adj = holm({k: v["p_rand"] for k, v in t.items()})
            for k in t:
                t[k]["p_holm"] = adj[k]
            fam[metric] = t
        return fam

    base0 = "DyVo baseline (REL, linked)"
    primary = family([(s, base0) for s in ["E1 alone (E1-occ v2, CV-K)", "E2 alone (REL, grounded v2 c=5)",
                                           "F1 REL + E1 (CV-K)", "F2 REL, interpolated E2 (CV-alpha)",
                                           "F1+F2 REL (CV-K x alpha)"]])
    secondary = family([(f"{p} {b}{suf}", bases_fixed[b]) for b in ("GPT-4", "Mixtral")
                        for p, suf in (("F1", " + E1 (CV-K)"), ("F2", ", interpolated E2 (CV-alpha)"),
                                       ("F1+F2", " (CV-K x alpha)"))])
    vs_dyvo = family([(s, base0) for s in systems if s != base0])  # descriptive, not a confirmatory family
    json.dump({"systems": res, "primary": primary, "secondary": secondary, "all_vs_dyvo_descriptive": vs_dyvo},
              open(out / "results.json", "w"), indent=1)
    for fname, fam in (("PRIMARY", primary), ("SECONDARY", secondary)):
        for metric, t in fam.items():
            for k, v in t.items():
                log(f"{fname} {metric} {k:75s} {v['mean_diff']*100:+.2f} p {v['p_rand']:.3f} holm {v['p_holm']:.3f} "
                    f"splits+ {v['frac_splits_better']:.2f}")


if __name__ == "__main__":
    main()
