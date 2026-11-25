"""Extension 1: nested topic-disjoint choice of K (council report §11, "small-topic discipline").

The training-dev proxy (synthetic queries) chooses K = 1 for every learned method because those queries are
lexically matched to their positive. Here K is chosen on HALF of the 42 test topics and applied to the OTHER half
(2-fold), so no topic is ever scored with a K selected on itself. Repeated over 50 random splits; we report the mean
and spread, and paired tests on the per-query CV-K scores of split 0 (+ the fraction of splits with the same sign).
Input: results{tag}/test_per_query.pkl (grid runs). Output: results{tag}/cvk.json
"""
import argparse
import json
import pickle

import numpy as np

from common import EXT, paired_tests, holm, log

KS = [1, 3, 5, 10, 20]


def cv_scores(per, method, qids, rng):
    perm = rng.permutation(qids)
    folds = [perm[: len(perm) // 2], perm[len(perm) // 2:]]
    out, chosen = {}, []
    for a, b in ((0, 1), (1, 0)):
        best = max(KS, key=lambda k: (np.mean([per[f"grid {method} K={k}"]["nDCG@10"][q] for q in folds[a]]), -k))
        chosen.append(best)
        for q in folds[b]:
            out[q] = {m: per[f"grid {method} K={best}"][m][q] for m in ("nDCG@10", "nDCG@20", "R@1000")}
    return out, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--splits", type=int, default=50)
    args = ap.parse_args()
    d = EXT / "e1" / f"results{args.tag}"
    per = pickle.load(open(d / "test_per_query.pkl", "rb"))
    methods = sorted({k.split(" K=")[0][5:] for k in per if k.startswith("grid ")})
    qids = sorted(per["B1 REL"]["nDCG@10"])
    res, split0 = {}, {}
    for m in methods:
        vals, ks = [], []
        for s in range(args.splits):
            sc, ch = cv_scores(per, m, qids, np.random.default_rng(s))
            vals.append({x: np.mean([sc[q][x] for q in qids]) for x in ("nDCG@10", "nDCG@20", "R@1000")})
            ks += ch
            if s == 0:
                split0[m] = sc
        res[m] = {x: {"mean": float(np.mean([v[x] for v in vals])), "sd": float(np.std([v[x] for v in vals]))}
                  for x in ("nDCG@10", "nDCG@20", "R@1000")}
        res[m]["K_chosen_counts"] = {str(k): ks.count(k) for k in KS}
        log(f"CV-K {m:15s} nDCG@10 {res[m]['nDCG@10']['mean']:.4f}±{res[m]['nDCG@10']['sd']:.4f} "
            f"R@1000 {res[m]['R@1000']['mean']:.4f}  K chosen {res[m]['K_chosen_counts']}")
    # paired tests on split 0, E1-loo / E1-occ vs each other method and REL / GPT-4 (fixed baselines)
    tests = {}
    for main_m in ("E1-loo", "E1-occ", "E1-pair"):
        a = {q: split0[main_m][q]["nDCG@10"] for q in qids}
        fam = {}
        for m in methods:
            if m != main_m:
                fam[f"vs {m}"] = paired_tests({q: split0[m][q]["nDCG@10"] for q in qids}, a)
        for b in ("B1 REL", "B2 GPT-4 (released)", "B2 Mixtral (released)"):
            fam[f"vs {b}"] = paired_tests(per[b]["nDCG@10"], a)
        adj = holm({k: v["p_rand"] for k, v in fam.items()})
        for k in fam:
            fam[k]["p_holm"] = adj[k]
        # sign stability across splits vs REL
        diffs = []
        for s in range(args.splits):
            sc, _ = cv_scores(per, main_m, qids, np.random.default_rng(s))
            diffs.append(np.mean([sc[q]["nDCG@10"] - per["B1 REL"]["nDCG@10"][q] for q in qids]))
        fam["frac_splits_better_than_REL"] = float(np.mean(np.array(diffs) > 0))
        tests[main_m] = fam
    json.dump({"cv": res, "tests_split0": tests}, open(d / "cvk.json", "w"), indent=1)
    for mm, fam in tests.items():
        for k, v in fam.items():
            if isinstance(v, dict):
                log(f"{mm} {k:28s} diff {v['mean_diff']:+.4f} p {v['p_rand']:.3f} holm {v['p_holm']:.3f}")
            else:
                log(f"{mm} {k}: {v:.2f}")


if __name__ == "__main__":
    main()
