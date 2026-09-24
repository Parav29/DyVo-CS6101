"""Big-PC step 4: collect test metrics for every finished experiment and compare with the paper.

For each DyVo/outputs/<experiment>/<date>/ with test_run.trec: recompute nDCG@10, nDCG@20, R@1000 with ir_measures
against the qrels of the experiment's test_dataset (--recompute, needs ir_datasets access), otherwise read
test_result.json written by HFTrainer.predict. Writes results/results.csv with the paper's numbers and the difference.

Usage (from the project root):
  python scripts/big_pc/04_collect_results.py [--recompute] [--dry-run]
"""
import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "DyVo"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import DS_LABEL, PAPER, all_paper_experiments, describe  # noqa: E402

METRICS = ("nDCG@10", "nDCG@20", "R@1000")


def read_result(run_dir):
    p = Path(run_dir) / "test_result.json"
    if not p.exists():
        return None
    m = json.load(open(p))
    return tuple(100 * m[f"test_{k}"] for k in METRICS)


def recompute(name, run_path):
    import ir_measures
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    sys.path.insert(0, str(REPO))
    from lsr.utils.dataset_utils import read_qrels
    GlobalHydra.instance().clear()
    os.chdir(REPO)
    with initialize_config_dir(config_dir=str(REPO / "lsr/configs"), version_base="1.2"):
        cfg = compose(config_name="config", overrides=[f"+experiment={name}"])
    qrels = read_qrels(cfg.test_dataset.qrels_path)
    run = {}
    for line in open(run_path):
        qid, _, did, _, score, _ = line.split()
        run.setdefault(qid, {})[did] = float(score)
    measures = [ir_measures.parse_measure(m) for m in METRICS]
    agg = ir_measures.calc_aggregate(measures, qrels, run)
    return tuple(100 * agg[m] for m in measures)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--output", default=str(ROOT / "results" / "results.csv"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    names = all_paper_experiments()
    if args.dry_run:
        print(f"[dry-run] would scan DyVo/outputs/<exp>/*/ for {len(names)} paper experiments and write {args.output}")
        for n in names[:5]:
            ds, method, l1 = describe(n)
            print(f"   {DS_LABEL[ds]:8s} {method:26s} L1={l1}  paper={PAPER[(method, l1)].get(ds)}  <- {n}")
        print("   ...")
        return 0
    rows = []
    for n in names:
        ds, method, l1 = describe(n)
        paper = PAPER.get((method, l1), {}).get(ds)
        dirs = sorted(glob.glob(str(REPO / "outputs" / n / "*" / "test_run.trec")))
        ours, src = None, ""
        if dirs:
            run_dir = Path(dirs[-1]).parent
            ours, src = (recompute(n, dirs[-1]), "recomputed") if args.recompute else (read_result(run_dir), "test_result.json")
        row = {"dataset": DS_LABEL[ds], "method": method, "l1": l1, "experiment": n, "status": "done" if ours else "not run",
               "source": src}
        for i, m in enumerate(METRICS):
            row[f"ours_{m}"] = f"{ours[i]:.2f}" if ours else ""
            row[f"paper_{m}"] = f"{paper[i]:.2f}" if paper else ""
            row[f"diff_{m}"] = f"{ours[i] - paper[i]:+.2f}" if ours and paper else ""
        rows.append(row)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    done = [r for r in rows if r["status"] == "done"]
    print(f"{len(done)}/{len(rows)} experiments finished -> {args.output}")
    for r in done:
        print(f"  {r['dataset']:8s} {r['method']:26s} L1={r['l1']}  nDCG@10 {r['ours_nDCG@10']} "
              f"(paper {r['paper_nDCG@10']}, {r['diff_nDCG@10']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
