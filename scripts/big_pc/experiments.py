"""Experiment names grouped by run priority, and the paper's reference numbers (prompt Section 7 / paper Tables 1-3).
Imported by 02_check_data.py, 04_collect_results.py and run_priority.sh (via `python experiments.py --group`)."""
import sys

L1 = {"1e-5": "0.00001", "1e-4": "0.0001", "1e-3": "0.001"}
DATASETS = ["codec", "robust04", "wapo"]  # wapo = TREC Core 2018
DS_LABEL = {"codec": "CODEC", "robust04": "Robust04", "wapo": "Core18"}


def lsrw(ds, l1):
    return f"qmlp_dmlm_{ds}_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_{L1[l1]}"


def dyvo(ds, l1, emb="laque", source=None):
    name = f"qmlp_dmlm_emlm_{emb}_{ds}_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_{L1[l1]}_entw_0.05"
    return name + (f"_{source}" if source else "")


PRIORITY = {
    "priority1": [lsrw("codec", "1e-5"), dyvo("codec", "1e-5")],
    "priority2": [lsrw("robust04", "1e-5"), dyvo("robust04", "1e-5"), lsrw("wapo", "1e-5"), dyvo("wapo", "1e-5")],
    "priority3": [f(ds, l1) for l1 in ("1e-3", "1e-4") for ds in DATASETS for f in (lsrw, dyvo)],
    "priority4": [dyvo(ds, "1e-5", "laque", s) for ds in DATASETS
                  for s in (["gpt4", "mistral", "bm25", "laquecand"] + (["human"] if ds == "codec" else []))],
    "priority5": [dyvo(ds, "1e-5", emb, "gpt4") for ds in DATASETS for emb in ("bert", "dpr", "jds", "wiki2vec", "blink")],
}


def all_paper_experiments():
    return [n for g in PRIORITY.values() for n in g]


# (method label, L1) -> {dataset: (nDCG@10, nDCG@20, R@1k)}; "-" rows omitted. Source: paper Tables 1-3.
PAPER = {
    ("LSR-w", "1e-3"): {"robust04": (40.37, 37.23, 55.66), "wapo": (34.50, 31.45, 52.66), "codec": (39.10, 35.32, 57.58)},
    ("DyVo (REL)", "1e-3"): {"robust04": (41.52, 38.62, 56.78), "wapo": (37.50, 34.61, 54.14), "codec": (42.67, 38.32, 59.81)},
    ("LSR-w", "1e-4"): {"robust04": (47.69, 44.48, 64.47), "wapo": (38.94, 37.37, 60.44), "codec": (50.54, 46.71, 66.39)},
    ("DyVo (REL)", "1e-4"): {"robust04": (48.15, 44.85, 64.72), "wapo": (43.10, 39.46, 60.43), "codec": (51.66, 47.95, 68.49)},
    ("LSR-w", "1e-5"): {"robust04": (49.13, 46.34, 66.86), "wapo": (40.99, 38.73, 63.22), "codec": (52.61, 49.22, 69.07)},
    ("DyVo (REL)", "1e-5"): {"robust04": (51.19, 47.65, 68.56), "wapo": (43.72, 40.56, 63.56), "codec": (53.40, 51.15, 70.60)},
    ("DyVo (BM25)", "1e-5"): {"robust04": (51.38, 47.72, 67.74), "wapo": (42.48, 38.89, 64.58), "codec": (53.25, 49.80, 69.83)},
    ("DyVo (LaQue)", "1e-5"): {"robust04": (49.42, 46.31, 68.25), "wapo": (40.24, 38.39, 64.83), "codec": (53.73, 50.34, 70.87)},
    ("DyVo (Mixtral)", "1e-5"): {"robust04": (52.97, 49.21, 69.28), "wapo": (43.80, 41.86, 68.27), "codec": (54.90, 52.82, 73.20)},
    ("DyVo (GPT4)", "1e-5"): {"robust04": (54.39, 50.89, 70.86), "wapo": (43.06, 42.25, 68.57), "codec": (56.46, 53.72, 74.47)},
    ("DyVo (Human)", "1e-5"): {"codec": (56.42, 52.96, 75.13)},
    ("DyVo (GPT4) Token Aggr.", "1e-5"): {"robust04": (51.35, 48.01, 67.46), "wapo": (41.63, 39.37, 64.01), "codec": (53.44, 50.39, 69.94)},
    ("DyVo (GPT4) DPR", "1e-5"): {"robust04": (48.68, 45.77, 75.21), "wapo": (40.26, 37.52, 70.81), "codec": (53.04, 49.18, 75.19)},
    ("DyVo (GPT4) JDS", "1e-5"): {"robust04": (51.21, 48.38, 73.79), "wapo": (44.29, 41.86, 70.16), "codec": (55.08, 50.93, 73.97)},
    ("DyVo (GPT4) Wiki2Vec", "1e-5"): {"robust04": (54.04, 50.21, 69.85), "wapo": (44.15, 43.13, 67.77), "codec": (56.30, 53.25, 73.03)},
    ("DyVo (GPT4) BLINK", "1e-5"): {"robust04": (55.56, 51.71, 71.81), "wapo": (44.63, 42.94, 69.11), "codec": (58.15, 54.83, 74.72)},
}
SOURCE_LABEL = {None: "REL", "gpt4": "GPT4", "mistral": "Mixtral", "bm25": "BM25", "laquecand": "LaQue", "human": "Human"}
EMB_LABEL = {"bert": "Token Aggr.", "dpr": "DPR", "jds": "JDS", "wiki2vec": "Wiki2Vec", "blink": "BLINK"}


def describe(name):
    """experiment name -> (dataset, method label, l1) or None if it is not a paper row."""
    inv_l1 = {v: k for k, v in L1.items()}
    for ds in DATASETS:
        for l1 in L1:
            if name == lsrw(ds, l1):
                return ds, "LSR-w", l1
            for emb in ["laque", *EMB_LABEL]:
                for src in SOURCE_LABEL:
                    if name == dyvo(ds, l1, emb, src):
                        if emb == "laque":
                            return ds, f"DyVo ({SOURCE_LABEL[src]})", l1
                        if src == "gpt4":
                            return ds, f"DyVo (GPT4) {EMB_LABEL[emb]}", l1
    del inv_l1
    return None


if __name__ == "__main__":
    # `python experiments.py --group priority1` prints one experiment name per line (used by run_priority.sh)
    if len(sys.argv) == 3 and sys.argv[1] == "--group":
        print("\n".join(PRIORITY[sys.argv[2]]))
    else:
        for g, names in PRIORITY.items():
            print(g, len(names))
