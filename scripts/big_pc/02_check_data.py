"""Big-PC step 2: verify that everything the chosen experiment configs need is present and consistent.

Checks (per experiment, composed with Hydra's compose API, nothing is trained):
  1. every local path the config reads exists (entity table, candidate files, init checkpoints, train TSV/JSON)
  2. RAM vs entity-table size: BertEntityEmbedding torch.load()s the WHOLE table into CPU RAM (LaQue/DPR/JDS/
     bert_agg 16.2 GB, BLINK 21.6 GB, Wiki2Vec 6.3 GB); warns if RAM is too small and points to an mmap load
  3. ir_datasets corpora are visible (Robust04 / Core 2018 need licensed data placed manually; instructions printed)
  4. ID alignment: max entity id in every candidate file < #rows of the embedding table; spot-check that
     "United States" is id 0 in knowledge_base/entity_list.json
  5. teacher scores (*_monot5_3b_scores.json) are non-negative (normalize="std" + KLDivLoss gives NaN otherwise; E1)
  6. robust04/queries_entities_gpt4.jsonl id-space sanity (NOTES S5.0 finding 5) + dyvo_data/robust04/docs.tsv note
  7. the configured entity_dim equals the table width (catches the upstream Wiki2Vec bug)

Usage (from the project root):
  python scripts/big_pc/02_check_data.py --dry-run
  python scripts/big_pc/02_check_data.py --experiments priority1      # CODEC LSR-w + DyVo(REL, LaQue)
  python scripts/big_pc/02_check_data.py --experiments <name>,<name>  # or 'all'
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "DyVo"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import PRIORITY, all_paper_experiments  # noqa: E402

TABLE_BYTES = {"laque": 16212.73e6, "dpr": 16212.73e6, "jds": 16212.73e6, "bert_agg": 16212.73e6,
               "blink": 21616.97e6, "wiki2vec": 6333.10e6}
LICENSED = {
    "disks45/nocr/trec-robust-2004": "TREC Disks 4&5 (minus CR): obtain from NIST (https://trec.nist.gov/data/cd45/), "
                                     "then follow https://ir-datasets.com/disks45.html to place the files under "
                                     "~/.ir_datasets/disks45/corpus",
    "wapo/v2/trec-core-2018": "Washington Post v2: request from NIST (https://trec.nist.gov/data/wapost/), then follow "
                              "https://ir-datasets.com/wapo.html (~/.ir_datasets/wapo/WashingtonPost.v2.tar.gz)",
    "codec": "CODEC documents: follow https://ir-datasets.com/codec.html (the corpus is obtained from the CODEC authors)",
}


class Report:
    def __init__(self):
        self.errors, self.warnings = [], []

    def ok(self, msg):
        print(f"  ok    {msg}")

    def warn(self, msg):
        self.warnings.append(msg)
        print(f"  WARN  {msg}")

    def err(self, msg):
        self.errors.append(msg)
        print(f"  FAIL  {msg}")


def compose(name):
    from hydra import compose as hcompose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "lsr/configs"), version_base="1.2"):
        return hcompose(config_name="config", overrides=[f"+experiment={name}"])


def config_paths(cfg):
    """(local paths, irds names, embedding path, candidate files, teacher-score file, entity dims)."""
    local, irds = [], set()

    def add(p):
        if p is None:
            return
        p = str(p)
        if p.startswith("irds:"):
            irds.add(p[5:])
        elif p.startswith("hfds:"):
            return
        else:
            local.append(p)

    tr = cfg.train_dataset
    for k in ("collection_path", "queries_path", "qrels_path", "ce_score_dict"):
        add(tr.get(k))
    for ds in (cfg.get("test_dataset"), cfg.get("eval_dataset")):
        if ds is not None:
            for k in ("docs_path", "queries_path", "qrels_path"):
                add(ds.get(k))
    for k in ("query_encoder_checkpoint", "doc_encoder_checkpoint"):
        add(cfg.model.get(k))
    emb, cands, dims = None, [], {}
    if "entity_embedding_model" in cfg:
        emb = cfg.entity_embedding_model.entity_emb_path
        add(emb)
        cands = [cfg.entity_candidate_retriever.query_ent_path, cfg.entity_candidate_retriever.doc_ent_path]
        for c in cands:
            add(c)
        dims = {s: cfg.model[s].config.get("entity_dim") for s in ("query_encoder", "doc_encoder")}
    return local, irds, emb, cands, tr.get("ce_score_dict"), dims


def total_ram():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def table_rows(path):
    import torch
    try:
        t = torch.load(path, map_location="cpu", mmap=True)
    except Exception:  # older serialization format: cannot mmap, report size-based estimate instead
        return None
    return tuple(t.shape)


def max_entity_id(path):
    mx, n = -1, 0
    with open(path) as f:
        for line in f:
            ents = json.loads(line)["entities"]
            n += 1
            if ents:
                mx = max(mx, max(int(e) for e in ents))
    return mx, n


def check_irds(name, rep):
    try:
        import ir_datasets
    except ImportError:
        rep.err("ir_datasets not installed")
        return
    base = name.split("/")[0]
    home = Path(os.environ.get("IR_DATASETS_HOME", Path.home() / ".ir_datasets"))
    root = home / base
    if root.exists() and any(root.iterdir()):
        rep.ok(f"ir_datasets:{name} has local data under {root}")
    else:
        key = next((k for k in LICENSED if name.startswith(k.split("/")[0])), None)
        rep.warn(f"ir_datasets:{name}: nothing under {root} yet. " + (LICENSED.get(key, "") if key else
                 "It may download automatically on first use."))


def check_experiment(name, rep, cache):
    print(f"\n== {name}")
    os.chdir(REPO)
    cfg = compose(name)
    local, irds, emb, cands, ce, dims = config_paths(cfg)
    for p in local:
        (rep.ok if Path(p).exists() else rep.err)(f"{'exists' if Path(p).exists() else 'MISSING'}: {p}")
    for n in sorted(irds):
        if n not in cache["irds"]:
            check_irds(n, rep)
            cache["irds"].add(n)
    if ce and not str(ce).startswith(("irds:", "hfds:")) and Path(ce).exists() and ce not in cache["ce"]:
        cache["ce"].add(ce)
        scores = json.load(open(ce))
        neg = sum(1 for q in scores.values() for s in q.values() if s < 0)
        norm = cfg.loss.get("normalize", "softmax")
        if neg and norm != "softmax":
            rep.err(f"{ce}: {neg} negative teacher scores with loss.normalize={norm!r} -> NaN KL (NOTES E1). "
                    "Use loss.normalize=softmax or transform the scores.")
        else:
            rep.ok(f"{ce}: teacher scores ok for normalize={norm!r} ({neg} negative)")
    if emb is None:
        return
    kind = Path(emb).stem.replace("entity_embs_", "")
    width = {"blink": 1024, "wiki2vec": 300}.get(kind, 768)
    for side, d in dims.items():
        (rep.ok if d == width else rep.err)(f"{side}.entity_dim={d} (table {kind} is {width}-d)")
    size = Path(emb).stat().st_size if Path(emb).exists() else TABLE_BYTES.get(kind)
    ram = total_ram()
    if size and ram:
        msg = f"entity table {kind}: {size / 1e9:.1f} GB loaded fully into CPU RAM; machine RAM {ram / 1e9:.0f} GB"
        if size * 1.5 > ram:
            rep.warn(msg + " -> too tight. Option: in lsr/models/entity_emb.py use torch.load(..., mmap=True) "
                           "(needs a file saved with torch>=2.1 zip format), or add RAM/swap.")
        else:
            rep.ok(msg)
    if Path(emb).exists():
        if emb not in cache["rows"]:
            shape = table_rows(emb)
            cache["rows"][emb] = shape
            if shape:
                rep.ok(f"{emb}: shape {shape}")
                if shape[1] != width:
                    rep.err(f"{emb}: width {shape[1]} != expected {width}")
        n_rows = cache["rows"][emb][0] if cache["rows"][emb] else int(size / (4 * width))
        for c in cands:
            if Path(c).exists():
                mx, n = max_entity_id(c)
                (rep.ok if mx < n_rows else rep.err)(f"{c}: {n} lines, max entity id {mx} < {n_rows} rows")


def check_globals(rep):
    print("\n== global data checks")
    kb = REPO / "dyvo_data/knowledge_base/entity_list.json"
    if kb.exists():
        sys.path.insert(0, str(REPO))
        from lsr.entity.generate_entities_llm import load_entity_vocab
        try:
            vocab = load_entity_vocab(kb)
            ids = {n: vocab.get(n) for n in ("United States", "Non-fungible token", "Bitcoin")}
            rep.ok(f"entity_list.json: {len(vocab)} names; spot check {ids}")
            if ids["United States"] != 0:
                rep.warn("'United States' is not id 0 -> check that entity_list.json indexes the same rows as the tables")
        except ValueError as e:
            rep.err(f"entity_list.json has an unexpected shape: {e} -> adapt load_entity_vocab()")
    else:
        rep.warn(f"{kb} missing (needed for --backend remap and name spot checks)")
    g4 = REPO / "dyvo_data/robust04/queries_entities_gpt4.jsonl"
    rel = REPO / "dyvo_data/robust04/queries_test_train_entities_rel.jsonl"
    if g4.exists() and rel.exists():
        relm = {str(json.loads(l)["id"]): set(json.loads(l)["entities"]) for l in open(rel)}
        rows = [json.loads(l) for l in open(g4)]
        overlap = sum(len(set(r["entities"]) & relm.get(str(r["id"]), set())) for r in rows)
        dup = sum(1 for r in rows if len(r["entities"]) != len(set(r["entities"])))
        if overlap == 0 and dup == len(rows):
            rep.warn("robust04/queries_entities_gpt4.jsonl: 0 ids shared with REL and every row has duplicates -> "
                     "likely a different id space (NOTES S5.0 #5). run_priority.sh re-maps it from entity_names.")
        else:
            rep.ok(f"robust04 GPT-4 file: {overlap} ids shared with REL, {dup} rows with duplicates")
    docs_tsv = REPO / "dyvo_data/robust04/docs.tsv"
    if docs_tsv.exists():
        rep.ok(f"{docs_tsv} present")
    else:
        print(f"  note  {docs_tsv} is not in lsr42/dyvo_data and NOT needed by the experiment configs (they read "
              "irds:disks45/nocr/trec-robust-2004). Only if you switch robust04_inparsv2_monot53b.yaml to the local TSV: "
              "build it from the licensed corpus with lsr/preprocess/export_robust04_docs.py "
              "(format: doc_id<TAB>title + ' ' + body, one line per doc, no tabs/newlines inside).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", default="priority1",
                    help="comma list of experiment names, a priority group (priority1..priority5), or 'all'")
    ap.add_argument("--dry-run", action="store_true", help="list the experiments and checks; read no data")
    args = ap.parse_args()
    if args.experiments == "all":
        names = all_paper_experiments()
    elif args.experiments in PRIORITY:
        names = PRIORITY[args.experiments]
    else:
        names = [n for n in args.experiments.split(",") if n]
    missing = [n for n in names if not (REPO / f"lsr/configs/experiment/{n}.yaml").exists()]
    if missing:
        print("unknown experiments:", missing)
        return 1
    if args.dry_run:
        print(f"[dry-run] would check {len(names)} experiment(s):")
        for n in names:
            print("   ", n)
        print("[dry-run] checks: paths exist, RAM vs entity table, ir_datasets corpora, entity-id alignment, "
              "non-negative teacher scores, entity_dim vs table width, robust04 GPT-4 id space, robust04 docs.tsv note")
        return 0
    rep = Report()
    cache = {"irds": set(), "ce": set(), "rows": {}}
    for n in names:
        check_experiment(n, rep, cache)
    check_globals(rep)
    print(f"\nSUMMARY: {len(rep.errors)} errors, {len(rep.warnings)} warnings")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
