"""Encode the CODEC corpus ONCE with the frozen DyVo full100k doc encoder and save the sparse document vectors.

Same model, collator, fp16 autocast, corpus order and doc batch (16) as the reproduction's fast_eval run, so the
saved vectors equal the ones behind the reported 53.70 nDCG@10. Query-side experiments (Extension 1) then re-score
the full corpus in seconds; document-side variants (Extension 2) are dumped with --doc-ent-path.

Output (<out>/): ids.npy int32, w.npy float16 (nonzero weights only), indptr.npy int64, doc_ids.json,
stats.json (n linked candidates, n positive entity postings per doc, timing). Resumable by shard.

  python scripts/ext/dump_index.py --out scratch/ext/index_linked
  python scripts/ext/dump_index.py --out scratch/ext/index_X --doc-ent-path scratch/ext/e2/docs_entities_X.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from common import REPO, compose_conf, load_model, log, to_dev, load_jsonl_entities, ListRetriever, make_collator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--doc-ent-path", default=None, help="jsonl {id, entities} replacing the linked doc entities")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard-docs", type=int, default=32000)
    ap.add_argument("--max-docs", type=int, default=-1)
    ap.add_argument("--only-listed", action="store_true",
                    help="encode only documents present in --doc-ent-path (delta dumps for Extension 2 v2)")
    args = ap.parse_args()
    out = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
    (out / "shards").mkdir(parents=True, exist_ok=True)

    conf = compose_conf()
    if args.doc_ent_path:
        p = Path(args.doc_ent_path)
        conf.entity_candidate_retriever.doc_ent_path = str(p if p.is_absolute() else REPO / p)
    log(f"doc entities: {conf.entity_candidate_retriever.doc_ent_path}")
    model = load_model(conf)
    from hydra.utils import instantiate
    test_dataset = instantiate(conf.test_dataset)
    retr = ListRetriever(doc_entities=load_jsonl_entities(conf.entity_candidate_retriever.doc_ent_path))
    docs = test_dataset.docs
    if args.only_listed:
        keep = [i for i in range(len(docs)) if docs.ids[i] in retr.doc_entities]
        doc_ids = [docs.ids[i] for i in keep]
        docs = Subset(docs, keep)
    n = len(docs) if args.max_docs < 0 else args.max_docs
    if not args.only_listed:
        doc_ids = [docs.ids[i] for i in range(n)]
    json.dump(doc_ids[:n], open(out / "doc_ids.json", "w"))
    collator = make_collator(conf, retr)
    log(f"{n} docs, {len(retr.doc_entities)} with entity lists")

    t0 = time.time()
    for s0 in range(0, n, args.shard_docs):
        f = out / "shards" / f"{s0:08d}.npz"
        if f.exists():
            continue
        s1 = min(n, s0 + args.shard_docs)
        dl = DataLoader(Subset(docs, range(s0, s1)), batch_size=args.batch, collate_fn=collator, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
        ids_l, w_l, lens, ncand, npos = [], [], [], [], []
        for b in dl:
            g = b["doc_groups"]
            nc = g["entity_masks"].sum(1).tolist()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                d_ids, d_ws, _ = model.encode_docs(**to_dev(g, "cuda"), to_dense=False)
            d_ids, d_ws = d_ids.int().cpu().numpy(), d_ws.float().cpu().numpy()
            for i in range(len(d_ids)):
                m = d_ws[i] > 0
                ids_l.append(d_ids[i][m]); w_l.append(d_ws[i][m].astype(np.float16)); lens.append(int(m.sum()))
                ncand.append(int(nc[i])); npos.append(int((d_ids[i][m] >= 30522).sum()))
        np.savez(f.with_suffix(".tmp.npz"), ids=np.concatenate(ids_l), w=np.concatenate(w_l), lens=np.array(lens),
                 ncand=np.array(ncand), npos=np.array(npos))
        f.with_suffix(".tmp.npz").rename(f)
        rate = (s1 - s0) / (time.time() - t0)
        t0 = time.time()
        log(f"shard {s0}-{s1} done ({rate:.0f} docs/s, eta {(n - s1) / rate / 3600:.2f} h)")

    log("merging shards")
    parts = [np.load(out / "shards" / f"{s0:08d}.npz") for s0 in range(0, n, args.shard_docs)]
    lens = np.concatenate([p["lens"] for p in parts])
    np.save(out / "ids.npy", np.concatenate([p["ids"] for p in parts]).astype(np.int32))
    np.save(out / "w.npy", np.concatenate([p["w"] for p in parts]).astype(np.float16))
    np.save(out / "indptr.npy", np.concatenate([[0], np.cumsum(lens)]).astype(np.int64))
    ncand = np.concatenate([p["ncand"] for p in parts]); npos = np.concatenate([p["npos"] for p in parts])
    np.save(out / "ent_counts.npy", np.stack([ncand, npos], 1))
    stats = {"n_docs": n, "postings_total": int(lens.sum()), "entity_postings": int(npos.sum()),
             "entity_candidates": int(ncand.sum()), "mean_entity_postings_per_doc": float(npos.mean()),
             "bytes_ids_plus_w": int(lens.sum()) * 6, "doc_ent_path": conf.entity_candidate_retriever.doc_ent_path}
    json.dump(stats, open(out / "stats.json", "w"), indent=1)
    log(json.dumps(stats))


if __name__ == "__main__":
    main()
