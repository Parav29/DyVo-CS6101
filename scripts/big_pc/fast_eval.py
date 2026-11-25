"""DYVO-REPRO fast test-time evaluation (same model, data, collator and scoring rule as HFTrainer.predict).

Why: the official evaluation_loop_iterative_vectorized scores each doc batch with a Python loop over queries plus
heap updates, and predict() also writes every doc vector to JSON (save_representation=True). For DyVo that took
~15 h per run on the shared GPUs, and it re-encodes the 729k-doc corpus once per query-candidate source.

What this does instead (numerically the same scoring: score = sum_ij q_w[i] * d_w[j] * [q_id[i] == d_id[j]]):
  * encodes the corpus ONCE with the doc encoder (fp16 autocast, natural corpus order, like predict()),
  * encodes the test queries once per candidate source (REL / GPT-4 / Mixtral share the doc side: doc vectors do not
    depend on query candidates),
  * scores all query sets with one sparse->dense matmul over the union of query ids,
  * keeps full score matrices, then top-10000 per query -> test_run.trec + test_result.json (same ir_measures metrics
    and key names as predict()), written to DyVo/outputs/<experiment>/fasteval/.
  * checkpoints partial scores every --ckpt-every batches so a killed job resumes where it stopped.

Usage (from repo root):
  python scripts/big_pc/fast_eval.py --model-dir DyVo/outputs/<train exp>/final/model \
      --exp <experiment used for model/doc side> --query-exps name1=<exp1> name2=<exp2> ...
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[2]
DYVO = REPO / "DyVo"
os.chdir(DYVO)
sys.path.insert(0, str(DYVO))

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
import ir_measures  # noqa: E402
from ir_measures import MRR, R, NDCG, MAP  # noqa: E402

METRICS = [MRR@10, R@5, R@10, R@100, R@1000, R@100000, NDCG@10, NDCG@20, MAP]


def log(msg):
    print(f"[fast_eval {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def compose_exp(exp):
    return compose(config_name="config", overrides=[f"+experiment={exp}"])


def to_dev(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment config for the model + doc side")
    ap.add_argument("--model-dir", default=None, help="dir with query_encoder/ and doc_encoder/ (overrides config ckpts)")
    ap.add_argument("--query-exps", nargs="+", required=True, help="outname=experiment (query candidate source)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--topk", type=int, default=10000)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--max-docs", type=int, default=-1, help="debug: only the first N docs")
    ap.add_argument("--work-dir", default=str(REPO / "scratch" / "fast_eval"))
    args = ap.parse_args()

    dev = torch.device("cuda")
    t0 = time.time()
    with initialize_config_dir(config_dir=str(DYVO / "lsr" / "configs"), version_base="1.2"):
        conf = compose_exp(args.exp)
        qconfs = {}
        for spec in args.query_exps:
            name, exp = spec.split("=", 1)
            qconfs[name] = compose_exp(exp)

    if args.model_dir:
        md = Path(args.model_dir) if Path(args.model_dir).is_absolute() else REPO / args.model_dir
        conf.model.query_encoder_checkpoint = str(md / "query_encoder" / "model.safetensors")
        conf.model.doc_encoder_checkpoint = str(md / "doc_encoder" / "model.safetensors")
    log(f"model ckpts: {conf.model.query_encoder_checkpoint} | {conf.model.doc_encoder_checkpoint}")
    for p in (conf.model.query_encoder_checkpoint, conf.model.doc_encoder_checkpoint):
        assert Path(p).exists(), p
    model = instantiate(conf.model).to(dev).eval()
    log(f"model loaded ({time.time()-t0:.0f}s)")

    test_dataset = instantiate(conf.test_dataset)
    docs = test_dataset.docs
    if args.max_docs > 0:
        docs = Subset(docs, range(args.max_docs))
    n_docs = len(docs)
    doc_ids = [test_dataset.docs.ids[i] for i in range(n_docs)]
    log(f"test set: {len(test_dataset.queries)} queries, {n_docs} docs ({time.time()-t0:.0f}s)")

    # ---- queries: one encoding per candidate source ----
    all_q = []  # (setname, qid, ids tensor, weights tensor)
    for name, qc in qconfs.items():
        collator = instantiate(qc.data_collator)
        dl = DataLoader(test_dataset.queries, batch_size=64, collate_fn=collator, shuffle=False)
        for b in dl:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                ids, ws, _ = model.encode_queries(**to_dev(b["queries"], dev), to_dense=False)
            for qid, i, w in zip(b["query_ids"], ids, ws):
                all_q.append((name, qid, i.long(), w.float()))
        log(f"encoded queries for {name} with {qc.data_collator.candidate_retriever.query_ent_path if 'candidate_retriever' in qc.data_collator else 'no entities'}")
    union = torch.unique(torch.cat([i for _, _, i, _ in all_q]))
    vocab_max = int(union.max().item()) + 1
    lookup = torch.full((max(vocab_max, 30522 + 5277580 + 1),), -1, dtype=torch.long, device=dev)
    lookup[union] = torch.arange(len(union), device=dev)
    U = len(union)
    Q = torch.zeros(len(all_q), U + 1, device=dev)
    for r, (_, _, i, w) in enumerate(all_q):
        Q[r].index_add_(0, lookup[i], w)  # sum over duplicate query ids == exact-match sum
    Q = Q[:, :U].contiguous()
    log(f"{len(all_q)} query vectors over {U} distinct ids")

    # ---- docs: encode once, score everything; resumable ----
    import hashlib
    key = f"{args.exp}|{conf.model.query_encoder_checkpoint}|{conf.model.doc_encoder_checkpoint}|{n_docs}|{sorted(args.query_exps)}"
    work = Path(args.work_dir) / hashlib.md5(key.encode()).hexdigest()[:16]
    work.mkdir(parents=True, exist_ok=True)
    (work / "key.txt").write_text(key)
    ck = work / "scores.pt"
    scores = torch.zeros(len(all_q), n_docs, dtype=torch.float32)
    start_batch = 0
    if ck.exists():
        state = torch.load(ck)
        if state["batch"] == args.batch:
            scores, start_batch = state["scores"], state["next_batch"]
            log(f"resuming at batch {start_batch}")
    n_batches = (n_docs + args.batch - 1) // args.batch
    doc_collator = instantiate(conf.data_collator)
    sub = Subset(docs, range(start_batch * args.batch, n_docs))
    dl = DataLoader(sub, batch_size=args.batch, collate_fn=doc_collator, shuffle=False,
                    num_workers=args.workers, pin_memory=True, prefetch_factor=4 if args.workers else None)
    tl = time.time()
    b_idx = start_batch
    for b in dl:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            d_ids, d_ws, _ = model.encode_docs(**to_dev(b["doc_groups"], dev), to_dense=False)
        cols = lookup[d_ids.long()]
        cols = torch.where(cols < 0, torch.full_like(cols, U), cols)
        D = torch.zeros(d_ids.shape[0], U + 1, device=dev)
        D.scatter_add_(1, cols, d_ws.float())
        s = D[:, :U] @ Q.T  # [B, nq]
        lo = b_idx * args.batch
        scores[:, lo:lo + s.shape[0]] = s.T.cpu()
        b_idx += 1
        if b_idx % 50 == 0:
            rate = (b_idx - start_batch) / (time.time() - tl)
            log(f"batch {b_idx}/{n_batches}  {rate:.2f} it/s  eta {(n_batches-b_idx)/max(rate,1e-9)/3600:.2f} h")
        if b_idx % args.ckpt_every == 0:
            torch.save({"scores": scores, "next_batch": b_idx, "batch": args.batch}, ck.with_suffix(".tmp"))
            os.replace(ck.with_suffix(".tmp"), ck)
    enc_secs = time.time() - tl
    torch.save({"scores": scores, "next_batch": b_idx, "batch": args.batch}, ck)
    log(f"docs done in {enc_secs/3600:.2f} h")

    # ---- runs + metrics per query set ----
    for name, qc in qconfs.items():
        exp_name = [s.split("=", 1)[1] for s in args.query_exps if s.split("=", 1)[0] == name][0]
        out = DYVO / "outputs" / exp_name / ("fasteval" if args.max_docs < 0 else f"fasteval_debug{args.max_docs}")
        out.mkdir(parents=True, exist_ok=True)
        run = {}
        with open(out / "test_run.trec", "w") as f:
            for r, (qn, qid, _, _) in enumerate(all_q):
                if qn != name:
                    continue
                k = min(args.topk, n_docs)
                vals, idx = scores[r].topk(k)
                run[qid] = {}
                for rank, (v, i) in enumerate(zip(vals.tolist(), idx.tolist())):
                    run[qid][doc_ids[i]] = v
                    f.write(f"{qid}\t0\t{doc_ids[i]}\t{rank}\t{v}\tLSR\n")
        m = ir_measures.calc_aggregate(METRICS, test_dataset.qrels, run)
        res = {f"test_{k}": v for k, v in m.items()}
        res.update({"test_runtime": time.time() - t0, "doc_encoding_hours": enc_secs / 3600,
                    "evaluator": "scripts/big_pc/fast_eval.py", "model_dir": str(args.model_dir),
                    "query_ent_path": str(qc.data_collator.get("candidate_retriever", {}).get("query_ent_path", "")),
                    "n_docs": n_docs})
        json.dump(res, open(out / "test_result.json", "w"), indent=1)
        log(f"{name}: nDCG@10 {res['test_nDCG@10']:.4f} nDCG@20 {res['test_nDCG@20']:.4f} R@1000 {res['test_R@1000']:.4f} -> {out}")


if __name__ == "__main__":
    main()
