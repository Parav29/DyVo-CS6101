"""Level-2 sanity check: the tiny DyVo dual encoder must be able to overfit ONE fixed batch.

Batch: 2 queries x (positive, negative) documents, built with the real DynamicDataCollator + BERTCandidate.
Teacher prefers the first doc for q1 and the second doc for q2, so the student has to learn both directions.
Prints the KL loss curve and the number of non-zero (unmasked) entity weights per step.

SANITY ONLY: random tiny model + synthetic text; nothing here says anything about retrieval quality.

Usage (from DyVo/):
  python sanity/overfit_one_batch.py                                    # overfit check (exit 1 if it fails)
  python sanity/overfit_one_batch.py --lambda-init 1.0 --no-train-lambda --probe   # collapse probe (S3.2)
"""
import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

for _k, _v in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "disabled",
               "CUDA_VISIBLE_DEVICES": ""}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "smoke_assets"
sys.path.insert(0, str(REPO))

from lsr.datasets.text_entity_collator import DynamicDataCollator  # noqa: E402
from lsr.entity.bert_candidates import BERTCandidate  # noqa: E402
from lsr.losses import EntityDistilKLLoss  # noqa: E402
from lsr.losses.regularizer import L1  # noqa: E402
from lsr.models import DualSparseEncoder  # noqa: E402
from lsr.models.entity_emb import BertEntityEmbedding  # noqa: E402
from lsr.models.mlm_emlm import TransformerMLMeMLMConfig, TransformerMLMeMLMSparseEncoder  # noqa: E402
from lsr.models.mlp_emlm import TransformerMLPeMLMConfig, TransformerMLPeMLMSparseEncoder  # noqa: E402
from lsr.tokenizer import HFTokenizer  # noqa: E402

VOCAB = 30522


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(lambda_init, entity_dim):
    base = str(ASSETS / "tiny-distilbert")
    ent = BertEntityEmbedding(entity_emb_path=str(ASSETS / f"entity_embs_tiny{entity_dim}.pt"))
    q = TransformerMLPeMLMSparseEncoder(TransformerMLPeMLMConfig(
        tf_base_model_name_or_dir=base, entity_dim=entity_dim, entity_weight=lambda_init), ent_emb_model=ent)
    d = TransformerMLMeMLMSparseEncoder(TransformerMLMeMLMConfig(
        tf_base_model_name_or_dir=base, entity_dim=entity_dim, entity_weight=lambda_init), ent_emb_model=ent)
    return DualSparseEncoder(q, d)


def build_batch(tmpdir):
    q_rows = [{"id": "q1", "entities": [1, 2, 4]}, {"id": "q2", "entities": [3, 7]}]
    d_rows = [{"id": "d1", "entities": [1, 4, 6]}, {"id": "d2", "entities": [10, 11]},
              {"id": "d3", "entities": [8, 9]}, {"id": "d4", "entities": [3, 7, 13]}]
    for name, rows in (("q.jsonl", q_rows), ("d.jsonl", d_rows)):
        (tmpdir / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    retriever = BERTCandidate(query_ent_path=str(tmpdir / "q.jsonl"), doc_ent_path=str(tmpdir / "d.jsonl"))
    tok = HFTokenizer(str(ASSETS / "tiny-tokenizer"))
    collator = DynamicDataCollator(tok, q_max_length=32, d_max_length=64, candidate_retriever=retriever)
    examples = [
        {"query_id": "q1", "query_text": "why are many commentators arguing nft are the next big investment category ?",
         "doc_id": ["d1", "d2"], "doc_text": ["the nft and bitcoin token market is the next big digital asset",
                                              "the gold standard and the exchange rate crisis"],
         "score": [3.0, 0.0]},
        {"query_id": "q2", "query_text": "how vital was french support during the american revolutionary war ?",
         "doc_id": ["d3", "d4"], "doc_text": ["electric car battery demand for lithium and cobalt",
                                              "french army and navy support in the american revolutionary war"],
         "score": [0.0, 3.0]},
    ]
    return collator(examples)


def count_entities(model, batch):
    """Non-zero entity weights among real (unmasked) candidates, for queries and all docs."""
    with torch.no_grad():
        counts, totals = 0, 0
        for enc, group in [(model.query_encoder, batch["queries"])] + [(model.doc_encoder, g) for g in batch["doc_groups"]]:
            ids, w, _ = enc(**group)
            n = group["entity_ids"].shape[1]
            m = group["entity_masks"].bool()
            counts += int((w[:, -n:][m] > 0).sum())
            totals += int(m.sum())
    return counts, totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-init", type=float, default=0.05)
    ap.add_argument("--no-train-lambda", action="store_true", help="freeze ent_weight (λ_ent) at its init value")
    ap.add_argument("--entity-dim", type=int, default=32, choices=[16, 32])
    ap.add_argument("--normalize", default="softmax", choices=["softmax", "std"],
                    help="teacher normalization; softmax gives a proper KL with minimum 0 (default for this check)")
    ap.add_argument("--threshold", type=float, default=0.5, help="pass if final_loss <= threshold * initial_loss")
    ap.add_argument("--probe", action="store_true", help="collapse probe: report only, never fail")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-json", default=None)
    args = ap.parse_args()

    seed_all(args.seed)
    model = build_model(args.lambda_init, args.entity_dim)
    model.train()
    if args.no_train_lambda:
        model.query_encoder.ent_weight.requires_grad_(False)
        model.doc_encoder.ent_weight.requires_grad_(False)
    loss_fn = EntityDistilKLLoss(q_regularizer=L1(weight=0.0), d_regularizer=L1(weight=0.0),
                                 normalize=args.normalize, reg_ent=False)
    with tempfile.TemporaryDirectory() as td:
        batch = build_batch(Path(td))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)

    history = []
    for step in range(args.steps + 1):
        kl, q_reg, d_reg, _ = model(loss_fn, batch["queries"], batch["doc_groups"], labels=batch["labels"])
        n_ent, n_tot = count_entities(model, batch)
        rec = {"step": step, "loss": float(kl), "nonzero_entities": n_ent, "entity_slots": n_tot,
               "lambda_q": float(model.query_encoder.ent_weight), "lambda_d": float(model.doc_encoder.ent_weight)}
        history.append(rec)
        if step % 10 == 0 or step == args.steps:
            print(f"step {step:4d}  kl={rec['loss']:.5f}  nonzero_entities={n_ent}/{n_tot}  "
                  f"lambda_q={rec['lambda_q']:.4f}  lambda_d={rec['lambda_d']:.4f}")
        if step == args.steps:
            break
        opt.zero_grad()
        (kl + q_reg + d_reg).backward()
        opt.step()

    first, last = history[0]["loss"], history[-1]["loss"]
    final_ent = history[-1]["nonzero_entities"]
    collapsed_steps = [h["step"] for h in history if h["nonzero_entities"] == 0]
    summary = {"initial_loss": first, "final_loss": last, "ratio": last / first if first else float("nan"),
               "final_nonzero_entities": final_ent, "entity_slots": history[-1]["entity_slots"],
               "first_collapse_step": collapsed_steps[0] if collapsed_steps else None,
               "args": vars(args), "sanity_only": True}
    if args.log_json:
        Path(args.log_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.log_json).write_text(json.dumps({"summary": summary, "history": history}, indent=1))
    print(f"[SANITY ONLY] loss {first:.5f} -> {last:.5f} (ratio {summary['ratio']:.3f}, threshold {args.threshold}); "
          f"non-zero entity weights at end: {final_ent}/{summary['entity_slots']}; "
          f"collapse (all entity weights 0) first seen at step: {summary['first_collapse_step']}")
    if args.probe:
        print("PROBE DONE")
        return 0
    ok = last <= args.threshold * first and final_ent > 0
    print("OVERFIT PASS" if ok else "OVERFIT FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
