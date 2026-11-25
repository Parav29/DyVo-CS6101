"""Shared helpers for the DyVo extensions (E1 query entity selection, E2 document entity repair).

Everything here reuses the frozen DyVo full100k model and the official scoring rule:
    score(q, d) = sum_i q_w[i] * d_w[j] * [q_id[i] == d_id[j]]
over word-piece ids (< 30522) and entity ids (30522 + entity index).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
DYVO = REPO / "DyVo"
EXT = REPO / "scratch" / "ext"
WORD_VOCAB = 30522
N_ENT = 5277580
V_TOTAL = WORD_VOCAB + N_ENT

DYVO_EXP = "qmlp_dmlm_emlm_laque_codec_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001_entw_0.05_full100k"
DYVO_MODEL = DYVO / "outputs" / DYVO_EXP / "10-01-2026" / "model"
CODEC = DYVO / "dyvo_data" / "codec"
KB = DYVO / "dyvo_data" / "knowledge_base"


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _enter_dyvo():
    os.chdir(DYVO)
    if str(DYVO) not in sys.path:
        sys.path.insert(0, str(DYVO))


def compose_conf(exp=DYVO_EXP):
    _enter_dyvo()
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(DYVO / "lsr" / "configs"), version_base="1.2"):
        return compose(config_name="config", overrides=[f"+experiment={exp}"])


def load_model(conf, dev="cuda", model_dir=DYVO_MODEL):
    from hydra.utils import instantiate
    conf.model.query_encoder_checkpoint = str(Path(model_dir) / "query_encoder" / "model.safetensors")
    conf.model.doc_encoder_checkpoint = str(Path(model_dir) / "doc_encoder" / "model.safetensors")
    return instantiate(conf.model).to(dev).eval()


class ListRetriever:
    """Candidate retriever backed by in-memory dicts {id: [entity indices]} (same interface as BERTCandidate)."""

    def __init__(self, query_entities=None, doc_entities=None):
        self.query_entities = query_entities or {}
        self.doc_entities = doc_entities or {}

    def retrieve_candidates(self, text, text_id, type="query"):
        src = self.query_entities if type == "query" else self.doc_entities
        ids = [int(e) for e in dict.fromkeys(src.get(str(text_id), []))]  # dedupe, keep order
        return ids, [1] * len(ids)


def make_collator(conf, retriever):
    from lsr.datasets.text_entity_collator import DynamicDataCollator
    from hydra.utils import instantiate
    tok = instantiate(conf.tokenizer)
    return DynamicDataCollator(tok, conf.query_max_length, conf.doc_max_length, retriever, entity_dim=768)


def to_dev(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def encode_queries(model, conf, queries, cand, batch_size=1, dev="cuda"):
    """queries: list of (qid, text); cand: {qid: [entity idx]}. Returns {qid: (ids int64 np, weights float32 np)}.

    batch_size=1 avoids the padding effect on entity max-pooling (assumption A7); the reproduction used 64.
    Zero weights are dropped; duplicate ids are summed (official query reduce="sum").
    """
    col = make_collator(conf, ListRetriever(query_entities=cand))
    out = {}
    for i in range(0, len(queries), batch_size):
        chunk = queries[i:i + batch_size]
        b = col([{"query_id": q, "query_text": t} for q, t in chunk])
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            ids, ws, _ = model.encode_queries(**to_dev(b["queries"], dev), to_dense=False)
        for (qid, _), qi, qw in zip(chunk, ids.long().cpu().numpy(), ws.float().cpu().numpy()):
            m = qw > 0
            u, inv = np.unique(qi[m], return_inverse=True)
            w = np.zeros(len(u), dtype=np.float32)
            np.add.at(w, inv, qw[m])
            out[qid] = (u, w)
    return out


# ---------------------------------------------------------------- sparse document index
class DocIndex:
    """Dumped document vectors (from dump_index.py) as a scipy CSC matrix [n_docs x V_TOTAL]."""

    def __init__(self, path, keep_csr=True):
        import scipy.sparse as sp
        path = Path(path)
        self.path = path
        self.doc_ids = json.load(open(path / "doc_ids.json"))
        n = len(self.doc_ids)
        self.csr = None
        if keep_csr or not (path / "csc_data.npy").exists():
            indptr = np.load(path / "indptr.npy")
            assert indptr[-1] < 2**31
            # int32 indptr keeps scipy from upcasting indices to int64 (would double memory)
            self.csr = sp.csr_matrix((np.load(path / "w.npy").astype(np.float32), np.load(path / "ids.npy").astype(np.int32),
                                      indptr.astype(np.int32)), shape=(n, V_TOTAL))
        if (path / "csc_data.npy").exists():  # precomputed column layout, memory-mapped (no conversion peak)
            self.csc = sp.csc_matrix((np.load(path / "csc_data.npy", mmap_mode="r"),
                                      np.load(path / "csc_indices.npy", mmap_mode="r"),
                                      np.load(path / "csc_indptr.npy")), shape=(n, V_TOTAL), copy=False)
        else:
            self.csc = self.csr.tocsc()
        if not keep_csr:
            self.csr = None
        self.did2row = {d: i for i, d in enumerate(self.doc_ids)}

    def score(self, qids, qvecs, topk=10000):
        """Exact term-at-a-time scoring of every document. Returns run {qid: {docid: score}}."""
        run = {}
        for qid in qids:
            ids, w = qvecs[qid]
            s = np.asarray(self.csc[:, ids] @ w).ravel() if len(ids) else np.zeros(len(self.doc_ids), np.float32)
            k = min(topk, len(s))
            top = np.argpartition(-s, k - 1)[:k]
            top = top[np.argsort(-s[top], kind="stable")]
            run[qid] = {self.doc_ids[j]: float(s[j]) for j in top}
        return run

    def doc_entity_weights(self, row):
        """{entity_idx: weight} for one document row (entity part only)."""
        a, b = self.csr.indptr[row], self.csr.indptr[row + 1]
        ids, w = self.csr.indices[a:b], self.csr.data[a:b]
        m = ids >= WORD_VOCAB
        return dict(zip((ids[m] - WORD_VOCAB).tolist(), w[m].tolist()))

    def entity_df(self):
        """Number of documents with a positive posting for every entity (array of length N_ENT)."""
        return np.diff(self.csc.indptr)[WORD_VOCAB:]


# ---------------------------------------------------------------- evaluation + statistics
def codec_test():
    """42 CODEC test queries [(qid, text)] and qrels {qid: {docid: rel}} from ir_datasets."""
    import ir_datasets
    ds = ir_datasets.load("codec")
    queries = [(q.query_id, q.query) for q in ds.queries_iter()]
    qrels = {}
    for r in ds.qrels_iter():
        qrels.setdefault(r.query_id, {})[r.doc_id] = r.relevance
    return queries, qrels


def evaluate(run, qrels):
    import ir_measures
    from ir_measures import NDCG, R
    ms = [NDCG @ 10, NDCG @ 20, R @ 1000]
    agg = ir_measures.calc_aggregate(ms, qrels, run)
    per = {}
    for m in ir_measures.iter_calc(ms, qrels, run):
        per.setdefault(str(m.measure), {})[m.query_id] = m.value
    return {str(k): v for k, v in agg.items()}, per


def paired_tests(a, b, n=10000, seed=0):
    """a, b: {qid: value}. Two-sided paired randomization test + 95% bootstrap CI of mean(b - a)."""
    q = sorted(set(a) & set(b))
    d = np.array([b[x] - a[x] for x in q])
    rng = np.random.default_rng(seed)
    obs = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    p = float((np.abs((signs * d).mean(1)) >= abs(obs) - 1e-12).mean())
    boots = rng.choice(d, size=(n, len(d))).mean(1)
    return {"mean_diff": float(obs), "p_rand": p, "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "wins": int((d > 1e-9).sum()), "losses": int((d < -1e-9).sum()), "n": len(d)}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values for a dict {name: p}."""
    items = sorted(pvals.items(), key=lambda x: x[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def load_jsonl_entities(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[str(r["id"])] = r["entities"]
    return out
