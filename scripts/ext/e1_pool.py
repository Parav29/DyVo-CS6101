"""Extension 1, step 1: build a cheap candidate pool for every training and test query.

Pool(q) = REL(q)  U  BM25-title top-K(q)  U  LaQue-dense top-K(q)
Both generators search only the "indexable catalog" (entities with >= 1 linked document in CODEC, 685,656 of 5.28M):
an entity without postings contributes exactly 0 to every score under the fixed index.
  * REL: the authors' linked query entities (empty for 3,319 of the 9,000 training queries).
  * BM25-title: BM25 over the 5.28M Wikipedia titles of the LaQue catalog (entity_list.json); no external index.
  * LaQue-dense: the authors' LaQue query encoder (mean pooling) dot the LaQue entity table, as in
    lsr/entity/retrieve_entity_with_laque.py.
  * s_e(q) = max_t log1p(relu(lambda_ent * h_t . E_e)) for each pool member, from the frozen DyVo full100k query
    head -- the exact weight DyVo gives e. (Using this head to search the whole catalog returns junk.)

For every pool member we store the features the selector may use (no entity-ID memorisation) and s_e(q).
Output: scratch/ext/e1/pool_{train,test}.jsonl
"""
import argparse
import json
import math
import re
import time
from collections import Counter

import numpy as np
import scipy.sparse as sp
import torch

from common import EXT, CODEC, KB, N_ENT, compose_conf, load_model, log, load_jsonl_entities, codec_test

STOP = set("""a an the of in on at to for from by with about as into over after before between through during without
within is are was were be been being has have had do does did what which who whom whose why how when where whether
this that these those it its it's their there they them he she his her we our you your i me my or and but not no nor
so than too very can could should would may might will shall must also any some such more most other others each
all both few many much own same just only s t don't vs versus between does did doing""".split())
TOK = re.compile(r"[a-z0-9]+")


def tokenize(s):
    return TOK.findall(s.lower())


class TitleBM25:
    def __init__(self, titles, allowed, k1=0.9, b=0.4):
        """BM25 over the titles of the `allowed` entities only (rows of the matrix = global entity indices)."""
        t0 = time.time()
        vocab, rows, cols = {}, [], []
        lens = np.zeros(len(titles), np.float32)
        for i in np.flatnonzero(allowed):
            toks = tokenize(titles[i])
            lens[i] = len(toks)
            for w in toks:
                rows.append(i)
                cols.append(vocab.setdefault(w, len(vocab)))
        m = sp.csr_matrix((np.ones(len(rows), np.float32), (np.array(rows), np.array(cols))),
                          shape=(len(titles), len(vocab)))
        m.sum_duplicates()
        self.csc = m.tocsc()
        self.vocab = vocab
        df = np.diff(self.csc.indptr)
        n = int(allowed.sum())
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.norm = (k1 * (1 - b + b * lens / max(lens[allowed].mean(), 1e-6))).astype(np.float32)
        self.k1 = k1
        log(f"title BM25 built: {len(vocab)} terms ({time.time()-t0:.0f}s)")

    def search(self, text, k):
        terms = [self.vocab[w] for w in dict.fromkeys(tokenize(text)) if w not in STOP and w in self.vocab]
        if not terms:
            return np.array([], int), np.array([], np.float32)
        s = np.zeros(self.csc.shape[0], np.float32)
        for t in terms:
            a, b = self.csc.indptr[t], self.csc.indptr[t + 1]
            rows, tf = self.csc.indices[a:b], self.csc.data[a:b]
            s[rows] += self.idf[t] * tf * (self.k1 + 1) / (tf + self.norm[rows])
        k = min(k, int((s > 0).sum()))
        if k == 0:
            return np.array([], int), np.array([], np.float32)
        top = np.argpartition(-s, k - 1)[:k]
        # ties broken by entity index (the catalog is ordered by popularity)
        top = top[np.lexsort((top, -s[top]))]
        return top, s[top]

    def score(self, text, ents):
        """BM25 score of specific entities (0 if no shared term)."""
        terms = [self.vocab[w] for w in dict.fromkeys(tokenize(text)) if w not in STOP and w in self.vocab]
        out = np.zeros(len(ents), np.float32)
        ents = np.asarray(ents)
        for t in terms:
            a, b = self.csc.indptr[t], self.csc.indptr[t + 1]
            rows, tf = self.csc.indices[a:b], self.csc.data[a:b]
            pos = np.searchsorted(rows, ents)
            ok = (pos < len(rows)) & (rows[np.minimum(pos, len(rows) - 1)] == ents)
            tfv = np.where(ok, tf[np.minimum(pos, len(rows) - 1)], 0)
            out += self.idf[t] * tfv * (self.k1 + 1) / (tfv + self.norm[ents])
        return out


class LaqueDense:
    """The authors' LaQue query encoder (lsr42/distilbert-base-uncased-dense-laque, mean pooling) . LaQue table."""

    def __init__(self, table, allowed=None, dev="cuda", chunk=1_000_000, name="lsr42/distilbert-base-uncased-dense-laque"):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.enc = AutoModel.from_pretrained(name).to(dev).eval().half()
        self.dev = dev
        idx = np.flatnonzero(allowed) if allowed is not None else np.arange(N_ENT)
        self.idx = torch.as_tensor(idx, dtype=torch.long)
        self.chunks = [table[self.idx[i:i + chunk]].to(dev, dtype=torch.float16) for i in range(0, len(idx), chunk)]
        self.offsets = list(range(0, len(idx), chunk))
        self.table = table

    @torch.no_grad()
    def embed(self, text):
        t = self.tok([text], truncation=True, max_length=300, return_tensors="pt").to(self.dev)
        h = self.enc(**t).last_hidden_state
        m = t["attention_mask"].unsqueeze(-1).half()
        return ((h * m).sum(1) / m.sum(1))[0]

    @torch.no_grad()
    def search(self, v, k):
        vs, ids = [], []
        for off, c in zip(self.offsets, self.chunks):
            s = (c @ v).float()
            a, b = s.topk(k)
            vs.append(a); ids.append(b + off)
        v_, i_ = torch.cat(vs), torch.cat(ids)
        o = v_.topk(k)
        return self.idx[i_[o.indices].cpu()].numpy(), o.values.cpu().numpy()

    @torch.no_grad()
    def score(self, v, ents):
        if len(ents) == 0:
            return np.zeros(0, np.float32)
        e = self.table[torch.as_tensor(np.asarray(ents), dtype=torch.long)].to(self.dev, dtype=torch.float16)
        return (e @ v).float().cpu().numpy()


class DyVoDense:
    """Scores all catalog entities with the frozen DyVo query entity head (batch 1, no padding)."""

    def __init__(self, model, conf, allowed, dev="cuda", chunk=1_000_000):
        from hydra.utils import instantiate
        self.qe = model.query_encoder
        self.tok = instantiate(conf.tokenizer)
        self.qlen = conf.query_max_length
        self.dev = dev
        t0 = time.time()
        table = self.qe.ent_emb_model.entity_emb  # mmap'd float32 [N_ENT, 768]
        self.lam = float(self.qe.ent_weight.detach())
        if allowed is None:  # per-candidate scoring only
            return
        self.allowed_idx = torch.as_tensor(np.flatnonzero(allowed), dtype=torch.long)
        self.chunks = [table[self.allowed_idx[i:i + chunk]].to(dev, dtype=torch.float16)
                       for i in range(0, len(self.allowed_idx), chunk)]
        self.offsets = list(range(0, len(self.allowed_idx), chunk))
        self.lam = float(self.qe.ent_weight.detach())
        log(f"LaQue table on GPU in fp16 ({time.time()-t0:.0f}s), lambda_ent={self.lam:.4f}")

    @torch.no_grad()
    def hidden(self, text):
        enc = self.tok([text], padding=True, truncation=True, max_length=self.qlen, return_tensors="pt",
                       return_special_tokens_mask=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = self.qe.model(input_ids=enc["input_ids"].to(self.dev), attention_mask=enc["attention_mask"].to(self.dev))
        return out.last_hidden_state[0].half()  # [L, 768]

    @torch.no_grad()
    def search(self, h, k):
        best_v, best_i = [], []
        for off, c in zip(self.offsets, self.chunks):
            s = torch.log1p(torch.relu((h @ c.T).float() * self.lam)).max(0).values  # [chunk]
            v, i = s.topk(k)
            best_v.append(v); best_i.append(i + off)
        v, i = torch.cat(best_v), torch.cat(best_i)
        o = v.topk(k)
        return self.allowed_idx[i[o.indices].cpu()].numpy(), o.values.cpu().numpy()

    @torch.no_grad()
    def score(self, h, ents):
        """(s_e(q), max raw logit) for given entity indices."""
        if len(ents) == 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        table = self.qe.ent_emb_model.entity_emb
        e = table[torch.as_tensor(np.asarray(ents), dtype=torch.long)].to(self.dev, dtype=torch.float16)
        raw = ((h @ e.T).float() * self.lam).max(0).values
        return torch.log1p(torch.relu(raw)).cpu().numpy(), raw.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--max-train", type=int, default=-1)
    args = ap.parse_args()
    out = EXT / "e1"
    out.mkdir(parents=True, exist_ok=True)

    titles = json.load(open(KB / "entity_list.json"))
    assert len(titles) == N_ENT, len(titles)
    # indexable catalog: entities with >= 1 linked document posting (others can never change a score)
    allowed = np.load(out / "df_linked.npy") > 0
    log(f"indexable catalog: {allowed.sum()} entities")
    bm25 = TitleBM25(titles, allowed)
    rel = load_jsonl_entities(CODEC / "queries_test_train_entities_rel.jsonl")

    train = [tuple(l.rstrip("\n").split("\t", 1)) for l in open(CODEC / "codec_mistral_topics.tsv")]
    if args.max_train > 0:
        train = train[:args.max_train]
    test, _ = codec_test()

    conf = compose_conf()
    model = load_model(conf)
    dense = DyVoDense(model, conf, None)  # only s_e(q) of pool members (its catalog-wide top-K is junk, see extension1.md)
    laque = LaqueDense(dense.qe.ent_emb_model.entity_emb, allowed)

    for split, qs in (("test", test), ("train", train)):
        t0 = time.time()
        timing = {"bm25": 0.0, "laque": 0.0}
        with open(out / f"pool_{split}.jsonl", "w") as f:
            for n, (qid, text) in enumerate(qs):
                ta = time.time()
                b_ids, b_s = bm25.search(text, args.k)
                tb = time.time()
                v = laque.embed(text)
                d_ids, d_s = laque.search(v, args.k)
                torch.cuda.synchronize()
                tc = time.time()
                timing["bm25"] += tb - ta; timing["laque"] += tc - tb
                r_ids = [int(e) for e in dict.fromkeys(rel.get(qid, []))]
                pool = list(dict.fromkeys(r_ids + d_ids.tolist() + b_ids.tolist()))
                h = dense.hidden(text)
                sq, raw = dense.score(h, pool)
                ls = laque.score(v, pool)
                bs = bm25.score(text, pool)
                b_rank = {int(e): r for r, e in enumerate(b_ids)}
                d_rank = {int(e): r for r, e in enumerate(d_ids)}
                cands = [{"e": int(e), "sq": float(sq[j]), "raw": float(raw[j]), "bm25": float(bs[j]),
                          "laque": float(ls[j]), "bm25_rank": b_rank.get(int(e), -1), "laque_rank": d_rank.get(int(e), -1),
                          "in_rel": int(e in r_ids)} for j, e in enumerate(pool)]
                f.write(json.dumps({"id": qid, "text": text, "cands": cands}) + "\n")
                if n % 1000 == 0:
                    log(f"{split} {n}/{len(qs)} ({time.time()-t0:.0f}s)")
        json.dump({k: v / len(qs) for k, v in timing.items()}, open(out / f"pool_{split}_timing.json", "w"))
        log(f"{split} done: {len(qs)} queries in {time.time()-t0:.0f}s; per-query s: {timing}")


if __name__ == "__main__":
    main()
