"""Extension 2, step 2: explicit-mention evidence for every document, inside the text DyVo actually sees.

Visible text = (title + " " + text) cut at the last of the 510 content word-pieces DyVo's doc encoder reads
(DistilBERT tokenizer, max_length 512, same field join as lsr.utils.dataset_utils.read_collection).
Mentions: greedy longest match (up to 6 tokens) of lexicon surface forms (e2_alias.py) in the visible text.
Single-token matches must be capitalised in the document (proper-noun evidence). Spans may not start/end with a
function word. All-lowercase multi-token spans count only if they are an exact/de-qualified title (not a redirect).

Output: scratch/ext/e2/mentions.jsonl  {"id", "m": [[ent, first_char, count, src, n_tok, surface], ...]}
        scratch/ext/e2/visible_len.json (chars visible per doc, for the local-span ablation)
Run with PYTHONHASHSEED=0.
"""
import json
import os
import re
import time
from multiprocessing import Pool

import numpy as np

from common import EXT, REPO, log

TOK = re.compile(r"[A-Za-z0-9]+")
MAXN = 6
# E2 v2 (exploratory, after the blinded review of v1 additions): stricter single-token evidence
V2 = os.environ.get("E2_V2") == "1"
SENT_END = set('.!?"\u201c\u201d\'|:')
# function words that may not start or end an evidence span ("on China", "it was", "the benefit" ...)
EDGE_STOP = set("""a an the of in on at to for from by with about as into onto over under after before between through
during without within upon is are was were be been being am has have had having do does did done doing will would
shall should can could may might must it its this that these those there here their theirs they them he him his
she her hers we us our you your yours i me my mine m s t d ll re ve not no nor and or but so than too very also just
only even still yet more most much many some any each every all both few other such own same what which who whom whose
why how when where if then because while up down out off again further once said says say""".split())
_LEX = None
_TOKZ = None


def _init():
    global _LEX, _TOKZ
    from transformers import AutoTokenizer
    z = np.load(EXT / "e2" / "lexicon.npz")
    _LEX = (z["key_hash"], z["ent"], z["src"])
    _TOKZ = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def visible(texts):
    enc = _TOKZ(texts, truncation=True, max_length=512, return_offsets_mapping=True, add_special_tokens=True)
    out = []
    for t, offs in zip(texts, enc["offset_mapping"]):
        ends = [b for a, b in offs if b > 0]
        out.append(t[: max(ends)] if ends else "")
    return out


def find_mentions(text):
    toks = [(m.group(0), m.start()) for m in TOK.finditer(text)]
    low = [t.lower() for t, _ in toks]
    keys, where = [], []
    for i in range(len(toks)):
        for n in range(1, MAXN + 1):
            if i + n > len(toks):
                break
            if n == 1 and not toks[i][0][0].isupper():
                continue
            if n == 1 and len(toks[i][0]) < 3 and not toks[i][0].isupper():  # "UK", "EU" yes; "Us", "It" no
                continue
            if low[i] in EDGE_STOP or low[i + n - 1] in EDGE_STOP:
                continue
            keys.append(hash(" ".join(low[i:i + n]))); where.append((i, n))
    if not keys:
        return []
    kh, ke, ks = _LEX
    h = np.array(keys, dtype=np.int64)
    pos = np.searchsorted(kh, h)
    pos = np.minimum(pos, len(kh) - 1)
    hit = kh[pos] == h
    found = {}
    for (i, n), p, ok in zip(where, pos, hit):
        if not ok:
            continue
        cap = any(toks[j][0][0].isupper() for j in range(i, i + n))
        if V2 and n == 1:
            tok, start = toks[i]
            prev = text[:start].rstrip()[-1:]
            sentence_initial = (prev == "" or prev in SENT_END)
            # single tokens: exact unqualified title only (no "Lowe" -> "Lowe (band)"), never sentence-initial
            # ("However" -> "However (song)") unless an all-caps acronym ("UK")
            if not tok.isupper() and (int(ks[p]) != 0 or sentence_initial):
                continue
        # an all-lowercase span counts as evidence only if it is a multi-word exact (or de-qualified) title
        if not cap and (n == 1 or int(ks[p]) == 2):
            continue
        found[(i, n)] = (int(ke[p]), int(ks[p]))
    res, i = {}, 0
    while i < len(toks):  # greedy longest match, left to right
        for n in range(MAXN, 0, -1):
            if (i, n) in found:
                e, s = found[(i, n)]
                if e in res:
                    res[e][2] += 1
                else:
                    res[e] = [e, toks[i][1], 1, s, n, text[toks[i][1]: toks[i + n - 1][1] + len(toks[i + n - 1][0])]]
                i += n
                break
        else:
            i += 1
    return list(res.values())


def work(batch):
    ids, texts = zip(*batch)
    vis = visible(list(texts))
    return [(i, len(v), find_mentions(v)) for i, v in zip(ids, vis)]


def main():
    assert os.environ.get("PYTHONHASHSEED") == "0"
    import ir_datasets
    d = EXT / "e2"
    t0 = time.time()
    docs = {}
    for doc in ir_datasets.load("codec").docs_iter():
        docs[doc.doc_id] = " ".join(x for x in (doc.title, doc.text) if x is not None)
    order = json.load(open(EXT / "index_linked" / "doc_ids.json"))
    assert len(order) == len(docs)
    log(f"{len(docs)} docs loaded ({time.time()-t0:.0f}s)")
    batches = [[(i, docs[i]) for i in order[a:a + 500]] for a in range(0, len(order), 500)]
    vis_len = {}
    n_m = 0
    out_name = "mentions_v2.jsonl" if V2 else "mentions.jsonl"
    with Pool(24, initializer=_init) as pool, open(d / out_name, "w") as f:
        for b, res in enumerate(pool.imap(work, batches, chunksize=2)):
            for i, vl, ms in res:
                vis_len[i] = vl; n_m += len(ms)
                f.write(json.dumps({"id": i, "m": ms}) + "\n")
            if b % 100 == 0:
                log(f"{(b+1)*500}/{len(order)} docs, {n_m} mention entities ({time.time()-t0:.0f}s)")
    if not V2:
        json.dump(vis_len, open(d / "visible_len.json", "w"))
    log(f"done: {n_m} (doc, entity) mention pairs, {n_m/len(order):.1f}/doc ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
