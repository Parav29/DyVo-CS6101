"""Extension 2, step 1: surface-form lexicon for the LaQue entity catalog.

Surface forms (lower-cased, tokenised with [a-z0-9]+):
  title       the Wikipedia title of the catalog entity           e.g. "open banking"
  base        title without a trailing "(...)" qualifier            "Mercury (planet)" -> "mercury"
  redirect    every English Wikipedia redirect to that title       "UK" -> "United Kingdom"
Ambiguous forms resolve to the most popular entity (smallest catalog index; entity_list.json is in popularity
order: United States, ..., see extension2.md). List/disambiguation pages are excluded.

Input: scratch/ext/e2/wiki/enwiki-latest-{redirect,page}.sql.gz (Wikimedia dumps)
Output: scratch/ext/e2/lexicon.npz  {key_hash int64 sorted, ent int32, src int8 (0 title, 1 base, 2 redirect), n_tok}
"""
import gzip
import json
import re
import time

import numpy as np

from common import EXT, KB, log

TOK = re.compile(r"[a-z0-9]+")
STOP = set("""a an the of in on at to for from by with about as into is are was were be it its this that and or but not
no he she they we you i his her their our new old one two first last all can will may""".split())
ROW_PAGE = re.compile(rb"\((\d+),0,'((?:[^'\\]|\\.)*)',1,")         # ns-0 redirect pages: page_id, title
ROW_RED = re.compile(rb"\((\d+),0,'((?:[^'\\]|\\.)*)',")             # rd_from, ns 0, rd_title


def unesc(b):
    return re.sub(rb"\\(.)", rb"\1", b).decode("utf-8", "replace").replace("_", " ")


def key_of(s):
    toks = TOK.findall(s.lower())
    return " ".join(toks), len(toks)


def stable_hash(s):
    # built-in (C) string hash; reproducible across processes only with a fixed PYTHONHASHSEED (checked in main)
    return hash(s)


def scan(path, rx):
    with gzip.open(path, "rb") as f:
        for line in f:
            # old dumps: many rows per INSERT line; current dumps: one "(...)," row per line
            if line.startswith(b"INSERT INTO") or line.startswith(b"("):
                yield from rx.findall(line)


def good(key, n):
    if n == 0 or key in STOP or key.isdigit():
        return False
    if n == 1 and len(key) < 2:  # 2-char forms (UK, EU) are only accepted upper-case in e2_mentions.py
        return False
    return True


def main():
    import os
    assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0 (lexicon keys are hash()es)"
    d = EXT / "e2"
    t0 = time.time()
    titles = json.load(open(KB / "entity_list.json"))
    title2idx = {}
    for i, t in enumerate(titles):
        title2idx.setdefault(t, i)
    bad = lambda t: t.startswith("List of") or t.endswith("(disambiguation)") or t.startswith("Index of")

    best = {}  # key -> (priority-sorted) (ent, src)

    def offer(surface, ent, src):
        k, n = key_of(surface)
        if not good(k, n):
            return
        cur = best.get(k)
        if cur is None or (ent, src) < cur:
            best[k] = (ent, src)

    for i, t in enumerate(titles):
        if bad(t):
            continue
        offer(t, i, 0)
        if t.endswith(")") and " (" in t:
            offer(t[: t.rfind(" (")], i, 1)
    log(f"titles: {len(best)} keys ({time.time()-t0:.0f}s)")

    pages = {}
    for pid, title in scan(d / "wiki" / "enwiki-latest-page.sql.gz", ROW_PAGE):
        pages[int(pid)] = unesc(title)
    log(f"redirect pages: {len(pages)} ({time.time()-t0:.0f}s)")
    n_red = 0
    for frm, target in scan(d / "wiki" / "enwiki-latest-redirect.sql.gz", ROW_RED):
        src_title = pages.get(int(frm))
        ent = title2idx.get(unesc(target))
        if src_title is None or ent is None or bad(titles[ent]):
            continue
        offer(src_title, ent, 2)
        n_red += 1
    log(f"redirects mapped to catalog entities: {n_red}; total keys {len(best)} ({time.time()-t0:.0f}s)")

    keys = list(best)
    h = np.array([stable_hash(k) for k in keys], dtype=np.int64)
    ent = np.array([best[k][0] for k in keys], dtype=np.int32)
    src = np.array([best[k][1] for k in keys], dtype=np.int8)
    ntok = np.array([k.count(" ") + 1 for k in keys], dtype=np.int8)
    o = np.argsort(h)
    np.savez(d / "lexicon.npz", key_hash=h[o], ent=ent[o], src=src[o], n_tok=ntok[o])
    stats = {"keys": len(keys), "from_title": int((src == 0).sum()), "from_base": int((src == 1).sum()),
             "from_redirect": int((src == 2).sum()), "entities_covered": int(len(np.unique(ent))),
             "redirects_mapped": n_red, "dup_hash": int(len(h) - len(np.unique(h)))}
    json.dump(stats, open(d / "lexicon_stats.json", "w"), indent=1)
    log(json.dumps(stats))


if __name__ == "__main__":
    main()
