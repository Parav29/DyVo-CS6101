"""Extension 2, step 4: build the document-entity variants under one matched budget.

Additions are entities NOT already linked to the document. Every variant gets the same per-document cap (--cap)
and the same global number of added candidates B (= the smallest variant's available total), so no variant wins by
storing more postings.
  E2-C1 alias-only   : entities with an explicit title/alias mention in the visible text, ranked by
                       (mention count desc, first position asc)                       -- no salience model
  E2-C2 unfiltered   : LaQue top-30 dense proposals, ranked by LaQue similarity           -- no evidence check
  E2    grounded     : proposals (LaQue top-30 U mention entities) KEPT ONLY IF explicitly mentioned in the
                       visible text, ranked by LaQue doc-entity similarity               -- evidence + salience
Global selection: pairs ranked by (within-doc rank, -score) and the first B kept (spreads the budget over docs).

Outputs (scratch/ext/e2/):
  docs_entities_union.jsonl  linked U all variants' capped additions (dumped once with dump_index.py)
  variants.npz               for each variant the selected (doc_row, entity) pairs
  build_stats.json
"""
import argparse
import json
import time

import numpy as np
import torch

from common import EXT, CODEC, KB, N_ENT, log, load_jsonl_entities  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=5)
    ap.add_argument("--v2", action="store_true",
                    help="exploratory v2: mentions_v2 (strict single-token evidence), additions restricted to entities "
                         "REL links somewhere in CODEC, grounded ranked by hubness-corrected (CSLS) LaQue similarity")
    args = ap.parse_args()
    sfx = "_v2" if args.v2 else ""
    d = EXT / "e2"
    t0 = time.time()
    order = json.load(open(EXT / "index_linked" / "doc_ids.json"))
    row = {x: i for i, x in enumerate(order)}
    N = len(order)
    linked = load_jsonl_entities(CODEC / "docs_entities.jsonl")
    linked = [set(int(e) for e in linked.get(x, [])) for x in order]
    _z = np.load(d / "dense_top30.npz")
    dense = {"ids": _z["ids"], "scores": _z["scores"]}  # materialise once (NpzFile re-reads on every [])
    emb = np.load(d / "doc_emb.npy", mmap_mode="r")
    table = torch.load(str(KB / "entity_embs_laque.pt"), map_location="cpu", mmap=True)
    mentions = [None] * N
    for l in open(d / f"mentions{sfx}.jsonl"):
        r = json.loads(l)
        mentions[row[r["id"]]] = r["m"]
    log(f"inputs loaded ({time.time()-t0:.0f}s)")
    linkable = None
    if args.v2:
        # keyphraseness (Mihalcea & Csomai 2007): P(linked | mentioned) = linked_df / (linked_df + unlinked_mention_df);
        # keep entities the linker usually links when they appear (>= 0.1, fixed a priori). Removes "Last Year",
        # weekdays, "Photograph", "File (tool)": mentioned everywhere, rarely linked.
        ldf = np.load(EXT / "e1" / "df_linked.npy").astype(np.float64)
        umdf = np.zeros(N_ENT, np.float64)
        for r in range(N):
            for m in mentions[r]:
                if m[0] not in linked[r]:
                    umdf[m[0]] += 1
        rate = np.where(ldf + umdf > 0, ldf / np.maximum(ldf + umdf, 1), 0)
        linkable = (ldf > 0) & (rate >= 0.1)
        log(f"keyphraseness filter: {int(linkable.sum())} entities eligible (of {int((ldf > 0).sum())} linked in CODEC)")
    if args.v2:  # mean document embedding: hub(e) = E_e . mu = mean similarity of e to all documents (CSLS-style)
        mu = np.zeros(768, np.float64)
        for a in range(0, N, 100_000):
            mu += np.asarray(emb[a:a + 100_000], dtype=np.float64).sum(0)
        mu = torch.from_numpy((mu / N).astype(np.float32))

    cand = {"alias": [], "unfiltered": [], "grounded": []}  # lists of (rank_in_doc, -score, doc, ent)
    st = {"mention_pairs": 0, "mention_pairs_already_linked": 0, "dense_pairs_already_linked": 0,
          "dense_pairs_with_evidence": 0}
    B = 4096
    for a in range(0, N, B):
        rows = range(a, min(N, a + B))
        # LaQue similarity for every mention entity of these docs (batched on CPU)
        pairs = [(r, m[0]) for r in rows for m in mentions[r]]
        sims = {}
        if pairs:
            pr = np.array([p[0] for p in pairs]); pe = np.array([p[1] for p in pairs])
            de = torch.from_numpy(np.asarray(emb[pr], dtype=np.float32))
            ee = table[torch.from_numpy(pe).long()]
            s = (de * ee).sum(1)
            if args.v2:
                s = s - ee @ mu
            s = s.numpy()
            sims = {(int(r), int(e)): float(v) for r, e, v in zip(pr, pe, s)}
        for r in rows:
            L = linked[r]
            ms = mentions[r]
            st["mention_pairs"] += len(ms)
            st["mention_pairs_already_linked"] += sum(m[0] in L for m in ms)
            ment = {m[0]: m for m in ms}
            if args.v2:  # repair only entities the linker knows in this corpus
                ms = [m for m in ms if linkable[m[0]]]
                ment = {m[0]: m for m in ms}
            # alias-only
            al = sorted([m for m in ms if m[0] not in L], key=lambda m: (-m[2], m[1]))[:args.cap]
            cand["alias"] += [(k, -m[2], r, m[0]) for k, m in enumerate(al)]
            # unfiltered dense
            ids, sc = dense["ids"][r], dense["scores"][r].astype(np.float32)
            st["dense_pairs_already_linked"] += sum(int(e) in L for e in ids)
            st["dense_pairs_with_evidence"] += sum(int(e) in ment for e in ids)
            un = [(int(e), float(s)) for e, s in zip(ids, sc) if int(e) not in L][:args.cap]
            cand["unfiltered"] += [(k, -s, r, e) for k, (e, s) in enumerate(un)]
            # grounded: proposals with explicit evidence, ranked by LaQue salience
            pool = {} if args.v2 else {int(e): float(s) for e, s in zip(ids, sc) if int(e) in ment and int(e) not in L}
            for e in ment:
                if e not in L and e not in pool:
                    pool[e] = sims[(r, e)]
            gr = sorted(pool.items(), key=lambda x: -x[1])[:args.cap]
            cand["grounded"] += [(k, -s, r, e) for k, (e, s) in enumerate(gr)]
        if (a // B) % 40 == 0:
            log(f"{a}/{N} ({time.time()-t0:.0f}s)")

    budget = min(len(v) for v in cand.values())
    out, stats = {}, {"cap_per_doc": args.cap, "available": {k: len(v) for k, v in cand.items()}, "budget_B": budget}
    for k, v in cand.items():
        v.sort(key=lambda x: (x[0], x[1]))
        sel = np.array([(x[2], x[3], x[0]) for x in v[:budget]], dtype=np.int64)  # doc_row, entity, rank in doc
        out[k] = sel
        stats[f"{k}_docs_touched"] = int(len(np.unique(sel[:, 0])))
    np.savez(d / f"variants{sfx}.npz", **out)
    extra = [set() for _ in range(N)]
    for sel in out.values():
        for r, e, _ in sel:
            extra[r].add(int(e))
    if not args.v2:  # union candidate file for one dump
        with open(d / "docs_entities_union.jsonl", "w") as f:
            for r, x in enumerate(order):
                f.write(json.dumps({"id": x, "entities": sorted(linked[r] | extra[r])}) + "\n")
    else:  # only (doc, entity) pairs not already encoded in the v1 union dump
        union = load_jsonl_entities(d / "docs_entities_union.jsonl")
        n_delta = 0
        with open(d / "docs_entities_delta_v2.jsonl", "w") as f:
            for r, x in enumerate(order):
                delta = extra[r] - set(int(e) for e in union.get(x, []))
                if delta:
                    n_delta += len(delta)
                    f.write(json.dumps({"id": x, "entities": sorted(delta)}) + "\n")
        stats["delta_pairs_to_encode"] = n_delta
    stats.update(st)
    stats["linked_pairs"] = int(sum(len(x) for x in linked))
    stats["union_pairs"] = int(sum(len(linked[r] | extra[r]) for r in range(N)))
    json.dump(stats, open(d / f"build_stats{sfx}.json", "w"), indent=1)
    log(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
