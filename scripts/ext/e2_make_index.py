"""Materialise one Extension 2 document variant as a standalone index directory (same layout as dump_index.py,
plus the memory-mapped CSC), so that Extension 1 can be re-run on the repaired index (plan step 8: recompute utility
targets after the index changes).

variant index = baseline postings (words + linked entities) + the variant's added entity postings
                (taken from the union dump and, for v2, the delta dump).
  python scripts/ext/e2_make_index.py --variant grounded --cap 5 --v2 --out index_grounded_v2
"""
import argparse
import json
import time

import numpy as np
import scipy.sparse as sp

from common import EXT, WORD_VOCAB, N_ENT, V_TOTAL, log
from e2_run import entity_postings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--cap", type=int, default=5)
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    out = EXT / args.out
    out.mkdir(exist_ok=True)
    base = EXT / "index_linked"
    order = json.load(open(base / "doc_ids.json"))
    N = len(order)
    ur, ue, uw = entity_postings(EXT / "index_union")
    if args.v2:
        rowmap = {x: i for i, x in enumerate(order)}
        dorder = json.load(open(EXT / "index_delta_v2" / "doc_ids.json"))
        dr, de, dw = entity_postings(EXT / "index_delta_v2")
        ur = np.concatenate([ur, np.array([rowmap[dorder[r]] for r in dr], dtype=ur.dtype)])
        ue, uw = np.concatenate([ue, de]), np.concatenate([uw, dw])
    sel = dict(np.load(EXT / "e2" / f"variants{'_v2' if args.v2 else ''}.npz"))[args.variant]
    sel = sel[sel[:, 2] < args.cap]
    vkey = sel[:, 0] * N_ENT + sel[:, 1]
    m = np.isin(ur.astype(np.int64) * N_ENT + ue, vkey)
    log(f"{args.variant}: {len(sel)} added candidates, {int(m.sum())} positive postings ({time.time()-t0:.0f}s)")
    ip = np.load(base / "indptr.npy")
    A = sp.csr_matrix((np.load(base / "w.npy").astype(np.float32), np.load(base / "ids.npy").astype(np.int32),
                       ip.astype(np.int32)), shape=(N, V_TOTAL))
    add = sp.csr_matrix((uw[m], (ur[m], ue[m] + WORD_VOCAB)), shape=(N, V_TOTAL))
    M = (A + add).tocsr()  # additions never overlap existing postings (they exclude linked entities)
    M.sort_indices()
    assert M.nnz == A.nnz + add.nnz, (M.nnz, A.nnz, add.nnz)
    np.save(out / "ids.npy", M.indices.astype(np.int32)); np.save(out / "w.npy", M.data.astype(np.float16))
    np.save(out / "indptr.npy", M.indptr.astype(np.int64))
    json.dump(order, open(out / "doc_ids.json", "w"))
    C = M.tocsc()
    np.save(out / "csc_data.npy", C.data.astype(np.float32)); np.save(out / "csc_indices.npy", C.indices.astype(np.int32))
    np.save(out / "csc_indptr.npy", C.indptr.astype(np.int32))
    json.dump({"variant": args.variant, "cap": args.cap, "v2": args.v2, "postings_total": int(M.nnz),
               "added_postings": int(add.nnz)}, open(out / "stats.json", "w"), indent=1)
    log(f"written {out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
