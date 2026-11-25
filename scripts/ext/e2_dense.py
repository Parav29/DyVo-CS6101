"""Extension 2, step 3: dense document entity proposals with the authors' LaQue encoder.

For every document: LaQue embedding of (title + text) truncated to 300 word-pieces (always inside DyVo's 512-token
visible window), top-30 entities over the full 5.28M LaQue catalog. Embeddings are kept so the similarity of any
(document, entity) pair -- e.g. for mention-evidence candidates -- can be computed later.

Output: scratch/ext/e2/dense_top30.npz {ids int32 [N,30], scores float16 [N,30]}, doc_emb.npy float16 [N,768]
"""
import json
import time

import numpy as np
import torch

from common import EXT, KB, N_ENT, log


def main():
    from transformers import AutoModel, AutoTokenizer
    import ir_datasets
    d = EXT / "e2"
    t0 = time.time()
    name = "lsr42/distilbert-base-uncased-dense-laque"
    tok = AutoTokenizer.from_pretrained(name)
    enc = AutoModel.from_pretrained(name).cuda().eval().half()
    table = torch.load(str(KB / "entity_embs_laque.pt"), map_location="cpu", mmap=True)
    chunks = [table[i:i + 1_000_000].cuda().half() for i in range(0, N_ENT, 1_000_000)]
    log(f"LaQue table on GPU ({time.time()-t0:.0f}s)")
    docs = {}
    for doc in ir_datasets.load("codec").docs_iter():
        docs[doc.doc_id] = " ".join(x for x in (doc.title, doc.text) if x is not None)
    order = json.load(open(EXT / "index_linked" / "doc_ids.json"))
    N = len(order)
    emb = np.lib.format.open_memmap(d / "doc_emb.npy", mode="w+", dtype=np.float16, shape=(N, 768))
    top_ids = np.zeros((N, 30), np.int32); top_s = np.zeros((N, 30), np.float16)
    B = 128
    with torch.no_grad():
        for a in range(0, N, B):
            t = tok([docs[i] for i in order[a:a + B]], truncation=True, max_length=300, padding=True,
                    return_tensors="pt").to("cuda")
            h = enc(**t).last_hidden_state
            m = t["attention_mask"].unsqueeze(-1).half()
            v = (h * m).sum(1) / m.sum(1)
            emb[a:a + len(v)] = v.cpu().numpy()
            vs, ids = [], []
            for off, c in zip(range(0, N_ENT, 1_000_000), chunks):
                s, i = (v @ c.T).float().topk(30, dim=1)
                vs.append(s); ids.append(i + off)
            s, i = torch.cat(vs, 1), torch.cat(ids, 1)
            o = s.topk(30, dim=1)
            top_ids[a:a + len(v)] = torch.gather(i, 1, o.indices).cpu().numpy()
            top_s[a:a + len(v)] = o.values.cpu().numpy()
            if (a // B) % 500 == 0:
                log(f"{a}/{N} ({time.time()-t0:.0f}s)")
    emb.flush()
    np.savez(d / "dense_top30.npz", ids=top_ids, scores=top_s)
    log(f"done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
