# compares fast_eval matmul scoring with HFTrainer.evaluation_loop_iterative_vectorized's formula on real encodings
import os, sys, torch
sys.path.insert(0, os.path.abspath("DyVo")); os.chdir("DyVo")
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Subset
E="qmlp_dmlm_emlm_laque_codec_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001_entw_0.05"
with initialize_config_dir(config_dir=os.path.abspath("lsr/configs"), version_base="1.2"):
    c = compose(config_name="config", overrides=[f"+experiment={E}_course20k_eval_gpt4"])
m = f"outputs/{E}_course20k/final/model"
c.model.query_encoder_checkpoint=m+"/query_encoder/model.safetensors"; c.model.doc_encoder_checkpoint=m+"/doc_encoder/model.safetensors"
model = instantiate(c.model).cuda().eval(); col = instantiate(c.data_collator); ds = instantiate(c.test_dataset)
dv = lambda b: {k: v.cuda() for k, v in b.items()}
qb = next(iter(DataLoader(ds.queries, batch_size=64, collate_fn=col)))
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    qi, qw, _ = model.encode_queries(**dv(qb["queries"]), to_dense=False)
qi_l, qw_l = list(qi), list(qw)
union = torch.unique(qi.long()); lk = torch.full((30522+5277581,), -1, dtype=torch.long, device="cuda"); lk[union]=torch.arange(len(union), device="cuda"); U=len(union)
Q = torch.zeros(len(qi_l), U+1, device="cuda")
for r in range(len(qi_l)): Q[r].index_add_(0, lk[qi_l[r].long()], qw_l[r].float())
Q = Q[:, :U]
maxd = 0; maxrel = 0
for bi, b in enumerate(DataLoader(Subset(ds.docs, range(100000, 100160)), batch_size=16, collate_fn=col)):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        di, dw, _ = model.encode_docs(**dv(b["doc_groups"]), to_dense=False)
        ref = []
        for q in range(len(qi_l)):
            mm = qi_l[q].unsqueeze(0).unsqueeze(-1) == di.unsqueeze(-2)
            it = qw_l[q].unsqueeze(0).unsqueeze(-1) * dw.unsqueeze(-2)
            ref.append((mm*it).sum(-1).sum(-1).float())
    ref = torch.stack(ref, 1)
    cols = lk[di.long()]; cols = torch.where(cols<0, torch.full_like(cols, U), cols)
    D = torch.zeros(di.shape[0], U+1, device="cuda"); D.scatter_add_(1, cols, dw.float())
    fast = D[:, :U] @ Q.T
    maxd = max(maxd, (fast-ref).abs().max().item()); maxrel = max(maxrel, ((fast-ref).abs()/(ref.abs()+1e-3)).max().item())
    print(bi, "ref mean", ref.mean().item(), "max abs diff", (fast-ref).abs().max().item())
print("MAX abs diff", maxd, "max rel diff", maxrel)
