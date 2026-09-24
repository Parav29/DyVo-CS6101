"""Load the smoke data through the OFFICIAL readers/datasets and print counts (S4.1 check)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from lsr.datasets import TripletIDDistilDataset  # noqa: E402
from lsr.datasets.prediction_dataset import PredictionDataset  # noqa: E402
from lsr.entity.bert_candidates import BERTCandidate  # noqa: E402
from lsr.utils.dataset_utils import read_ce_score, read_collection, read_qrels, read_queries  # noqa: E402

D = str(HERE)
docs = read_collection(f"{D}/docs.tsv")
train_q = read_queries(f"{D}/train_queries.tsv")
qrels = read_qrels(f"{D}/train_qrels.json")
ce = read_ce_score(f"{D}/train_monot5_scores.json")
assert all(s >= 0 for q in ce.values() for s in q.values()), "teacher scores must be non-negative (normalize=std)"
train = TripletIDDistilDataset(f"{D}/docs.tsv", f"{D}/train_queries.tsv", f"{D}/train_qrels.json",
                               f"{D}/train_monot5_scores.json")
ex = train[0]
assert len(ex["doc_text"]) == 2 and len(ex["score"]) == 2
test = PredictionDataset(f"{D}/test_qrels.json", f"{D}/test_queries.tsv", f"{D}/docs.tsv", num_documents=-1,
                         query_field=["query"], doc_field=["text"])
cands = BERTCandidate(query_ent_path=f"{D}/queries_entities.jsonl", doc_ent_path=f"{D}/docs_entities.jsonl")
n_q_ent = sum(1 for q, _ in train_q if cands.retrieve_candidates("", q, "query")[0])
print(f"docs={len(docs)} train_queries={len(train_q)} train_qrels={len(qrels)} ce_queries={len(ce)} "
      f"train_dataset={len(train)} test_queries={len(test.queries)} test_docs={len(test.docs)} "
      f"test_qrels={len(test.qrels)} train_queries_with_entities={n_q_ent}")
print("READERS OK")
