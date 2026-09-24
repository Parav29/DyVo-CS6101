"""Build a query entity-candidate file for the paper's Table 2/3 rows:
REL entities for TRAINING query ids + entities from a chosen retriever/LLM source for TEST query ids.

Why: the authors' LLM/BM25/LaQue candidate files only cover the evaluation topics (test-only), while every official
experiment config points `query_ent_path` at the REL file. The paper uses GPT-4/Mixtral/... candidates for the test
queries at inference, and training always uses REL (see NOTES.md "LLMs and API keys").

Output: one {"id": ..., "entities": [...]} line per id in the REL file (same line count), readable by BERTCandidate.
Test ids missing from the source fall back to REL (and are counted/logged).

Usage (from DyVo/):
  python -m lsr.entity.merge_candidates --dataset codec --source gpt4
  python -m lsr.entity.merge_candidates --dataset robust04 --source gpt4 \
      --source-file dyvo_data/robust04/queries_entities_gpt4_remapped.jsonl     # e.g. after re-mapping ids
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("merge_candidates")
SOURCES = ("gpt4", "mistral", "bm25", "laque", "human")


def read_jsonl(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows[str(obj["id"])] = [int(e) for e in obj["entities"]]
    return rows


def default_paths(data_dir, dataset, source):
    base = Path(data_dir) / dataset
    rel = base / "queries_test_train_entities_rel.jsonl"
    if source == "human":
        src = base / "queries_test_train_entities_human.jsonl"
        out = base / "queries_test_train_entities_human_merged.jsonl"  # never overwrite the authors' human file
    else:
        src = base / f"queries_entities_{source}.jsonl"
        out = base / f"queries_test_train_entities_{source}.jsonl"
    test_ids = base / "queries_entities_gpt4.jsonl"  # the LLM files are test-only: their ids define the test set
    return rel, src, out, test_ids


def merge(rel_path, source_path, test_ids_path, output_path):
    rel = read_jsonl(rel_path)
    src = read_jsonl(source_path)
    test_ids = set(read_jsonl(test_ids_path)) if test_ids_path else set(src)
    unknown = test_ids - set(rel)
    if unknown:
        raise ValueError(f"{len(unknown)} test ids are not in the REL file, e.g. {sorted(unknown)[:5]}")
    out_rows, n_test, n_fallback, fallback_ids = [], 0, 0, []
    for qid, rel_ents in rel.items():
        if qid in test_ids:
            n_test += 1
            if qid in src:
                ents = src[qid]
            else:
                ents, n_fallback = rel_ents, n_fallback + 1
                fallback_ids.append(qid)
        else:
            ents = rel_ents
        out_rows.append({"id": qid, "entities": ents})
    for p in (rel_path, source_path, test_ids_path):
        if p and Path(p).resolve() == Path(output_path).resolve():
            raise ValueError(f"refusing to overwrite an input file: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    stats = {"output": str(output_path), "lines": len(out_rows), "train_ids": len(out_rows) - n_test,
             "test_ids": n_test, "test_ids_from_source": n_test - n_fallback,
             "test_ids_fallback_to_rel": n_fallback, "fallback_ids": fallback_ids}
    if n_fallback:
        logger.warning("%d test ids had no entry in %s and fell back to REL: %s", n_fallback, source_path, fallback_ids)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["codec", "robust04", "wapo"])
    ap.add_argument("--source", required=True, choices=SOURCES)
    ap.add_argument("--data-dir", default="dyvo_data")
    ap.add_argument("--rel-file", default=None, help="override the REL (train+test) file")
    ap.add_argument("--source-file", default=None, help="override the source file for test ids")
    ap.add_argument("--test-ids-file", default=None, help="file whose ids define the test set (default: the GPT-4 file)")
    ap.add_argument("--output", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rel, src, out, test_ids = default_paths(args.data_dir, args.dataset, args.source)
    rel = args.rel_file or rel
    src = args.source_file or src
    out = args.output or out
    test_ids = args.test_ids_file or test_ids
    if args.dry_run:
        print(f"[dry-run] would merge REL={rel} + {args.source}={src} (test ids from {test_ids}) -> {out}")
        return 0
    stats = merge(rel, src, test_ids, out)
    print(json.dumps({k: v for k, v in stats.items() if k != "fallback_ids"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
