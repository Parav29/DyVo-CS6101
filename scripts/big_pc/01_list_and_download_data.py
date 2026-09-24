"""Big-PC step 1: list lsr42/dyvo_data (a DATASET repo) with sizes, check free disk, then download selectively.

The README's `huggingface-cli download lsr42/dyvo_data` is wrong: it lacks `--repo-type dataset` and lands in the HF
cache. This uses `hf download lsr42/dyvo_data --repo-type dataset --local-dir DyVo/dyvo_data --include <pattern>`.
Idempotent: files already present with the right size are skipped.

Usage (from the project root):
  python scripts/big_pc/01_list_and_download_data.py --dry-run
  python scripts/big_pc/01_list_and_download_data.py --datasets codec,robust04,wapo --embeddings laque
  python scripts/big_pc/01_list_and_download_data.py --embeddings laque,wiki2vec,blink,dpr,jds,bert_agg   # Table 3
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO_ID = "lsr42/dyvo_data"
CACHED_LISTING = ROOT / "logs" / "hf_listing.tsv"  # written on the build machine (S5.0)
EMBEDDINGS = ["laque", "wiki2vec", "blink", "dpr", "jds", "bert_agg"]
DATASETS = ["codec", "robust04", "wapo"]


def list_repo(offline=False):
    if offline:
        if not CACHED_LISTING.exists():
            return {}
        return {p: int(s) for p, s in (l.split("\t") for l in CACHED_LISTING.read_text().splitlines() if l.strip())}
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile
    return {x.path: x.size for x in HfApi().list_repo_tree(REPO_ID, repo_type="dataset", recursive=True)
            if isinstance(x, RepoFile)}


def select(listing, datasets, embeddings, entity_list=True, init=True):
    chosen = []
    for path in sorted(listing):
        top = path.split("/")[0]
        if top in datasets:
            chosen.append(path)
        elif top == "dyvo_init" and init:
            chosen.append(path)
        elif top == "knowledge_base":
            name = Path(path).name
            if name == "entity_list.json" and entity_list:
                chosen.append(path)
            elif any(name == f"entity_embs_{e}.pt" for e in embeddings):
                chosen.append(path)
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="codec,robust04,wapo")
    ap.add_argument("--embeddings", default="laque", help=f"comma list from {EMBEDDINGS} (or 'all')")
    ap.add_argument("--no-entity-list", action="store_true", help="skip knowledge_base/entity_list.json (129 MB)")
    ap.add_argument("--no-init", action="store_true", help="skip dyvo_init/ (MSMARCO-pretrained LSR checkpoints)")
    ap.add_argument("--local-dir", default=str(ROOT / "DyVo" / "dyvo_data"))
    ap.add_argument("--safety-gb", type=float, default=10.0, help="free space to keep after download")
    ap.add_argument("--dry-run", action="store_true", help="plan only; uses the cached listing, no network")
    args = ap.parse_args()

    datasets = [d for d in args.datasets.split(",") if d]
    embeddings = EMBEDDINGS if args.embeddings == "all" else [e for e in args.embeddings.split(",") if e]
    for d in datasets:
        assert d in DATASETS, d
    for e in embeddings:
        assert e in EMBEDDINGS, e
    listing = list_repo(offline=args.dry_run)
    if not listing:
        print("no listing available (dry-run without logs/hf_listing.tsv)")
        return 0 if args.dry_run else 1
    chosen = select(listing, datasets, embeddings, not args.no_entity_list, not args.no_init)
    local = Path(args.local_dir)
    todo = [p for p in chosen if not ((local / p).exists() and (local / p).stat().st_size == listing[p])]
    need = sum(listing[p] for p in todo)
    total = sum(listing[p] for p in chosen)
    print(f"repo total {sum(listing.values()) / 1e9:.2f} GB; selected {len(chosen)} files = {total / 1e9:.2f} GB; "
          f"to download {len(todo)} files = {need / 1e9:.2f} GB")
    for p in chosen:
        print(f"  {'TODO' if p in todo else 'have'}  {listing[p] / 1e9:8.3f} GB  {p}")
    local.mkdir(parents=True, exist_ok=True) if not args.dry_run else None
    free = shutil.disk_usage(local if local.exists() else ROOT).free
    print(f"free disk: {free / 1e9:.1f} GB (need {need / 1e9:.1f} GB + {args.safety_gb} GB margin)")
    if need + args.safety_gb * 1e9 > free:
        print("NOT ENOUGH DISK: drop embeddings (--embeddings laque) or free space")
        return 0 if args.dry_run else 1
    for p in todo:
        cmd = ["hf", "download", REPO_ID, "--repo-type", "dataset", "--local-dir", str(local), "--include", p]
        if args.dry_run:
            print("[dry-run]", " ".join(cmd))
            continue
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        got = (local / p).stat().st_size
        if got != listing[p]:
            print(f"SIZE MISMATCH for {p}: {got} != {listing[p]}")
            return 1
    print("download plan complete" + (" (dry-run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
