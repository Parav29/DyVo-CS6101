"""Level-0 static check: in every DyVo experiment config, both encoders' `entity_dim` must equal the true width of the
entity embedding table they load. Otherwise no projection layer is created and the first forward pass fails with a
shape mismatch (this caught the upstream Wiki2Vec configs, see NOTES.md).

True widths are derived from the HF file sizes (all tables have 5,277,xxx float32 rows):
  laque/dpr/jds/bert_agg 16.21 GB -> 768, blink 21.62 GB -> 1024, wiki2vec 6.33 GB -> 300; smoke tables by name.
Composes configs with Hydra's compose API (no training, nothing loaded). Usage: python scripts/check_entity_dims.py
"""
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "DyVo"
KNOWN_DIMS = {"entity_embs_laque.pt": 768, "entity_embs_dpr.pt": 768, "entity_embs_jds.pt": 768,
              "entity_embs_bert_agg.pt": 768, "entity_embs_blink.pt": 1024, "entity_embs_wiki2vec.pt": 300,
              "entity_embs_tiny32.pt": 32, "entity_embs_tiny16.pt": 16}


def main():
    os.chdir(REPO)
    from hydra import compose, initialize_config_dir

    checked, bad, skipped = 0, [], []
    with initialize_config_dir(config_dir=str(REPO / "lsr/configs"), version_base="1.2"):
        for f in sorted(glob.glob(str(REPO / "lsr/configs/experiment/*.yaml"))):
            name = Path(f).stem
            cfg = compose(config_name="config", overrides=[f"+experiment={name}"])
            if "entity_embedding_model" not in cfg:
                continue
            emb = Path(cfg.entity_embedding_model.entity_emb_path).name
            if emb not in KNOWN_DIMS:
                skipped.append((name, emb))
                continue
            want = KNOWN_DIMS[emb]
            for side in ("query_encoder", "doc_encoder"):
                got = cfg.model[side].config.get("entity_dim")
                if got != want:
                    bad.append(f"{name}: {side}.config.entity_dim={got} but {emb} is {want}-d")
            checked += 1
    for b in bad:
        print("MISMATCH:", b)
    for n, e in skipped:
        print(f"skipped (unknown table {e}): {n}")
    print(f"entity_dim check: {checked} DyVo configs checked, {len(bad)} mismatches, {len(skipped)} skipped")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
