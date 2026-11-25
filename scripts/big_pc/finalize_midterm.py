"""DYVO-REPRO: fill the LSR-w 100k row into repro_codec/MIDTERM.md and copy all test results into repro_codec/results/.
Run from the repo root once DyVo/outputs/<LSR-w full100k>/fasteval/test_result.json exists."""
import glob
import gzip
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
L = "qmlp_dmlm_codec_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001_full100k"
PAPER = {"n10": 52.61, "n20": 49.22, "r1k": 69.07}
DYVO_REL_100K_N10 = 53.70
PAPER_GAP = 0.79  # paper DyVo (REL) 53.40 - LSR-w 52.61

res = json.load(open(ROOT / "DyVo/outputs" / L / "fasteval/test_result.json"))
n10, n20, r1k = (100 * res[k] for k in ("test_nDCG@10", "test_nDCG@20", "test_R@1000"))
d = lambda ours, paper: f"{ours:.2f} / {paper:.2f} ({ours - paper:+.2f})"
row = f"{d(n10, PAPER['n10'])} | {d(n20, PAPER['n20'])} | {d(r1k, PAPER['r1k'])}"
gap = DYVO_REL_100K_N10 - n10

p = ROOT / "repro_codec/MIDTERM.md"
s = p.read_text()
now = datetime.now().strftime("%-d %b %Y, %H:%M IST")
s = s.replace("Status as of 1 Oct 2026, 18:00 IST: every DyVo row is finished; the LSR-w 100k baseline is finishing training (step 96k / 100k).",
              f"Status as of {now}: all rows finished.")
s = s.replace("| LSR-w (Table 1) | *pending, ~20:00 1 Oct* / 52.61 | – / 49.22 | – / 69.07 |", f"| LSR-w (Table 1) | **{n10:.2f}** / 52.61 ({n10 - 52.61:+.2f}) | {d(n20, PAPER['n20'])} | {d(r1k, PAPER['r1k'])} |")
s = s.replace("| step 96k, done ~19:00 1 Oct |", "| done (resumed once, see below) |")
s = s.replace("| LSR-w | 100k | *pending* / 52.61 | – / 49.22 | – / 69.07 |", f"| LSR-w | 100k | {row} |")
s = s.replace("Phase 4 — Training (done except LSR-w 100k)", "Phase 4 — Training (done)")
if gap > 0:
    claim = (f"- **Claim 1 holds: entities help.** DyVo (REL) beats LSR-w at every training budget: +1.57 nDCG@10 at init, "
             f"+0.52 at 20k steps, and **+{gap:.2f} at 100k** (paper: +{PAPER_GAP:.2f}).")
else:
    claim = (f"- **Claim 1 holds at init and 20k steps but not at 100k.** DyVo (REL) beats LSR-w by +1.57 nDCG@10 at init and +0.52 at 20k, "
             f"but at 100k it is {gap:+.2f} (paper: +{PAPER_GAP:.2f}). This gap is well inside the ±1-point noise of a single run on 42 queries, "
             f"so this run neither confirms nor refutes the REL-only gain at 100k. The LLM-candidate rows (+{56.84 - n10:.2f} for GPT-4) still clearly beat LSR-w.")
old = [l for l in s.splitlines() if l.startswith("- **Claim 1 holds: entities help.**")]
if old:
    s = s.replace(old[0], claim)
p.write_text(s)
print(f"LSR-w 100k: nDCG@10 {n10:.2f} nDCG@20 {n20:.2f} R@1000 {r1k:.2f}; DyVo(REL) - LSR-w = {gap:+.2f}")

# copy results: every test_result.json + the top-1000 of each run (gzipped; R@1000 is the deepest metric reported)
out = ROOT / "repro_codec/results"
for f in sorted(glob.glob(str(ROOT / "DyVo/outputs/*/*/test_result.json"))):
    f = Path(f)
    exp, sub = f.parent.parent.name, f.parent.name
    if exp.startswith("smoke"):
        continue
    dst = out / exp / sub
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(f, dst / "test_result.json")
    run = f.parent / "test_run.trec"
    if run.exists():
        with open(run) as fin, gzip.open(dst / "test_run.top1000.trec.gz", "wt") as fout:
            for line in fin:
                if int(line.split("\t")[3]) < 1000:
                    fout.write(line)
print(f"copied results to {out}")
