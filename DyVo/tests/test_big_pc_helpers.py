"""The big-PC experiment list covers every paper row and maps to existing configs."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "big_pc"))
import experiments as ex  # noqa: E402


def test_every_paper_experiment_has_a_config_and_a_paper_row():
    names = ex.all_paper_experiments()
    assert len(names) == len(set(names)) == 46
    for n in names:
        assert (ROOT / "DyVo/lsr/configs/experiment" / f"{n}.yaml").exists(), n
        ds, method, l1 = ex.describe(n)
        assert ds in ex.PAPER[(method, l1)], n
    covered = {ex.describe(n)[1:] for n in names}
    assert covered == set(ex.PAPER)


def test_reference_numbers_spot_check():
    assert ex.PAPER[("DyVo (GPT4)", "1e-5")]["codec"] == (56.46, 53.72, 74.47)
    assert ex.PAPER[("LSR-w", "1e-5")]["robust04"][0] == 49.13
    assert ex.describe("smoke_dyvo") is None
