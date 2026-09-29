# ERRORS

Format:
```
## E<id> | task <task-id> | attempts: <n> | status: open|fixed|blocked
- symptom:
- hypothesis:
- tried:
- result:
```

## E1 | task S2.3 | attempts: 1 | status: fixed
- symptom: `tests/test_loss.py::test_loss_finite_and_grads_reach_everything` → `tensor(nan)` from `EntityDistilKLLoss(normalize="std")`.
- hypothesis: "std" is not a real option; it falls into `F.normalize(labels)` (L2). The test's teacher label -1.0 stays negative, so KLDivLoss's `xlogy(target, target)` is NaN. Not a model bug.
- tried: use non-negative teacher scores in the test (the real configs need non-negative MonoT5 scores anyway); added `test_std_normalize_requires_nonnegative_teacher_scores` documenting the pitfall; big-PC `02_check_data.py` will check the real score files for negatives.
- result: 27 passed.

## E2 | task S4.2 | attempts: 1 | status: fixed
- symptom: `python -m lsr.train +experiment=smoke_dyvo` → `pydantic_core._pydantic_core.ValidationError: 1 validation error for Settings / start_method  Extra inputs are not permitted` at `lsr/train.py:32 wandb.Settings(start_method="fork")`.
- hypothesis: wandb 0.30 (latest, pulled unpinned) removed `start_method`; the 2024 code targets an older wandb.
- tried: pin `wandb==0.16.6` (no code change) and refresh requirements-lock.txt.
- result: smoke_dyvo and smoke_lsrw both exit 0.

## E3 | task S5.2 | attempts: 1 | status: fixed
- symptom: all 28 generated configs failed composition: `In 'config': Could not find 'loss/triplet_margin'`.
- hypothesis: my DYVO-REPRO header was inserted above `# @package _global_`; Hydra only honours the package directive on the first line, so the experiment was merged under `experiment.` and the base defaults (triplet_margin, splade, toy_triplets) were never overridden.
- tried: moved `# @package _global_` back to line 1.
- result: 73/73 compose.

## E4 | task S5.2 | attempts: 1 | status: fixed
- symptom: (static) Wiki2Vec configs set encoder `entity_dim: 768` while `entity_embs_wiki2vec.pt` is 300-d (from file size) → no projection → matmul shape error at step 1 on the big PC.
- hypothesis: upstream config copy-paste bug.
- tried: set 300 for the affected encoders (3 official + 3 new configs); added scripts/check_entity_dims.py to verify.sh; confirmed the checker flags the unfixed file.
- result: 53 DyVo configs checked, 0 mismatches.

## E5 | task big-PC setup | attempts: 2 | status: fixed
- symptom: `00_setup_env.sh` → pip `ResolutionImpossible`: lock pins `protobuf==4.25.9` but also `googleapis-common-protos==1.75.3` (needs protobuf>=6.33.5).
- hypothesis: the lock is a full `pip freeze` of the build machine, which pip never re-checked; wandb 0.16.6 needs protobuf 4.
- tried: install the lock with `--no-deps` (exact replica of the build env). Also needed Python 3.10 (machine has only 3.13): `uv python install 3.10`, passed via `--python`.
- result: env OK (torch 2.2.2+cu121, CUDA visible); `pip check` lists only the two known protobuf mismatches; `verify.sh` PASSED on the big PC (129 s).

## E6 | task big-PC data | attempts: 1 | status: fixed
- symptom: ir_datasets says the CODEC corpus is "available on request".
- hypothesis: outdated; the CODEC README now links a public copy.
- tried: downloaded `huggingface.co/datasets/macavaney/codec/documents.jsonl.gz` (sha256 c567900f…ab90), gunzipped to `~/.ir_datasets/codec/v1/comets_documents.jsonl`.
- result: 729,824 docs, ir_datasets loads it, 5917/5917 qrel docs found; `02_check_data.py --experiments priority1` → 0 errors.

## E7 | task big-PC verify | attempts: 1 | status: fixed
- symptom: `verify.sh` L3 smoke_dyvo → `wandb.sdk.service.service.ServiceStartTimeoutError: Timed out waiting for wandb service to start after 30.0 seconds`. Whole gate took ~30 min instead of ~2 (shared machine, load ≈ 16).
- hypothesis: machine load, not code; real runs would hit the same 30 s timeout at start-up.
- tried: `WANDB__SERVICE_WAIT=300` exported in `verify.sh` and `03_train.sh`.
- result: see logs/verify_bigpc.log.
