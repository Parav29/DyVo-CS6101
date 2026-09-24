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
