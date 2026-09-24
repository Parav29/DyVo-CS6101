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
