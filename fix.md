The hierarchical pipeline (`train_service.py`, `train_category.py`,
`features.py`, `clean.py`) is producing a catastrophic regression: the Category
model scores **macro-F1 ≈ 0.22 / accuracy ≈ 0.69**, versus a known-good baseline
of **~0.85 accuracy** from an earlier script (`13b_svc_baseline_experiment.py`).

The data cleaning is correct. The architecture is correct. The regression is a
**plumbing bug in feature handling plus lost class balancing** — not a data or
modeling problem. Do NOT change `clean.py`'s label logic or re-investigate the
data. Fix the specific issues below.

## Root causes (confirmed — fix exactly these)

### 1. The `max_features` sweep evaluates on training data and is meaningless
In `train_category.py` and `train_service.py`, `_best_max_features` /
`_sweep_max_features` call:
```python
X, _, _ = build_features(df_train, df_train, mf, purpose="service_category")
```
This passes `df_train` as BOTH train and test. `build_features` fits a fresh
TF-IDF on every call, so the vectorizer is fit on the same rows it scores. The
resulting CV macro-F1 values (~0.13–0.15) are noise, and the sweep ends up
picking the smallest grid value (500). This is also the source of the
`train=(83520, 501) test=(83520, 501)` log line (test size == train size).

**Fix:** The sweep must fit TF-IDF on a train fold and score on a held-out fold.
Either:
- (preferred, simplest) Remove the sweep entirely and hardcode a large
  `max_features` (see issue #2), OR
- Implement the sweep correctly: split `df_train` into a sub-train/sub-val,
  fit features on sub-train only via `build_features(sub_train, sub_val, mf)`,
  and score the model on sub-val. Never fit and score on the same frame.

### 2. The feature budget is ~100x too small
The new `TFIDF_MAX_FEATURES_GRID` is {500, 1000, 2000, 5000}; the sweep picked
**500**. The known-good `13b` script used **55,000** TF-IDF features
(5k name + 15k subject + 35k symptom) and achieved ~85%. 500 features cannot
separate ~33 Italian-text categories.

**Fix:** Raise the budget dramatically. To match 13b, target ~50,000 features on
the combined text field. If keeping a grid, use something like
{20000, 40000, 60000}; otherwise hardcode ~50000. Confirm the final fitted
matrix has tens of thousands of columns, not ~500.

### 3. Class balancing was dropped at the category/service level
`13b` used `LinearSVC(class_weight="balanced", max_iter=10_000)`. The new
`train_category.py` and `train_service.py` use `LogisticRegression` with **no
`class_weight`** and `max_iter=2000`, which both depresses rare-class recall
(many 0.00 rows) and triggers `ConvergenceWarning`.

**Fix:**
- Add `class_weight="balanced"` to the `LogisticRegression` in BOTH
  `train_category.py` and `train_service.py`.
- Raise `max_iter` to at least 5000 to eliminate the convergence warnings.

## Primary instruction: match the 13b baseline first

Before any redesign, make the new pipeline reproduce 13b's result so we confirm
recovery. Concretely, for the Category stage:
- ~50,000 TF-IDF features (no broken sweep).
- `class_weight="balanced"`, `max_iter >= 5000`.
- A proper train/test split where **test size is clearly smaller than train**
  (e.g. ~20% test).

Keep the rest of the new design (separate `FrequencySenderEncoder`, the
`purpose`-based context features) as-is for now — just fix the three issues
above.

## Add guardrail assertions (so this can't silently recur)
In `build_features` and in each training script, add asserts:
- `X_train.shape[0] == len(df_train)` and `X_test.shape[0] == len(df_test)`.
- `X_train.shape[0] != X_test.shape[0]` unless train and test genuinely have
  equal row counts (they won't) — i.e. assert the final eval is not run with a
  test frame equal in size to the train frame.
- `X_train.shape[1] > 1000` (catch a collapsed feature budget early).
- Log `X_train.shape` and `X_test.shape` immediately before `model.fit`.

## Acceptance criteria
- Category model returns to **~0.85 accuracy** (and a sane macro-F1 in line with
  the historical ~0.59) on the held-out test set.
- No `ConvergenceWarning`.
- Final feature matrix has tens of thousands of columns.
- Test row count != train row count in all logged shapes.

## Do NOT do
- Do not modify the label-decision / cleaning logic in `clean.py`.
- Do not add synthetic data.
- Do not "simplify" the hierarchy into a flat model.
- Do not proceed to tune the subcategory stage until the Category stage matches
  the ~0.85 baseline — stop and report the recovered numbers first.

## After recovery (separate step — wait for human go-ahead)
Once Category is back to ~0.85, we will decide whether to migrate from
LogisticRegression to LinearSVC and tune the sender feature. Do not start that
until the recovery is confirmed.
