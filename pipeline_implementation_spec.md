# Implementation Spec — Hierarchical Ticket Classification Pipeline

**For:** the implementing engineer (Claude Code).
**Goal:** build, from raw data, a three-stage hierarchical classifier that
predicts **Service → Category → Subcategory** for support tickets.

This is NOT a subcategory-only task. All three stages are built from scratch as
modular scripts. The subcategory stage gets the most attention because that is
where accuracy currently degrades, but the pipeline is the deliverable.

---

## 1. Background the implementer needs

- ~129,388 raw tickets, Italian + English free text.
- Prediction is **hierarchical**: first predict Service (2 classes), then
  Category within service, then Subcategory within category. This architecture
  empirically beats a single flat classifier by ~5%, so preserve it.
- Each stage's error is unrecoverable downstream: a wrong Category means the
  Subcategory can never be right. End-to-end accuracy is the product of the
  three stages.
- The data has heavy class imbalance and many "trashbin"/placeholder labels that
  must be removed before training. What counts as a trashbin is provided by a
  human-audited file (see §3), NOT hardcoded.

### Known facts
- Text fields used: `Subject` + `Symptom`. Also use `ProfileFullName` (sender)
  and the parent `Category`/`Service` as features for the subcategory stage.
- Do **NOT** use `SCMTitle` or `CustomerLocation` (14–15% missing, weak signal,
  dropping rows for them harms rare classes).
- `Altro` and same-name placeholder subcategories (e.g. `66-ERP-365FO` →
  `66-ERP-365FO`) are defaults meaning "unclassified," not real subcategories.
- 16 categories are **flat** (exactly one subcategory = the category name).
  These are NOT modeled; subcategory is assigned by rule.

---

## 2. Deliverables (modular scripts)

```
pipeline/
  config.py            # all constants, paths, thresholds
  audit.py             # Stage 0: dump label counts for human review
  clean.py             # Stages 1-3: load, integrity, label drops, text cleaning
  features.py          # Stage 4: TF-IDF + encoders (reusable train/inference)
  train_service.py     # Stage 5a: Service model
  train_category.py    # Stage 5b: Category model (per service)
  train_subcategory.py # Stage 5c: Subcategory models (per category)
  evaluate.py          # Stage 6: per-stage + end-to-end metrics
  pipeline.py          # inference: glue all three stages + rules
  README.md
```

All randomness seeded. All cleaning logic lives in `clean.py`/`features.py` and
is imported by both training and inference — never duplicated.

---

## 3. config.py — single source of truth

```python
RANDOM_SEED = 42
RAW_DATA_PATH = "data/raw_tickets.parquet"
LABEL_DECISIONS_PATH = "data/label_decisions.csv"  # produced by human audit

TEXT_FIELDS = ["Subject", "Symptom"]
SENDER_FIELD = "ProfileFullName"
DROP_FIELDS = ["IncidentNumber", "SCMTitle", "CustomerLocation", "Email"]

PRIMARY_METRIC = "macro_f1"          # optimize this
MIN_SUBCAT_SUPPORT = 50              # per (category, subcategory) pair; tunable
ABSTAIN_CONFIDENCE = 0.40            # below this, subcat -> "unspecified"; tune on val
TEST_SIZE = 0.20
```

---

## 4. Stage-by-stage

### Stage 0 — `audit.py` (run once, human in the loop)
- Output `value_counts()` for `Category`, `Subcategory`, and each
  `(Category, Subcategory)` pair to CSVs.
- A human tags each label in `label_decisions.csv` with one of:
  `REAL | TRASHBIN | DEFAULT | FLAT`.
- The pipeline reads this file. **No label names are hardcoded in code.**
- Also emit the list of flat categories (single subcategory) for verification.

### Stage 1 — load & integrity (`clean.py`)
- Load raw, drop `DROP_FIELDS`.
- Drop rows with null `Service`, `Category`, or `Subcategory`.
- Verify no duplicate `IncidentNumber` / rows.
- Maintain a row-count ledger logged after every drop.

### Stage 2 — apply label decisions (`clean.py`)
Driven by `label_decisions.csv`:
- Drop rows where Category is `TRASHBIN`.
- Drop rows where Subcategory is `DEFAULT` (includes `Altro` and same-name
  placeholders) — **for subcategory training only**; these rows still feed the
  Service and Category models.
- Tag `FLAT` categories; exclude them from subcategory training and record them
  for rule-based inference.

> v1 rule: drop `Altro` globally. A stakeholder may later identify categories
> where `Altro` is meaningful; if so, change its tag in `label_decisions.csv`
> for those categories — no code change needed.

### Stage 3 — text cleaning (`clean.py`)
One reusable `clean_text(s)` function, applied to `Subject` and `Symptom`,
used identically at train and inference. Remove, in order:
1. Email signatures/footers (address blocks, phone lines, `www.scmgroup.com`).
2. Disclaimers (e.g. "this email is from an unusual correspondent…").
3. `[cid:…]` tags, `BEGIN:VCALENDAR…END:VCALENDAR` blocks.
4. Emails, phone numbers, URLs → token or removed.
5. Lowercase, collapse whitespace, strip control chars.

Start from RAW data — do not assume `Symptom` is pre-cleaned. Output combined
field `text = clean(Subject) + " " + clean(Symptom)`.

### Stage 4 — features (`features.py`)
Fit on train only; transform train+test with the fitted objects.
- **TF-IDF** on `text`, unigrams+bigrams. Sweep `max_features`
  {500,1000,2000,5000} on macro-F1.
- **ProfileFullName**: frequency-encode (or hash top-N, bucket rest).
  **Leakage guard:** flag senders appearing in only one subcategory in train;
  confirm metrics hold on multi-subcategory senders and that test senders exist
  in train. Document the check in the README.
- **Category + Service**: passed as features to the subcategory stage.

### Stage 5 — train (three scripts)
Stratified 80/20 split done **after** all cleaning. Test set 100% real.

- **`train_service.py`** — 2-class (Application/Infrastructure). TF-IDF +
  features → Logistic Regression / Linear SVM. Expect ~0.95.
- **`train_category.py`** — Category within each service. Same feature stack.
  Use Logistic Regression (need probabilities downstream). Expect ~0.85.
- **`train_subcategory.py`** — one model per **non-flat** category, trained only
  on that category's `REAL` subcategories with support ≥ `MIN_SUBCAT_SUPPORT`.
  Subcategories below the floor are excluded from the label set; at inference
  they fall into the abstain bucket. Logistic Regression preferred (probabilities
  drive the abstention threshold).

### Stage 6 — `evaluate.py`
- **Decompose:** report each stage in isolation. Critically, evaluate each
  subcategory model **on tickets routed by their TRUE parent category**, so
  subcat quality is measured separately from upstream errors.
- **Primary metric: macro-F1** over REAL subcategories, per category + overall.
  Always report accuracy and weighted-F1 alongside, never alone.
- Report end-to-end (predicted-category-fed) numbers too, to show real-world
  performance including upstream error propagation.
- Per-category table: support, macro-F1, abstention rate.
- **Expect end-to-end accuracy to drop vs. the old 69.5%** because the easy
  `Altro` class (18% of data) is removed. This is correct. Judge by macro-F1.

### `pipeline.py` — inference glue
For a new ticket:
1. clean_text → features.
2. Service model → service.
3. Category model → category (keep probability).
4. If category is FLAT → subcategory = category (rule, no model).
   Else → that category's subcategory model.
5. If subcategory model's top probability < `ABSTAIN_CONFIDENCE` →
   subcategory = "unspecified (review)".
6. Return (service, category, subcategory, confidences).

---

## 5. Guardrails the implementer must respect

- **No synthetic data anywhere, especially not in test/val.**
- **No hardcoded label names** — all label decisions come from
  `label_decisions.csv`.
- **One cleaning code path** shared by train and inference.
- **Fit feature encoders on train only** (no leakage from test).
- **Optimize macro-F1, report accuracy too** — never optimize accuracy alone.
- **Seed everything** for reproducibility.

---

## 6. Open items (flagged, non-blocking)
1. `Altro` deep-dive by stakeholder — may reopen specific categories (handled via
   the audit file, no code change).
2. Confirm all 16 flat categories truly have one subcategory in raw data
   (Stage 0 verifies).
3. ProfileFullName leakage must be validated in Stage 4, not assumed.
