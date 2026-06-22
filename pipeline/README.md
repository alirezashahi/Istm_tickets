# Hierarchical Ticket Classification Pipeline

Predicts **Service → Category → Subcategory** for SCM support tickets (Italian + English).

---

## Directory layout

```
pipeline/
  config.py            # all constants, paths, thresholds
  audit.py             # Stage 0: dump label counts for human review
  clean.py             # Stages 1–3: load, integrity, label drops, text cleaning
  features.py          # Stage 4: TF-IDF + encoders (reusable train/inference)
  train_service.py     # Stage 5a: Service classifier
  train_category.py    # Stage 5b: Category classifier (per service)
  train_subcategory.py # Stage 5c: Subcategory classifiers (per category)
  evaluate.py          # Stage 6: per-stage + end-to-end metrics
  pipeline.py          # Inference: glue all three stages + rules
  data/
    label_decisions.csv         # human-audited label tags (REAL/TRASHBIN/DEFAULT/FLAT)
    audit/                      # outputs of audit.py
  models/
    service_model.joblib
    category_model.joblib
    subcategory/                # one .joblib per non-flat category + _index.joblib
```

---

## Setup

```bash
pip install scikit-learn pandas numpy scipy joblib
```

Optionally, to use parquet:
```bash
pip install pyarrow
```

---

## Running the pipeline

### Step 0 — Audit (run once; human review required)

```bash
python audit.py
```

Writes CSVs to `data/audit/` and a `data/label_decisions.csv` template.  
**Human action required:** open `label_decisions.csv` and tag every row as one of:

| Tag | Meaning |
|-----|---------|
| `REAL` | A real, meaningful subcategory — include in training |
| `TRASHBIN` | The whole category is junk — drop rows |
| `DEFAULT` | A placeholder subcategory (Altro, same-name) — drop from subcat training |
| `FLAT` | Exactly one subcategory per category — assign by rule, no model needed |

Rows pre-filled automatically (based on rules): `FLAT` categories, `Altro`, same-name placeholders. Fill in the blanks.

### Step 1–3 (data loading, cleaning) — embedded in training scripts

No separate run needed; `clean.py` is imported by all training scripts.

### Step 4–5 — Train

```bash
python train_service.py
python train_category.py
python train_subcategory.py
```

All three can be run independently. Subcategory training depends only on the cleaned data, not on the trained service/category models.

### Step 6 — Evaluate

```bash
python evaluate.py
```

Reports:
- Service stage: macro-F1, weighted-F1, accuracy
- Category stage: same, per service and overall (true-service-routed)
- Subcategory stage: same, per category and overall (true-category-routed)
- End-to-end: same, with predicted routing (real-world numbers)
- Per-category table: support, macro-F1, abstention rate

**Why end-to-end accuracy drops vs the old 69.5%:**  
`Altro` (≈18% of tickets) was a trivially predictable default class that inflated accuracy.  
It is now excluded from training. Judge model quality by **macro-F1**, not accuracy.

### Inference

```python
from pipeline import Pipeline

p = Pipeline.load()
result = p.predict(
    subject="Workflow fermi nella worklist",
    symptom="Due workflow sembrano bloccati...",
    sender="CARLO VANNUCCI",
)
print(result.service, result.category, result.subcategory)
print(result.confidences)  # {"service": 0.99, "category": 0.87, "subcategory": 0.72}
print(result.is_flat)      # True if category is flat (no model, rule-based)
```

Or from the CLI:

```bash
python pipeline.py --subject "Workflow bloccato" --symptom "Worklist non risponde" --sender "MARIO ROSSI"
```

---

## Architecture

```
Ticket text
  └─ clean_text()                    (clean.py — same function at train + inference)
       └─ TF-IDF + sender + context  (features.py — fit on train only)
            ├─ Service model          → Application | Infrastructure
            ├─ Category model (×2)    → one per service
            └─ Subcategory model (×N) → one per non-flat category
                                         ↳ no model for category → "unspecified (review)"
```

**Flat categories** (single subcategory = category name) are handled by a rule:  
`subcategory = category`. No model is trained for them.

---

## Leakage

`ProfileFullName` is used as a frequency-encoded feature. This carries a leakage risk:
if a sender always submits the same subcategory type, the model could memorise sender → subcategory.

`features.check_sender_leakage()` runs automatically during training and logs:

| Metric | What it tells you |
|--------|-------------------|
| `n_train_senders_single_subcat` | Senders seen with only one subcat in train — highest leakage risk |
| `test_fraction_multi_subcat_sender` | Fraction of test rows where the sender has ≥2 subcats in train — "clean" test population |
| `test_oov_sender_fraction` | Fraction of test senders not seen in train → bucket to `__other__` |

**How to validate:** compare macro-F1 on multi-subcategory-sender test rows vs all test rows.  
If the gap is large (>5pp), sender feature is carrying disproportionate signal and should be down-weighted or removed.

---

## Key configuration (config.py)

| Constant | Default | Effect |
|----------|---------|--------|
| `MIN_SUBCAT_SUPPORT` | 50 | Subcategories with fewer train rows are excluded from the label set |
| `SENDER_TOP_N` | 200 | Top-N senders get explicit encoding; rest → `__other__` |
| `TFIDF_MAX_FEATURES_GRID` | [500,1000,2000,5000] | CV sweep; best value auto-selected per model |
| `TEST_SIZE` | 0.20 | Stratified hold-out fraction |
| `RANDOM_SEED` | 42 | All randomness seeded |

---

## Guardrails

- No label names are hardcoded. All label decisions come from `label_decisions.csv`.
- One cleaning code path (`clean_text`) is shared by training and inference.
- Feature encoders are fit on train only (no test leakage).
- Primary metric is macro-F1. Accuracy is always reported alongside, never optimised alone.
- No synthetic data anywhere.
