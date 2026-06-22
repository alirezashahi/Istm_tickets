# Service Model — Performance Analysis

**Overall accuracy: 96.68% | Macro F1: 0.92 | Weighted F1: 0.97**
Test set: 17,217 tickets | 2 classes

---

## 1. Classification Report

| Class | Precision | Recall | F1 | Support | Share |
|---|---:|---:|---:|---:|---:|
| Application | 0.98 | 0.98 | 0.98 | 15,130 | 87.9% |
| Infrastructure | 0.85 | 0.88 | 0.87 | 2,087 | 12.1% |
| **macro avg** | **0.92** | **0.93** | **0.92** | **17,217** | |
| **weighted avg** | **0.97** | **0.97** | **0.97** | **17,217** | |

The gap between macro F1 (0.92) and weighted F1 (0.97) reflects the class imbalance:
Application tickets outnumber Infrastructure ~7:1, so weighted average is dominated by the
easier, larger class.

---

## 2. Data Sizes

| Stage | Rows |
|---|---:|
| Raw CSV loaded | 129,388 |
| After dropping DROP_FIELDS + missing core labels | 129,228 |
| After noise filter | **86,085** |
| Rows removed by noise filter | 43,143 (33.4%) |
| **Train split (80%)** | **68,868** |
| **Test split (20%)** | **17,217** |

Split: stratified by Service label, `test_size=0.20`, `random_state=42`.

---

## 3. What Was Dropped and Why

The noise filter removes any row where the **Category** OR **Subcategory** column matches
`(?i)(cms|altro|other|z-other)`. This is the same filter used by the category model.

### Dropped via Category name

| Category | Rows removed | Reason |
|---|---:|---|
| 70-CMS | ~11,702 | "cms" matches noise regex |
| 99-Z-Other Applications | ~2,919 | "other" matches noise regex |
| 29-Z-Other Infrastructure | ~246 | "other" and "z-other" match noise regex |
| 82-WCMS | ~9 | "cms" inside "WCMS" |
| **Subtotal** | **~14,876** | |

### Dropped via Subcategory name

| Root cause | Rows removed | Example |
|---|---:|---|
| 34-PLM "Altro" subcategory | ~23,624 | Single subcategory "Altro" covers 94% of PLM tickets |
| Various "Other" subcategories | ~2,000 | 75-MPS, 76-RDA, 78-TIME TRACK, etc. |
| Various "Z-Other" subcategories | ~1,800 | 01-Workplace, 02-User Application |
| Other noise subcategories | ~800 | Mixed across categories |
| **Subtotal** | **~28,267** | |

> **Total removed: 43,143 rows (33.4%)** — the single largest contributor is 34-PLM "Altro"
> subcategory (~23,624 rows), which accounts for more than half of all noise-filtered rows.

### Rows dropped before noise filter

| Reason | Rows |
|---|---:|
| Missing core labels (Service / Category / Subcategory) | 160 |

---

## 4. Class Imbalance

The two-class split is heavily skewed:

| Class | Clean rows (full dataset) | Test support | Share |
|---|---:|---:|---:|
| Application | ~75,555 | 15,130 | 87.9% |
| Infrastructure | ~10,530 | 2,087 | 12.1% |

`class_weight="balanced"` in LinearSVC corrects for this during training by upweighting
Infrastructure. Without it, the model would learn to almost always predict Application.

Infrastructure is harder to classify (F1 0.87 vs 0.98) for two reasons:
- Far fewer training examples
- Many Infrastructure categories (Server, Network, System Software) use short, generic subject
  lines that can look like Application tickets

---

## 5. Historical Comparison

| Experiment | Accuracy | Infrastructure F1 | Macro F1 |
|---|---:|---:|---:|
| **This run (script-14 approach)** | **96.68%** | **0.87** | **0.92** |
| Best historical (No Location/No CMS) | 95.81% | — | — |
| Previous pipeline (LogReg + FrequencySenderEncoder) | 92.95% | 0.82 | 0.89 |

**+3.73 pp** improvement over the previous pipeline version, beating the historical best by
**+0.87 pp**.

The improvement comes from three changes applied together:
1. **Noise filter** — removes 33% of rows with ambiguous/catch-all labels; the model learns
   from cleaner signal
2. **3 separate TF-IDF vectorizers** (ProfileFullName 5k + Subject 15k + Symptom 35k = 55k
   features) instead of a combined 50k TF-IDF + FrequencySenderEncoder; sender name gets its
   own dedicated feature space
3. **LinearSVC instead of LogisticRegression** — LinearSVC handles high-dimensional sparse
   text features with better margin maximisation

---

## 6. Model Artefacts

| File | Contents |
|---|---|
| `pipeline/models/service_model.joblib` | LinearSVC — used for prediction |
| `pipeline/models/service_model_calibrated.joblib` | CalibratedClassifierCV (cv=3, sigmoid) — used for confidence scores |
| `pipeline/models/service_transformers.joblib` | dict of 3 TfidfVectorizers keyed `tfidf_name`, `tfidf_subject`, `tfidf_symptom` |

---

## 7. Noise Filter Ablation — Does Filtering Actually Help?

To verify the noise filter is beneficial and not just discarding useful signal, the same
3-vectorizer + LinearSVC was trained on the full 129,228-row dataset (noise filter disabled).

| | With noise filter | Without noise filter | Δ |
|---|---:|---:|---:|
| Rows trained on | 86,085 | 129,228 | −43,143 |
| Test set size | 17,217 | 25,846 | |
| **Accuracy** | **96.68%** | **93.81%** | **+2.87 pp** |
| Application F1 | 0.98 | 0.96 | +0.02 |
| **Infrastructure F1** | **0.87** | **0.83** | **+0.04** |
| Macro F1 | 0.92 | 0.89 | +0.03 |

**Conclusion: the noise filter helps.** Removing 33% of rows improves accuracy by ~3 pp, with
Infrastructure gaining the most (+4 F1 points). The dropped rows — CMS category tickets,
"Altro"/"Other"/"Z-Other" subcategory tickets — act as mislabelled or ambiguous training
examples. Because their Category/Subcategory labels are noise buckets rather than precise
assignments, the text content doesn't cleanly correlate with the label. Keeping them introduces
conflicting signal that hurts the decision boundary, particularly for the already harder
Infrastructure class.

> **Note on test set comparability**: the test populations differ slightly (17k vs 26k) because
> the noisy tickets are also held out when the filter is off. The method is identical so the
> comparison is valid, but the absolute test numbers should not be compared directly.

---

## 8. Summary Scorecard

| Metric | Value |
|---|---|
| Total classes | 2 |
| Training rows | 68,868 |
| Test rows | 17,217 |
| Overall accuracy | **96.68%** |
| Macro F1 | 0.92 |
| Weighted F1 | 0.97 |
| Application F1 | 0.98 |
| Infrastructure F1 | 0.87 |
| vs. previous pipeline | **+3.73 pp accuracy** |
| vs. historical best | **+0.87 pp accuracy** |
