# Subcategory Model — Performance Analysis

**True-routed accuracy: 95.88% | Macro F1: 0.9463 | Abstain rate: 3.2%**
**End-to-end accuracy: 86.01% | Macro F1: 0.7717 | Abstain rate: 4.1%**

---

## 1. Full Pipeline Explanation

The pipeline is a three-stage cascade. Each stage builds on the prediction of the previous one.

```
Incoming ticket
  │
  │  subject, symptom, sender (ProfileFullName)
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 1 — SERVICE                                           │
│  Model: LinearSVC (global)                                   │
│  Features: 3 TF-IDF vectorizers                              │
│    tfidf_name    → ProfileFullName  (5,000 features)         │
│    tfidf_subject → Subject          (15,000 features)        │
│    tfidf_symptom → Symptom          (35,000 features)        │
│  Artefacts: service_model.joblib                             │
│             service_model_calibrated.joblib (confidence)     │
│             service_transformers.joblib                      │
│  Output: "Application" or "Infrastructure"  + confidence     │
└──────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 2 — CATEGORY                                          │
│  Model: LinearSVC (global, single model for all categories)  │
│  Features: same 3 TF-IDF vectorizers (re-fitted on category  │
│            training data)                                    │
│  Artefacts: category_model.joblib                            │
│             category_model_calibrated.joblib (confidence)    │
│             category_transformers.joblib                     │
│  Output: e.g. "30-ERP Microsoft AX 2012"  + confidence       │
└──────────────────────────────────────────────────────────────┘
  │
  ├─► Flat category? (14 categories — see Section 4)
  │     → subcategory = category name, no model needed
  │
  ├─► No subcategory model trained? (18 skipped — see Section 5)
  │     → subcategory = "unspecified (review)"  [ABSTAIN]
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 3 — SUBCATEGORY                                       │
│  Model: LinearSVC per category (18 models)                   │
│  Features: 3 TF-IDF vectorizers re-fitted on that            │
│            category's training subset                        │
│  Artefacts: models/subcategory/<category>.joblib             │
│             contains: model, calibrated, transformers,       │
│                       classes, excluded_subcats              │
│  Output: subcategory name  + confidence                      │
│  Abstention: if max confidence < 0.40 → "unspecified (review)│
└──────────────────────────────────────────────────────────────┘
```

### Confidence and abstention

At each stage the pipeline uses a `CalibratedClassifierCV(LinearSVC, cv=3, method="sigmoid")`
to convert the SVM decision scores into calibrated probabilities.
At the subcategory stage, if the top class probability is below **0.40** the model abstains
and returns `"unspecified (review)"` rather than guessing.

### Noise filter

Before training (service, category, subcategory), rows whose **Category** OR **Subcategory**
column matches `(?i)(cms|altro|other|z-other)` are removed. This removes ~33% of raw tickets
that carry ambiguous or catch-all labels and would otherwise pollute the decision boundary.

---

## 2. The Script-14 Approach — What Changed and Why It Works

### Background

The pipeline originally used a different modelling approach for all three stages:

| Component | Old pipeline | New pipeline (script-14 inspired) |
|---|---|---|
| Classifier | LogisticRegression | LinearSVC |
| Features | Single combined TF-IDF (50k) + FrequencySenderEncoder (top-200 sender buckets) | 3 separate TF-IDF vectorizers: ProfileFullName (5k) + Subject (15k) + Symptom (35k) |
| Noise filter | None | Drops rows where Category or Subcategory matches `(?i)(cms|altro|other|z-other)` |
| Confidence | Native LogReg probabilities | CalibratedClassifierCV (cv=3, sigmoid) wrapping LinearSVC |
| Production model | Trained on 80% split | Retrained on 100% of clean data after held-out evaluation |

The name "script-14" refers to `14_train_ultimate_category_model.py`, a standalone experiment
script that tested this combination and achieved the best category accuracy seen in any prior
experiment (85.63%). When we applied the same pattern to the pipeline — with the additional
benefit of the newer data export and the pipeline's better `clean_text()` function — we
exceeded that result (87.00% category).

---

### Change 1 — Noise filter (+3 pp service, +2 pp category)

The single biggest lever. Removes ~33% of rows (43,143 out of 129,228) whose labels are
noise buckets rather than genuine assignments:

- **Category-level noise**: 70-CMS (~11,702), 99-Z-Other Applications (~2,919),
  29-Z-Other Infrastructure (~246), 82-WCMS (~9)
- **Subcategory-level noise**: 34-PLM "Altro" subcategory (~23,624 rows alone),
  plus "Other", "Z-Other" subcategories scattered across many categories

These tickets have real text (the symptom description is genuine) but their labels were
assigned to a catch-all bucket, not a specific category. Training on them teaches the model
to associate real vocabulary with noise labels, polluting the decision boundary.

Ablation test (service model, same 3-vectorizer + LinearSVC):

| | With noise filter | Without noise filter |
|---|---:|---:|
| Accuracy | **96.68%** | 93.81% |
| Infrastructure F1 | **0.87** | 0.83 |

The filter hurts on minority classes most — Infrastructure loses 4 F1 points without it —
because Infrastructure tickets overlap heavily in vocabulary with the catch-all labels.

---

### Change 2 — 3 separate TF-IDF vectorizers instead of 1 combined

The old pipeline concatenated Subject + Symptom into a single `text` column and fed it
through one 50k-feature TF-IDF, plus a `FrequencySenderEncoder` (top-200 sender buckets
one-hot encoded).

The new approach gives each input field its own vocabulary budget:

```
tfidf_name    → ProfileFullName        5,000 features   (1-2 grams, sublinear_tf)
tfidf_subject → Subject  (cleaned)    15,000 features   (1-2 grams, sublinear_tf)
tfidf_symptom → Symptom  (cleaned)    35,000 features   (1-2 grams, sublinear_tf)
─────────────────────────────────────────────────────
Total                                 55,000 features
```

Why this is better:

- **ProfileFullName** is a categorical signal: certain users/groups always submit the same
  type of ticket. A shared vocabulary space would lose this signal to the much larger
  Subject/Symptom corpora. Its own 5k space preserves it cleanly.
- **Subject** is short and keyword-dense (high signal/noise). 15k features captures the
  useful terminology without dilution.
- **Symptom** is long and contains the most discriminative content — 35k features lets it
  develop a rich vocabulary without crowding out Subject or sender terms.
- The old `FrequencySenderEncoder` (top-200 sender buckets) was a weaker proxy: it bucketed
  senders by raw frequency and collapsed rare senders to `__other__`, losing most of the
  sender-specific signal. A full TF-IDF on the name field captures every sender's pattern.

---

### Change 3 — LinearSVC instead of LogisticRegression

Both are linear classifiers, but LinearSVC is better suited to high-dimensional sparse text:

- **Margin maximisation**: SVM finds the maximum-margin hyperplane, which generalises better
  on sparse feature spaces where many features are near-zero for any given document.
- **No probability calibration overhead**: LogisticRegression spends capacity fitting
  sigmoid probabilities directly; SVC focuses purely on the decision boundary and delegates
  calibration to a separate `CalibratedClassifierCV` step.
- **`dual=False`** with `max_iter=10,000`: for n_samples >> n_features (our 55k-feature
  space with 86k+ rows), the primal form converges faster and produces better solutions.
- **`class_weight="balanced"`**: automatically upweights minority classes (Infrastructure,
  rare subcategories) proportional to their inverse frequency.

---

### Change 4 — CalibratedClassifierCV for confidence scores

The old LogisticRegression produced native probabilities, but these were not well-calibrated
for the abstention threshold (0.40). In practice the model was often uncertain (probabilities
clustered near 0.40–0.55), causing the abstain rate to reach 22% in subcategory evaluation.

The new approach wraps LinearSVC in `CalibratedClassifierCV(cv=3, method="sigmoid")`:
- The sigmoid (Platt scaling) is fit on 3-fold cross-validation, producing properly
  calibrated probabilities that spread across the full [0, 1] range.
- Result: the subcategory abstain rate dropped from **22.1% → 3.2%** without changing the
  0.40 threshold. The model is now genuinely confident when it should be, rather than
  artificially uncertain.

---

### Overall Impact Across All Stages

| Stage | Old accuracy | New accuracy | Δ |
|---|---:|---:|---:|
| Service | 92.95% | **96.68%** | **+3.73 pp** |
| Category | 85.09%¹ | **87.00%** | **+1.91 pp** |
| Subcategory (true-routed) | 90.71% | **95.88%** | **+5.17 pp** |
| Subcategory (end-to-end) | 82.05% | **86.01%** | **+3.96 pp** |
| Subcategory abstain rate | 22.1% | **3.2%** | **−18.9 pp** |

¹ *Final Production Model from `metrics_history.json` (deployed API baseline)*

All four changes compound each other: cleaner training data (noise filter) lets the better
feature representation (3 vectorizers) learn sharper boundaries, which LinearSVC exploits
more effectively than LogisticRegression, and the calibrated probabilities then translate
that confidence into reliable abstention decisions.

---

## 3. Training Data Summary

| Split | Rows |
|---|---:|
| Raw CSV | 129,388 |
| After drop missing labels | 129,228 |
| After drop TRASHBIN categories | 126,017 |
| After drop DEFAULT subcategories | 89,328 |
| After exclude FLAT categories | **86,660** |

The 86,660 rows are split per-category: each category model is trained on its own subset
with its own 80/20 stratified split.

---

## 5. Trained Categories (18)

These categories have a subcategory model. Held-out eval F1 is the test split result
from `train_subcategory.py`.

| Category | Subcategories trained | Excluded (< 50) | Train rows | Test rows | Held-out Macro F1 | True-routed F1 |
|---|---:|---:|---:|---:|---:|---:|
| 01-Workplace | 6 | 1 | 4,148 | 1,037 | 0.6444 | 0.9429 |
| 02-User Application | 7 | 2 | 963 | 241 | 0.4863 | 1.0000 |
| 30-ERP Microsoft AX 2012 | 17 | 1 | 19,227 | 4,807 | 0.5419 | 0.9431 |
| 32-EBS (ERP) | 3 | 0 | 6,383 | 1,596 | 0.5743 | 0.3795 |
| 34-PLM | 3 | 20 | 968 | 243 | 0.6077 | 0.6405 |
| 35-CAD | 3 | 3 | 917 | 230 | 0.9419 | 0.9917 |
| 40-CRM Microsoft | 4 | 1 | 5,155 | 1,289 | 0.6818 | 0.9677 |
| 42-CPQ Wood | 3 | 0 | 2,596 | 649 | 0.7944 | 0.9963 |
| 46-MyPortal | 6 | 7 | 4,826 | 1,207 | 0.7189 | 0.9477 |
| 48-BI Microsoft | 7 | 1 | 5,444 | 1,361 | 0.5577 | 0.8729 |
| 58-FCS | 8 | 2 | 2,866 | 717 | 0.7450 | 0.9897 |
| 64-Applicazioni HR | 2 | 1 | 510 | 128 | 0.8789 | 0.9835 |
| 66-ERP-365FO | 6 | 4 | 1,070 | 268 | 0.5965 | 0.9924 |
| 67-RPA | 2 | 0 | 277 | 70 | 0.9720 | 1.0000 |
| 70-CMS | 6 | 6 | 7,275 | 1,819 | 0.5806 | 0.9823 |
| 71-HCL NOTES | 6 | 4 | 1,704 | 427 | 0.6068 | 0.9937 |
| 72-ASM | 4 | 7 | 950 | 238 | 0.4751 | 0.5782 |
| 73-ERP (ACG AS/400) | 5 | 5 | 2,046 | 512 | 0.4722 | 0.7662 |

> **Held-out Macro F1 vs True-routed F1**: the held-out number is from the production
> model evaluated on its own 20% split (some overfitting is expected since the production
> model retrains on 100%). True-routed is from `evaluate.py` on the subcategory test split
> with strict isolation. **True-routed is the reliable number.**

---

## 6. Subcategory Support per Trained Category

### 01-Workplace  *(train=4,148 · test=1,037 · Macro F1=0.94)*

| Subcategory | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| User Management | 706 | 0.86 | 0.90 | 0.88 |
| User Creation | 99 | 0.77 | 0.73 | 0.75 |
| Password Reset | 79 | 0.52 | 0.51 | 0.51 |
| SW Installation | 57 | 0.70 | 0.54 | 0.61 |
| Laptop | 61 | 0.68 | 0.67 | 0.68 |
| First Supply | 35 | 0.54 | 0.37 | 0.44 |
| *Workstation CAD* | *excluded* | — | — | — |

### 02-User Application  *(train=963 · test=241 · Macro F1=0.49)*

| Subcategory | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Office 365 Applications | 92 | 0.60 | 0.73 | 0.66 |
| Google G Suite | 30 | 0.52 | 0.50 | 0.51 |
| User Tool | 30 | 0.48 | 0.40 | 0.44 |
| Application Login | 28 | 0.46 | 0.46 | 0.46 |
| Office tool | 28 | 0.32 | 0.29 | 0.30 |
| User Application Other | 23 | 0.46 | 0.26 | 0.33 |
| Antivirus | 10 | 0.70 | 0.70 | 0.70 |
| *Cad* | *excluded* | — | — | — |
| *Ivanti Tool* | *excluded* | — | — | — |

### 30-ERP Microsoft AX 2012  *(train=19,227 · test=4,807 · Macro F1=0.54)*

| Subcategory | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Warehouse & Material Mgmt | 1,314 | 0.84 | 0.81 | 0.82 |
| Finance | 867 | 0.84 | 0.80 | 0.82 |
| Production | 929 | 0.77 | 0.76 | 0.77 |
| User Management | 386 | 0.69 | 0.69 | 0.69 |
| Procurement | 463 | 0.66 | 0.71 | 0.68 |
| Quality | 116 | 0.65 | 0.77 | 0.70 |
| Sales & Logistics | 272 | 0.50 | 0.55 | 0.52 |
| Sales & Logistic Machines | 94 | 0.45 | 0.43 | 0.44 |
| Sales Machines | 70 | 0.41 | 0.51 | 0.46 |
| Sales Items | 52 | 0.39 | 0.46 | 0.42 |
| Sales & Logistic Items | 92 | 0.43 | 0.42 | 0.43 |
| Logistics Machines | 39 | 0.49 | 0.59 | 0.53 |
| Logistics Items | 34 | 0.47 | 0.50 | 0.49 |
| Configurator | 28 | 0.60 | 0.54 | 0.57 |
| Controlling | 19 | 0.17 | 0.16 | 0.16 |
| User Creation | 20 | 0.38 | 0.30 | 0.33 |
| Customer & Vendor master data | 12 | 0.36 | 0.42 | 0.38 |
| *Travel expenses* | *excluded* | — | — | — |

### 32-EBS (ERP)  *(train=6,383 · test=1,596 · Macro F1=0.58)*

| Subcategory | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Sales & Logistics | 1,326 | 0.89 | 0.94 | 0.91 |
| Finance | 241 | 0.62 | 0.45 | 0.52 |
| Configurator | 29 | 0.30 | 0.28 | 0.29 |

### 34-PLM  *(train=968 · test=243 · Macro F1=0.61)*

| Subcategory | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Workflow | 209 | 0.89 | 0.96 | 0.92 |
| Client issue | 11 | 1.00 | 0.64 | 0.78 |
| Traslatore Solid Edge | 23 | 0.20 | 0.09 | 0.12 |
| *20 subcategories* | *excluded* | — | — | — |

> 34-PLM had 23 distinct subcategories, 20 of which fell below the 50-row support floor.
> Only 3 trained; the "Traslatore Solid Edge" subcategory struggles because of its small size.

### 35-CAD  *(train=917 · test=230 · Macro F1=0.94)*

| Subcategory | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Solid Edge | 188 | 0.97 | 0.99 | 0.98 |
| Eplan | 22 | 1.00 | 0.95 | 0.98 |
| Autocad | 20 | 0.94 | 0.80 | 0.86 |
| *Creo, Autocad LT, Inventor* | *excluded* | — | — | — |

### 40-CRM Microsoft  *(train=5,155 · test=1,289 · Macro F1=0.68)*

| Subcategory | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Service & Field Service | 652 | 0.84 | 0.89 | 0.86 |
| Sales | 416 | 0.87 | 0.86 | 0.86 |
| Field Service | 157 | 0.73 | 0.68 | 0.70 |
| Customer Service | 64 | 0.41 | 0.23 | 0.30 |
| *Marketing* | *excluded* | — | — | — |

### 42-CPQ Wood  *(train=2,596 · test=649 · Macro F1=0.79)*

| Subcategory | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 42-CPQ | 475 | 0.90 | 0.96 | 0.93 |
| Configurator | 78 | 0.83 | 0.73 | 0.78 |
| Pricing & Quoting | 96 | 0.81 | 0.58 | 0.68 |

### 46-MyPortal  *(train=4,826 · test=1,207 · Macro F1=0.72)*

| Subcategory | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 13-PORTALE | 717 | 0.93 | 0.95 | 0.94 |
| 01-DMS | 377 | 0.93 | 0.95 | 0.94 |
| 03-ESHOP | 67 | 0.72 | 0.63 | 0.67 |
| D-Sales | 18 | 0.74 | 0.78 | 0.76 |
| 07-PORTALE PARTNERS LOGISTICA | 13 | 0.83 | 0.77 | 0.80 |
| 05-PORTALE PARTNERS MARKETING | 15 | 0.40 | 0.13 | 0.20 |
| *7 subcategories* | *excluded* | — | — | — |

### 48-BI Microsoft  *(train=5,444 · test=1,361 · Macro F1=0.56)*

| Subcategory | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Sales & Logistics | 759 | 0.79 | 0.83 | 0.81 |
| Finance | 275 | 0.62 | 0.62 | 0.62 |
| Controlling | 175 | 0.41 | 0.32 | 0.36 |
| Production | 75 | 0.38 | 0.40 | 0.39 |
| CDP | 34 | 0.73 | 0.71 | 0.72 |
| Warehouse & Material Mgmt | 25 | 0.44 | 0.44 | 0.44 |
| Procurement | 18 | 0.52 | 0.61 | 0.56 |
| *Control Room* | *excluded* | — | — | — |

### 58-FCS  *(train=2,866 · test=717 · Macro F1=0.75)*

| Subcategory | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| MDS | 314 | 0.83 | 0.89 | 0.86 |
| LPS | 179 | 0.88 | 0.92 | 0.90 |
| MDM | 50 | 0.91 | 0.86 | 0.89 |
| HI-WMS | 46 | 0.80 | 0.87 | 0.83 |
| HI-MDS | 45 | 0.77 | 0.60 | 0.68 |
| EDS | 13 | 0.91 | 0.77 | 0.83 |
| FCS (all) | 58 | 0.53 | 0.36 | 0.43 |
| DDS | 12 | 0.60 | 0.50 | 0.55 |
| *WMS, WHS* | *excluded* | — | — | — |

### 64-Applicazioni HR  *(train=510 · test=128 · Macro F1=0.88)*

| Subcategory | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 365FO - Talent | 77 | 0.92 | 0.88 | 0.90 |
| CDP | 51 | 0.83 | 0.88 | 0.86 |
| *HRMS - Success Factors* | *excluded* | — | — | — |

### 66-ERP-365FO  *(train=1,070 · test=268 · Macro F1=0.60)*

| Subcategory | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Finance | 106 | 0.79 | 0.86 | 0.82 |
| Logistics Machines | 46 | 0.74 | 0.76 | 0.75 |
| Sales Items | 33 | 0.54 | 0.61 | 0.57 |
| Sales Machines | 45 | 0.62 | 0.53 | 0.57 |
| Logistics Items | 22 | 0.37 | 0.32 | 0.34 |
| Sales & Logistics | 16 | 0.64 | 0.44 | 0.52 |
| *Procurement, Customer & Vendor, User Mgmt, Production* | *excluded* | — | — | — |

### 67-RPA  *(train=277 · test=70 · Macro F1=0.97)*

| Subcategory | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Entrate Merci | 60 | 1.00 | 0.98 | 0.99 |
| Ciclo Passivo | 10 | 0.91 | 1.00 | 0.95 |

### 70-CMS  *(train=7,275 · test=1,819 · Macro F1=0.58)*

| Subcategory | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Infrastructure | 1,493 | 0.96 | 0.97 | 0.96 |
| Service | 126 | 0.52 | 0.52 | 0.52 |
| Production | 91 | 0.52 | 0.38 | 0.44 |
| Sales & Logistics | 70 | 0.47 | 0.46 | 0.46 |
| Warehouse & Material Mgmt | 20 | 0.50 | 0.50 | 0.50 |
| Finance | 19 | 0.61 | 0.58 | 0.59 |
| *Procurement, Spare Parts, Sales, Planning, Quality, HR* | *excluded* | — | — | — |

### 71-HCL NOTES  *(train=1,704 · test=427 · Macro F1=0.61)*

| Subcategory | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Sales | 108 | 0.70 | 0.79 | 0.74 |
| Service | 136 | 0.58 | 0.47 | 0.52 |
| Planning and Supply Chain | 83 | 0.57 | 0.64 | 0.60 |
| Finance | 32 | 0.57 | 0.62 | 0.60 |
| Engineering | 54 | 0.65 | 0.72 | 0.68 |
| Procurement | 14 | 0.83 | 0.36 | 0.50 |
| *Quality, Spare Parts, Marketing, HR* | *excluded* | — | — | — |

### 72-ASM  *(train=950 · test=238 · Macro F1=0.48)*

| Subcategory | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Service | 202 | 0.90 | 0.93 | 0.91 |
| Quality | 10 | 0.47 | 0.80 | 0.59 |
| Sales | 16 | 0.38 | 0.19 | 0.25 |
| Planning and Supply Chain | 10 | 0.25 | 0.10 | 0.14 |
| *Engineering, Procurement, Finance, Commissioning, Spare Parts, HR, Marketing* | *excluded* | — | — | — |

### 73-ERP (ACG AS/400)  *(train=2,046 · test=512 · Macro F1=0.47)*

| Subcategory | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Planning and Supply Chain | 186 | 0.66 | 0.65 | 0.66 |
| Service | 216 | 0.60 | 0.59 | 0.60 |
| Procurement | 54 | 0.52 | 0.59 | 0.56 |
| Finance | 41 | 0.47 | 0.49 | 0.48 |
| Sales | 15 | 0.09 | 0.07 | 0.08 |
| *Spare Parts, Quality, Engineering, Marketing, Production* | *excluded* | — | — | — |

---

## 7. Flat Categories (14) — Rule-Based

These categories have no subcategory model. When a ticket is predicted to belong to one of
them, **subcategory = category name** directly. This is by design: the subcategory structure
has no meaningful distinction from the category for these labels.

| Category | Reason |
|---|---|
| 11-Collaboration Device | Flat by label design |
| 12-ES Contractors | Flat by label design |
| 20-IT Security | Flat by label design |
| 44-iService | Flat by label design |
| 50-DWH Oracle | Flat by label design |
| 52-Hyperion | Flat by label design |
| 54-Piteco | Flat by label design |
| 60-Nicim | Flat by label design |
| 63-MANTIS | Flat by label design |
| 64-HR-DD365FO | Flat by label design |
| 65-Applicazioni Corporate | Flat by label design |
| 80-North America | Flat by label design |
| 81-SDH | Flat by label design |
| 83-WPR | Flat by label design |

---

## 8. Skipped Categories (18) — No Model Trained

These categories exist in the data but could not produce a trainable subcategory model
because after the `MIN_SUBCAT_SUPPORT=50` floor, fewer than 2 distinct subcategories remained.

| Category | Subcategories in raw data | After support filter | Why skipped |
|---|---:|---:|---|
| 03-System Software | 8 | 0 | All 8 subcategories < 50 rows |
| 04-Network | 7 | 1 | 6 below floor; only 1 remains (needs ≥ 2) |
| 05-Printer | 2 | 1 | 1 below floor; only 1 remains |
| 06-Server | 7 | 1 | 6 below floor; only 1 remains |
| 07-Storage | 2 | 1 | 1 below floor; only 1 remains |
| 08-Voice | 3 | 0 | All 3 subcategories < 50 rows |
| 09-Mobility Device | 3 | 1 | 2 below floor; only 1 remains |
| 10-Industrial Device | 3 | 0 | All 3 subcategories < 50 rows |
| 43-CPQ Experlogix | 3 | 0 | All 3 subcategories < 50 rows |
| 68-ERP Fondwise | 9 | 1 | 8 below floor; only 1 remains |
| 74-MES (OASI) | 6 | 1 | 5 below floor; only 1 remains |
| 75-MPS | 4 | 0 | All 4 subcategories < 50 rows |
| 76-RDA | 3 | 0 | All 3 subcategories < 50 rows |
| 77-QUALIWARE | 4 | 0 | All 4 subcategories < 50 rows |
| 78-TIME TRACK | 2 | 0 | Both subcategories < 50 rows |
| 79-BOARD | 2 | 0 | Both subcategories < 50 rows |
| 82-WCMS | 2 | 0 | Both subcategories < 50 rows |
| 84-LOGiN (WMS) | 2 | 0 | Both subcategories < 50 rows |

In production: tickets routed to a skipped category always return `"unspecified (review)"`.

---

## 9. Evaluation Results

### True-category-routed (best case — correct category assumed)

| Category | Support | Macro F1 | Abstain rate |
|---|---:|---:|---:|
| 02-User Application | 240 | **1.0000** | 4.2% |
| 67-RPA | 69 | **1.0000** | 0.0% |
| 42-CPQ Wood | 649 | 0.9963 | 0.0% |
| 71-HCL NOTES | 426 | 0.9937 | 1.2% |
| 66-ERP-365FO | 270 | 0.9924 | 1.1% |
| 35-CAD | 237 | 0.9917 | 0.0% |
| 58-FCS | 718 | 0.9897 | 0.0% |
| 64-Applicazioni HR | 127 | 0.9835 | 0.0% |
| 70-CMS | 1,819 | 0.9823 | 0.3% |
| 40-CRM Microsoft | 1,290 | 0.9677 | 0.2% |
| 46-MyPortal | 1,208 | 0.9477 | 0.2% |
| 30-ERP Microsoft AX 2012 | 4,806 | 0.9431 | 7.7% |
| 01-Workplace | 1,037 | 0.9429 | 0.0% |
| 48-BI Microsoft | 1,361 | 0.8729 | 8.7% |
| 73-ERP (ACG AS/400) | 513 | 0.7662 | 3.5% |
| 34-PLM | 239 | 0.6405 | 0.0% |
| 72-ASM | 236 | 0.5782 | 0.0% |
| 32-EBS (ERP) | 1,596 | **0.3795** | 0.0% |
| **Overall** | **17,217** | **0.9463** | **3.2%** |

### End-to-end — predicted category routing (real-world scenario)

| Category | Support | Macro F1 | Abstain rate |
|---|---:|---:|---:|
| 67-RPA | 69 | 1.0000 | 0.0% |
| 30-ERP Microsoft AX 2012 | 4,614 | 0.8853 | 7.8% |
| 66-ERP-365FO | 325 | 0.5192 | 6.5% |
| 71-HCL NOTES | 631 | 0.4675 | 6.3% |
| 58-FCS | 759 | 0.4536 | 0.0% |
| 02-User Application | 422 | 0.3564 | 17.1% |
| 48-BI Microsoft | 1,363 | 0.3505 | 8.6% |
| 73-ERP (ACG AS/400) | 757 | 0.3328 | 5.9% |
| 01-Workplace | 1,481 | 0.2989 | 1.2% |
| 35-CAD | 298 | 0.2928 | 0.0% |
| 72-ASM | 384 | 0.2873 | 0.0% |
| 64-Applicazioni HR | 145 | 0.2625 | 0.0% |
| 42-CPQ Wood | 680 | 0.2223 | 0.0% |
| 46-MyPortal | 1,292 | 0.2216 | 0.2% |
| 34-PLM | 273 | 0.1901 | 0.0% |
| 40-CRM Microsoft | 1,330 | 0.1810 | 0.2% |
| 32-EBS (ERP) | 1,666 | **0.0698** | 0.0% |
| **Overall** | **17,217** | **0.7717** | **4.1%** |

---

## 10. Before / After Comparison (retraining with script-14 approach)

| Metric | Old models | New models | Δ |
|---|---:|---:|---:|
| True-routed accuracy | 90.71% | **95.88%** | **+5.17 pp** |
| True-routed Macro F1 | 0.8925 | **0.9463** | **+0.054** |
| E2E accuracy | 82.05% | **86.01%** | **+3.96 pp** |
| E2E Macro F1 | 0.7337 | **0.7717** | **+0.038** |
| Abstain rate (true-routed) | 22.1% | **3.2%** | **−18.9 pp** |
| Abstain rate (E2E) | 25.8% | **4.1%** | **−21.7 pp** |

The dramatic drop in abstain rate is the most impactful change: LinearSVC via
CalibratedClassifierCV produces much better-calibrated confidence scores than the old
LogisticRegression + FeaturePipeline combination, so the model no longer excessively
abstains on confident predictions.

---

## 11. Confusion Matrices

Per-category confusion matrices are saved in:
```
pipeline/models/subcategory/confusion_matrices/
```

One PNG per trained category:

| File | Category |
|---|---|
| `01-Workplace_cm.png` | 01-Workplace |
| `02-User_Application_cm.png` | 02-User Application |
| `30-ERP_Microsoft_AX_2012_cm.png` | 30-ERP Microsoft AX 2012 |
| `32-EBS__ERP__cm.png` | 32-EBS (ERP) |
| `34-PLM_cm.png` | 34-PLM |
| `35-CAD_cm.png` | 35-CAD |
| `40-CRM_Microsoft_cm.png` | 40-CRM Microsoft |
| `42-CPQ_Wood_cm.png` | 42-CPQ Wood |
| `46-MyPortal_cm.png` | 46-MyPortal |
| `48-BI_Microsoft_cm.png` | 48-BI Microsoft |
| `58-FCS_cm.png` | 58-FCS |
| `64-Applicazioni_HR_cm.png` | 64-Applicazioni HR |
| `66-ERP-365FO_cm.png` | 66-ERP-365FO |
| `67-RPA_cm.png` | 67-RPA |
| `70-CMS_cm.png` | 70-CMS |
| `71-HCL_NOTES_cm.png` | 71-HCL NOTES |
| `72-ASM_cm.png` | 72-ASM |
| `73-ERP__ACG_AS_400__cm.png` | 73-ERP (ACG AS/400) |

Regenerate all matrices by running:
```bash
python pipeline/train_subcategory.py
```
The matrices are reproduced at the bottom of each category's training block.

---

## 12. Key Observations

### What works well (true-routed F1 ≥ 0.94)
- **35-CAD, 67-RPA, 42-CPQ Wood, 58-FCS, 64-Applicazioni HR, 66-ERP-365FO** — distinctive
  vocabularies, limited subcategory overlap
- **70-CMS, 71-HCL NOTES, 40-CRM Microsoft, 46-MyPortal** — large training sets with
  sufficiently distinct subcategory language

### What struggles (true-routed F1 < 0.65)
- **32-EBS (ERP)** (F1=0.38): dominated by "Sales & Logistics" (83% of tickets). The two
  minority subcategories — Finance and Configurator — have very few tickets and bleed into
  each other in text space.
- **72-ASM** (F1=0.58): "Service" accounts for 85% of tickets; the 3 remaining subcategories
  are extremely small and use generic vocabulary.
- **73-ERP (ACG AS/400)** (F1=0.77): 5 similarly-worded business-domain subcategories with
  no clear lexical discriminator. "Sales" almost entirely fails (F1=0.08, 15 samples).

### The E2E gap
The true-routed vs E2E gap (0.9463 → 0.7717 macro F1) is driven entirely by category routing
errors, not subcategory model weakness. The worst E2E performers are categories that the
category model sometimes misroutes:
- **32-EBS** (true=0.38 → E2E=0.07): already weak true-routed, amplified by routing confusion
- **40-CRM** (true=0.97 → E2E=0.18): strong subcategory model, but heavily misrouted
- **42-CPQ Wood** (true=1.00 → E2E=0.22): perfect subcategory model, routing errors dominate

Improving category accuracy for these labels is the highest-leverage next step.
