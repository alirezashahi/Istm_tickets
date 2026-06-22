# Category Model — Performance Analysis

**Overall accuracy: 87.00% | Macro F1: 0.63 | Weighted F1: 0.87**
Test set: 17,205 tickets | 41 categories

Tiers: ✅ F1 ≥ 0.80 · ⚠️ F1 0.50–0.79 · ❌ F1 < 0.50

---

## 1. Classification Report (sorted by volume)

| Category | Support | Precision | Recall | F1 | Tier | Note |
|---|---:|---:|---:|---:|:---:|---|
| 30-ERP Microsoft AX 2012 | 4,816 | 0.94 | 0.89 | 0.92 | ✅ | |
| 01-Workplace | 1,607 | 0.82 | 0.77 | 0.79 | ⚠️ | High volume, borderline |
| 32-EBS (ERP) | 1,596 | 0.91 | 0.92 | 0.92 | ✅ | |
| 48-BI Microsoft | 1,367 | 0.88 | 0.86 | 0.87 | ✅ | |
| 40-CRM Microsoft | 1,297 | 0.87 | 0.89 | 0.88 | ✅ | |
| 46-MyPortal | 1,234 | 0.87 | 0.90 | 0.89 | ✅ | |
| 66-ERP-365FO | 904 | 0.88 | 0.93 | 0.90 | ✅ | |
| 58-FCS | 729 | 0.84 | 0.95 | 0.89 | ✅ | |
| 42-CPQ Wood | 649 | 0.82 | 0.89 | 0.85 | ✅ | |
| 73-ERP (ACG AS/400) | 537 | 0.90 | 0.90 | 0.90 | ✅ | |
| 71-HCL NOTES | 438 | 0.91 | 0.93 | 0.92 | ✅ | |
| 34-PLM | 278 | 0.78 | 0.81 | 0.79 | ⚠️ | Raw count was 25,038 — see Section 4 |
| 35-CAD | 255 | 0.79 | 0.84 | 0.81 | ✅ | |
| 72-ASM | 252 | 0.95 | 0.92 | 0.94 | ✅ | |
| 52-Hyperion | 231 | 0.72 | 0.82 | 0.77 | ⚠️ | |
| **02-User Application** | **223** | **0.35** | **0.33** | **0.34** | **❌** | **★ Highest-impact problem** |
| 64-Applicazioni HR | 132 | 0.60 | 0.75 | 0.67 | ⚠️ | |
| 64-HR-DD365FO | 83 | 0.57 | 0.71 | 0.63 | ⚠️ | |
| 67-RPA | 70 | 0.98 | 0.93 | 0.96 | ✅ | Best performing category |
| 44-iService | 69 | 0.60 | 0.62 | 0.61 | ⚠️ | |
| 06-Server | 67 | 0.41 | 0.36 | 0.38 | ❌ | |
| 65-Applicazioni Corporate | 54 | 0.90 | 0.81 | 0.85 | ✅ | |
| 09-Mobility Device | 49 | 0.53 | 0.67 | 0.59 | ⚠️ | |
| 80-North America | 43 | 0.89 | 0.77 | 0.82 | ✅ | |
| 68-ERP Fondwise | 41 | 0.89 | 0.95 | 0.92 | ✅ | |
| 04-Network | 27 | 0.23 | 0.19 | 0.20 | ❌ | |
| 60-Nicim | 25 | 0.30 | 0.48 | 0.37 | ❌ | |
| 03-System Software | 24 | 0.39 | 0.29 | 0.33 | ❌ | |
| 05-Printer | 20 | 0.60 | 0.45 | 0.51 | ⚠️ | |
| 74-MES (OASI) | 19 | 0.59 | 0.68 | 0.63 | ⚠️ | |
| 07-Storage | 16 | 0.76 | 0.81 | 0.79 | ⚠️ | |
| 50-DWH Oracle | 15 | 0.31 | 0.33 | 0.32 | ❌ | |
| 77-QUALIWARE | 11 | 0.82 | 0.82 | 0.82 | ✅ | |
| 63-MANTIS | 10 | 0.55 | 0.60 | 0.57 | ⚠️ | |
| 08-Voice | 4 | 0.20 | 0.25 | 0.22 | ❌ | Too few samples |
| 10-Industrial Device | 3 | 0.50 | 0.33 | 0.40 | ❌ | Too few samples |
| 43-CPQ Experlogix | 3 | 0.00 | 0.00 | 0.00 | ❌ | Effectively untrained — see Section 3 |
| 20-IT Security | 2 | 0.00 | 0.00 | 0.00 | ❌ | Effectively untrained — see Section 3 |
| 75-MPS | 2 | 0.00 | 0.00 | 0.00 | ❌ | Effectively untrained — see Section 3 |
| 84-LOGiN (WMS) | 2 | 1.00 | 0.50 | 0.67 | ⚠️ | Too few samples |
| 76-RDA | 1 | 0.00 | 0.00 | 0.00 | ❌ | Effectively untrained — see Section 3 |

---

## 2. High-Impact Struggling Categories

These have enough tickets to matter in production but are predicted poorly.

| Category | Support | F1 | Why it struggles |
|---|---:|---:|---|
| **02-User Application** | 223 | 0.34 | Broad catch-all label — "User Application" covers many different software types that look like other specific categories (Office 365, Google G Suite, Antivirus, CAD tools). Tickets bleed into 01-Workplace, 35-CAD, etc. |
| **06-Server** | 67 | 0.38 | Infrastructure tickets (Application Server, DB Server, Firewall, VMware) are short and generic; overlap with 04-Network, 03-System Software. |
| **04-Network** | 27 | 0.20 | Very generic subject lines ("Internet not working", "VPN issue") — hard to distinguish from 01-Workplace, 06-Server. |
| **03-System Software** | 24 | 0.33 | Small category covering OS, database, firewall software — overlaps with 06-Server. |
| **60-Nicim** | 25 | 0.37 | Niche application, tickets may not contain distinctive vocabulary. |
| **50-DWH Oracle** | 15 | 0.32 | Very small training set after noise filter; also short ticket text for DWH issues. |

> **Action recommendation**: For 02-User Application (the biggest problem), consider reviewing whether it should be split into subcategories more aggressively, or whether some subcategories (Office 365, G Suite) should be folded into 01-Workplace instead.

---

## 3. Categories in Training but Effectively Untrained

These survived all filters and appear in training, but lost almost all their rows to the subcategory noise filter, leaving too few clean samples to learn from.

| Category | Raw rows | After noise filter | Test support | F1 | Root cause |
|---|---:|---:|---:|---:|---|
| 75-MPS | 64 | ~11 | 2 | 0.00 | 53 rows had subcategory "Other" → dropped |
| 76-RDA | 43 | ~7 | 1 | 0.00 | 36 rows had subcategory "Other" → dropped |
| 43-CPQ Experlogix | 25 | ~14 | 3 | 0.00 | Some subcategories were "Other" type |
| 20-IT Security | 11 | 11 (FLAT) | 2 | 0.00 | No noise loss, just inherently tiny |
| 84-LOGiN (WMS) | 53 | ~8 | 2 | 0.67 | 45 rows had subcategory "Other" → dropped |

> These categories have a model entry but cannot be predicted reliably. In production they will almost always be misclassified. Consider either collecting more labelled data or merging them into a broader category.

---

## 4. The PLM Effect — Massive Data Loss from Noise Filtering

The subcategory noise filter (`altro|other|z-other`) causes extreme data loss for several top categories because their dominant subcategory is a noise bucket:

| Category | Raw rows | Rows dropped by noise filter | Remaining | Drop % |
|---|---:|---:|---:|---:|
| **34-PLM** | 25,038 | 23,624 (subcategory "Altro") | ~1,414 | **94.4%** |
| 01-Workplace | 10,545 | ~5,044 (Z-Others + other DEFAULT) | ~5,501 | ~47.8% |
| 02-User Application | 1,802 | ~569 (subcategory "Z-Others") | ~1,233 | ~31.6% |
| 75-MPS | 64 | 53 (subcategory "Other") | ~11 | 82.8% |
| 84-LOGiN (WMS) | 53 | 45 (subcategory "Other") | ~8 | 84.9% |
| 76-RDA | 43 | 36 (subcategory "Other") | ~7 | 83.7% |

**34-PLM** is the most striking case: it appears to be the #1 category by raw count (25,038 tickets) but shows up with only 278 test samples in the report. Its F1 of 0.79 is reasonable given the reduced data, but the model has almost no signal about what a "generic PLM" ticket looks like because that data was all labelled "Altro".

---

## 5. Categories Completely Excluded from Training

These categories have **zero** presence in the trained model and will never be predicted.

### Excluded by TRASHBIN label
Marked as noise/garbage in `label_decisions.csv` — removed by `load_and_clean("category")` before any training:

| Category | Raw rows | Reason |
|---|---:|---|
| 99-Z-Other Applications | 2,919 | Intentional noise bucket |
| 29-Z-Other Infrastructure | 246 | Intentional noise bucket |
| 36-SMC | 46 | Single subcategory "smc" tagged TRASHBIN |

### Excluded by category-name noise regex
Category name itself matches `(?i)(cms|altro|other|z-other)`:

| Category | Raw rows | Reason |
|---|---:|---|
| 70-CMS | 11,702 | "cms" in category name — largest exclusion |
| 82-WCMS | 9 | "cms" inside "WCMS" |

> **70-CMS** is the second largest category in the raw data (11,702 rows, 9% of all tickets). These tickets are intentionally excluded because CMS is a noise label. However, this means ~9% of production tickets will never match any trained category — worth monitoring.

### Excluded by subcategory noise cascade → min_count
All non-noisy subcategory rows fall below the min_count=5 threshold after the noise filter:

| Category | Raw rows | Non-noisy rows remaining | Why |
|---|---:|---:|---|
| 78-TIME TRACK | 17 | ~3 | Only subcategory "Other" (14 rows) dropped |
| 79-BOARD | 12 | ~0 | All subcategories appear to be "Other"/"Altro" type |

### Excluded by absolute min_count (< 5 total rows)
| Category | Raw rows |
|---|---:|
| 11-Collaboration Device | 2 |
| 54-Piteco | 2 |
| 12-ES Contractors | 1 |
| 81-SDH | 1 |
| 83-WPR | 1 |

---

## 6. Confusion Matrix

Generated by `train_category.py` and saved to:
```
pipeline/models/category_confusion_matrix.png
```

Run `python pipeline/train_category.py` to regenerate it. The matrix shows which categories are being confused with each other — useful for diagnosing the high-impact failures in Section 2.

---

## 7. Summary Scorecard

| Metric | Value |
|---|---|
| Total categories in dataset | 54 |
| Categories trained on | 41 |
| Categories completely excluded | 13 |
| ✅ High performers (F1 ≥ 0.80) | 20 |
| ⚠️ Medium performers (F1 0.50–0.79) | 13 |
| ❌ Low performers (F1 < 0.50) | 8 |
| Effectively untrained (F1 = 0.00) | 4 |
| Overall accuracy | 87.00% |
| Macro F1 | 0.63 |
| Weighted F1 | 0.87 |

The gap between macro F1 (0.63) and weighted F1 (0.87) reflects the imbalance: the model is excellent on the high-volume ERP/CRM/Portal categories that dominate the weighted average, but struggles on the long tail of infrastructure and niche application categories with limited training data.
