#!/usr/bin/env python3
"""
Ultimate Category-Only Production Model Training Script

This script trains the Category-level LinearSVC model.
It strictly adheres to the rule:
1. No subcategory/category merging whatsoever.
2. Complete removal of noise categories and subcategories (cms, altro, other, z-other).
3. 55k TF-IDF Sparse matrix features.

Output: final_model_report/models_production/ultimate_category/
"""

import os
import re
import warnings
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE       = Path(__file__).parent
DATA_DIR   = BASE / "Data"
OUTPUT_DIR = BASE / "final_model_report" / "models_production" / "ultimate_category"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────

SVC_PARAMS = dict(class_weight="balanced", max_iter=10_000, dual=False, random_state=42)

def banner(msg: str) -> None:
    print(f"\n{'='*70}\n  {msg}\n{'='*70}")


# ── Phase 1: Load & Prepare Data ──────────────────────────────────────────────

banner("PHASE 1: Load & Prepare Dataset")

def load_and_clean(splits_to_load, label):
    dfs = []
    for split in splits_to_load:
        try:
            df = pd.read_csv(DATA_DIR / f"{split}.csv", low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"  Error loading {split}.csv: {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined.dropna(subset=["Category", "Subcategory"], inplace=True)
    
    for col in ("ProfileFullName", "Subject", "Symptom"):
        if col in combined.columns:
            combined[col] = combined[col].fillna("")
            
    if "Symptom_clean" not in combined.columns:
        import sys
        sys.path.insert(0, str(BASE / "ticket_routing_api"))
        try:
            from preprocessing import clean_symptom
            combined["Symptom_clean"] = combined["Symptom"].apply(clean_symptom)
        except ImportError:
            combined["Symptom_clean"] = combined["Symptom"]
            
    combined["Symptom_clean"] = combined["Symptom_clean"].fillna("")
    
    noise_pattern = r"(?i)(cms|altro|other|z-other)"
    mask_noise_cat = combined["Category"].str.contains(noise_pattern, regex=True, na=False)
    mask_noise_subcat = combined["Subcategory"].str.contains(noise_pattern, regex=True, na=False)
    mask_keep = ~(mask_noise_cat | mask_noise_subcat)
    
    cleaned = combined[mask_keep].copy()
    print(f"  [{label}] Cleaned: {len(cleaned):,} rows (removed {(~mask_keep).sum():,} noise tickets)")
    return cleaned

train_val = load_and_clean(["train", "val"], "Train+Val")
test = load_and_clean(["test"], "Test")

# ── Phase 2: Feature Engineering & Evaluation ─────────────────────────────────

banner("PHASE 2: Train on Train+Val & Evaluate on Held-Out Test")

print("  Fitting tfidf_name    (max_features=5,000)  …")
tfidf_name = TfidfVectorizer(max_features=5_000, ngram_range=(1, 2), sublinear_tf=True)
X_tr_name = tfidf_name.fit_transform(train_val["ProfileFullName"])
X_te_name = tfidf_name.transform(test["ProfileFullName"])

print("  Fitting tfidf_subject (max_features=15,000) …")
tfidf_subject = TfidfVectorizer(max_features=15_000, ngram_range=(1, 2), sublinear_tf=True)
X_tr_subj = tfidf_subject.fit_transform(train_val["Subject"])
X_te_subj = tfidf_subject.transform(test["Subject"])

print("  Fitting tfidf_symptom (max_features=35,000) …")
tfidf_symptom = TfidfVectorizer(max_features=35_000, ngram_range=(1, 2), sublinear_tf=True)
X_tr_symp = tfidf_symptom.fit_transform(train_val["Symptom_clean"])
X_te_symp = tfidf_symptom.transform(test["Symptom_clean"])

X_train = hstack([X_tr_name, X_tr_subj, X_tr_symp])
X_test = hstack([X_te_name, X_te_subj, X_te_symp])

print(f"  Training eval model (LinearSVC) on {X_train.shape[0]:,} samples...")
eval_model = LinearSVC(**SVC_PARAMS)
eval_model.fit(X_train, train_val["Category"])

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

eval_preds = eval_model.predict(X_test)
test_acc = accuracy_score(test["Category"], eval_preds)
print(f"\n  [HELD-OUT EVALUATION] Test Set Accuracy: {test_acc*100:.2f}%\n")

print("  Generating Confusion Matrix Plot...")
cat_labels = sorted(test["Category"].unique())
cm = confusion_matrix(test["Category"], eval_preds, labels=cat_labels)

plt.figure(figsize=(24, 20))
sns.heatmap(cm, annot=False, cmap="Blues", fmt="d",
            xticklabels=cat_labels, yticklabels=cat_labels)
plt.title(f"Confusion Matrix (Test Accuracy: {test_acc*100:.2f}%)", fontsize=18)
plt.xlabel("Predicted Category", fontsize=14)
plt.ylabel("True Category", fontsize=14)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()

plot_path = OUTPUT_DIR / "category_confusion_matrix.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"  Saved plot: {plot_path}")

# ── Phase 3: Train Final Production Model ─────────────────────────────────────

banner("PHASE 3: Train Final Production Model on ALL Data")

full_clean = pd.concat([train_val, test], ignore_index=True)
print(f"  Combined full clean data: {len(full_clean):,} rows")

X_name = tfidf_name.fit_transform(full_clean["ProfileFullName"])
X_subj = tfidf_subject.fit_transform(full_clean["Subject"])
X_symp = tfidf_symptom.fit_transform(full_clean["Symptom_clean"])
X_full = hstack([X_name, X_subj, X_symp])

vectorizers = {
    "tfidf_name":    tfidf_name,
    "tfidf_subject": tfidf_subject,
    "tfidf_symptom": tfidf_symptom,
}
joblib.dump(vectorizers, OUTPUT_DIR / "feature_transformers.joblib")
print(f"  Saved: {OUTPUT_DIR}/feature_transformers.joblib")

print("  Training base category_model (LinearSVC) …")
category_model = LinearSVC(**SVC_PARAMS)
category_model.fit(X_full, full_clean["Category"])
joblib.dump(category_model, OUTPUT_DIR / "category_model.joblib")
print(f"  Saved: category_model.joblib ({len(category_model.classes_)} classes)")

print("  Training category_model_calibrated (CalibratedClassifierCV cv=3) …")
cat_counts = full_clean["Category"].value_counts()
calibratable_cats = cat_counts[cat_counts >= 3].index.tolist()
rare_cats = cat_counts[cat_counts < 3].index.tolist()

if rare_cats:
    print(f"  WARNING: {len(rare_cats)} categories have <3 samples — excluded from calibration:")
    for c in rare_cats:
        print(f"    {c}: {cat_counts[c]} samples")

calib_mask = full_clean["Category"].isin(calibratable_cats).values
print(f"  Calibratable categories: {len(calibratable_cats)}/{len(cat_counts)}")

base_svc = LinearSVC(**SVC_PARAMS)
calibrated = CalibratedClassifierCV(base_svc, cv=3, method="sigmoid")
calibrated.fit(X_full[calib_mask], full_clean.loc[calib_mask, "Category"])
joblib.dump(calibrated, OUTPUT_DIR / "category_model_calibrated.joblib")
print("  Saved: category_model_calibrated.joblib")

# ── Evaluation Readout ────────────────────────────────────────────────────────

banner("COMPLETE")
print(f"  Ultimate Category models successfully saved to:\n  {OUTPUT_DIR}")
