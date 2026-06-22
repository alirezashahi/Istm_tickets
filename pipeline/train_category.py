"""Stage 5b — train single global Category classifier (script-14 approach).

Single LinearSVC across all categories with:
- Aggressive noise removal (cms / altro / other / z-other on both Category and Subcategory)
- 3 separate TF-IDF vectorizers: ProfileFullName (5k) + Subject (15k) + Symptom (35k)
- Calibrated probability variant for confidence scoring
"""
import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).parent))

from clean import clean_text, load_and_clean
from config import (
    CATEGORY_MODEL_CALIBRATED_PATH,
    CATEGORY_MODEL_PATH,
    CATEGORY_TRANSFORMERS_PATH,
    MODEL_DIR,
    RANDOM_SEED,
    TARGET_CATEGORY,
    TARGET_SUBCAT,
    TEST_SIZE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SVC_PARAMS = dict(class_weight="balanced", max_iter=10_000, dual=False, random_state=RANDOM_SEED)
NOISE_PATTERN = r"(?i)(cms|altro|other|z-other)"


def _drop_noise(df: pd.DataFrame) -> pd.DataFrame:
    mask_cat = df[TARGET_CATEGORY].str.contains(NOISE_PATTERN, regex=True, na=False)
    mask_sub = df[TARGET_SUBCAT].str.contains(NOISE_PATTERN, regex=True, na=False)
    kept = df[~(mask_cat | mask_sub)].copy()
    removed = len(df) - len(kept)
    log.info("  Noise filter removed %d rows → %d remaining", removed, len(kept))
    return kept


def _build_X(df: pd.DataFrame, vectorizers: dict, fit: bool) -> any:
    name_col = df["ProfileFullName"].fillna("").values
    subj_col = df["Subject"].map(clean_text).values
    symp_col = df["Symptom"].map(clean_text).values

    if fit:
        X_name = vectorizers["tfidf_name"].fit_transform(name_col)
        X_subj = vectorizers["tfidf_subject"].fit_transform(subj_col)
        X_symp = vectorizers["tfidf_symptom"].fit_transform(symp_col)
    else:
        X_name = vectorizers["tfidf_name"].transform(name_col)
        X_subj = vectorizers["tfidf_subject"].transform(subj_col)
        X_symp = vectorizers["tfidf_symptom"].transform(symp_col)

    return hstack([X_name, X_subj, X_symp])


def _make_vectorizers() -> dict:
    return {
        "tfidf_name":    TfidfVectorizer(max_features=5_000,  ngram_range=(1, 2), sublinear_tf=True),
        "tfidf_subject": TfidfVectorizer(max_features=15_000, ngram_range=(1, 2), sublinear_tf=True),
        "tfidf_symptom": TfidfVectorizer(max_features=35_000, ngram_range=(1, 2), sublinear_tf=True),
    }


def train() -> None:
    # ── Load & clean ──────────────────────────────────────────────────────────
    df, _ = load_and_clean("category")
    df = _drop_noise(df)

    # Drop categories with fewer samples than needed for a stratified split
    min_count = max(2, int(np.ceil(1 / TEST_SIZE)))
    counts = df[TARGET_CATEGORY].value_counts()
    too_few = counts[counts < min_count].index.tolist()
    if too_few:
        log.warning("Dropping %d categories with < %d samples: %s", len(too_few), min_count, too_few)
        df = df[~df[TARGET_CATEGORY].isin(too_few)].copy()

    log.info("Final training data: %d rows, %d categories", len(df), df[TARGET_CATEGORY].nunique())

    # ── Held-out evaluation split ─────────────────────────────────────────────
    y = df[TARGET_CATEGORY].values
    df_tr, df_te, y_tr, y_te = train_test_split(
        df, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    log.info("Train: %d  Test: %d", len(df_tr), len(df_te))

    eval_vecs = _make_vectorizers()
    X_tr = _build_X(df_tr, eval_vecs, fit=True)
    X_te = _build_X(df_te, eval_vecs, fit=False)
    log.info("Feature matrix: train=%s  test=%s", X_tr.shape, X_te.shape)

    log.info("Fitting eval LinearSVC…")
    eval_model = LinearSVC(**SVC_PARAMS)
    eval_model.fit(X_tr, y_tr)

    y_pred = eval_model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    log.info("\n[HELD-OUT EVALUATION] Accuracy: %.4f (%.2f%%)", acc, acc * 100)
    log.info("\n%s", classification_report(y_te, y_pred, zero_division=0))

    cat_labels = sorted(set(y_te))
    cm = confusion_matrix(y_te, y_pred, labels=cat_labels)
    fig, ax = plt.subplots(figsize=(28, 24))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d",
                xticklabels=cat_labels, yticklabels=cat_labels, ax=ax)
    ax.set_title(f"Category Confusion Matrix — Accuracy {acc*100:.2f}%", fontsize=16)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    cm_path = MODEL_DIR / "category_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    log.info("Confusion matrix saved → %s", cm_path)

    # ── Production model — retrain on full data ───────────────────────────────
    log.info("Retraining on full clean data (%d rows)…", len(df))
    prod_vecs = _make_vectorizers()
    X_full = _build_X(df, prod_vecs, fit=True)

    log.info("Fitting production LinearSVC…")
    prod_model = LinearSVC(**SVC_PARAMS)
    prod_model.fit(X_full, y)

    # ── Calibrated model ──────────────────────────────────────────────────────
    cat_counts = df[TARGET_CATEGORY].value_counts()
    calibratable = cat_counts[cat_counts >= 3].index.tolist()
    rare = cat_counts[cat_counts < 3].index.tolist()
    if rare:
        log.warning("Excluding %d categories with < 3 samples from calibration: %s", len(rare), rare)

    calib_mask = df[TARGET_CATEGORY].isin(calibratable).values
    log.info("Fitting calibrated model (%d/%d categories)…", len(calibratable), len(cat_counts))
    calibrated = CalibratedClassifierCV(LinearSVC(**SVC_PARAMS), cv=3, method="sigmoid")
    calibrated.fit(X_full[calib_mask], df.loc[calib_mask, TARGET_CATEGORY])

    # ── Save artefacts ────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(prod_model, CATEGORY_MODEL_PATH)
    log.info("Saved → %s  (%d classes)", CATEGORY_MODEL_PATH, len(prod_model.classes_))
    joblib.dump(calibrated, CATEGORY_MODEL_CALIBRATED_PATH)
    log.info("Saved → %s", CATEGORY_MODEL_CALIBRATED_PATH)
    joblib.dump(prod_vecs, CATEGORY_TRANSFORMERS_PATH)
    log.info("Saved → %s", CATEGORY_TRANSFORMERS_PATH)


if __name__ == "__main__":
    train()
