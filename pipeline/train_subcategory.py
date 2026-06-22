"""Stage 5c — train one Subcategory classifier per non-flat category (script-14 approach).

Same pattern as train_service.py / train_category.py:
- 3 separate TF-IDF vectorizers: ProfileFullName (5k) + Subject (15k) + Symptom (35k)
- LinearSVC with class_weight=balanced
- CalibratedClassifierCV for probability / abstention scoring

Saves per category:
    models/subcategory/<safe_category_name>.joblib
        {
          "model":          LinearSVC            (predict),
          "calibrated":     CalibratedClassifierCV (predict_proba / confidence),
          "transformers":   dict of 3 TfidfVectorizers,
          "classes":        [...],
          "test_macro_f1":  float,
          "excluded_subcats": [...],
        }

    models/subcategory/_index.joblib
        {
          "trained_categories": [...],
          "flat_categories":    [...],
          "skipped_categories": [...],
        }

Usage:
    python train_subcategory.py
"""
import logging
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).parent))

from clean import clean_text, load_and_clean
from config import (
    MIN_SUBCAT_SUPPORT,
    RANDOM_SEED,
    SUBCAT_MODEL_DIR,
    TARGET_CATEGORY,
    TARGET_SUBCAT,
    TEST_SIZE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SVC_PARAMS = dict(class_weight="balanced", max_iter=10_000, dual=False, random_state=RANDOM_SEED)


def _safe_name(category: str) -> str:
    return re.sub(r"[^\w\-]", "_", category)


def _make_vectorizers() -> dict:
    return {
        "tfidf_name":    TfidfVectorizer(max_features=5_000,  ngram_range=(1, 2), sublinear_tf=True),
        "tfidf_subject": TfidfVectorizer(max_features=15_000, ngram_range=(1, 2), sublinear_tf=True),
        "tfidf_symptom": TfidfVectorizer(max_features=35_000, ngram_range=(1, 2), sublinear_tf=True),
    }


def _build_X(df: pd.DataFrame, vectorizers: dict, fit: bool):
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


def train() -> None:
    df, flat_cats = load_and_clean("subcategory")

    SUBCAT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    trained: list[str] = []
    skipped: list[str] = []

    all_categories = sorted(df[TARGET_CATEGORY].unique())
    log.info("Non-flat categories to train: %d", len(all_categories))

    for cat in all_categories:
        log.info("\n══ Category: %s ══════════════════════════════════", cat)
        sub = df[df[TARGET_CATEGORY] == cat].copy()

        # Apply support floor per subcategory
        counts = sub[TARGET_SUBCAT].value_counts()
        low_support = counts[counts < MIN_SUBCAT_SUPPORT].index.tolist()
        excluded_subcats = low_support
        if low_support:
            log.info(
                "  Excluding %d subcategories below support=%d: %s",
                len(low_support), MIN_SUBCAT_SUPPORT, low_support,
            )
        sub = sub[~sub[TARGET_SUBCAT].isin(low_support)].copy()

        y = sub[TARGET_SUBCAT].values
        n_classes = len(set(y))

        if n_classes < 2:
            log.warning("  Only %d usable subcategory after support filter — skipping.", n_classes)
            skipped.append(cat)
            continue

        df_tr, df_te, y_tr, y_te = train_test_split(
            sub, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED,
        )
        log.info("  Subcategories: %d  Train: %d  Test: %d", n_classes, len(df_tr), len(df_te))

        # ── Held-out evaluation ───────────────────────────────────────────────
        eval_vecs = _make_vectorizers()
        X_tr = _build_X(df_tr, eval_vecs, fit=True)
        X_te = _build_X(df_te, eval_vecs, fit=False)

        eval_model = LinearSVC(**SVC_PARAMS)
        eval_model.fit(X_tr, y_tr)
        y_pred = eval_model.predict(X_te)
        macro_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
        log.info(
            "  Test Macro-F1: %.4f  Weighted-F1: %.4f  Acc: %.4f",
            macro_f1,
            f1_score(y_te, y_pred, average="weighted", zero_division=0),
            (y_pred == y_te).mean(),
        )
        log.info("\n%s", classification_report(y_te, y_pred, zero_division=0))

        # ── Production model — retrain on full subset ─────────────────────────
        prod_vecs = _make_vectorizers()
        X_full = _build_X(sub, prod_vecs, fit=True)

        prod_model = LinearSVC(**SVC_PARAMS)
        prod_model.fit(X_full, y)

        # ── Calibrated model ──────────────────────────────────────────────────
        # cv=3 needs at least 3 samples per class; fall back to cv=2 if needed
        min_class_count = counts[counts >= MIN_SUBCAT_SUPPORT].min()
        cv = 3 if min_class_count >= 3 else 2
        calibrated = CalibratedClassifierCV(LinearSVC(**SVC_PARAMS), cv=cv, method="sigmoid")
        calibrated.fit(X_full, y)

        artefact = {
            "model":            prod_model,
            "calibrated":       calibrated,
            "transformers":     prod_vecs,
            "classes":          prod_model.classes_.tolist(),
            "test_macro_f1":    macro_f1,
            "excluded_subcats": excluded_subcats,
        }
        out_path = SUBCAT_MODEL_DIR / f"{_safe_name(cat)}.joblib"
        joblib.dump(artefact, out_path)
        log.info("  Saved → %s", out_path)
        trained.append(cat)

    index = {
        "trained_categories": trained,
        "flat_categories":    sorted(flat_cats),
        "skipped_categories": skipped,
    }
    joblib.dump(index, SUBCAT_MODEL_DIR / "_index.joblib")
    log.info(
        "\nDone. Trained: %d  Skipped: %d  Flat (rule-based): %d",
        len(trained), len(skipped), len(flat_cats),
    )


if __name__ == "__main__":
    train()
