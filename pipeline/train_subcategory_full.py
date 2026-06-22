"""Train subcategory models WITHOUT the MIN_SUBCAT_SUPPORT floor.

Identical to train_subcategory.py except:
  - Every subcategory with >= 2 samples is included (no 50-example floor).
  - Singleton subcategories (1 sample) are added to train only — they cannot
    appear in a stratified split but the model will still have seen them.
  - Models are saved to models/subcategory_full/ so they coexist with the
    floored models for comparison.

Purpose: understand how rare classes affect confidence and whether the model
can still make useful predictions on them (vs abstaining).

Usage:
    python train_subcategory_full.py

Evaluate afterwards with:
    python evaluate_confidence.py
"""
import logging
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from clean import load_and_clean
from config import (
    MODEL_DIR,
    RANDOM_SEED,
    SUBCAT_FULL_MODEL_DIR,
    TARGET_CATEGORY,
    TARGET_SERVICE,
    TARGET_SUBCAT,
    TEST_SIZE,
    TFIDF_MAX_FEATURES,
)
from features import build_features, check_sender_leakage

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MIN_SPLIT_COUNT = 2  # minimum samples per class to include in stratified split


def _safe_name(category: str) -> str:
    return re.sub(r"[^\w\-]", "_", category)


def _split_with_singletons(
    sub: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Split df into train/test, handling rare classes gracefully.

    Classes with < MIN_SPLIT_COUNT samples go into train only (can't stratify).
    Returns (df_train, df_test, y_train, y_test).
    """
    counts = sub[TARGET_SUBCAT].value_counts()
    too_rare = counts[counts < MIN_SPLIT_COUNT].index
    rare_rows = sub[sub[TARGET_SUBCAT].isin(too_rare)]
    splittable = sub[~sub[TARGET_SUBCAT].isin(too_rare)]

    n_classes = splittable[TARGET_SUBCAT].nunique()
    if n_classes < 2:
        # Nothing to split — everything goes to train
        return sub, pd.DataFrame(columns=sub.columns), sub[TARGET_SUBCAT].values, np.array([])

    # sklearn stratified split requires test_size >= n_classes.
    # If the dataset is too small to satisfy this, put everything in train.
    test_n = max(1, int(len(splittable) * TEST_SIZE))
    if test_n < n_classes:
        log.info(
            "  Dataset too small to stratify (test_size=%d < n_classes=%d) — train-only.",
            test_n, n_classes,
        )
        return sub, pd.DataFrame(columns=sub.columns), sub[TARGET_SUBCAT].values, np.array([])

    y_split = splittable[TARGET_SUBCAT].values
    df_tr_split, df_te, y_tr_split, y_te = train_test_split(
        splittable, y_split,
        test_size=TEST_SIZE,
        stratify=y_split,
        random_state=RANDOM_SEED,
    )
    # Rare rows always go to train
    df_tr = pd.concat([df_tr_split, rare_rows], ignore_index=True)
    y_tr = df_tr[TARGET_SUBCAT].values

    if len(too_rare) > 0:
        log.info(
            "  %d singleton subcategories added to train only (not splittable): %s",
            len(too_rare), too_rare.tolist(),
        )

    return df_tr, df_te, y_tr, y_te


def train() -> None:
    df, flat_cats = load_and_clean("subcategory")

    SUBCAT_FULL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    trained: list[str] = []
    skipped: list[str] = []
    singleton_only: list[str] = []

    all_categories = sorted(dlides .f[TARGET_CATEGORY].unique())
    log.info("Non-flat categories to train (full, no support floor): %d", len(all_categories))

    for cat in all_categories:
        out_path = SUBCAT_FULL_MODEL_DIR / f"{_safe_name(cat)}.joblib"
        if out_path.exists():
            log.info("Skipping %s — model already saved.", cat)
            trained.append(cat)
            continue

        log.info("\n══ Category: %s ══════════════════════════════════", cat)
        sub = df[df[TARGET_CATEGORY] == cat].copy()

        n_subcats = sub[TARGET_SUBCAT].nunique()
        if n_subcats < 2:
            log.warning("  Only %d subcategory — skipping.", n_subcats)
            skipped.append(cat)
            continue

        counts = sub[TARGET_SUBCAT].value_counts()
        rare = counts[counts < MIN_SPLIT_COUNT]
        if len(rare) > 0:
            log.info(
                "  %d subcategories with < %d samples (train-only): %s",
                len(rare), MIN_SPLIT_COUNT, rare.index.tolist(),
            )

        df_tr, df_te, y_tr, y_te = _split_with_singletons(sub)

        has_test = len(df_te) > 0
        log.info(
            "  Total subcategories: %d  Train: %d  Test: %d",
            n_subcats, len(df_tr), len(df_te),
        )

        X_tr, X_te_or_dummy, feat_pipeline = build_features(
            df_tr,
            df_te if has_test else df_tr,  # build_features needs a non-empty test frame
            TFIDF_MAX_FEATURES,
            purpose="subcategory",
        )
        X_te = X_te_or_dummy if has_test else None

        log.info("  X_train: %s  X_test: %s", X_tr.shape, X_te_or_dummy.shape if has_test else "(none)")

        if has_test:
            check_sender_leakage(df_tr, df_te, target_col=TARGET_SUBCAT)

        model = LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_SEED,
            C=1.0,
            n_jobs=-1,
            class_weight="balanced",
        )
        model.fit(X_tr, y_tr)

        if has_test and len(y_te) > 0:
            y_pred = model.predict(X_te)
            proba = model.predict_proba(X_te)
            max_conf = proba.max(axis=1)

            macro_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
            log.info(
                "  Test Macro-F1: %.4f  Weighted-F1: %.4f  Acc: %.4f  Avg-conf: %.3f",
                macro_f1,
                f1_score(y_te, y_pred, average="weighted", zero_division=0),
                (y_pred == y_te).mean(),
                max_conf.mean(),
            )
            log.info("\n%s", classification_report(y_te, y_pred, zero_division=0))
        else:
            macro_f1 = float("nan")
            log.info("  No test set (all rows are singletons or too rare to split)")
            singleton_only.append(cat)

        artefact = {
            "model": model,
            "feature_pipeline": feat_pipeline,
            "classes": model.classes_.tolist(),
            "test_macro_f1": macro_f1,
            "max_features": TFIDF_MAX_FEATURES,
            "excluded_subcats": [],  # nothing excluded — full model
            "rare_subcats": rare.index.tolist() if len(rare) > 0 else [],
        }
        joblib.dump(artefact, out_path)
        log.info("  Saved → %s", out_path)
        trained.append(cat)

    index = {
        "trained_categories": trained,
        "flat_categories": sorted(flat_cats),
        "skipped_categories": skipped,
        "singleton_only_categories": singleton_only,
    }
    joblib.dump(index, SUBCAT_FULL_MODEL_DIR / "_index.joblib")
    log.info(
        "\nDone. Trained: %d  Skipped: %d  Singleton-only (no test): %d  Flat: %d",
        len(trained), len(skipped), len(singleton_only), len(flat_cats),
    )


if __name__ == "__main__":
    train()
